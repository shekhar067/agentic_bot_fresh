# app/simulation_engine.py
from datetime import datetime, time
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Callable
from pathlib import Path
import calendar
import sys
from collections import Counter # MODIFIED (2025-05-09): Added for exit reason counting
from typing import Dict, List, Optional, Tuple

from app.agentic_core import RuleBasedAgent
from app.config import config
from app.performance_logger_mongo import log_backtest_run_results
from app.utils.expiry_utils import get_expiry_type, is_expiry_day as util_is_expiry_day, get_expiry_date_for_week_of

engine_logger = logging.getLogger("SimulationEngine")
if not engine_logger.hasHandlers():
    log_level_sim = getattr(config, "LOG_LEVEL", "INFO")
    log_format_sim = getattr(config, "LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    engine_logger.addHandler(logging.StreamHandler(sys.stdout))
    engine_logger.setLevel(log_level_sim)
engine_logger.propagate = False


class SimpleBacktester:
    """
    A backtesting engine that can run simulations using either a predefined
    strategy function or a more complex RuleBasedAgent.
    Logs detailed trade traces and summary performance to MongoDB via performance_logger_mongo.
    """
    def __init__(self,
                 agent: Optional[RuleBasedAgent] = None,
                 strategy_func: Optional[Callable[[pd.Series, Optional[pd.DataFrame]], str]] = None,
                 strategy_name: str = "UnnamedStrategy"):

        if agent and strategy_func:
            raise ValueError("Provide either an agent or a strategy_func, not both.")
        if not agent and not strategy_func:
            raise ValueError("Must provide either an agent or a strategy_func.")

        if agent and not isinstance(agent, RuleBasedAgent):
             raise TypeError("Agent must be an instance of RuleBasedAgent.")
        if strategy_func and not callable(strategy_func):
             raise TypeError("strategy_func must be callable.")

        self.agent = agent
        self.strategy_func = strategy_func
        self.strategy_name = strategy_name
        if agent and (strategy_name == "UnnamedStrategy" or not strategy_name) :
            self.strategy_name = getattr(agent, 'name', 'RuleBasedAgentRun')
        elif not strategy_name or strategy_name == "UnnamedStrategy":
            self.strategy_name = "DefaultSingleStrategyRun"

        self.results = None
        self.sim_logger = None
        self.log_file_path = None

        self.default_sl_mult = config.DEFAULT_SL_ATR_MULT
        self.default_tp_mult = config.DEFAULT_TP_ATR_MULT
        self.default_tsl_mult = getattr(config, "DEFAULT_TSL_ATR_MULT", 1.5)

        log_msg_init_mode = f"AGENT mode for Agent: '{self.strategy_name}'" if self.agent else f"SINGLE STRATEGY mode for '{self.strategy_name}'"
        engine_logger.info(
            f"SimpleBacktester initialized in {log_msg_init_mode}. "
            f"Initial default risk: SL={self.default_sl_mult}, TP={self.default_tp_mult}, TSL={self.default_tsl_mult}"
        )

    def _setup_simulation_logging(self, base_log_dir: Path, timeframe_suffix: str):
        # MODIFIED (2025-05-09): No changes here from simulation_engine_debug_exits_20250509
        try:
            self.log_run_folder = base_log_dir
            self.log_run_folder.mkdir(parents=True, exist_ok=True)

            log_file_name = f"{self.strategy_name}__{timeframe_suffix}.log"
            self.log_file_path = self.log_run_folder / log_file_name

            logger_name = f"SimTrace_{self.strategy_name}_{timeframe_suffix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            self.sim_logger = logging.getLogger(logger_name)
            self.sim_logger.setLevel(logging.DEBUG)
            self.sim_logger.propagate = False

            for handler in self.sim_logger.handlers[:]:
                self.sim_logger.removeHandler(handler)
                handler.close()

            fh = logging.FileHandler(self.log_file_path, mode='w')
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter(getattr(config, "LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            fh.setFormatter(formatter)
            self.sim_logger.addHandler(fh)

            self.sim_logger.info(f"✅ Simulation logger initialized for Strategy: '{self.strategy_name}', Suffix: '{timeframe_suffix}'")
            self.sim_logger.info(f"✅ Trace logs: {self.log_file_path}")

        except Exception as e:
            engine_logger.error(f"❌ Failed to set up simulation logger for {self.strategy_name} (Suffix: {timeframe_suffix}) at {base_log_dir}: {e}", exc_info=True)
            self.sim_logger = logging.getLogger(f"SimTrace_Error_{datetime.now().strftime('%Y%m%d%H%M%S%f')}")
            self.sim_logger.addHandler(logging.NullHandler())


    def run_simulation(self, df: pd.DataFrame, log_dir: Path, timeframe: str,
                       run_id: Optional[str] = None,
                       optuna_trial_params: Optional[Dict] = None,
                       optuna_study_name: Optional[str] = None,
                       optuna_trial_number: Optional[int] = None
                       ) -> Dict:
        # MODIFIED (2025-05-09): No changes here from simulation_engine_debug_exits_20250509
        log_file_suffix = f"{timeframe}"
        if optuna_study_name and optuna_trial_number is not None:
            log_file_suffix += f"_trial_{optuna_trial_number}"
        elif run_id:
             log_file_suffix += f"_run_{run_id.split('_')[-1]}"


        self._setup_simulation_logging(log_dir, log_file_suffix)
        self.sim_logger.info(f"--- run_simulation START for Strategy: '{self.strategy_name}', Timeframe: '{timeframe}', Run ID: {run_id} ---")
        if optuna_trial_params:
            self.sim_logger.info(f"Optuna Trial Parameters: {optuna_trial_params}")
            self.sim_logger.info(f"Optuna Study: {optuna_study_name}, Trial: {optuna_trial_number}")
        else:
            self.sim_logger.info(f"Running with instance risk parameters: SL={self.default_sl_mult}, TP={self.default_tp_mult}, TSL={self.default_tsl_mult}")

        if df.empty:
            self.sim_logger.error("Input DataFrame 'df' is empty. Cannot run simulation.")
            self._close_sim_logger()
            return {"total_pnl": 0, "trade_count": 0, "win_rate": 0, "trades_details": [], "exit_reasons": {}, "error": "Input DataFrame empty", "performance_score": -np.inf}

        if not isinstance(df.index, pd.DatetimeIndex):
             self.sim_logger.warning("DataFrame index is not a DatetimeIndex. Attempting conversion.")
             try:
                 df.index = pd.to_datetime(df.index)
             except Exception as e_idx:
                 self.sim_logger.error(f"Failed to convert index to DatetimeIndex: {e_idx}. Sim may fail.")
                 self._close_sim_logger()
                 return {"total_pnl": 0, "trade_count": 0, "win_rate": 0, "trades_details": [], "exit_reasons": {}, "error": "Invalid DataFrame Index", "performance_score": -np.inf}

        self.sim_logger.debug(f"🔍 DataFrame shape for simulation: {df.shape}")
        if not df.empty:
            self.sim_logger.debug(f"🔍 Columns in DataFrame: {df.columns.tolist()}")

        # MODIFIED (2025-05-09): This incremental counter is still useful for debugging during the loop
        # The final summary will be calculated from trades_details_list using collections.Counter
        incremental_exit_reason_counter = {"SL": 0, "TP": 0, "REV": 0, "TSL": 0, "EOD": 0, "MANUAL": 0, "OOR": 0, "ERROR": 0}
        position = 0
        entry_price = np.nan
        stop_loss = np.nan
        take_profit = np.nan
        cumulative_pnl = 0.0
        trade_count = 0
        winning_trades = 0
        trades_details_list = []
        prev_trade_entry_time = pd.NaT
        entry_strategy_name = self.strategy_name
        max_price_since_entry = -np.inf
        min_price_since_entry = np.inf

        sim_col_prefix = config.SIM_DF_COL_PREFIX

        for col_name in [f'{sim_col_prefix}position', f'{sim_col_prefix}trade_pnl',
                         f'{sim_col_prefix}cumulative_pnl', f'{sim_col_prefix}signal',
                         f'{sim_col_prefix}sl_level', f'{sim_col_prefix}tp_level', f'{sim_col_prefix}tsl_level']:
            if "pnl" in col_name or "level" in col_name : df[col_name] = np.nan
            elif "pos" in col_name: df[col_name] = 0
            else: df[col_name] = 'hold'

        atr_col = None
        atr_candidates = [col for col in df.columns if 'atr' in col.lower()]
        if atr_candidates:
            atr_col = min(atr_candidates, key=len)

        if not atr_col or atr_col not in df.columns:
            error_msg = f"❌ CRITICAL: ATR column (tried: {atr_candidates}, selected: {atr_col}) not found. Cannot apply ATR-based SL/TP/TSL."
            self.sim_logger.error(error_msg)
            self.sim_logger.error(f"Available columns: {df.columns.tolist()}")
            self._close_sim_logger()
            return {"total_pnl": 0, "trade_count": 0, "win_rate": 0, "trades_details": [], 
                    "exit_reasons": incremental_exit_reason_counter, # Return the counter even on error
                    "error": "ATR Column Missing", "performance_score": -np.inf}

        self.sim_logger.debug(f"✅ Using ATR column: '{atr_col}' for SL/TP/TSL calculations.")

        self.sim_logger.info("--- Starting main simulation bar-by-bar loop ---")
        for i in range(1, len(df)):
            current_idx = df.index[i]
            current_row = df.iloc[i]
            data_history_for_decision = df

            current_price = current_row.get('close', np.nan)
            low_price = current_row.get('low', np.nan)
            high_price = current_row.get('high', np.nan)
            current_atr = current_row.get(atr_col, np.nan)

            if pd.isna(current_price) or pd.isna(low_price) or pd.isna(high_price):
                self.sim_logger.warning(f"Skipping bar {current_idx} due to NaN in OHLC.")
                df.loc[current_idx, f'{sim_col_prefix}position'] = position
                df.loc[current_idx, f'{sim_col_prefix}cumulative_pnl'] = cumulative_pnl
                continue

            potential_signal = 'hold'
            sl_atr_mult_bar = self.default_sl_mult
            tp_atr_mult_bar = self.default_tp_mult
            tsl_atr_mult_bar = self.default_tsl_mult
            current_signal_strategy_name = self.strategy_name

            if self.agent:
                try:
                    decision_output = self.agent.decide(current_row, data_history=data_history_for_decision)
                    potential_signal = decision_output[0]
                    sl_atr_mult_bar = decision_output[1] if len(decision_output) > 1 and pd.notna(decision_output[1]) and decision_output[1] > 0 else self.default_sl_mult
                    tp_atr_mult_bar = decision_output[2] if len(decision_output) > 2 and pd.notna(decision_output[2]) and decision_output[2] > 0 else self.default_tp_mult
                    tsl_atr_mult_bar = decision_output[3] if len(decision_output) > 3 and pd.notna(decision_output[3]) and decision_output[3] > 0 else self.default_tsl_mult
                    current_signal_strategy_name = decision_output[4] if len(decision_output) > 4 and decision_output[4] else getattr(self.agent, 'name', 'AgentDecision')
                except Exception as agent_e:
                    self.sim_logger.error(f"Agent failed to decide at {current_idx}: {agent_e}", exc_info=True)
                    potential_signal = 'hold'
            elif self.strategy_func:
                try:
                    potential_signal = self.strategy_func(current_row, data_history=data_history_for_decision)
                    sl_atr_mult_bar = self.default_sl_mult
                    tp_atr_mult_bar = self.default_tp_mult
                    tsl_atr_mult_bar = self.default_tsl_mult
                except Exception as strat_e:
                     self.sim_logger.error(f"Strategy '{self.strategy_name}' failed at {current_idx}: {strat_e}", exc_info=True)
                     potential_signal = 'hold'

            exit_triggered = False
            exit_price = np.nan
            exit_reason = "N/A"
            df.loc[current_idx, f'{sim_col_prefix}tsl_level'] = np.nan

            if position != 0:
                #self.sim_logger.debug(f"Bar {current_idx}: In position {position}. Checking exits. L={low_price}, H={high_price}, C={current_price}, ATR={current_atr:.2f if pd.notna(current_atr) else 'NaN'}")
                atr_str = f"{current_atr:.2f}" if pd.notna(current_atr) else "NaN"
                self.sim_logger.debug(f"Bar {current_idx}: In position {position}. Checking exits. L={low_price}, H={high_price}, C={current_price}, ATR={atr_str}")

                current_tsl_level = np.nan
                if pd.notna(current_atr) and current_atr > 0 and pd.notna(tsl_atr_mult_bar) and tsl_atr_mult_bar > 0:
                    trailing_sl_buffer = current_atr * tsl_atr_mult_bar
                    if position == 1:
                        max_price_since_entry = max(max_price_since_entry, high_price)
                        current_tsl_level = max_price_since_entry - trailing_sl_buffer
                        df.loc[current_idx, f'{sim_col_prefix}tsl_level'] = current_tsl_level
                        self.sim_logger.debug(f"  TSL Check (Long): MaxPrice={max_price_since_entry:.2f}, TSL_Level={current_tsl_level:.2f}, Low={low_price:.2f}, Buffer={trailing_sl_buffer:.2f}")
                        if low_price <= current_tsl_level:
                            exit_reason = "TSL"; exit_price = current_tsl_level
                            self.sim_logger.debug(f"    TSL (Long) TRIGGERED at {exit_price:.2f}")
                    elif position == -1:
                        min_price_since_entry = min(min_price_since_entry, low_price)
                        current_tsl_level = min_price_since_entry + trailing_sl_buffer
                        df.loc[current_idx, f'{sim_col_prefix}tsl_level'] = current_tsl_level
                        self.sim_logger.debug(f"  TSL Check (Short): MinPrice={min_price_since_entry:.2f}, TSL_Level={current_tsl_level:.2f}, High={high_price:.2f}, Buffer={trailing_sl_buffer:.2f}")
                        if high_price >= current_tsl_level:
                            exit_reason = "TSL"; exit_price = current_tsl_level
                            self.sim_logger.debug(f"    TSL (Short) TRIGGERED at {exit_price:.2f}")
                else:
                    self.sim_logger.debug(f"  TSL Check: Skipped (ATR NaN/zero, or TSL mult NaN/zero/None. ATR={current_atr}, TSL_Mult={tsl_atr_mult_bar})")

                if exit_reason == "N/A" and pd.notna(stop_loss):
                    self.sim_logger.debug(f"  SL Check: SL_Level={stop_loss:.2f}, Low={low_price:.2f}, High={high_price:.2f}")
                    if position == 1 and low_price <= stop_loss:
                        exit_reason = "SL"; exit_price = stop_loss
                        self.sim_logger.debug(f"    SL (Long) TRIGGERED at {exit_price:.2f}")
                    elif position == -1 and high_price >= stop_loss:
                        exit_reason = "SL"; exit_price = stop_loss
                        self.sim_logger.debug(f"    SL (Short) TRIGGERED at {exit_price:.2f}")
                elif exit_reason == "N/A": # MODIFIED (2025-05-09): Added this else if for clarity
                     self.sim_logger.debug(f"  SL Check: Skipped (stop_loss is NaN or TSL already triggered)")

                if exit_reason == "N/A" and pd.notna(take_profit):
                    self.sim_logger.debug(f"  TP Check: TP_Level={take_profit:.2f}, Low={low_price:.2f}, High={high_price:.2f}")
                    if position == 1 and high_price >= take_profit:
                        exit_reason = "TP"; exit_price = take_profit
                        self.sim_logger.debug(f"    TP (Long) TRIGGERED at {exit_price:.2f}")
                    elif position == -1 and low_price <= take_profit:
                        exit_reason = "TP"; exit_price = take_profit
                        self.sim_logger.debug(f"    TP (Short) TRIGGERED at {exit_price:.2f}")
                elif exit_reason == "N/A": # MODIFIED (2025-05-09): Added this else if
                     self.sim_logger.debug(f"  TP Check: Skipped (take_profit is NaN or prior exit triggered)")

                if exit_reason == "N/A":
                    self.sim_logger.debug(f"  REV Check: Position={position}, PotentialSignal='{potential_signal}'")
                    if position == 1 and potential_signal == 'sell_potential':
                        exit_reason = "REV"; exit_price = current_price
                        self.sim_logger.debug(f"    REV (Long to Sell) TRIGGERED at {exit_price:.2f}")
                    elif position == -1 and potential_signal == 'buy_potential':
                        exit_reason = "REV"; exit_price = current_price
                        self.sim_logger.debug(f"    REV (Short to Buy) TRIGGERED at {exit_price:.2f}")

                if exit_reason != "N/A":
                    exit_triggered = True
                    self.sim_logger.info(f"    Exit reason determined: {exit_reason}. Incrementing incremental_exit_reason_counter.") # Debug log
                    incremental_exit_reason_counter[exit_reason] += 1
                    exit_price = np.clip(exit_price, low_price, high_price)
                    self.sim_logger.info(f"EXIT: Pos={position}, Reason={exit_reason}, EntryP={entry_price:.2f}, ExitP={exit_price:.2f} at {current_idx}")

                    trade_pnl_gross = (exit_price - entry_price) * position
                    commission_pct = getattr(config, "COMMISSION_PCT", 0.0005)
                    slippage_pct = getattr(config, "SLIPPAGE_PCT", 0.0002)
                    entry_cost = abs(entry_price * (commission_pct + slippage_pct))
                    exit_cost = abs(exit_price * (commission_pct + slippage_pct))
                    trade_pnl_net = trade_pnl_gross - (entry_cost + exit_cost)

                    cumulative_pnl += trade_pnl_net
                    trade_count += 1
                    if trade_pnl_net > 0: winning_trades += 1

                    if pd.notna(prev_trade_entry_time):
                         trades_details_list.append({
                            'EntryTime': str(prev_trade_entry_time), 'ExitTime': str(current_idx),
                            'Position': 'long' if position == 1 else 'short',
                            'EntryPrice': round(entry_price, 4), 'ExitPrice': round(exit_price, 4),
                            'PnL_Net': round(trade_pnl_net, 4),
                            'PnL_Gross': round(trade_pnl_gross, 4),
                            'ExitReason': exit_reason, # This is key for the Counter method
                            'StrategyName': entry_strategy_name
                        })

                    df.loc[current_idx, f'{sim_col_prefix}trade_pnl'] = trade_pnl_net
                    df.loc[current_idx, f'{sim_col_prefix}position'] = 0
                    df.loc[current_idx, f'{sim_col_prefix}signal'] = f'exit_{exit_reason.lower()}'

                    position = 0; entry_price = stop_loss = take_profit = np.nan
                    prev_trade_entry_time = pd.NaT; entry_strategy_name = None
                    max_price_since_entry = -np.inf; min_price_since_entry = np.inf

            if not exit_triggered and position == 0 and potential_signal in ['buy_potential', 'sell_potential']:
                 if pd.notna(current_atr) and current_atr > 0 and \
                    pd.notna(sl_atr_mult_bar) and sl_atr_mult_bar > 0 and \
                    pd.notna(tp_atr_mult_bar) and tp_atr_mult_bar > 0:

                    position = 1 if potential_signal == 'buy_potential' else -1
                    entry_price = current_price
                    prev_trade_entry_time = current_idx
                    entry_strategy_name = current_signal_strategy_name

                    sl_distance = current_atr * sl_atr_mult_bar
                    tp_distance = current_atr * tp_atr_mult_bar
                    if position == 1:
                        stop_loss = entry_price - sl_distance
                        take_profit = entry_price + tp_distance
                    else:
                        stop_loss = entry_price + sl_distance
                        take_profit = entry_price - tp_distance

                    max_price_since_entry = entry_price
                    min_price_since_entry = entry_price

                    self.sim_logger.info(
                        f"ENTRY: Signal={potential_signal.split('_')[0].upper()}, Strategy='{entry_strategy_name}', "
                        f"Price={entry_price:.2f}, ATR={current_atr:.3f}, "
                        f"SL={stop_loss:.2f} (mult:{sl_atr_mult_bar:.2f}), "
                        f"TP={take_profit:.2f} (mult:{tp_atr_mult_bar:.2f}), "
                        f"TSL_Mult_Active={tsl_atr_mult_bar if pd.notna(tsl_atr_mult_bar) and tsl_atr_mult_bar > 0 else 'N/A'} at {current_idx}"
                    )
                    df.loc[current_idx, f'{sim_col_prefix}position'] = position
                    df.loc[current_idx, f'{sim_col_prefix}signal'] = potential_signal.split('_')[0]
                    df.loc[current_idx, f'{sim_col_prefix}sl_level'] = stop_loss
                    df.loc[current_idx, f'{sim_col_prefix}tp_level'] = take_profit
                 else:
                     self.sim_logger.warning(
                         f"Skipping entry signal '{potential_signal}' at {current_idx} due to invalid ATR ({current_atr if pd.notna(current_atr) else 'NaN'}) "
                         f"or SL/TP multipliers (SL={sl_atr_mult_bar}, TP={tp_atr_mult_bar})"
                     )
                     df.loc[current_idx, f'{sim_col_prefix}signal'] = 'hold_no_risk_params'

            df.loc[current_idx, f'{sim_col_prefix}cumulative_pnl'] = cumulative_pnl
            if df.loc[current_idx, f'{sim_col_prefix}position'] == 0 and position != 0 :
                 df.loc[current_idx, f'{sim_col_prefix}position'] = position
            elif position == 0 and df.loc[current_idx, f'{sim_col_prefix}position'] != 0:
                 pass
            elif position == 0:
                 df.loc[current_idx, f'{sim_col_prefix}position'] = 0

            if df.loc[current_idx, f'{sim_col_prefix}signal'] == 'hold' and potential_signal != 'hold':
                 df.loc[current_idx, f'{sim_col_prefix}signal'] = potential_signal

        self.sim_logger.info("--- Finished main simulation bar-by-bar loop ---")

        win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0.0
        self.sim_logger.info(f"--- Simulation Summary for '{self.strategy_name}' ({timeframe}) ---")
        self.sim_logger.info(f"Incrementally built exit_reason_counter: {incremental_exit_reason_counter}") # Log the incremental one
        self.sim_logger.info(f"Total Trades: {trade_count}, Win Rate: {win_rate:.2f}%, Final Net PnL: {cumulative_pnl:.2f}")

        final_results = self._log_and_prepare_final_results(
            df=df, original_timeframe=timeframe, cumulative_pnl=cumulative_pnl,
            trade_count=trade_count, win_rate=win_rate,
            # Pass the incrementally built counter here; it will be used if trades_details_list is empty,
            # otherwise, the Counter method will be preferred inside _log_and_prepare_final_results
            incremental_exit_reason_counter=incremental_exit_reason_counter, 
            trades_details_list=trades_details_list,
            sl_mult_used=self.default_sl_mult,
            tp_mult_used=self.default_tp_mult,
            tsl_mult_used=self.default_tsl_mult,
            run_id=run_id,
            optuna_study_name=optuna_study_name,
            optuna_trial_number=optuna_trial_number,
            optuna_trial_params=optuna_trial_params
        )

        self._close_sim_logger()
        return final_results

    def _log_and_prepare_final_results(self, df: pd.DataFrame, original_timeframe: str, cumulative_pnl: float,
                               trade_count: int, win_rate: float,
                               incremental_exit_reason_counter: Dict, # MODIFIED (2025-05-09): Received from main loop
                               trades_details_list: List[Dict],
                               sl_mult_used: float, tp_mult_used: float, tsl_mult_used: Optional[float],
                               run_id: Optional[str],
                               optuna_study_name: Optional[str],
                               optuna_trial_number: Optional[int],
                               optuna_trial_params: Optional[Dict] = None) -> Dict:
        try:
            first_valid_idx = df.first_valid_index()
            session_from_data, day_of_week_from_data, is_expiry_flag_from_data = "Unknown", "Unknown", False
            market_cond_from_data, vol_stat_from_data = "Unknown", "Unknown"
            current_run_symbol = optuna_trial_params.get("symbol", getattr(config, "DEFAULT_SYMBOL", "NIFTY")).upper() if optuna_trial_params else getattr(config, "DEFAULT_SYMBOL", "NIFTY").upper()
            current_run_market = optuna_trial_params.get("market", getattr(config, "DEFAULT_MARKET", "NSE")).upper() if optuna_trial_params else getattr(config, "DEFAULT_MARKET", "NSE").upper()

            if first_valid_idx and isinstance(first_valid_idx, pd.Timestamp):
                timestamp = first_valid_idx
                def infer_session(ts: pd.Timestamp) -> str:
                    if not isinstance(ts, pd.Timestamp): return "Unknown"
                    tm = ts.time()
                    if tm >= time(9, 15) and tm <= time(10, 59): return "Morning"
                    elif tm >= time(11, 0) and tm <= time(13, 29): return "Midday"
                    elif tm >= time(13, 30) and tm <= time(15, 30): return "Afternoon"
                    return "OffMarket"
                session_from_data = infer_session(timestamp)
                day_of_week_from_data = calendar.day_name[timestamp.weekday()]
                is_expiry_flag_from_data = util_is_expiry_day(current_run_symbol, timestamp, current_run_market)

            if not df.empty:
                market_cond_from_data = df['regime'].mode()[0] if 'regime' in df.columns and not df['regime'].mode().empty else df.iloc[0].get('regime', 'Unknown')
                vol_stat_from_data = df['volatility_status'].mode()[0] if 'volatility_status' in df.columns and not df['volatility_status'].mode().empty else df.iloc[0].get('volatility_status', 'Unknown')

            sim_col_prefix = config.SIM_DF_COL_PREFIX
            max_drawdown = 0.0
            if f"{sim_col_prefix}cumulative_pnl" in df.columns and not df[f"{sim_col_prefix}cumulative_pnl"].dropna().empty:
                cumulative_pnl_series = df[f"{sim_col_prefix}cumulative_pnl"].dropna()
                if not cumulative_pnl_series.empty:
                     max_drawdown = float((cumulative_pnl_series.cummax() - cumulative_pnl_series).max())

            avg_trade_duration_minutes = np.mean([(pd.to_datetime(t["ExitTime"]) - pd.to_datetime(t["EntryTime"])).total_seconds() / 60.0 for t in trades_details_list if "ExitTime" in t and "EntryTime" in t]) if trades_details_list else 0.0
            gross_pnl_sum = sum(t.get('PnL_Gross', 0) for t in trades_details_list if pd.notna(t.get('PnL_Gross')))
            total_profit = sum(t.get('PnL_Net', 0) for t in trades_details_list if t.get('PnL_Net', 0) > 0)
            total_loss = abs(sum(t.get('PnL_Net', 0) for t in trades_details_list if t.get('PnL_Net', 0) < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else np.inf if total_profit > 0 else 0

            # MODIFIED (2025-05-09): Use collections.Counter for final exit reason summary
            final_exit_reasons_summary = {}
            if trades_details_list:
                exit_reason_values_from_log = [
                    trade.get('ExitReason') for trade in trades_details_list if trade.get('ExitReason') and trade.get('ExitReason') != "N/A"
                ]
                calculated_exit_counts = Counter(exit_reason_values_from_log)
                final_exit_reasons_summary = {
                    key: calculated_exit_counts.get(key, 0)
                    for key in ["SL", "TP", "REV", "TSL", "EOD", "MANUAL", "OOR", "ERROR"]
                }
                self.sim_logger.info(f"Final exit reasons summary from trade_details_list: {final_exit_reasons_summary}")
            else: # Fallback if no trades in list, use the incremental one (should also be all zeros if no trades)
                final_exit_reasons_summary = incremental_exit_reason_counter
                self.sim_logger.warning(f"No trades in trades_details_list. Using incremental exit_reason_counter: {final_exit_reasons_summary}")


            pnl_for_score = float(cumulative_pnl)
            score_trade_count = int(trade_count)
            score_win_rate = float(win_rate) / 100.0
            score_max_drawdown = float(max_drawdown)

            if score_trade_count < getattr(config, "MIN_TRADES_FOR_RELIABLE_SCORE", 3):
                performance_score_calculated = -100.0
                self.sim_logger.warning(f"Low trade count ({score_trade_count}) for score calculation. Min required: {getattr(config, 'MIN_TRADES_FOR_RELIABLE_SCORE', 3)}. Assigning low score.")
            else:
                normalized_pnl = pnl_for_score / getattr(config, "EXPECTED_PNL_NORMALIZER", 1000)
                normalized_drawdown = score_max_drawdown / getattr(config, "EXPECTED_DRAWDOWN_NORMALIZER", 2000)
                trade_count_factor = np.log1p(score_trade_count) / np.log1p(getattr(config, "TARGET_TRADE_COUNT_FOR_SCORE", 20))
                trade_count_factor = min(trade_count_factor, 1.0)
                performance_score_calculated = (
                    0.5 * normalized_pnl + 0.3 * score_win_rate - 0.2 * normalized_drawdown
                ) * trade_count_factor * 100
            performance_score_final = round(float(performance_score_calculated), 4)
            self.sim_logger.info(f"Calculated Performance Score: {performance_score_final}")

            params_logged_for_run = {}
            if optuna_trial_params:
                params_logged_for_run.update(optuna_trial_params)
            else:
                params_logged_for_run.update({
                    "sl_mult": sl_mult_used, "tp_mult": tp_mult_used,
                    "tsl_mult": tsl_mult_used if tsl_mult_used is not None else "N/A"
                })
                if self.strategy_func and not self.agent:
                    params_logged_for_run["strategy_base_params"] = "default_or_fixed"

            log_backtest_run_results(
                strategy_name=self.strategy_name,
                parameters_used=params_logged_for_run,
                performance_metrics={
                    "total_pnl": float(cumulative_pnl), "gross_pnl": float(gross_pnl_sum),
                    "max_drawdown": float(score_max_drawdown),
                    "avg_trade_duration_minutes": float(round(avg_trade_duration_minutes, 2)),
                    "trade_count": int(score_trade_count), "win_rate": float(round(win_rate, 2)),
                    "profit_factor": float(round(profit_factor, 2)),
                },
                symbol=current_run_symbol,
                timeframe=original_timeframe,
                market_condition=market_cond_from_data,
                session=session_from_data,
                day=day_of_week_from_data,
                is_expiry=is_expiry_flag_from_data,
                performance_score=performance_score_final,
                optuna_study_name=optuna_study_name,
                optuna_trial_number=optuna_trial_number,
                run_id=run_id,
                custom_data={
                    "exit_reasons_summary": final_exit_reasons_summary, # MODIFIED (2025-05-09): Use the one from Counter
                    "volatility_status_from_data": vol_stat_from_data
                }
            )

            return_dict = {
                "total_pnl": float(cumulative_pnl), "trade_count": int(score_trade_count),
                "win_rate": float(round(win_rate, 2)), 
                "exit_reasons": final_exit_reasons_summary, # MODIFIED (2025-05-09): Return the Counter-based summary
                "performance_score": performance_score_final,
                "max_drawdown": float(score_max_drawdown), "profit_factor": float(round(profit_factor,2)),
                "indicator_config": params_logged_for_run,
                "params_used_this_run": params_logged_for_run,
                "trades_details": trades_details_list
            }
            return return_dict

        except Exception as final_log_e:
             self.sim_logger.error(f"Error during final results prep/logging: {final_log_e}", exc_info=True)
             return {"performance_score": -np.inf, "error": str(final_log_e)}


    def _close_sim_logger(self):
        # MODIFIED (2025-05-09): No changes here from simulation_engine_debug_exits_20250509
        if self.sim_logger and self.sim_logger.handlers:
            for handler in self.sim_logger.handlers[:]:
                try:
                    handler.flush(); handler.close()
                    self.sim_logger.removeHandler(handler)
                except Exception as e_close:
                    engine_logger.error(f"Error closing sim log handler for {getattr(self, 'log_file_path', 'unknown')}: {e_close}")
        self.sim_logger = None
