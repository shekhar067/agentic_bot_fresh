# app/simulation_engine.py

from datetime import datetime
import pandas as pd
import numpy as np
import logging
from typing import Dict, Callable, List, Tuple, Any
from pathlib import Path
import time # For potential timing checks

# Use absolute imports
from app.config import config
from app.agentic_core import RuleBasedAgent # Import the Agent

# Configure logger for this module
engine_logger = logging.getLogger("SimulationEngine") # Use a specific name
engine_logger.setLevel(logging.DEBUG) # Ensure debug messages are processed
# Prevent duplicating logs if root logger is also configured by Flask/Orchestrator
engine_logger.propagate = False
# Add a NullHandler to prevent "No handler found" warnings if not configured by caller
if not engine_logger.hasHandlers():
    engine_logger.addHandler(logging.NullHandler())


class SimpleBacktester:
    """Runs a simulation driven by an Agentic Core with detailed bar-by-bar logging."""

    def __init__(self, agent: RuleBasedAgent):
        """
        Args:
            agent: An instance of the agentic core that makes decisions.
        """
        if not isinstance(agent, RuleBasedAgent):
             raise TypeError("Agent must be an instance of RuleBasedAgent or similar.")
        self.agent = agent
        self.results = None # This will store the result dict for the run
        self.sim_logger = None # Placeholder for the dedicated simulation logger
        self.log_file_path = None

    def _setup_simulation_logging(self, log_dir: Path, timeframe: str):
        """Sets up the dedicated logger for this simulation run."""
        self.log_file_path = log_dir / f"simulation_trace_{timeframe}.log"
        # Create a dedicated logger instance for this simulation
        # Use a more specific name to avoid conflicts if run multiple times
        logger_name = f"SimTrace_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.sim_logger = logging.getLogger(logger_name)
        self.sim_logger.setLevel(logging.DEBUG) # Log everything for trace

        # Prevent propagation
        self.sim_logger.propagate = False

        # Remove existing handlers for this specific logger instance
        for handler in self.sim_logger.handlers[:]:
            self.sim_logger.removeHandler(handler)
            handler.close()

        # Create file handler for this simulation's trace log
        fh = logging.FileHandler(self.log_file_path, mode='w') # Overwrite log each time
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.sim_logger.addHandler(fh)
        self.sim_logger.info(f"--- Simulation Trace Log Initialized for Timeframe: {timeframe} ---")
        self.sim_logger.info(f"Log file: {self.log_file_path}")


    # ==================================================================
    # COMPLETE simulate_agent_run METHOD
    # ==================================================================
    def simulate_agent_run(self, df: pd.DataFrame, log_dir: Path, timeframe: str) -> Dict:
        """
        Simulates trades based on decisions from the agentic core with detailed bar-by-bar logging.

        Args:
            df: DataFrame with OHLCV, indicators, and 'regime' column.
            log_dir: Path to the directory where the detailed simulation log should be saved.
            timeframe: String representing the timeframe (e.g., "5min") for the log filename.

        Returns:
            Dictionary containing backtest results (PnL, trade count, win rate, trades list).
        """
        # Setup dedicated logger for this run
        self._setup_simulation_logging(log_dir, timeframe)

        self.sim_logger.info(f"Starting agent simulation loop on {len(df)} bars...")
        self.sim_logger.info(f"Using Config SL ATR Mult={config.DEFAULT_SL_ATR_MULT}, TP Mult={config.DEFAULT_TP_ATR_MULT}")
        self.sim_logger.info(f"Using Config Commission={config.COMMISSION_PCT}, Slippage={config.SLIPPAGE_PCT}")

        # --- State Variables ---
        position = 0  # 0: Flat, 1: Long, -1: Short
        entry_price = np.nan
        stop_loss = np.nan
        take_profit = np.nan
        trade_pnl = 0.0
        cumulative_pnl = 0.0
        trade_count = 0
        winning_trades = 0
        trades_details_list = [] # Store trades as dicts
        prev_trade_entry_time = pd.NaT # Timestamp of entry
        entry_strategy_name = None # Name of strategy triggering entry

        # --- Results Columns ---
        df['agent_position'] = 0
        df['agent_trade_pnl'] = 0.0
        df['agent_cumulative_pnl'] = 0.0
        df['agent_signal'] = 'hold' # Store the agent's final confirmed signal
        df['sl_level'] = np.nan
        df['tp_level'] = np.nan

        # --- Get ATR column name ---
        atr_col = 'atr'
        if atr_col not in df.columns:
             atr_col_long = f'ATRr_{config.INDICATOR_ATR_PERIOD}'
             if atr_col_long in df.columns:
                  atr_col = atr_col_long
                  self.sim_logger.warning(f"Using ATR column '{atr_col}' as 'atr' not found.")
             else:
                  self.sim_logger.error(f"Required ATR column ('atr' or '{atr_col_long}') not found. Cannot apply ATR SL/TP.")
                  return {"total_pnl": 0,"trade_count": 0,"win_rate": 0,"trades_summary_list": [], "error": "ATR Column Missing"}

        self.sim_logger.debug(f"Starting loop using ATR column: '{atr_col}'")

        # --- Simulation Loop ---
        for i in range(1, len(df)): # Start from 1 to allow comparing with prev_row
            # --- Get current/previous data ---
            current_idx = df.index[i]
            prev_idx = df.index[i - 1]
            current_row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            # Basic data checks
            current_price = current_row.get('close', np.nan)
            low_price = current_row.get('low', np.nan)
            high_price = current_row.get('high', np.nan)
            current_atr = current_row.get(atr_col, np.nan)

            self.sim_logger.debug(f"--- Bar: {current_idx} ---")
            self.sim_logger.debug(f"DATA: O={current_row.get('open', np.nan):.2f} H={high_price:.2f} L={low_price:.2f} C={current_price:.2f} V={current_row.get('volume', 0)} ATR={current_atr:.3f}")

            if pd.isna(current_price) or pd.isna(low_price) or pd.isna(high_price):
                self.sim_logger.warning("Skipping bar due to NaN in OHLC.")
                df.loc[current_idx, 'agent_position'] = position # Carry position state
                df.loc[current_idx, 'agent_cumulative_pnl'] = cumulative_pnl # Carry PnL state
                continue

            # --- Get Decision from Agent ---
            current_regime = current_row.get('regime', 'Unknown')
            agent_potential_signal, sl_atr_mult, tp_atr_mult, selected_strategy_name = self.agent.decide(current_row, data_history=df)
            # Store potential signal for analysis, actual signal below
            # df.loc[current_idx, 'agent_potential_signal'] = agent_potential_signal

            self.sim_logger.debug(f"AGENT: Regime='{current_regime}', Strategy='{selected_strategy_name}', PotentialSignal='{agent_potential_signal}', SL Mult={sl_atr_mult:.2f}, TP Mult={tp_atr_mult:.2f}")

            # --- Check Exits FIRST (if in a position) ---
            exit_triggered = False
            exit_price = np.nan
            exit_reason = "N/A"

            if position != 0:
                self.sim_logger.debug(f"EXIT CHECK: Position={position}, Price={current_price:.2f}, Low={low_price:.2f}, High={high_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}")
                exit_cond_sl = False; exit_cond_tp = False; exit_cond_rev = False

                # Check SL
                if pd.notna(stop_loss):
                    if position == 1 and low_price <= stop_loss: exit_cond_sl = True; exit_price = stop_loss
                    elif position == -1 and high_price >= stop_loss: exit_cond_sl = True; exit_price = stop_loss
                self.sim_logger.debug(f"  SL Check: Condition Met = {exit_cond_sl}")

                # Check TP (only if SL not hit)
                if not exit_cond_sl and pd.notna(take_profit):
                    if position == 1 and high_price >= take_profit: exit_cond_tp = True; exit_price = take_profit
                    elif position == -1 and low_price <= take_profit: exit_cond_tp = True; exit_price = take_profit
                self.sim_logger.debug(f"  TP Check: Condition Met = {exit_cond_tp}")

                # Check Reversal (only if SL/TP not hit)
                if not exit_cond_sl and not exit_cond_tp:
                    if position == 1 and (agent_potential_signal == 'sell_potential' or agent_potential_signal == 'sell'): exit_cond_rev = True; exit_price = current_price
                    elif position == -1 and (agent_potential_signal == 'buy_potential' or agent_potential_signal == 'buy'): exit_cond_rev = True; exit_price = current_price
                self.sim_logger.debug(f"  Reversal Check: Condition Met = {exit_cond_rev}")

                # Determine final exit reason
                if exit_cond_sl: exit_triggered=True; exit_reason="Stop Loss"
                elif exit_cond_tp: exit_triggered=True; exit_reason="Take Profit"
                elif exit_cond_rev: exit_triggered=True; exit_reason="Agent Signal Reversal"

                # --- Process Exit ---
                if exit_triggered:
                    # Clip exit price to ensure it's within the bar's range
                    exit_price = np.clip(exit_price, low_price, high_price)
                    self.sim_logger.info(f"EXIT: Pos={position}, Reason={exit_reason}, Price={exit_price:.2f}")

                    # Calculate PnL including costs
                    trade_pnl = (exit_price - entry_price) * position # PnL in points
                    entry_cost = abs(entry_price * config.COMMISSION_PCT) + abs(entry_price * config.SLIPPAGE_PCT)
                    exit_cost = abs(exit_price * config.COMMISSION_PCT) + abs(exit_price * config.SLIPPAGE_PCT)
                    trade_pnl -= (entry_cost + exit_cost)

                    cumulative_pnl += trade_pnl
                    trade_count += 1
                    if trade_pnl > 0: winning_trades += 1

                    # Log and Append Trade Detail
                    if not pd.isna(prev_trade_entry_time):
                        trade_detail = {
                            'EntryTime': prev_trade_entry_time.isoformat() if isinstance(prev_trade_entry_time, pd.Timestamp) else str(prev_trade_entry_time),
                            'ExitTime': current_idx.isoformat() if isinstance(current_idx, pd.Timestamp) else str(current_idx),
                            'Position': ('long' if position == 1 else 'short'),
                            'EntryPrice': round(entry_price, 2),
                            'ExitPrice': round(exit_price, 2),
                            'PnL_Points': round(trade_pnl, 2),
                            'ExitReason': exit_reason,
                            'StrategyName': entry_strategy_name # Strategy that triggered entry
                        }
                        trades_details_list.append(trade_detail)
                        self.sim_logger.debug(f" -> Trade logged. List length: {len(trades_details_list)}")
                    else:
                        self.sim_logger.warning(f" Exit triggered but prev_trade_entry_time invalid. Trade not logged.")

                    # Update DataFrame state for this bar
                    df.loc[current_idx, 'agent_trade_pnl'] = trade_pnl
                    df.loc[current_idx, 'agent_position'] = 0 # Mark flat now
                    df.loc[current_idx, 'agent_signal'] = 'exit'

                    # Reset state variables for the next bar
                    position = 0
                    entry_price = np.nan
                    stop_loss = np.nan
                    take_profit = np.nan
                    prev_trade_entry_time = pd.NaT
                    entry_strategy_name = None

            # --- Check Entries (Only if Flat and NO exit happened on this bar) ---
            if not exit_triggered and position == 0:
                entry_signal = None # Actual confirmed signal ('buy' or 'sell')
                confirmed_strategy_name = None

                # --- Confirmation Logic Based on Agent's Potential Signal ---
                if agent_potential_signal == 'buy_potential':
                    self.sim_logger.debug(f"ENTRY CHECK: Potential Buy. Confirming strategy: '{selected_strategy_name}'")
                    if selected_strategy_name == "EMA_Crossover":
                        ema_fast_col=f'EMA_{config.EMA_FAST_PERIOD}'; ema_slow_col=f'EMA_{config.EMA_SLOW_PERIOD}'
                        if ema_fast_col in current_row.index and ema_slow_col in current_row.index and ema_fast_col in prev_row.index and ema_slow_col in prev_row.index:
                             if not (pd.isna(current_row[ema_fast_col]) or pd.isna(prev_row[ema_fast_col]) or pd.isna(current_row[ema_slow_col]) or pd.isna(prev_row[ema_slow_col])):
                                  c_fast=current_row[ema_fast_col]; c_slow=current_row[ema_slow_col]; p_fast=prev_row[ema_fast_col]; p_slow=prev_row[ema_slow_col]
                                  self.sim_logger.debug(f"  Confirming EMA Buy: Cur={c_fast:.2f}>{c_slow:.2f}?({c_fast>c_slow}), Prev={p_fast:.2f}<={p_slow:.2f}?({p_fast<=p_slow})")
                                  if c_fast > c_slow and p_fast <= p_slow:
                                      entry_signal = 'buy'; confirmed_strategy_name = selected_strategy_name; self.sim_logger.debug("  -> Confirmed: EMA Buy")
                             # else: self.sim_logger.debug("  EMA Buy Check Failed (NaN values)")
                        # else: self.sim_logger.debug("  EMA Buy Check Failed (Missing columns)")

                    elif selected_strategy_name == "RSI_Basic":
                         rsi_col=f'RSI_{config.INDICATOR_RSI_PERIOD}'; rsi_oversold=30
                         if rsi_col in current_row.index and rsi_col in prev_row.index:
                             if not (pd.isna(current_row[rsi_col]) or pd.isna(prev_row[rsi_col])):
                                  c_rsi=current_row[rsi_col]; p_rsi=prev_row[rsi_col]
                                  self.sim_logger.debug(f"  Confirming RSI Buy: Cur={c_rsi:.2f}>{rsi_oversold}?({c_rsi>rsi_oversold}), Prev={p_rsi:.2f}<={rsi_oversold}?({p_rsi<=rsi_oversold})")
                                  if c_rsi > rsi_oversold and p_rsi <= rsi_oversold:
                                      entry_signal = 'buy'; confirmed_strategy_name = selected_strategy_name; self.sim_logger.debug("  -> Confirmed: RSI Buy")
                             # else: self.sim_logger.debug("  RSI Buy Check Failed (NaN values)")
                         # else: self.sim_logger.debug("  RSI Buy Check Failed (Missing columns)")
                    # Add elif for other strategies here...

                elif agent_potential_signal == 'sell_potential':
                     self.sim_logger.debug(f"ENTRY CHECK: Potential Sell. Confirming strategy: '{selected_strategy_name}'")
                     if selected_strategy_name == "EMA_Crossover":
                        ema_fast_col=f'EMA_{config.EMA_FAST_PERIOD}'; ema_slow_col=f'EMA_{config.EMA_SLOW_PERIOD}'
                        if ema_fast_col in current_row.index and ema_slow_col in current_row.index and ema_fast_col in prev_row.index and ema_slow_col in prev_row.index:
                             if not (pd.isna(current_row[ema_fast_col]) or pd.isna(prev_row[ema_fast_col]) or pd.isna(current_row[ema_slow_col]) or pd.isna(prev_row[ema_slow_col])):
                                  c_fast=current_row[ema_fast_col]; c_slow=current_row[ema_slow_col]; p_fast=prev_row[ema_fast_col]; p_slow=prev_row[ema_slow_col]
                                  self.sim_logger.debug(f"  Confirming EMA Sell: Cur={c_fast:.2f}<{c_slow:.2f}?({c_fast<c_slow}), Prev={p_fast:.2f}>={p_slow:.2f}?({p_fast>=p_slow})")
                                  if c_fast < c_slow and p_fast >= p_slow:
                                      entry_signal = 'sell'; confirmed_strategy_name = selected_strategy_name; self.sim_logger.debug("  -> Confirmed: EMA Sell")
                     elif selected_strategy_name == "RSI_Basic":
                          rsi_col=f'RSI_{config.INDICATOR_RSI_PERIOD}'; rsi_overbought=70
                          if rsi_col in current_row.index and rsi_col in prev_row.index:
                              if not (pd.isna(current_row[rsi_col]) or pd.isna(prev_row[rsi_col])):
                                   c_rsi=current_row[rsi_col]; p_rsi=prev_row[rsi_col]
                                   self.sim_logger.debug(f"  Confirming RSI Sell: Cur={c_rsi:.2f}<{rsi_overbought}?({c_rsi<rsi_overbought}), Prev={p_rsi:.2f}>={rsi_overbought}?({p_rsi>=rsi_overbought})")
                                   if c_rsi < rsi_overbought and p_rsi >= rsi_overbought:
                                       entry_signal = 'sell'; confirmed_strategy_name = selected_strategy_name; self.sim_logger.debug("  -> Confirmed: RSI Sell")
                     # Add elif for other strategies...


                # --- Process Confirmed Entry ---
                if entry_signal:
                    if pd.isna(current_atr) or current_atr <= 0:
                        self.sim_logger.warning(f"[{current_idx}] Skipping confirmed entry '{entry_signal}': Invalid ATR ({current_atr})")
                    else:
                        # Set state for the NEXT bar
                        position = 1 if entry_signal == 'buy' else -1
                        entry_price = current_price # Entry at current close
                        prev_trade_entry_time = current_idx # Timestamp for this entry
                        entry_strategy_name = confirmed_strategy_name # Store confirmed strategy name

                        # Calculate SL/TP using agent's multipliers and current ATR
                        sl_distance = current_atr * sl_atr_mult
                        tp_distance = current_atr * tp_atr_mult
                        if position == 1:
                            stop_loss = entry_price - sl_distance
                            take_profit = entry_price + tp_distance
                        else: # Short
                            stop_loss = entry_price + sl_distance
                            take_profit = entry_price - tp_distance

                        # Log the entry with calculated levels
                        self.sim_logger.info(f"ENTRY: Signal={entry_signal.upper()}, Strategy='{entry_strategy_name or 'N/A'}', Price={entry_price:.2f}, ATR={current_atr:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}")

                        # Update DataFrame for CURRENT bar's state
                        df.loc[current_idx, 'agent_position'] = position
                        df.loc[current_idx, 'agent_signal'] = entry_signal # Store confirmed signal
                        df.loc[current_idx, 'sl_level'] = stop_loss
                        df.loc[current_idx, 'tp_level'] = take_profit
                # else: # Log if potential signal wasn't confirmed
                #     if agent_potential_signal not in ['hold', None]:
                #          self.sim_logger.debug(f"[{current_idx}] Potential signal '{agent_potential_signal}' from agent strategy '{selected_strategy_name}' not confirmed.")


            # --- Update State Columns in DataFrame ---
            # Update cumulative PnL regardless of trade action
            df.loc[current_idx, 'agent_cumulative_pnl'] = cumulative_pnl
            # Ensure position column reflects state for NEXT bar correctly
            # If exit happened, position is 0. If entry happened, position is +/-1.
            # If neither happened, carry forward the existing position state.
            if not exit_triggered and df.loc[current_idx, 'agent_position'] == 0:
                 df.loc[current_idx, 'agent_position'] = position


        # --- Compile Final Results ---
        win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0.0
        self.sim_logger.info(f"--- Simulation Finished. Total Trades: {trade_count}, Win Rate: {win_rate:.2f}%, Final Cum PnL: {cumulative_pnl:.2f} ---")
        if trades_details_list:
            self.sim_logger.debug(f"Last trade detail: {trades_details_list[-1]}")
        else:
            self.sim_logger.debug("No trades were generated during this simulation.")


        # Close the dedicated log file handler
        if self.sim_logger:
            for handler in self.sim_logger.handlers[:]:
                try:
                    handler.close()
                    self.sim_logger.removeHandler(handler)
                except Exception as e:
                    # Log to engine_logger or print, as sim_logger handler might be closed
                    engine_logger.error(f"Error closing simulation log handler: {e}")


        # Return results including the trade list
        return {
            "total_pnl": cumulative_pnl,
            "trade_count": trade_count,
            "win_rate": win_rate,
            "trades_summary_list": trades_details_list, # Key name matches run_simulation_step.py
        }

    # --- End of simulate_agent_run method ---

# (You might have helper methods like _apply_slippage, _calculate_pnl_points here - keep them if they exist)