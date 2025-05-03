# app/backtester.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Callable, List, Tuple, Any
# Use absolute import
from app.config import config

logger = logging.getLogger(__name__)
# Ensure logger is configured (might be done by caller, but good practice)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Set level to DEBUG to see detailed logs for this diagnosis
logger.setLevel(logging.DEBUG)

class SimpleBacktester:
    # __init__ and run methods remain the same...
    def __init__(self, strategies: Dict[str, Callable]):
        self.strategies = strategies
        self.results = {}

    def run(self, data: pd.DataFrame):
        # ... run method ...
        if data.empty: logger.error("Input data is empty."); return None # Return None on error
        logger.info(f"Running backtest on {len(data)} bars for {len(self.strategies)} strategies.")
        logger.info(f"Backtester using SL ATR Mult: {config.DEFAULT_SL_ATR_MULT}, TP ATR Mult: {config.DEFAULT_TP_ATR_MULT}")
        logger.info(f"Backtester using Commission PCT: {config.COMMISSION_PCT}, Slippage PCT: {config.SLIPPAGE_PCT}")
        for name, strategy_func in self.strategies.items():
            logger.info(f"--- Backtesting Strategy: {name} ---")
            try: # Add try/except around simulation
                 strategy_results = self._simulate_strategy(data.copy(), name, strategy_func)
                 self.results[name] = strategy_results
                 # Check if result is valid before accessing keys
                 if strategy_results and isinstance(strategy_results, dict):
                      pnl = strategy_results.get('total_pnl', 'N/A');
                      trades = strategy_results.get('trade_count', 'N/A');
                      win_rate = strategy_results.get('win_rate', 'N/A')
                      pnl_str = f"{pnl:.2f}" if isinstance(pnl, (int,float)) else pnl
                      wr_str = f"{win_rate:.2f}%" if isinstance(win_rate, (int,float)) else win_rate
                      logger.info(f"Strategy '{name}' Results: PnL={pnl_str}, Trades={trades}, Win Rate={wr_str}")
                 else:
                      logger.error(f"Strategy '{name}' simulation did not return valid results.")
            except Exception as sim_e:
                 logger.error(f"Error during simulation for strategy '{name}': {sim_e}", exc_info=True)
                 self.results[name] = {"total_pnl": 0,"trade_count": 0,"win_rate": 0,"trades_summary_list": [], "error": str(sim_e)}

        return self.results

    def _simulate_strategy(self, df: pd.DataFrame, strategy_name: str, strategy_func: Callable) -> Dict:
        """Simulates trades for a single strategy with enhanced logging."""

        potential_signals = df.apply(strategy_func, axis=1)
        position = 0; entry_price = np.nan; stop_loss = np.nan; take_profit = np.nan
        trade_pnl = 0.0; cumulative_pnl = 0.0; trade_count = 0; winning_trades = 0
        trades_details_list = [] # Store trades as dicts
        prev_trade_entry_time = pd.NaT # Initialize timestamp for entry

        df[f'{strategy_name}_position'] = 0
        df[f'{strategy_name}_trade_pnl'] = 0.0
        df[f'{strategy_name}_cumulative_pnl'] = 0.0

        atr_col = 'atr'
        if atr_col not in df.columns:
             atr_col_long = f'ATRr_{config.INDICATOR_ATR_PERIOD}'
             if atr_col_long in df.columns: atr_col = atr_col_long
             else: logger.error(f"ATR column not found. Cannot apply ATR SL/TP."); return {"total_pnl": 0,"trade_count": 0,"win_rate": 0,"trades_summary_list": [],"error": "ATR Column Missing"}

        logger.debug(f"[{strategy_name}] Starting simulation loop...")

        for i in range(1, len(df)): # Start from 1
            # --- Get current/previous data ---
            current_idx = df.index[i]; prev_idx = df.index[i - 1]
            current_row = df.iloc[i]; prev_row = df.iloc[i - 1]
            current_potential_signal = potential_signals.iloc[i]; prev_potential_signal = potential_signals.iloc[i - 1]
            current_price = current_row['close']; low_price = current_row['low']; high_price = current_row['high']
            current_atr = current_row.get(atr_col, np.nan)

            # --- Check Exits ---
            exit_triggered = False; exit_price = np.nan; exit_reason = "N/A"
            if position != 0: # Only check exits if in a position
                if position == 1: # Long exit checks
                    logger.debug(f"[{strategy_name}][{current_idx}] Checking Long Exit: Low={low_price:.2f} vs SL={stop_loss:.2f}; High={high_price:.2f} vs TP={take_profit:.2f}; Signal='{current_potential_signal}'")
                    if pd.notna(stop_loss) and low_price <= stop_loss: exit_triggered = True; exit_price = stop_loss; exit_reason = "Stop Loss"
                    elif pd.notna(take_profit) and high_price >= take_profit: exit_triggered = True; exit_price = take_profit; exit_reason = "Take Profit"
                    elif current_potential_signal == 'sell_potential' and prev_potential_signal != 'sell_potential': exit_triggered = True; exit_price = current_price; exit_reason = "Signal Reversal"
                elif position == -1: # Short exit checks
                    logger.debug(f"[{strategy_name}][{current_idx}] Checking Short Exit: High={high_price:.2f} vs SL={stop_loss:.2f}; Low={low_price:.2f} vs TP={take_profit:.2f}; Signal='{current_potential_signal}'")
                    if pd.notna(stop_loss) and high_price >= stop_loss: exit_triggered = True; exit_price = stop_loss; exit_reason = "Stop Loss"
                    elif pd.notna(take_profit) and low_price <= take_profit: exit_triggered = True; exit_price = take_profit; exit_reason = "Take Profit"
                    elif current_potential_signal == 'buy_potential' and prev_potential_signal != 'buy_potential': exit_triggered = True; exit_price = current_price; exit_reason = "Signal Reversal"

                # --- Process Exit ---
                if exit_triggered:
                    logger.info(f"[{strategy_name}][{current_idx}] EXIT TRIGGERED due to {exit_reason} at price {exit_price:.2f}")
                    trade_pnl = (exit_price - entry_price) * position
                    # Apply costs
                    trade_pnl -= (abs(entry_price * config.COMMISSION_PCT) + abs(exit_price * config.COMMISSION_PCT))
                    trade_pnl -= (abs(entry_price * config.SLIPPAGE_PCT) + abs(exit_price * config.SLIPPAGE_PCT))
                    cumulative_pnl += trade_pnl; trade_count += 1
                    if trade_pnl > 0: winning_trades += 1

                    # --- Log and Append Trade Detail ---
                    if not pd.isna(prev_trade_entry_time):
                        trade_detail = {
                            'EntryTime': prev_trade_entry_time.isoformat() if isinstance(prev_trade_entry_time, pd.Timestamp) else str(prev_trade_entry_time),
                            'ExitTime': current_idx.isoformat() if isinstance(current_idx, pd.Timestamp) else str(current_idx),
                            'Position': ('long' if position == 1 else 'short'),
                            'EntryPrice': round(entry_price, 2),
                            'ExitPrice': round(exit_price, 2),
                            'PnL_Points': round(trade_pnl, 2),
                            'ExitReason': exit_reason
                        }
                        trades_details_list.append(trade_detail)
                        logger.debug(f"[{strategy_name}] Appended trade detail. List length now: {len(trades_details_list)}") # Log append success
                    else:
                        logger.warning(f"[{strategy_name}][{current_idx}] Exit triggered but previous entry time was invalid. Trade not logged.")
                    # --- End Log and Append ---

                    df.loc[current_idx, f'{strategy_name}_trade_pnl'] = trade_pnl
                    df.loc[current_idx, f'{strategy_name}_position'] = 0 # Mark flat on this bar
                    position = 0; entry_price = np.nan; stop_loss = np.nan; take_profit = np.nan; prev_trade_entry_time = pd.NaT # Reset state

            # --- Check Entries (Only if Flat) ---
            # Important: Use 'elif' to prevent entry on the same bar as exit
            elif position == 0:
                entry_signal = None
                # --- Signal Confirmation Logic (remains same) ---
                if strategy_name == "EMA_Crossover": # ... (EMA logic) ...
                     ema_fast_col=f'EMA_{config.EMA_FAST_PERIOD}'; ema_slow_col=f'EMA_{config.EMA_SLOW_PERIOD}'
                     if ema_fast_col in current_row.index and ema_slow_col in current_row.index and ema_fast_col in prev_row.index and ema_slow_col in prev_row.index:
                         if not (pd.isna(current_row[ema_fast_col]) or pd.isna(prev_row[ema_fast_col]) or pd.isna(current_row[ema_slow_col]) or pd.isna(prev_row[ema_slow_col])):
                              if current_row[ema_fast_col]>current_row[ema_slow_col] and prev_row[ema_fast_col]<=prev_row[ema_slow_col]: entry_signal = 'buy'
                              elif current_row[ema_fast_col]<current_row[ema_slow_col] and prev_row[ema_fast_col]>=prev_row[ema_slow_col]: entry_signal = 'sell'
                elif strategy_name == "RSI_Basic": # ... (RSI logic) ...
                    rsi_col=f'RSI_{config.RSI_PERIOD}'; rsi_oversold=30; rsi_overbought=70
                    if rsi_col in current_row.index and rsi_col in prev_row.index:
                         if not (pd.isna(current_row[rsi_col]) or pd.isna(prev_row[rsi_col])):
                              if current_row[rsi_col]>rsi_oversold and prev_row[rsi_col]<=rsi_oversold: entry_signal = 'buy'
                              elif current_row[rsi_col]<rsi_overbought and prev_row[rsi_col]>=rsi_overbought: entry_signal = 'sell'

                # Process Entry
                if entry_signal:
                    if pd.isna(current_atr) or current_atr <= 0:
                        logger.warning(f"[{strategy_name}][{current_idx}] Skipping entry: Invalid ATR value ({current_atr})")
                    else:
                        position = 1 if entry_signal == 'buy' else -1
                        entry_price = current_price
                        prev_trade_entry_time = current_idx # Store entry time

                        sl_distance = current_atr * config.DEFAULT_SL_ATR_MULT
                        tp_distance = current_atr * config.DEFAULT_TP_ATR_MULT
                        if position == 1: stop_loss = entry_price - sl_distance; take_profit = entry_price + tp_distance
                        else: stop_loss = entry_price + sl_distance; take_profit = entry_price - tp_distance

                        logger.info(f"[{strategy_name}][{current_idx}] ENTRY {entry_signal.upper()} at {entry_price:.2f} | ATR={current_atr:.2f} | SL={stop_loss:.2f} | TP={take_profit:.2f}")
                        df.loc[current_idx, f'{strategy_name}_position'] = position # Mark position entry on this bar
                # else: logger.debug(f"[{strategy_name}][{current_idx}] No entry signal.") # Uncomment if needed

            # --- Update DataFrame for plotting/analysis ---
            df.loc[current_idx, f'{strategy_name}_cumulative_pnl'] = cumulative_pnl
            # Carry forward position state if it didn't change *on this specific bar*
            # If an exit happened, position is already 0. If an entry happened, it's +/-1.
            # If nothing happened and we were in a position, carry it forward.
            if df.loc[current_idx, f'{strategy_name}_position'] == 0 and position != 0 : # Check if state is inconsistent (should not happen with elif)
                 logger.warning(f"State mismatch at {current_idx}!") # Should not happen
            elif df.loc[current_idx, f'{strategy_name}_position'] == 0 and position == 0 : # If flat, ensure it's marked flat
                 pass # Already 0
            elif df.loc[current_idx, f'{strategy_name}_position'] != 0: # If entry happened on this bar
                 pass # Already marked +/- 1
            else: # If hold while in position from previous bar
                 df.loc[current_idx, f'{strategy_name}_position'] = position


        # --- Compile Final Results ---
        win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0.0
        logger.debug(f"[{strategy_name}] Finishing simulation. Total trades generated: {len(trades_details_list)}")
        if trades_details_list: logger.debug(f"[{strategy_name}] Last trade detail: {trades_details_list[-1]}")

        return {
            "total_pnl": cumulative_pnl,
            "trade_count": trade_count,
            "win_rate": win_rate,
            "trades_summary_list": trades_details_list, # Ensure this is returned
        }