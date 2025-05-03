import pandas as pd
import numpy as np
import logging
from typing import Dict, Callable, List, Tuple
from .config import config  # Use relative import

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SimpleBacktester:
    """Runs a simple vectorized or iterative backtest for multiple strategies."""

    def __init__(self, strategies: Dict[str, Callable]):
        """
        Args:
            strategies: Dictionary mapping strategy names to signal functions.
                        Signal functions take a row (pd.Series) and return
                        'buy_potential', 'sell_potential', or 'hold'.
        """
        self.strategies = strategies
        self.results = {}  # Store results per strategy

    def run(self, data: pd.DataFrame):
        """
        Runs the backtest for all configured strategies.

        Args:
            data: DataFrame with OHLCV and calculated indicator columns.
        """
        if data.empty:
            logger.error("Input data is empty. Cannot run backtest.")
            return

        logger.info(
            f"Running backtest on {len(data)} bars for {len(self.strategies)} strategies."
        )

        for name, strategy_func in self.strategies.items():
            logger.info(f"--- Backtesting Strategy: {name} ---")
            # Run simulation for the individual strategy
            strategy_results = self._simulate_strategy(data.copy(), name, strategy_func)
            self.results[name] = strategy_results
            # Print basic summary for this strategy
            pnl = strategy_results["total_pnl"]
            trades = strategy_results["trade_count"]
            win_rate = strategy_results["win_rate"]
            logger.info(
                f"Strategy '{name}' Results: PnL={pnl:.2f}, Trades={trades}, Win Rate={win_rate:.2f}%"
            )

        return self.results

    def _simulate_strategy(
        self, df: pd.DataFrame, strategy_name: str, strategy_func: Callable
    ) -> Dict:
        """Simulates trades for a single strategy."""

        # --- Generate Potential Signals ---
        # Apply the strategy function row-by-row (less efficient but clearer for complex logic)
        # A vectorized approach might be possible for simpler strategies
        potential_signals = df.apply(strategy_func, axis=1)

        # --- Initialize State and Results Columns ---
        position = 0  # 0: Flat, 1: Long, -1: Short
        entry_price = np.nan
        stop_loss = np.nan
        take_profit = np.nan
        trade_pnl = 0.0
        cumulative_pnl = 0.0
        trade_count = 0
        winning_trades = 0

        # Use lists to store trade details efficiently
        entry_times = []
        exit_times = []
        entry_prices = []
        exit_prices = []
        pnls = []
        positions_held = []  # 'long' or 'short'

        # Create columns to store state for analysis (optional but helpful)
        df[f"{strategy_name}_position"] = 0
        df[f"{strategy_name}_trade_pnl"] = 0.0
        df[f"{strategy_name}_cumulative_pnl"] = 0.0

        # --- Simulation Loop ---
        for i in range(1, len(df)):  # Start from 1 to check previous state
            current_idx = df.index[i]
            prev_idx = df.index[i - 1]
            current_row = df.iloc[i]
            prev_row = df.iloc[i - 1]  # Needed for confirming crosses

            current_potential_signal = potential_signals.iloc[i]
            prev_potential_signal = potential_signals.iloc[i - 1]

            current_price = current_row[
                "close"
            ]  # Execute trades at close for simplicity
            low_price = current_row["low"]
            high_price = current_row["high"]

            # --- Check Exits ---
            exit_triggered = False
            exit_price = np.nan

            if position == 1:  # If Long
                # Check Stop Loss
                if low_price <= stop_loss:
                    exit_triggered = True
                    exit_price = stop_loss  # Exit at SL price
                # Check Take Profit
                elif high_price >= take_profit:
                    exit_triggered = True
                    exit_price = take_profit  # Exit at TP price
                # Check for signal reversal (e.g., potential sell signal appears)
                elif (
                    current_potential_signal == "sell_potential"
                    and prev_potential_signal != "sell_potential"
                ):
                    exit_triggered = True
                    exit_price = current_price  # Exit at current close on reversal

            elif position == -1:  # If Short
                # Check Stop Loss
                if high_price >= stop_loss:
                    exit_triggered = True
                    exit_price = stop_loss  # Exit at SL price
                # Check Take Profit
                elif low_price <= take_profit:
                    exit_triggered = True
                    exit_price = take_profit  # Exit at TP price
                # Check for signal reversal (e.g., potential buy signal appears)
                elif (
                    current_potential_signal == "buy_potential"
                    and prev_potential_signal != "buy_potential"
                ):
                    exit_triggered = True
                    exit_price = current_price  # Exit at current close on reversal

            # Process Exit
            if exit_triggered:
                trade_pnl = (exit_price - entry_price) * position  # PnL in points
                # Apply Commission & Slippage (simplified: once on entry, once on exit)
                trade_pnl -= (entry_price * config.COMMISSION_PCT) + (
                    exit_price * config.COMMISSION_PCT
                )  # Commission
                trade_pnl -= (entry_price * config.SLIPPAGE_PCT) + (
                    exit_price * config.SLIPPAGE_PCT
                )  # Slippage

                cumulative_pnl += trade_pnl
                trade_count += 1
                if trade_pnl > 0:
                    winning_trades += 1

                # Record trade details
                entry_times.append(
                    prev_trade_entry_time
                )  # Store entry time from previous state
                exit_times.append(current_idx)
                entry_prices.append(entry_price)
                exit_prices.append(exit_price)
                pnls.append(trade_pnl)
                positions_held.append("long" if position == 1 else "short")

                # Update DataFrame state for this bar
                df.loc[current_idx, f"{strategy_name}_trade_pnl"] = trade_pnl
                df.loc[current_idx, f"{strategy_name}_position"] = 0  # Flat after exit

                # Reset state
                position = 0
                entry_price = np.nan
                stop_loss = np.nan
                take_profit = np.nan

            # --- Check Entries (Only if Flat) ---
            if position == 0:
                entry_signal = None
                # Confirm actual crossover for EMA strategy
                if strategy_name == "EMA_Crossover":
                    ema_fast_col = f"EMA_{config.EMA_FAST_PERIOD}"
                    ema_slow_col = f"EMA_{config.EMA_SLOW_PERIOD}"
                    if (
                        current_row[ema_fast_col] > current_row[ema_slow_col]
                        and prev_row[ema_fast_col] <= prev_row[ema_slow_col]
                    ):
                        entry_signal = "buy"
                    elif (
                        current_row[ema_fast_col] < current_row[ema_slow_col]
                        and prev_row[ema_fast_col] >= prev_row[ema_slow_col]
                    ):
                        entry_signal = "sell"
                # Confirm actual crossover for RSI strategy
                elif strategy_name == "RSI_Basic":
                    rsi_col = f"RSI_{config.RSI_PERIOD}"
                    rsi_oversold = 30
                    rsi_overbought = 70
                    if (
                        current_row[rsi_col] > rsi_oversold
                        and prev_row[rsi_col] <= rsi_oversold
                    ):
                        entry_signal = "buy"
                    elif (
                        current_row[rsi_col] < rsi_overbought
                        and prev_row[rsi_col] >= rsi_overbought
                    ):
                        entry_signal = "sell"
                # Add confirmation logic for other strategies here...

                # Process Entry
                if entry_signal:
                    position = 1 if entry_signal == "buy" else -1
                    entry_price = current_price  # Enter at close
                    prev_trade_entry_time = current_idx  # Store entry time

                    # Calculate static SL/TP based on percentage
                    if position == 1:
                        stop_loss = entry_price * (1 - config.DEFAULT_SL_PCT / 100.0)
                        take_profit = entry_price * (1 + config.DEFAULT_TP_PCT / 100.0)
                    else:  # Short
                        stop_loss = entry_price * (1 + config.DEFAULT_SL_PCT / 100.0)
                        take_profit = entry_price * (1 - config.DEFAULT_TP_PCT / 100.0)

                    # Update DataFrame state for this bar
                    df.loc[current_idx, f"{strategy_name}_position"] = position

            # Update cumulative PnL column for plotting
            df.loc[current_idx, f"{strategy_name}_cumulative_pnl"] = cumulative_pnl
            # Carry forward position state if unchanged
            if position != 0 and not exit_triggered:
                df.loc[current_idx, f"{strategy_name}_position"] = position

        # --- Compile Results ---
        win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0.0
        trades_summary = pd.DataFrame(
            {
                "EntryTime": entry_times,
                "ExitTime": exit_times,
                "Position": positions_held,
                "EntryPrice": entry_prices,
                "ExitPrice": exit_prices,
                "PnL_Points": pnls,
            }
        )

        return {
            "total_pnl": cumulative_pnl,
            "trade_count": trade_count,
            "win_rate": win_rate,
            "trades_summary_df": trades_summary,
            "results_df": df,  # Return the dataframe with signals/pnl columns
        }
