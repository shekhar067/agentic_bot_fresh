# app/run_backtest.py

import logging
import argparse
from pathlib import Path
import pandas as pd

# Use relative imports
from app.config import config
from app.strategies import strategy_functions
from app.backtester import SimpleBacktester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_single_backtest(indicator_file_path: Path):
    """Loads indicator data and runs the backtest."""
    logger.info(
        f"--- Starting Backtest Run using Indicator File: {indicator_file_path} ---"
    )

    if not indicator_file_path.is_file():
        logger.error(f"Indicator file not found: {indicator_file_path}")
        return None

    try:
        # 1. Load Data with Indicators
        logger.info(f"Loading data with indicators from {indicator_file_path}...")
        # Load data, ensuring datetime index is parsed correctly
        data_with_indicators = pd.read_csv(
            indicator_file_path, index_col=0, parse_dates=True
        )
        if not isinstance(data_with_indicators.index, pd.DatetimeIndex):
            raise ValueError("Index of indicator file is not a DatetimeIndex.")
        logger.info(f"Loaded {len(data_with_indicators)} rows.")

        if data_with_indicators.empty:
            logger.error("Indicator data file is empty. Cannot run backtest.")
            return None

        # 2. Initialize Backtester
        # strategy_functions is imported from strategies.py
        backtester = SimpleBacktester(strategies=strategy_functions)

        # 3. Run Backtest
        results = backtester.run(data_with_indicators)  # Pass the loaded DataFrame

        # 4. Basic Result Summary
        logger.info("--- Backtest Run Complete ---")
        if results:
            for strategy_name, summary in results.items():
                print(f"\nStrategy: {strategy_name}")
                print(f"  Total PnL (Points): {summary['total_pnl']:.2f}")
                print(f"  Total Trades: {summary['trade_count']}")
                print(f"  Win Rate: {summary['win_rate']:.2f}%")
                # TODO: Add saving results/reports here later if needed
                # For now, just prints summary
        else:
            logger.warning("Backtester did not return results.")

        return results  # Return the results dictionary

    except FileNotFoundError:
        logger.error(
            f"Indicator data file not found during backtest run: {indicator_file_path}"
        )
        return None
    except ValueError as ve:
        logger.error(
            f"Data validation or configuration error during backtest run: {ve}"
        )
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during the backtest run: {e}", exc_info=True
        )
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run backtest using pre-calculated indicator data."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file path with indicators (e.g., data/nifty_with_indicators_5min.csv)",
    )
    args = parser.parse_args()

    indicator_path = Path(args.input)
    run_single_backtest(indicator_path)
