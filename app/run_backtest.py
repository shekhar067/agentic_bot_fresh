# app/run_backtest.py

import logging
import argparse
from pathlib import Path
import sys
import pandas as pd
import json # Import json

# Use absolute imports
from app.config import config
from app.strategies import strategy_functions
from app.backtester import SimpleBacktester

logging.basicConfig(level=logging.INFO, # Basic config for standalone run
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- FUNCTION TO RUN AND SAVE ---
def run_and_save_backtest(indicator_file_path: Path, output_json_path: Path):
    """Loads indicator data, runs the backtest, and saves summary metrics."""
    logger.info(f"--- Starting Backtest & Save for: {indicator_file_path} -> {output_json_path} ---")

    if not indicator_file_path.is_file():
        logger.error(f"Indicator file not found: {indicator_file_path}")
        raise FileNotFoundError(f"Indicator file not found: {indicator_file_path}")

    try:
        # 1. Load Data
        logger.info(f"Loading data with indicators from {indicator_file_path}...")
        data_with_indicators = pd.read_csv(indicator_file_path, index_col=0, parse_dates=True)
        if not isinstance(data_with_indicators.index, pd.DatetimeIndex):
            raise ValueError("Index of indicator file is not a DatetimeIndex.")
        logger.info(f"Loaded {len(data_with_indicators)} rows.")
        logger.info(f"Columns loaded: {data_with_indicators.columns.tolist()}")

        if data_with_indicators.empty:
            raise ValueError("Indicator data file is empty.")

        # 2. Initialize Backtester
        backtester = SimpleBacktester(strategies=strategy_functions)

        # 3. Run Backtest
        # The backtester.run method returns a dictionary where keys are strategy names
        # and values are dictionaries containing results like 'total_pnl', 'trade_count', etc.
        results_dict = backtester.run(data_with_indicators)

        if not results_dict:
            raise ValueError("Backtester did not return results.")

        # 4. Prepare Summary for Saving (Extract key metrics)
        summary_to_save = {}
        for strategy_name, summary in results_dict.items():
            summary_to_save[strategy_name] = {
                'total_pnl': summary.get('total_pnl'),
                'trade_count': summary.get('trade_count'),
                'win_rate': summary.get('win_rate'),
                # Add other key metrics you want quick access to
            }
            # Optionally save the full trades_summary_df if needed later
            # summary['trades_summary_df'].to_csv(output_json_path.parent / f"trades_{strategy_name}_{output_json_path.stem}.csv")


        # 5. Save Summary to JSON
        output_json_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
        with open(output_json_path, 'w') as f:
            json.dump(summary_to_save, f, indent=4)
        logger.info(f"Successfully ran backtest and saved summary to {output_json_path}")

        return summary_to_save # Return the summary data

    except Exception as e:
        logger.error(f"An unexpected error occurred during backtest run/save: {e}", exc_info=True)
        raise # Re-raise exception to signal failure

# --- Main block for standalone execution (if needed for debugging) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run backtest using indicator data and save results.')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path with indicators')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path for summary results')
    args = parser.parse_args()

    indicator_path = Path(args.input)
    output_path = Path(args.output)

    try:
        run_and_save_backtest(indicator_path, output_path)
    except Exception:
         sys.exit(1) # Exit with error code if function raised exception