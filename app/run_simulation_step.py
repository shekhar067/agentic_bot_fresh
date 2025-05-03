# app/run_backtest.py

import logging
import argparse
from pathlib import Path
import pandas as pd
import json
import sys

# Use absolute imports
from app.config import config
from app.strategies import strategy_functions
from app.simulation_engine import SimpleBacktester

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    # Use specific handlers if needed for file logging from here
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
# Set higher level for this specific logger if needed for debugging
logger.setLevel(logging.DEBUG) # Make sure debug messages show up


# --- FUNCTION TO RUN AND SAVE ---
def run_and_save_backtest(indicator_file_path: Path, output_json_path: Path):
    """Loads indicator data, runs the backtest, and saves summary metrics & trades."""
    logger.info(f"--- Starting Backtest & Save for: {indicator_file_path} -> {output_json_path} ---")

    if not indicator_file_path.is_file():
        logger.error(f"Indicator file not found: {indicator_file_path}")
        raise FileNotFoundError(f"Indicator file not found: {indicator_file_path}")

    try:
        # 1. Load Data
        # ... (loading logic remains the same) ...
        logger.info(f"Loading data with indicators from {indicator_file_path}...")
        data_with_indicators = pd.read_csv(indicator_file_path, index_col=0, parse_dates=True)
        if not isinstance(data_with_indicators.index, pd.DatetimeIndex):
            raise ValueError("Index of indicator file is not a DatetimeIndex.")
        logger.info(f"Loaded {len(data_with_indicators)} rows.")
        logger.info(f"Columns loaded: {data_with_indicators.columns.tolist()}")
        if data_with_indicators.empty: raise ValueError("Indicator data file is empty.")


        # 2. Initialize Backtester
        backtester = SimpleBacktester(strategies=strategy_functions)

        # 3. Run Backtest
        results_dict = backtester.run(data_with_indicators)

        if not results_dict:
            raise ValueError("Backtester did not return results.")

        # 4. Prepare Summary for Saving
        summary_to_save = {}
        for strategy_name, summary in results_dict.items():
            # --- ADD DETAILED LOGGING HERE ---
            logger.debug(f"Preparing summary for strategy: {strategy_name}")
            trades_list = summary.get('trades_summary_list', None) # Get the list
            logger.debug(f"  Raw 'trades_summary_list' from backtester: Type={type(trades_list)}, Length={len(trades_list) if isinstance(trades_list, list) else 'N/A'}")
            if isinstance(trades_list, list) and trades_list:
                 logger.debug(f"  First trade detail item: {trades_list[0]}")
            # --- END LOGGING ---

            summary_to_save[strategy_name] = {
                'total_pnl': summary.get('total_pnl'),
                'trade_count': summary.get('trade_count'),
                'win_rate': summary.get('win_rate'),
                # Use the retrieved list, default to empty list if None/missing
                'trades_details': trades_list if isinstance(trades_list, list) else []
            }
            # Optional check after assignment
            logger.debug(f"  'trades_details' key in summary_to_save: {'trades_details' in summary_to_save[strategy_name]}, Length: {len(summary_to_save[strategy_name].get('trades_details', []))}")


        # --- ADD LOGGING BEFORE SAVING ---
        logger.debug(f"Final structure being saved to JSON for {output_json_path.name}:")
        try:
            # Use pformat for potentially cleaner multi-line logging of the dict
            import pprint
            logger.debug(pprint.pformat(summary_to_save))
        except ImportError:
            logger.debug(summary_to_save) # Fallback if pprint not available
        # --- END LOGGING ---

        # 5. Save Summary to JSON
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(summary_to_save, f, indent=4, default=str)
        logger.info(f"Successfully ran backtest and saved summary to {output_json_path}")

        return summary_to_save

    except Exception as e:
        logger.error(f"An unexpected error occurred during backtest run/save: {e}", exc_info=True)
        raise

# --- Main block ---
if __name__ == "__main__":
    # ... (argparse remains same) ...
    parser = argparse.ArgumentParser(description='Run backtest using indicator data and save results.')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path with indicators')
    parser.add_argument('--output-json', type=str, required=True, help='Output JSON file path for summary results')
    args = parser.parse_args()
    indicator_path = Path(args.input)
    output_path = Path(args.output_json)
    try: run_and_save_backtest(indicator_path, output_path)
    except Exception: sys.exit(1)