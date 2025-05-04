# app/run_feature_engine.py

import sys
import argparse
import logging
from pathlib import Path

# Use absolute imports
from app.config import config
from app.data_io import load_historical_data
from app.feature_engine import IndicatorCalculator

# Configure logging for this script
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """ Main function to run indicator calculation as a script """
    parser = argparse.ArgumentParser(description='Calculate technical indicators for historical data.')
    parser.add_argument('--input', type=str, required=True, help='Input raw CSV file path (e.g., data/nifty_historical_data_5min.csv)')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path (e.g., data/nifty_with_indicators_5min.csv)')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info(f"Feature Engine Script Started: {input_path} -> {output_path}")

    try:
        # Load raw data using the loader (pass filename relative to DATA_FOLDER)
        raw_df = load_historical_data(input_path.name)

        # Calculate indicators
        calculator = IndicatorCalculator() # Uses defaults from config
        df_with_indicators = calculator.calculate_all_indicators(raw_df)

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_with_indicators.to_csv(output_path, index=True) # Keep datetime index
        logger.info(f"Successfully calculated features and saved to {output_path}")

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to calculate features: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()