# # app/run_feature_engine.py

# import sys
# import argparse
# import logging
# from pathlib import Path

# # Use absolute imports
# from app.config import config
# from app.data_io import load_historical_data
# from app.feature_engine import IndicatorCalculator
# from utils.expiry_utils import enrich_expiry_flags
# # Configure logging for this script
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--input", required=True)
# parser.add_argument("--output", required=True)

# # ✅ Add these two:
# parser.add_argument("--symbol", required=False, default="nifty", help="Trading symbol (e.g. nifty)")
# parser.add_argument("--exchange", required=False, default="NSE", help="Exchange (e.g. NSE)")

# args = parser.parse_args()


# def main():
#     """ Main function to run indicator calculation as a script """
#     parser = argparse.ArgumentParser(description='Calculate technical indicators for historical data.')
#     parser.add_argument('--input', type=str, required=True, help='Input raw CSV file path (e.g., data/nifty_historical_data_5min.csv)')
#     parser.add_argument('--output', type=str, required=True, help='Output CSV file path (e.g., data/nifty_with_indicators_5min.csv)')
#     args = parser.parse_args()

#     input_path = Path(args.input)
#     output_path = Path(args.output)

#     logger.info(f"Feature Engine Script Started: {input_path} -> {output_path}")

#     try:
#         # Load raw data using the loader (pass filename relative to DATA_FOLDER)
#         raw_df = load_historical_data(input_path.name)

#         # Calculate indicators
#         calculator = IndicatorCalculator() # Uses defaults from config
#         df_with_indicators = calculator.calculate_all_indicators(raw_df)
#         # After calculating indicators
#         df = enrich_expiry_flags(df, symbol="nifty", exchange="NSE")
#         # Save results
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         df_with_indicators.to_csv(output_path, index=True) # Keep datetime index
#         logger.info(f"Successfully calculated features and saved to {output_path}")

#     except FileNotFoundError:
#         logger.error(f"Input file not found: {input_path}")
#         sys.exit(1)
#     except Exception as e:
#         logger.error(f"Failed to calculate features: {e}", exc_info=True)
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

# app/run_feature_engine.py

import sys
import argparse
import logging
from pathlib import Path

from app.config import config
from app.data_io import load_historical_data
from app.feature_engine import IndicatorCalculator
from app.utils.expiry_utils import enrich_expiry_flags

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Parse CLI Arguments (Only once, globally) ---
parser = argparse.ArgumentParser(description='Calculate indicators and enrich with expiry flags')
parser.add_argument('--input', required=True, help='Path to input raw CSV file')
parser.add_argument('--output', required=True, help='Path to save output file with indicators')
parser.add_argument('--symbol', required=False, default='nifty', help='Trading symbol (e.g., nifty)')
parser.add_argument('--exchange', required=False, default='NSE', help='Exchange name (e.g., NSE)')
args = parser.parse_args()


def main():
    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info(f"Feature Engine Script Started: {input_path} -> {output_path}")

    try:
        raw_df = load_historical_data(input_path.name)
        calculator = IndicatorCalculator()
       # df = calculator.calculate_all_indicators(raw_df)
        df = calculator.calculate_all_indicators(raw_df, symbol=args.symbol)


        # ✅ Use parsed symbol and exchange
        df = enrich_expiry_flags(df, symbol=args.symbol, exchange_segment=args.exchange)
        # Save results

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=True)

        logger.info(f"✅ Successfully saved enriched feature file to: {output_path}")
    except FileNotFoundError:
        logger.error(f"❌ Input file not found: {input_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to generate features: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
