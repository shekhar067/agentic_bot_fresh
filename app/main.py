import logging
from pathlib import Path

# Use relative imports for modules within the 'app' package
from app.config import config
from app.data_io import load_historical_data
from app.analysis_engine import add_indicators
from app.strategies import strategy_functions  # Import the dictionary
from app.simulation_engine import SimpleBacktester

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)  # Output logs to console

logger = logging.getLogger(__name__)


def run_pipeline():
    """Main pipeline function for Phase 1."""
    logger.info("--- Starting Phase 1 Backtesting Pipeline ---")

    try:
        # 1. Load Data
        # Uses NIFTY_5MIN_CSV from config by default
        raw_data = load_historical_data()
        if raw_data.empty:
            return  # Exit if data loading failed

        # 2. Add Indicators
        data_with_indicators = add_indicators(raw_data)
        if data_with_indicators.empty:
            logger.error("Indicator calculation resulted in empty DataFrame. Exiting.")
            return

        # 3. Initialize Backtester with Strategies
        # strategy_functions is imported from strategies.py
        backtester = SimpleBacktester(strategies=strategy_functions)

        # 4. Run Backtest
        results = backtester.run(data_with_indicators)

        # 5. Basic Result Summary (more detailed report in Phase 2/3)
        logger.info("--- Backtesting Complete ---")
        if results:
            for strategy_name, summary in results.items():
                print(f"\nStrategy: {strategy_name}")
                print(f"  Total PnL (Points): {summary['total_pnl']:.2f}")
                print(f"  Total Trades: {summary['trade_count']}")
                print(f"  Win Rate: {summary['win_rate']:.2f}%")
                # Optionally print the trades summary DataFrame
                # print("  Trades Summary:")
                # print(summary['trades_summary_df'].to_string())
        else:
            logger.warning("Backtester did not return results.")

    except FileNotFoundError:
        logger.error(
            "Required data file not found. Please check config and data folder."
        )
    except ValueError as ve:
        logger.error(f"Data validation or configuration error: {ve}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in the pipeline: {e}", exc_info=True
        )


if __name__ == "__main__":
    run_pipeline()
