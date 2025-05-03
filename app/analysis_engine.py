# app/analysis_engine.py

import pandas as pd
import pandas_ta as ta
import logging
import numpy as np  # Import numpy
from app.config import config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds technical indicators to the DataFrame."""
    logger.info(f"Calculating indicators for DataFrame with shape {df.shape}...")
    if df.empty:
        logger.warning("Input DataFrame is empty. Cannot calculate indicators.")
        return df

    # Create a copy to avoid modifying the original DataFrame passed to the function
    df_out = df.copy()

    try:
        # Calculate indicators using pandas_ta, allowing errors for individual indicators
        logger.info("Calculating EMAs...")
        df_out.ta.ema(length=config.EMA_FAST_PERIOD, append=True)
        df_out.ta.ema(length=config.EMA_SLOW_PERIOD, append=True)

        logger.info("Calculating SMA...")
        df_out.ta.sma(length=config.SMA_PERIOD, append=True)

        logger.info("Calculating RSI...")
        df_out.ta.rsi(length=config.RSI_PERIOD, append=True)

        logger.info("Calculating ATR...")
        # Store the result of ta.atr to check its success
        atr_result = df_out.ta.atr(length=config.ATR_PERIOD, append=True)

        # --- Verification and Handling ATR ---
        atr_col_name = (
            f"ATR_{config.ATR_PERIOD}"  # Default name pandas_ta usually creates
        )
        if atr_result is None or atr_col_name not in df_out.columns:
            logger.warning(
                f"pandas_ta failed to calculate or append '{atr_col_name}'. Checking common alternative names..."
            )
            # Check common alternative names pandas_ta might use
            possible_atr_names = [
                "ATR",
                f"ATRr_{config.ATR_PERIOD}",
            ]  # Add others if known
            found_atr_col = None
            for name in possible_atr_names:
                if name in df_out.columns:
                    found_atr_col = name
                    logger.info(
                        f"Found ATR column as '{found_atr_col}'. Using this column."
                    )
                    break
            if not found_atr_col:
                logger.error(
                    "ATR calculation failed completely. Cannot proceed without ATR for SL/TP."
                )
                # Return the DataFrame without ATR, subsequent steps will likely fail or ignore SL/TP
                return df_out
            atr_col_name = found_atr_col  # Use the name that was actually found

        # Ensure ATR column is numeric and handle NaNs before dropna
        if atr_col_name in df_out.columns:
            df_out[atr_col_name] = pd.to_numeric(df_out[atr_col_name], errors="coerce")
            if df_out[atr_col_name].isnull().all():
                logger.error(f"ATR column '{atr_col_name}' contains only NaN values.")
                # Decide how to handle: return, fill with a default, or raise error
                return df_out  # Return partially calculated for now
        else:
            # This case should be caught above, but added for safety
            logger.error(
                f"Critical error: Expected ATR column '{atr_col_name}' not found."
            )
            return df_out

        # --- Drop rows with NaN ATR (using the identified column name) ---
        initial_rows = len(df_out)
        df_out.dropna(subset=[atr_col_name], inplace=True)
        rows_dropped = initial_rows - len(df_out)
        if rows_dropped > 0:
            logger.info(
                f"Dropped {rows_dropped} rows due to initial NaN in '{atr_col_name}'."
            )

        if df_out.empty:
            logger.error("DataFrame is empty after dropping NaN ATR values.")
            return df_out

        logger.info(
            f"Indicators calculated. DataFrame shape after NaN drop: {df_out.shape}"
        )
        return df_out

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}", exc_info=True)
        # Return the DataFrame as it is, possibly with partial indicators
        return df_out
