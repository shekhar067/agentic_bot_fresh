# app/indicator_calculator.py

import sys
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import pandas_ta as ta
from typing import Optional, Dict
from multiprocessing import cpu_count

# Assuming config is in the same parent directory structure
from app.config import config
from app.data_loader import load_historical_data  # Import loader to get raw data

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class IndicatorCalculator:
    """
    Calculates technical indicators using pandas_ta.
    (Based on the comprehensive snippet you provided earlier)
    """

    # Using fewer defaults here, relying on config.py for periods
    DEFAULT_PARAMS = {
        "macd_params": (
            config.INDICATOR_MACD_FAST,
            config.INDICATOR_MACD_SLOW,
            config.INDICATOR_MACD_SIGNAL,
        ),
        "bollinger_std": 2.0,
        "vwap_enabled": config.VWAP_ENABLED,
        "vwap_type": config.VWAP_TYPE,
        "vol_ma_enabled": True,
        "supertrend_length": 10,  # Example, adjust if needed
        "supertrend_multiplier": 3.0,  # Example, adjust if needed
        # Add other specific defaults if not covered by Config
    }

    def __init__(self, params: Optional[Dict] = None):
        # Combine defaults with specific config values and any overrides
        self.params = {**self.DEFAULT_PARAMS}
        # Add periods from config
        self.params["sma_periods"] = config.INDICATOR_SMA_PERIODS
        self.params["ema_periods"] = config.INDICATOR_EMA_PERIODS
        self.params["rsi_period"] = config.INDICATOR_RSI_PERIOD
        self.params["atr_period"] = config.INDICATOR_ATR_PERIOD
        self.params["bollinger_period"] = config.INDICATOR_BBANDS_PERIOD
        # ... add all other relevant periods from config ...
        self.params["vol_ma_len"] = config.INDICATOR_VOL_MA_LEN

        # Override with any explicitly passed params
        if params:
            self.params.update(params)
        # self._validate_params() # Optional: Add validation if needed
        logger.info("IndicatorCalculator initialized.")

    def calculate_session_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate session-based VWAP using the DatetimeIndex."""
        # (Add the calculate_session_vwap method from your snippet here if needed)
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DatetimeIndex required for session VWAP.")
            df["vwap"] = np.nan
            return df
        try:
            # Ensure volume is numeric before calculation
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            # Calculate Typical Price * Volume
            tp = (df["high"] + df["low"] + df["close"]) / 3
            tpv = tp * df["volume"]
            # Group by date and calculate cumulative sum within each group
            vol_cumsum = (
                df.groupby(df.index.date, group_keys=False)["volume"]
                .cumsum()
                .replace(0, np.nan)
            )
            tpv_cumsum = df.groupby(df.index.date, group_keys=False)[
                tpv.name
            ].cumsum()  # Need to ensure tpv has a name if it's a Series
            df["vwap"] = tpv_cumsum / vol_cumsum
            return df
        except Exception as e:
            logger.error(f"Session VWAP calculation error: {e}", exc_info=True)
            df["vwap"] = np.nan
            return df

    # In app/indicator_calculator.py

# (Keep imports and IndicatorCalculator class definition as before)

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculates all configured indicators using pandas_ta """
        logger.info(f"Calculating all indicators for DataFrame shape {df.shape}...")
        if df.empty:
            logger.warning("Input DataFrame is empty.")
            return df
        df_out = df.copy()
        try:
            # --- Build the list of indicators for pandas_ta Strategy ---
            ta_list = []

            # EMAs from config
            ema_periods = self.params.get('ema_periods', [])
            logger.debug(f"Configured EMA periods: {ema_periods}")
            for p in ema_periods:
                if isinstance(p, int) and p > 0:
                    ta_list.append({"kind": "ema", "length": p})
                else:
                    logger.warning(f"Invalid EMA period skipped: {p}")

            # SMAs from config
            sma_periods = self.params.get('sma_periods', [])
            logger.debug(f"Configured SMA periods: {sma_periods}")
            for p in sma_periods:
                 if isinstance(p, int) and p > 0:
                     ta_list.append({"kind": "sma", "length": p})
                 else:
                     logger.warning(f"Invalid SMA period skipped: {p}")

            # RSI from config
            rsi_period = self.params.get('rsi_period')
            logger.debug(f"Configured RSI period: {rsi_period}")
            if isinstance(rsi_period, int) and rsi_period > 0:
                 ta_list.append({"kind": "rsi", "length": rsi_period})

            # ATR from config
            atr_period = self.params.get('atr_period')
            logger.debug(f"Configured ATR period: {atr_period}")
            if isinstance(atr_period, int) and atr_period > 0:
                 ta_list.append({"kind": "atr", "length": atr_period})

            # Bollinger Bands from config
            bb_period = self.params.get('bollinger_period')
            bb_std = self.params.get('bollinger_std')
            logger.debug(f"Configured BBands: period={bb_period}, std={bb_std}")
            if isinstance(bb_period, int) and bb_period > 0 and isinstance(bb_std, (float, int)) and bb_std > 0:
                 ta_list.append({"kind": "bbands", "length": bb_period, "std": bb_std})

            # MACD from config
            macd_params = self.params.get('macd_params')
            logger.debug(f"Configured MACD params: {macd_params}")
            if isinstance(macd_params, tuple) and len(macd_params) == 3 and all(isinstance(p, int) and p > 0 for p in macd_params):
                 ta_list.append({"kind": "macd", "fast": macd_params[0], "slow": macd_params[1], "signal": macd_params[2]})

            # Volume MA from config
            vol_ma_enabled = self.params.get('vol_ma_enabled', True)
            vol_ma_len = self.params.get('vol_ma_len')
            logger.debug(f"Configured Vol MA: enabled={vol_ma_enabled}, length={vol_ma_len}")
            if vol_ma_enabled and isinstance(vol_ma_len, int) and vol_ma_len > 0:
                 # Important: specify the input column for SMA on volume
                 ta_list.append({"kind": "sma", "close": "volume", "length": vol_ma_len, "prefix": "VOL"}) # Use prefix to avoid name clash

            # SuperTrend example from config
            st_len = self.params.get('supertrend_length', 10)
            st_mult = self.params.get('supertrend_multiplier', 3.0)
            logger.debug(f"Configured SuperTrend: length={st_len}, multiplier={st_mult}")
            if isinstance(st_len, int) and st_len > 0 and isinstance(st_mult, (float, int)) and st_mult > 0:
                 ta_list.append({"kind": "supertrend", "length": st_len, "multiplier": st_mult})

            # --- Create and Run pandas_ta Strategy ---
            if not ta_list:
                logger.warning("No valid indicators configured to calculate.")
                return df_out # Return early if no indicators

            MyStrategy = ta.Strategy(
                name="All Indicators",
                description="Calculate standard indicators",
                ta=ta_list # Use the explicitly built list
            )

            logger.info(f"Applying pandas_ta strategy with {len(ta_list)} indicator definitions...")
            df_out.ta.strategy(MyStrategy)
            # --- Log columns *after* calculation ---
            logger.info(f"Columns AFTER pandas_ta strategy: {df_out.columns.tolist()}")

            # --- Handle VWAP separately ---
            if self.params.get('vwap_enabled'):
                 # ... (keep VWAP calculation logic as before) ...
                 pass # Add VWAP logic here if needed

            # --- Rename Columns / Aliases ---
            # Rename Volume MA if prefix was used
            vol_ma_col = f'VOL_SMA_{vol_ma_len}'
            if vol_ma_enabled and vol_ma_col in df_out.columns:
                 df_out.rename(columns={vol_ma_col: 'volume_sma'}, inplace=True) # Use a distinct name
                 logger.info(f"Renamed '{vol_ma_col}' to 'volume_sma'.")

            # Check and rename ATR column (pandas_ta often uses 'ATRr_')
            atr_period = self.params.get("atr_period", 14)
            atr_col_name_long = f'ATRr_{atr_period}'
            atr_col_name_short = 'atr' # The name the backtester expects
            actual_atr_col = None

            if atr_col_name_short in df_out.columns:
                actual_atr_col = atr_col_name_short
                logger.info(f"ATR column already exists as '{actual_atr_col}'.")
            elif atr_col_name_long in df_out.columns:
                 df_out.rename(columns={atr_col_name_long: atr_col_name_short}, inplace=True)
                 actual_atr_col = atr_col_name_short
                 logger.info(f"Renamed '{atr_col_name_long}' to '{actual_atr_col}'.")
            else:
                 logger.error(f"ATR column ('{atr_col_name_short}' or '{atr_col_name_long}') not found after calculation!")
                 # Cannot proceed without ATR if needed for stops
                 raise ValueError("ATR Calculation failed or column not found.")

            # --- Drop initial NaNs (based on confirmed ATR column) ---
            initial_rows = len(df_out)
            df_out.dropna(subset=[actual_atr_col], inplace=True)
            rows_dropped = initial_rows - len(df_out)
            if rows_dropped > 0:
                logger.info(f"Dropped {rows_dropped} rows due to initial NaNs in '{actual_atr_col}'.")

            if df_out.empty:
                 logger.error("DataFrame is empty after dropping NaN ATR values.")
                 raise ValueError("No data remaining after dropping initial NaN values.")

            logger.info(f"Finished calculating indicators. Final shape: {df_out.shape}")
            return df_out.copy() # Return a copy

        except Exception as e:
            logger.error(f"Error during indicator calculation: {e}", exc_info=True)
            raise # Re-raise the error

# (Keep the main function and argparse stuff as before for standalone execution)
# ...


def main():
    """Main function to run indicator calculation as a script"""
    parser = argparse.ArgumentParser(
        description="Calculate technical indicators for historical data."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input raw CSV file path (e.g., data/nifty_historical_data_5min.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path (e.g., data/nifty_with_indicators_5min.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info(f"Indicator Calculator Script Started: {input_path} -> {output_path}")

    try:
        # Load raw data using the loader
        raw_df = load_historical_data(
            input_path.name
        )  # Assuming filename only is needed if using config path

        # Calculate indicators
        calculator = IndicatorCalculator()  # Uses defaults from config
        df_with_indicators = calculator.calculate_all_indicators(raw_df)

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_with_indicators.to_csv(output_path, index=True)  # Keep datetime index
        logger.info(f"Successfully calculated indicators and saved to {output_path}")

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to calculate indicators: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
