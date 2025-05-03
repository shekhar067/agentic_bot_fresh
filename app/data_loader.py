# Near the top of app/data_loader.py

# Keep imports:
import pandas as pd
import logging
from pathlib import Path
from typing import Optional # Import Optional if using None default
from app.config import config # Keep this to get DATA_FOLDER

logger = logging.getLogger(__name__)
# Keep logging basicConfig if not configured elsewhere

# --- CHANGE THIS FUNCTION DEFINITION ---
# Old: def load_historical_data(file_name: str = config.NIFTY_5MIN_CSV) -> pd.DataFrame:
# New:
def load_historical_data(file_name: str) -> pd.DataFrame:
    """
    Loads historical data from a CSV file in the configured data folder.
    Requires file_name to be provided.
    """
    # No need to check if file_name is None now, as it's required

    file_path = config.DATA_FOLDER / file_name # Construct path using config.DATA_FOLDER
    logger.info(f"Attempting to load data from: {file_path}")

    # ... the rest of the function (file checks, loading, processing) remains exactly the same ...
    if not file_path.is_file():
        logger.error(f"Data file not found at: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    try:
        # (Keep the try block with pd.read_csv, datetime handling, column checks, etc.)
        df = pd.read_csv(file_path)
        datetime_col = None
        possible_dt_cols = ['datetime', 'date', 'timestamp', 'time']
        for col in possible_dt_cols:
            if col in df.columns:
                datetime_col = col
                break
        if datetime_col:
            logger.info(f"Using column '{datetime_col}' as datetime source.")
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df.set_index(datetime_col, inplace=True)
        elif isinstance(df.index, pd.DatetimeIndex):
             logger.info("Using existing DataFrame index as datetime source.")
        else:
             try:
                  logger.warning("No standard datetime column found, attempting to parse first column as index.")
                  df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                  if not isinstance(df.index, pd.DatetimeIndex):
                       raise ValueError("First column could not be parsed as DatetimeIndex.")
             except Exception as e:
                  logger.error(f"Failed to identify or parse a datetime index: {e}")
                  raise ValueError("Could not find or parse a datetime index in the CSV.")

        df.columns = [col.lower() for col in df.columns]
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"CSV file missing required columns: {missing_cols}")
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=required_cols, inplace=True)
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} rows with NaN in OHLCV columns.")
        df.sort_index(inplace=True)
        logger.info(f"Successfully loaded {len(df)} rows from {file_path}. Index: {df.index.min()} to {df.index.max()}")
        return df

    except Exception as e:
        logger.error(f"Failed to load or process data from {file_path}: {e}", exc_info=True)
        raise

# --- END OF FUNCTION ---