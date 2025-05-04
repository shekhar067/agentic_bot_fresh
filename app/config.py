# app/config.py (Revised for Multi-Timeframe and Cleaner Indicators)

import os
from dotenv import load_dotenv
from pathlib import Path
import ast # To safely evaluate list/tuple strings from env vars
import logging # Added for warning

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Load environment variables from .env file in the parent directory
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

def get_list_from_env(env_var_name, default_value_str):
    """Safely gets a list/tuple from env var string."""
    env_val = os.getenv(env_var_name, default_value_str)
    try:
        # Use ast.literal_eval for safe evaluation of list/tuple strings
        evaluated = ast.literal_eval(env_val)
        if isinstance(evaluated, (list, tuple)):
            return evaluated
        else:
            logger.warning(f"Env var {env_var_name}='{env_val}' did not evaluate to a list/tuple. Using default: {default_value_str}")
            return ast.literal_eval(default_value_str)
    except (ValueError, SyntaxError, TypeError):
        logger.warning(f"Could not parse env var {env_var_name}='{env_val}'. Using default: {default_value_str}")
        # Ensure default is returned correctly even if it's also a string representation
        try:
            return ast.literal_eval(default_value_str)
        except (ValueError, SyntaxError, TypeError):
            logger.error(f"Default value '{default_value_str}' for {env_var_name} is also invalid!")
            return [] # Return empty list as a safe fallback


class Config:
    """Loads configuration from environment variables."""

    # --- Credentials ---
    ANGELONE_API_KEY = os.getenv("ANGELONE_API_KEY")
    ANGELONE_CLIENT_CODE = os.getenv("ANGELONE_CLIENT_CODE")
    ANGELONE_PASSWORD = os.getenv("ANGELONE_PASSWORD") # Ensure .env uses this name
    ANGELONE_TOTP_SECRET = os.getenv("ANGELONE_TOTP_SECRET")
    INDICATOR_MACD_SLOW = int(os.getenv("INDICATOR_MACD_SLOW", 26))
    INDICATOR_MACD_SIGNAL = int(os.getenv("INDICATOR_MACD_SIGNAL", 9))
    # --- Logging ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE = os.getenv("LOG_FILE", "app.log")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
    LOGGING_ENABLED = os.getenv("LOGGING_ENABLED", "True").lower() in ('true', '1', 't', 'yes')
    if LOGGING_ENABLED:
        logging.basicConfig(
            level=LOG_LEVEL,
            format=LOG_FORMAT,
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler()
            ]
        )
    # --- API Configuration ---
    # --- Data Paths and Filenames ---
    DATA_FOLDER = Path(__file__).parent.parent / os.getenv("DATA_FOLDER", "data")

    # Dictionary mapping timeframe suffix to RAW data filename
    # Allows overriding individual files via environment variables in .env
    RAW_DATA_FILES = {
        "1min":  os.getenv("RAW_1MIN_CSV", "nifty_historical_data_1min.csv"),
        "3min":  os.getenv("RAW_3MIN_CSV", "nifty_historical_data_3min.csv"),
        "5min":  os.getenv("RAW_5MIN_CSV", "nifty_historical_data_5min.csv"),
        "15min": os.getenv("RAW_15MIN_CSV", "nifty_historical_data_15min.csv"),
        # Example for adding more:
        # "1hour": os.getenv("RAW_1HOUR_CSV", "nifty_historical_data_1hour.csv"),
    }

    # --- Simulation Parameters ---
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 100000))
    # --- REMOVE or Comment Out Percentage SL/TP ---
    # DEFAULT_SL_PCT = float(os.getenv("DEFAULT_SL_PCT", 1.0))
    # DEFAULT_TP_PCT = float(os.getenv("DEFAULT_TP_PCT", 2.0))
    # --- ADD ATR Multipliers ---
    DEFAULT_SL_ATR_MULT = float(os.getenv("DEFAULT_SL_ATR_MULT", 1.5)) # Default: 1.5 * ATR for Stop Loss
    DEFAULT_TP_ATR_MULT = float(os.getenv("DEFAULT_TP_ATR_MULT", 2.0)) # Default: 2.0 * ATR for Take Profit
    # DEFAULT_TSL_ATR_MULT = float(os.getenv("DEFAULT_TSL_ATR_MULT", 1.2)) # Add later for Trailing Stops

    COMMISSION_PCT = float(os.getenv("COMMISSION_PCT", 0.0)) # Per side
    SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", 0.0)) # Per side

    # --- Indicator Parameters (remain the same) ---
    # ... (keep all your indicator parameters) ...
    INDICATOR_SMA_PERIODS = get_list_from_env("INDICATOR_SMA_PERIODS", "(10, 20, 50)")
    INDICATOR_EMA_PERIODS = get_list_from_env("INDICATOR_EMA_PERIODS", "(9, 14, 21, 50)")
    INDICATOR_RSI_PERIOD  = int(os.getenv("INDICATOR_RSI_PERIOD", 14))
    INDICATOR_ATR_PERIOD  = int(os.getenv("INDICATOR_ATR_PERIOD", 14)) # Ensure this matches ATR calc
    INDICATOR_BBANDS_PERIOD= int(os.getenv("INDICATOR_BBANDS_PERIOD", 20))
    INDICATOR_BBANDS_STDDEV= float(os.getenv("INDICATOR_BBANDS_STDDEV", 2.0))
    INDICATOR_MACD_PARAMS = get_list_from_env("INDICATOR_MACD_PARAMS", "(12, 26, 9)")
    INDICATOR_VOL_MA_LEN  = int(os.getenv("INDICATOR_VOL_MA_LEN", 20))
    INDICATOR_ADX_PERIOD = int(os.getenv("INDICATOR_ADX_PERIOD", 14))
    INDICATOR_ADX_SMOOTHING = int(os.getenv("INDICATOR_ADX_SMOOTHING", 14))
    INDICATOR_STOCH_PERIOD = int(os.getenv("INDICATOR_STOCH_PERIOD", 14))
    INDICATOR_STOCH_SMOOTHING = int(os.getenv("INDICATOR_STOCH_SMOOTHING", 3))
    INDICATOR_CCI_PERIOD = int(os.getenv("INDICATOR_CCI_PERIOD", 20))
    INDICATOR_ADX_PERIOD = int(os.getenv("INDICATOR_ADX_PERIOD", 14)) # Also used for DMI +/- length
  
    INDICATOR_SUPERTREND_LENGTH = int(os.getenv("INDICATOR_SUPERTREND_LENGTH", 10))
    INDICATOR_SUPERTREND_MULTIPLIER = float(os.getenv("INDICATOR_SUPERTREND_MULTIPLIER", 3.0))
    VWAP_ENABLED = os.getenv("VWAP_ENABLED", "False").lower() in ('true', '1', 't', 'yes')
    VWAP_TYPE = os.getenv("VWAP_TYPE", 'session')
    # --- Regime Detection Parameters (NEW SECTION) ---
    REGIME_ADX_PERIOD = INDICATOR_ADX_PERIOD # Use same period as ADX indicator by default
    REGIME_ADX_THRESHOLD_TREND = int(os.getenv("REGIME_ADX_THRESHOLD_TREND", 25))
    REGIME_ADX_THRESHOLD_RANGE = int(os.getenv("REGIME_ADX_THRESHOLD_RANGE", 20))
    # Optional: Add ATR or BBW thresholds if needed for more complex rules later
    REGIME_ATR_MA_PERIOD = 20
    REGIME_ATR_VOL_HIGH_MULT = 1.2 # e.g., ATR > 1.2 * SMA(ATR)
    REGIME_ATR_VOL_LOW_MULT = 0.8  # e.g., ATR < 0.8 * SMA(ATR)   

  
   



    # Add other parameters used by your IndicatorCalculator and strategies
    # Ensure names here are consistent with how they are accessed/used elsewhere
  
    INDICATOR_MACD_FAST= int(os.getenv("INDICATOR_MACD_FAST", 12))
    # --- VWAP Config ---
    # Use .lower() and check against common "true" values for boolean env vars
   
    EMA_FAST_PERIOD = int(os.getenv("EMA_FAST_PERIOD", 9))
    EMA_SLOW_PERIOD = int(os.getenv("EMA_SLOW_PERIOD", 21))
    RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
    # app/config.py
# ... other params
    DEFAULT_SL_ATR_MULT = 1.5 # Example: 1.5 * ATR for Stop Loss
    DEFAULT_TP_ATR_MULT = 2.0 # Example: 2.0 * ATR for Take Profit
    DEFAULT_TSL_ATR_MULT = 1.2 # Optional: For trailing stop logic later
# Create a single instance for the app to use
config = Config()

# Optional: Print warning if credentials missing
# if not all([config.ANGELONE_API_KEY, config.ANGELONE_CLIENT_CODE, config.ANGELONE_PASSWORD, config.ANGELONE_TOTP_SECRET]):
#     logger.warning(f"\nAngel One API credentials not fully set in {dotenv_path}. Online features will fail.\n")