# app/config.py (with suggested additions)

import os
import ast
import logging
from pathlib import Path
from dotenv import load_dotenv

# ... (logger setup, dotenv_path, load_dotenv, get_env_var function - keep as is) ...
logger = logging.getLogger(__name__)
# Minimal basicConfig here, your class logic below will reconfigure if LOGGING_ENABLED
if not logger.handlers: # Avoid adding handlers if already configured by another module
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format='%(asctime)s - %(levelname)s - %(message)s')


dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    logger.warning(f".env file not found at {dotenv_path}. Using defaults or environment variables.")

def get_env_var(env_var_name, default_value, expected_type=str):
    """Helper to get and cast environment variables."""
    env_val = os.getenv(env_var_name)
    if env_val is None:
        # If default_value itself is a string representation of list/tuple for list/tuple type
        if expected_type in (list, tuple) and isinstance(default_value, str):
            try:
                return ast.literal_eval(default_value)
            except (ValueError, SyntaxError, TypeError):
                logger.error(f"Default string value '{default_value}' for {env_var_name} is invalid for list/tuple! Returning empty list.")
                return []
        return default_value

    try:
        if expected_type == bool:
            return env_val.lower() in ('true', '1', 't', 'yes')
        elif expected_type == list or expected_type == tuple:
            evaluated = ast.literal_eval(env_val)
            if isinstance(evaluated, (list, tuple)):
                return evaluated
            else:
                logger.warning(f"Env var {env_var_name}='{env_val}' did not evaluate to a list/tuple. Using default: {default_value}")
                # If default_value itself is a string representation of list/tuple
                if isinstance(default_value, str):
                    try:
                        return ast.literal_eval(default_value)
                    except (ValueError, SyntaxError, TypeError):
                        logger.error(f"Default string value '{default_value}' for {env_var_name} is also invalid for list/tuple! Returning empty list.")
                        return []
                return default_value # default_value is already in correct type
        return expected_type(env_val)
    except (ValueError, SyntaxError, TypeError):
        logger.warning(f"Could not parse env var {env_var_name}='{env_val}' as {expected_type.__name__}. Using default: {default_value}")
        # If default_value itself is a string representation of list/tuple
        if expected_type in (list, tuple) and isinstance(default_value, str):
             try:
                return ast.literal_eval(default_value)
             except (ValueError, SyntaxError, TypeError):
                logger.error(f"Default string value '{default_value}' for {env_var_name} is also invalid for list/tuple! Returning empty list.")
                return []
        return default_value


class Config:
    """Loads configuration from environment variables."""

    # --- Credentials ---
    ANGELONE_API_KEY = get_env_var("ANGELONE_API_KEY", None)
    ANGELONE_CLIENT_CODE = get_env_var("ANGELONE_CLIENT_CODE", None)
    ANGELONE_PASSWORD = get_env_var("ANGELONE_PASSWORD", None)
    ANGELONE_TOTP_SECRET = get_env_var("ANGELONE_TOTP_SECRET", None)

    # --- Logging ---
    LOG_LEVEL = get_env_var("LOG_LEVEL", "INFO").upper()
    LOG_FILE = get_env_var("LOG_FILE", "app.log")
    LOG_FORMAT = get_env_var("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    LOGGING_ENABLED = get_env_var("LOGGING_ENABLED", True, expected_type=bool)

    if LOGGING_ENABLED:
        # Clear existing handlers from root logger if any from module-level basicConfig
        # to prevent duplicate messages if this config module is imported multiple times
        # or if other modules also call basicConfig.
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
        
        logging.basicConfig(
            level=LOG_LEVEL,
            format=LOG_FORMAT,
            handlers=[
                logging.FileHandler(LOG_FILE, mode='a'), # Append mode
                logging.StreamHandler()
            ]
        )
        logger.info(f"Logging configured to level {LOG_LEVEL}, format '{LOG_FORMAT}', and file {LOG_FILE}")
    else:
        # If logging is not enabled via this config, ensure a NullHandler to prevent "No handler found" warnings
        # if other modules try to log without their own setup and this config is imported.
        logger.addHandler(logging.NullHandler())
        # Prevent propagation if we are explicitly disabling logging via this config.
        # This assumes this config module is the central point of logging decision.
        # If other modules might independently set up logging, this line could be problematic.
        # logging.getLogger().propagate = False # Be cautious with this line.

    # --- Data Paths ---
    DATA_FOLDER = Path(__file__).resolve().parent.parent / get_env_var("DATA_FOLDER", "data")
   
    RAW_DATA_FILES = get_env_var("RAW_DATA_FILES", 
        {
            "1min": "nifty_historical_data_1min.csv",
            "3min": "nifty_historical_data_3min.csv",
            "5min": "nifty_historical_data_5min.csv",
            "15min": "nifty_historical_data_15min.csv",
        }, expected_type=dict) # Assuming get_env_var can handle dicts if they are simple strings
                               # Or load them individually as you had:
    # RAW_DATA_FILES = {
    #     "1min":  get_env_var("RAW_1MIN_CSV", "nifty_historical_data_1min.csv"),
    #     "3min":  get_env_var("RAW_3MIN_CSV", "nifty_historical_data_3min.csv"),
    #     "5min":  get_env_var("RAW_5MIN_CSV", "nifty_historical_data_5min.csv"),
    #     "15min": get_env_var("RAW_15MIN_CSV", "nifty_historical_data_15min.csv"),
    # }
    DATA_DIR_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "datawithindicator"
    

    # --- Simulation Parameters ---
    INITIAL_CAPITAL = get_env_var('INITIAL_CAPITAL', 100000.0, expected_type=float)
    DEFAULT_SL_ATR_MULT = get_env_var("DEFAULT_SL_ATR_MULT", 1.5, expected_type=float)
    DEFAULT_TP_ATR_MULT = get_env_var("DEFAULT_TP_ATR_MULT", 2.0, expected_type=float)
    DEFAULT_TSL_ATR_MULT = get_env_var("DEFAULT_TSL_ATR_MULT", 1.2, expected_type=float)
    TRAILING_SL_MULT = get_env_var("TRAILING_SL_MULT", 0.5, expected_type=float)

    COMMISSION_PCT = get_env_var("COMMISSION_PCT", 0.0, expected_type=float)
    SLIPPAGE_PCT = get_env_var("SLIPPAGE_PCT", 0.0, expected_type=float)


    # --- Core Indicator Parameters for Feature Engine ---
    INDICATOR_SMA_PERIODS = get_env_var("INDICATOR_SMA_PERIODS", [10, 20, 50], expected_type=list)
    INDICATOR_EMA_PERIODS = get_env_var("INDICATOR_EMA_PERIODS", [9, 11, 14, 16, 21, 50], expected_type=list)
    INDICATOR_RSI_PERIOD  = get_env_var("INDICATOR_RSI_PERIOD", 14, expected_type=int)
    INDICATOR_ATR_PERIOD  = get_env_var("INDICATOR_ATR_PERIOD", 14, expected_type=int)
    INDICATOR_BBANDS_PERIOD= get_env_var("INDICATOR_BBANDS_PERIOD", 20, expected_type=int)
    INDICATOR_BBANDS_STDDEV= get_env_var("INDICATOR_BBANDS_STDDEV", 2.0, expected_type=float)
    INDICATOR_MACD_PARAMS = get_env_var("INDICATOR_MACD_PARAMS", [12, 26, 9], expected_type=list)
    
    INDICATOR_VOL_MA_LEN  = get_env_var("INDICATOR_VOL_MA_LEN", 20, expected_type=int)
    VOL_MA_ENABLED = get_env_var("VOL_MA_ENABLED", True, expected_type=bool)

    INDICATOR_ADX_PERIOD = get_env_var("INDICATOR_ADX_PERIOD", 14, expected_type=int)
    INDICATOR_ADX_SMOOTHING = get_env_var("INDICATOR_ADX_SMOOTHING", 14, expected_type=int) # For ADX line smoothing

    INDICATOR_STOCH_PERIOD = get_env_var("INDICATOR_STOCH_PERIOD", 14, expected_type=int)
    INDICATOR_STOCH_SMOOTHING = get_env_var("INDICATOR_STOCH_SMOOTHING", 3, expected_type=int)
    INDICATOR_STOCH_D_SMOOTHING = get_env_var("INDICATOR_STOCH_D_SMOOTHING", 3, expected_type=int)

    INDICATOR_CCI_PERIOD = get_env_var("INDICATOR_CCI_PERIOD", 20, expected_type=int)
  
    INDICATOR_SUPERTREND_LENGTH = get_env_var("INDICATOR_SUPERTREND_LENGTH", 10, expected_type=int)
    INDICATOR_SUPERTREND_MULTIPLIER = get_env_var("INDICATOR_SUPERTREND_MULTIPLIER", 3.0, expected_type=float)
    
    VWAP_ENABLED = get_env_var("VWAP_ENABLED", True, expected_type=bool)
    VWAP_TYPE = get_env_var("VWAP_TYPE", 'session')

    INDICATOR_OBV_ENABLED = get_env_var("INDICATOR_OBV_ENABLED", True, expected_type=bool)
    INDICATOR_OBV_EMA_PERIOD = get_env_var("INDICATOR_OBV_EMA_PERIOD", 21, expected_type=int)

    INDICATOR_VWMA_ENABLED = get_env_var("INDICATOR_VWMA_ENABLED", True, expected_type=bool)
    INDICATOR_VWMA_PERIOD = get_env_var("INDICATOR_VWMA_PERIOD", 20, expected_type=int)

    INDICATOR_ICHIMOKU_ENABLED = get_env_var("INDICATOR_ICHIMOKU_ENABLED", True, expected_type=bool)
    INDICATOR_ICHIMOKU_TENKAN = get_env_var("INDICATOR_ICHIMOKU_TENKAN", 9, expected_type=int)
    INDICATOR_ICHIMOKU_KIJUN = get_env_var("INDICATOR_ICHIMOKU_KIJUN", 26, expected_type=int)
    INDICATOR_ICHIMOKU_SENKOU_B = get_env_var("INDICATOR_ICHIMOKU_SENKOU_B", 52, expected_type=int)
    INDICATOR_ICHIMOKU_CHIKOU_OFFSET = get_env_var("INDICATOR_ICHIMOKU_CHIKOU_OFFSET", 26, expected_type=int)

    INDICATOR_MFI_ENABLED = get_env_var("INDICATOR_MFI_ENABLED", True, expected_type=bool)
    INDICATOR_MFI_PERIOD = get_env_var("INDICATOR_MFI_PERIOD", 14, expected_type=int)

    INDICATOR_CHAIKIN_OSC_ENABLED = get_env_var("INDICATOR_CHAIKIN_OSC_ENABLED", True, expected_type=bool)
    INDICATOR_CHAIKIN_OSC_FAST = get_env_var("INDICATOR_CHAIKIN_OSC_FAST", 3, expected_type=int)
    INDICATOR_CHAIKIN_OSC_SLOW = get_env_var("INDICATOR_CHAIKIN_OSC_SLOW", 10, expected_type=int)

    INDICATOR_KELTNER_ENABLED = get_env_var("INDICATOR_KELTNER_ENABLED", True, expected_type=bool)
    INDICATOR_KELTNER_LENGTH = get_env_var("INDICATOR_KELTNER_LENGTH", 20, expected_type=int)
    INDICATOR_KELTNER_ATR_LENGTH = get_env_var("INDICATOR_KELTNER_ATR_LENGTH", 10, expected_type=int)
    INDICATOR_KELTNER_MULTIPLIER = get_env_var("INDICATOR_KELTNER_MULTIPLIER", 2.0, expected_type=float)
    INDICATOR_KELTNER_MAMODE = get_env_var("INDICATOR_KELTNER_MAMODE", "ema")

    INDICATOR_DONCHIAN_ENABLED = get_env_var("INDICATOR_DONCHIAN_ENABLED", True, expected_type=bool)
    INDICATOR_DONCHIAN_LOWER_PERIOD = get_env_var("INDICATOR_DONCHIAN_LOWER_PERIOD", 20, expected_type=int)
    INDICATOR_DONCHIAN_UPPER_PERIOD = get_env_var("INDICATOR_DONCHIAN_UPPER_PERIOD", 20, expected_type=int)
    
    # --- Regime Detection and Feature Engineering Toggles ---
    REGIME_ADX_PERIOD = INDICATOR_ADX_PERIOD 
    REGIME_ADX_THRESHOLD_TREND = get_env_var("REGIME_ADX_THRESHOLD_TREND", 25, expected_type=int)
    REGIME_ADX_THRESHOLD_RANGE = get_env_var("REGIME_ADX_THRESHOLD_RANGE", 20, expected_type=int)
    
    REGIME_ATR_MA_PERIOD = get_env_var("REGIME_ATR_MA_PERIOD", 20, expected_type=int) # For Volatility Status
    REGIME_ATR_VOL_HIGH_MULT = get_env_var("REGIME_ATR_VOL_HIGH_MULT", 1.2, expected_type=float)
    REGIME_ATR_VOL_LOW_MULT = get_env_var("REGIME_ATR_VOL_LOW_MULT", 0.8, expected_type=float)

    ADD_EXPIRY_FEATURES = get_env_var("ADD_EXPIRY_FEATURES", True, expected_type=bool) # For feature_engine


    # --- MongoDB Configuration ---
    MONGO_URI = get_env_var("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME = get_env_var("MONGO_DB_NAME", "trading_bot")
    MONGO_COLLECTION_BACKTEST_RESULTS = get_env_var("MONGO_COLLECTION_BACKTEST_RESULTS", "strategy_backtest_runs")
    MONGO_COLLECTION_TUNED_PARAMS = get_env_var("MONGO_COLLECTION_TUNED_PARAMS", "strategy_tuned_params")
    MONGO_TIMEOUT_MS = int(get_env_var("MONGO_TIMEOUT_MS", 5000))  # Default to 5000 ms if not set
    MONGO_URI_DISPLAY = MONGO_URI.replace("mongodb://", "mongodb://*****@") if "@" in MONGO_URI else MONGO_URI


    # --- Optuna Configuration ---
    OPTUNA_TRIALS_PER_CONTEXT = get_env_var("OPTUNA_TRIALS_PER_CONTEXT", 50, expected_type=int)
    OPTUNA_STUDY_TIMEOUT_SECONDS = get_env_var("OPTUNA_STUDY_TIMEOUT_SECONDS", 3600, expected_type=int) # Default 1hr
    MAX_OPTUNA_WORKERS = get_env_var("MAX_OPTUNA_WORKERS", 4, expected_type=int)
    LOG_DIR = get_env_var("LOG_DIR", str(Path(__file__).resolve().parent.parent / "logs"))
    
    SIM_DF_COL_PREFIX = "sim_"  # Used to prefix simulation columns like sim_position, sim_trade_pnl, etc.
   
    # --- Miscellaneous ---
    USE_MULTIPROCESSING = get_env_var("USE_MULTIPROCESSING", False, expected_type=bool)
    
    DEFAULT_SYMBOL = get_env_var("DEFAULT_SYMBOL", "nifty")
    DEFAULT_MARKET = get_env_var("DEFAULT_MARKET", "NSE")
    DEFAULT_SEGMENT = get_env_var("DEFAULT_SEGMENT", "Index")

    # --- Potentially Redundant or Strategy-Specific Parameters ---
    # These are kept for reference or if used by other modules directly.
    # Feature_engine primarily uses the more general list/tuple based configurations above.
    # INDICATOR_MACD_FAST = get_env_var("INDICATOR_MACD_FAST", 12, expected_type=int)
    # INDICATOR_MACD_SLOW = get_env_var("INDICATOR_MACD_SLOW", 26, expected_type=int)
    # INDICATOR_MACD_SIGNAL = get_env_var("INDICATOR_MACD_SIGNAL", 9, expected_type=int)
    # EMA_FAST_PERIOD = get_env_var("EMA_FAST_PERIOD", 9, expected_type=int)
    # EMA_SLOW_PERIOD = get_env_var("EMA_SLOW_PERIOD", 21, expected_type=int)
    # RSI_PERIOD = get_env_var("RSI_PERIOD", 14, expected_type=int) # This is a duplicate of INDICATOR_RSI_PERIOD


# Create a single config instance for the application to use
config = Config()

# Optional: Check and warn if essential API credentials are not set
if not all([config.ANGELONE_API_KEY, config.ANGELONE_CLIENT_CODE, config.ANGELONE_PASSWORD, config.ANGELONE_TOTP_SECRET]):
    logger.warning(
        f"\nAngel One API credentials not fully set (loaded from .env at {dotenv_path if dotenv_path.exists() else 'not found'}). "
        "Online features might fail."
    )