# app/agent_trade_logger.py

import logging
from datetime import datetime, timedelta, timezone # Use timezone-aware UTC
from typing import Dict, Any

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, PyMongoError

# Assuming config is accessible via standard import path
try:
    from app.config import config
except ImportError:
    # Fallback for running script directly or if app module isn't found easily
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from app.config import config

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Ensure logger has handlers if run standalone
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Define standard collection names - adjusted slightly for clarity based on your modes
# You can customize these names further in config.py if needed
DEFAULT_COLLECTION_MAP = {
    "historical_test": "agent_hist_test_trades", # 1) agent mode with historical data
    "dry_sim": "agent_dry_sim_trades",        # 3) agent live data dry run (Simulated execution, no broker)
    "live_paper": "agent_live_paper_trades",  # 3b) Alternative: Live data, using broker's paper trading API
    "live_exec": "agent_live_executed"       # 4) agent live mode (Real execution via broker)
    # Mode 2) "agent dry trades with random data" isn't standard.
    # If needed, add a mode like "synthetic_test": "agent_synth_test_trades"
    # Assuming "dry_sim" is the most likely intended mode for live data without execution.
}
# Optionally override with values from config
COLLECTION_MAP = getattr(config, "AGENT_TRADE_LOG_COLLECTIONS", DEFAULT_COLLECTION_MAP)


class AgentTradeLogger:
    """
    Manages logging of agent trades to MongoDB with efficient connection handling using a Singleton pattern.
    Ensures essential indexes are present on trade log collections.
    """
    _instance = None
    _client = None
    _db = None

    def __new__(cls, *args, **kwargs):
        # Singleton pattern to reuse client/db connection across the application
        if cls._instance is None:
            cls._instance = super(AgentTradeLogger, cls).__new__(cls)
            cls._connect_and_setup() # Initialize connection only once
        # Re-check connection if instance exists but client was lost
        elif cls._client is None or cls._db is None:
             cls._connect_and_setup()

        return cls._instance

    @classmethod
    def _connect_and_setup(cls):
        """Establishes MongoDB connection and ensures indexes."""
        if cls._client is not None: # Already connected or connection attempt failed previously
             # Try pinging again if db is None, maybe connection recovered
             if cls._db is None:
                 try:
                     cls._client.admin.command('ping')
                     cls._db = cls._client[config.MONGO_DB_NAME]
                     logger.info("AgentTradeLogger re-established database handle.")
                 except Exception:
                     logger.error("AgentTradeLogger failed to re-establish database handle.")
                     cls._client = None # Mark as disconnected
                     cls._db = None
             return # Don't try to reconnect multiple times rapidly

        try:
            cls._client = MongoClient(
                config.MONGO_URI,
                serverSelectionTimeoutMS=config.MONGO_TIMEOUT_MS,
                uuidRepresentation='standard' # Recommended setting
            )
            # Ping server to ensure connection before proceeding
            cls._client.admin.command('ping')
            cls._db = cls._client[config.MONGO_DB_NAME]
            logger.info(f"AgentTradeLogger connected to MongoDB: {config.MONGO_URI_DISPLAY}/{config.MONGO_DB_NAME}")

            # Ensure indexes for all known collections
            for mode, collection_name in COLLECTION_MAP.items():
                 cls._ensure_indexes(collection_name)

        except ConnectionFailure as e:
            logger.error(f"AgentTradeLogger MongoDB connection failed: {e}", exc_info=True)
            cls._client = None
            cls._db = None
            # Optionally raise an error to halt execution if DB logging is critical
            # raise ConnectionFailure("Critical: Failed to connect to MongoDB for Trade Logger.") from e
        except Exception as e:
            logger.error(f"AgentTradeLogger initialization error: {e}", exc_info=True)
            cls._client = None
            cls._db = None
            # raise e # Optional: re-raise other unexpected init errors

    @classmethod
    def _ensure_indexes(cls, collection_name: str):
        """Ensure essential indexes exist on the collection."""
        if cls._db is None:
            logger.warning(f"Cannot ensure indexes for {collection_name}: DB connection not available.")
            return
        try:
            collection = cls._db[collection_name]
            # Define desired indexes - adjust fields based on your actual payload and query needs
            indexes_to_ensure = {
                "timestamp_desc": ([("timestamp", DESCENDING)], False),
                "symbol_asc": ([("symbol", ASCENDING)], False),
                "order_id_unique": ([("order_id", ASCENDING)], True), # Unique if order_id is always present and unique per collection
                "client_order_id_unique": ([("client_order_id", ASCENDING)], True), # If using unique client IDs
                "run_id_asc": ([("run_id", ASCENDING)], False),
                "strategy_name_asc": ([("strategy_name", ASCENDING)], False),
                "status_asc": ([("status", ASCENDING)], False),
                "logged_at_desc": ([("logged_at", DESCENDING)], False),
            }
            
            existing_indexes = collection.index_information()
            existing_index_names = list(existing_indexes.keys())

            for index_name, (index_keys, unique) in indexes_to_ensure.items():
                # Check if an index with the same keys already exists (name might differ)
                key_match_exists = False
                for existing_name, index_info in existing_indexes.items():
                     # Compare keys ignoring direction for existence check, but respect uniqueness
                     if index_info['key'] == index_keys:
                          key_match_exists = True
                          break
                
                # Create index if no matching key found
                if not key_match_exists:
                    try:
                        collection.create_index(index_keys, name=index_name, background=True, unique=unique, sparse=unique) # Sparse if unique
                        logger.info(f"Created index '{index_name}' on {collection_name}.")
                    except PyMongoError as idx_e:
                         logger.warning(f"Could not create index '{index_name}' on {collection_name} (might exist with different options?): {idx_e}")

        except PyMongoError as e:
            logger.warning(f"Could not check/ensure indexes for {collection_name}: {e}")
        except Exception as e:
             logger.error(f"Unexpected error ensuring indexes for {collection_name}: {e}", exc_info=False)


    def log_trade(self, trade_payload: Dict[str, Any], mode: str) -> bool:
        """
        Logs a trade event executed or simulated by the agent in the given mode.

        Args:
            trade_payload (dict): Dictionary containing trade event details. 
                                  Essential fields might include: run_id, timestamp (event time), symbol, 
                                  strategy_name, agent_context, direction, quantity, price, 
                                  order_type, status (e.g., PLACED, FILLED, CANCELLED, REJECTED, 
                                  COMPLETED, SL_TRIGGERED, TP_TRIGGERED), order_id, client_order_id, 
                                  pnl (if closed), commission, sl_price, tp_price, reason.
            mode (str): Key from COLLECTION_MAP (e.g., 'historical_test', 'dry_sim', 'live_paper', 'live_exec').

        Returns:
            bool: True if logging was successful, False otherwise.
        """
        if self._db is None:
            logger.error("Cannot log trade: MongoDB connection not available.")
            # Optionally try to reconnect here? Or rely on next call to __new__
            AgentTradeLogger._connect_and_setup() # Attempt reconnect on next call
            if self._db is None: # Check if reconnect worked
                 return False

        collection_name = COLLECTION_MAP.get(mode)
        if not collection_name:
            logger.error(f"Invalid trade logging mode '{mode}'. Valid modes are: {list(COLLECTION_MAP.keys())}")
            return False

        try:
            collection = self._db[collection_name]
            
            # --- Standardize Payload ---
            payload_to_log = trade_payload.copy() # Avoid modifying original dict

            # Add timestamp (use UTC) if not present - use timezone.utc
            if "timestamp" not in payload_to_log:
                 payload_to_log["timestamp"] = datetime.now(timezone.utc)
            elif payload_to_log["timestamp"].tzinfo is None: # Ensure timestamp is timezone-aware
                logger.warning("Trade payload 'timestamp' lacks timezone info. Assuming UTC.")
                payload_to_log["timestamp"] = payload_to_log["timestamp"].replace(tzinfo=timezone.utc)

            # Add logged_at timestamp
            payload_to_log["logged_at"] = datetime.now(timezone.utc)
            payload_to_log["log_mode"] = mode # Store the mode

            # Optional: Add schema version if you evolve the payload structure
            # payload_to_log["schema_version"] = "1.0"

            # --- Insert into DB ---
            result = collection.insert_one(payload_to_log)
            # Use specific fields for more informative logging
            log_msg = (
                 f"Agent trade logged to '{collection_name}'. "
                 f"Mode: {mode}, "
                 f"Symbol: {payload_to_log.get('symbol', 'N/A')}, "
                 f"Status: {payload_to_log.get('status', 'N/A')}, "
                 f"Strategy: {payload_to_log.get('strategy_name', 'N/A')}, "
                 f"DB_ID: {result.inserted_id}"
             )

            logger.info(log_msg)
            return True

        except PyMongoError as e:
            logger.error(f"Failed to log agent trade to '{collection_name}' (Mode: {mode}): {e}", exc_info=True)
            return False
        except Exception as e:
             logger.error(f"Unexpected error logging agent trade to '{collection_name}' (Mode: {mode}): {e}", exc_info=True)
             return False

    def close_connection(self):
        """Closes the MongoDB connection if open. Should be called on application exit."""
        if AgentTradeLogger._client:
            try:
                AgentTradeLogger._client.close()
                AgentTradeLogger._client = None
                AgentTradeLogger._db = None
                logger.info("AgentTradeLogger MongoDB connection closed.")
            except Exception as e:
                logger.error(f"Error closing AgentTradeLogger MongoDB connection: {e}", exc_info=True)

# --- Convenience Function Wrapper (optional but maintains simple interface) ---
# This ensures that the singleton instance is used when calling the function.

def log_agent_trade(trade_payload: dict, mode: str) -> bool:
    """
    Convenience function to log a trade executed by the agent.
    Uses a shared AgentTradeLogger instance.

    Args:
        trade_payload (dict): Dictionary containing trade details.
        mode (str): Key from COLLECTION_MAP (e.g., 'historical_test', 'dry_sim', 'live_paper', 'live_exec').

    Returns:
        bool: True if logging was successful, False otherwise.
    """
    try:
        logger_instance = AgentTradeLogger() # Get or create the singleton instance
        if logger_instance._db is None: # Check if connection failed during init
             logger.error("Trade logger DB connection failed previously. Cannot log trade.")
             return False
        return logger_instance.log_trade(trade_payload, mode)
    except Exception as e:
         # Handles potential errors during AgentTradeLogger instantiation if it wasn't already created
         logger.error(f"Failed to get/use AgentTradeLogger instance: {e}", exc_info=True)
         return False

def close_trade_logger_connection():
     """Closes the shared MongoDB connection used by the trade logger."""
     try:
         logger_instance = AgentTradeLogger() # Get instance
         logger_instance.close_connection()
     except Exception as e:
          logger.warning(f"Attempted to close trade logger connection, but failed or instance wasn't ready: {e}")


# Example Usage (demonstrates using the function):
if __name__ == "__main__":
    print("Running AgentTradeLogger examples...")

    if not hasattr(config, 'MONGO_URI') or not config.MONGO_URI:
        print("MONGO_URI not configured in app/config.py. Exiting example.")
        sys.exit(1)

    # Example 1: Logging a historical test trade
    print("\nLogging historical test trade...")
    trade1 = {
        "run_id": "hist_test_001",
        "timestamp": datetime.now(timezone.utc) - timedelta(days=1), # Event timestamp
        "symbol": "NIFTY", "timeframe": "5min",
        "strategy_name": "SuperTrend_ADX",
        "agent_context": {"market_condition": "Trending", "session": "Morning", "day": "Thursday"},
        "order_type": "LIMIT", "direction": "BUY", "quantity": 50, "price": 18500.50,
        "status": "FILLED", "order_id": "TEST_ORD_1", "client_order_id": "TEST_CLI_1",
        "sl_price": 18450.00, "tp_price": 18600.00,
        "reason": "Entry signal generated"
    }
    success1 = log_agent_trade(trade1, mode="historical_test")
    print(f"Log success (hist_test): {success1}")

    # Example 2: Logging a simulated dry run trade
    print("\nLogging dry run (simulated) trade...")
    trade2 = {
        "run_id": "dry_sim_run_002",
        "timestamp": datetime.now(timezone.utc),
        "symbol": "BANKNIFTY", "timeframe": "15min",
        "strategy_name": "EMA_Crossover",
        "agent_context": {"market_condition": "Ranging", "session": "Midday", "day": "Friday"},
        "order_type": "MARKET", "direction": "SELL", "quantity": 25,
        "status": "SIMULATED_FILL", "fill_price": 44100.00, # Use fill_price for simulated fills
        "order_id": "DRY_SIM_ORD_2", "client_order_id": "DRY_CLI_2",
        "tp_price": 43900.00,
        "reason": "Entry signal generated"
    }
    success2 = log_agent_trade(trade2, mode="dry_sim")
    print(f"Log success (dry_sim): {success2}")

    # Example 3: Logging a live paper trade execution acknowledgement
    print("\nLogging live paper trade...")
    trade3 = {
        "run_id": "live_paper_run_003",
        "timestamp": datetime.now(timezone.utc),
        "symbol": "NIFTY", "timeframe": "5min",
        "strategy_name": "BB_MeanReversion",
        "agent_context": {"market_condition": "Ranging", "session": "Afternoon", "day": "Friday"},
        "order_type": "LIMIT", "direction": "BUY", "quantity": 50, "price": 18480.00,
        "status": "FILLED", # Status from paper trading account
        "order_id": "PAPER_BROKER_ORD_3", # Broker's paper order ID
        "client_order_id": "PAPER_CLI_3",
        "fill_price": 18480.00, "commission": 5.0,
        "sl_price": 18440.00
    }
    success3 = log_agent_trade(trade3, mode="live_paper")
    print(f"Log success (live_paper): {success3}")

    # Example 4: Logging a real live executed trade closure
    print("\nLogging real live executed trade closure...")
    trade4 = {
        "run_id": "live_exec_run_004",
        "timestamp": datetime.now(timezone.utc),
        "symbol": "RELIANCE", "timeframe": "5min",
        "strategy_name": "VolatilityBreakout_BBS",
        "agent_context": {"market_condition": "Volatile", "session": "Morning", "day": "Monday"},
        "order_type": "SL_MARKET", "direction": "SELL", "quantity": 10, # Closing a long position
        "status": "COMPLETED", # Trade closed via SL
        "order_id": "REAL_BROKER_ORD_4_EXIT",
        "client_order_id": "REAL_CLI_4_EXIT",
        "fill_price": 2505.50, "commission": 2.50,
        "pnl": -44.50, # Calculated PnL for the closed trade
        "entry_time": datetime.now(timezone.utc) - timedelta(minutes=30), # Approx entry time
        "exit_time": datetime.now(timezone.utc),
        "reason": "Stop loss triggered"
    }
    success4 = log_agent_trade(trade4, mode="live_exec")
    print(f"Log success (live_exec): {success4}")


    # Clean up shared connection used by convenience function/singleton
    # This should ideally be called once when your application shuts down cleanly.
    close_trade_logger_connection()
    print("\nExamples finished.")