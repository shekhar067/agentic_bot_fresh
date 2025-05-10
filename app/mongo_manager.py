# app/mongo_manager.py
import logging
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ConfigurationError
import sys # MODIFIED (2025-05-09): Added for logger fallback
from pathlib import Path # MODIFIED (2025-05-09): Added for path manipulation

try:
    from app.config import config
except ImportError:
    # MODIFIED (2025-05-09): More robust fallback for path
    current_dir_mongo_mgr = Path(__file__).resolve().parent
    project_root_mongo_mgr = current_dir_mongo_mgr.parent
    if str(project_root_mongo_mgr) not in sys.path:
        sys.path.insert(0, str(project_root_mongo_mgr))
    from app.config import config

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): 
    # MODIFIED (2025-05-09): Use config for log level/format if available
    log_level_mm = getattr(config, "LOG_LEVEL", "INFO")
    log_format_mm = getattr(config, "LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=log_level_mm, format=log_format_mm, handlers=[logging.StreamHandler(sys.stdout)])


class MongoManager:
    """
    Manages a singleton MongoDB client instance for the application.
    The client handles connection pooling internally.
    """
    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None 

    @classmethod
    def get_client(cls) -> Optional[MongoClient]:
        """
        Returns the singleton MongoClient instance.
        Establishes the connection on the first call.
        """
        if cls._client is None: # This check is already correct
            if not hasattr(config, 'MONGO_URI') or not config.MONGO_URI:
                logger.error("MONGO_URI not configured. Cannot establish MongoDB connection.")
                return None
            try:
                logger.info(f"Attempting to connect to MongoDB at {config.MONGO_URI_DISPLAY}...")
                cls._client = MongoClient(
                    config.MONGO_URI,
                    serverSelectionTimeoutMS=config.MONGO_TIMEOUT_MS,
                    uuidRepresentation='standard' 
                )
                cls._client.admin.command('ping')
                logger.info(f"Successfully connected to MongoDB: {config.MONGO_URI_DISPLAY}")
            except ConnectionFailure as e:
                logger.error(f"MongoDB connection failed: {e}", exc_info=True)
                cls._client = None 
            except ConfigurationError as e:
                logger.error(f"MongoDB configuration error: {e}", exc_info=True)
                cls._client = None
            except Exception as e:
                logger.error(f"An unexpected error occurred during MongoDB connection: {e}", exc_info=True)
                cls._client = None
        return cls._client

    @classmethod
    def get_database(cls, db_name: Optional[str] = None) -> Optional[Database]:
        """
        Returns a Database object from the singleton client.
        Uses the default DB name from config if not specified.
        """
        client = cls.get_client()
        # MODIFIED (2025-05-09): Changed 'if client:' to 'if client is not None:'
        if client is not None:
            target_db_name = db_name or getattr(config, "MONGO_DB_NAME", None) # Ensure MONGO_DB_NAME exists in config
            if not target_db_name:
                logger.error("MongoDB database name not configured (config.MONGO_DB_NAME) or provided.")
                return None
            
            # Optimization: Cache the default database object
            if cls._db is None or (cls._db.name != target_db_name): # If no cached DB or different DB requested
                cls._db = client[target_db_name]
            # No need for the 'elif cls._db.name != target_db_name and db_name is not None:'
            # as the above condition handles it. If a specific db_name is given and it's different,
            # it will fetch it. If db_name is None, it uses default. If db_name is same as cached, it uses cache.
            
            return cls._db # Return the (potentially cached) default DB or the specifically requested one
        
        logger.debug("MongoManager.get_database() returning None because client is None.")
        return None

    @classmethod
    def close_client(cls):
        """
        Closes the singleton MongoClient connection if it's open.
        Should be called on application shutdown.
        """
        if cls._client is not None: # Correct check
            try:
                cls._client.close()
                logger.info("MongoDB client connection closed via MongoManager.")
            except Exception as e:
                logger.error(f"Error closing MongoDB client connection: {e}", exc_info=True)
            finally:
                cls._client = None
                cls._db = None 

def close_mongo_connection_on_exit():
    """Ensures the global MongoDB client is closed when the application exits."""
    logger.info("Attempting to close global MongoDB connection on application exit (via MongoManager)...")
    MongoManager.close_client()

if __name__ == '__main__':
    logger.info("Testing MongoManager...")
    db_instance = MongoManager.get_database()
    if db_instance is not None: # MODIFIED (2025-05-09): Correct check
        logger.info(f"Successfully obtained database instance: {db_instance.name}")
        try:
            collections = db_instance.list_collection_names()
            logger.info(f"Collections in '{db_instance.name}': {collections[:5]}...")
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
    else:
        logger.error("Failed to get database instance on first try.")

    client_instance_2 = MongoManager.get_client()
    if client_instance_2 is not None:  # MODIFIED (2025-05-09): Correct check
        logger.info("Successfully obtained client instance again (should be reused).")
    else:
        logger.error("Failed to get client instance on second try.")

    MongoManager.close_client()
    logger.info("Attempting to get client after closing...")
    client_after_close = MongoManager.get_client() # This might reconnect if MONGO_URI is valid
    if client_after_close is not None: # MODIFIED (2025-05-09): Correct check
        logger.info("Client obtained after explicit close (reconnected).")
        MongoManager.close_client() 
    else:
        logger.info("Client is None after explicit close or failed to reconnect.")
