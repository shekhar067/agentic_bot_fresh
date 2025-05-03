import os
import pyotp
import logging
import time

# Only import SmartConnect if needed, handle potential ImportError
try:
    from SmartApi import SmartConnect
except ImportError:
    SmartConnect = None  # Define as None if library is not installed

from app.config import config  # Use relative import

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Global variable to hold the session object
smart_api_obj = None
MAX_RETRIES = 3
RETRY_DELAY = 5


def get_session():
    """Initializes and returns an authenticated SmartConnect session object."""
    global smart_api_obj
    if smart_api_obj:
        # TODO: Add a check here to see if the session is still valid
        # e.g., by making a simple API call like getProfile
        # If invalid, set smart_api_obj = None and re-authenticate
        logger.info("Returning existing SmartAPI session (validity check pending).")
        return smart_api_obj

    if not SmartConnect:
        logger.error("SmartApi library not installed. Cannot create session.")
        # You might want to raise an exception or handle this case differently
        return None  # Cannot proceed

    logger.info("Attempting to create new SmartAPI session...")
    api_key = config.ANGELONE_API_KEY
    client_code = config.ANGELONE_CLIENT_CODE
    password = config.ANGELONE_PASSWORD
    totp_secret = config.ANGELONE_TOTP_SECRET

    if not all([api_key, client_code, password, totp_secret]):
        logger.error("Missing Angel One credentials in config/environment.")
        return None  # Cannot proceed without credentials

    try:
        api = SmartConnect(api_key)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Authentication attempt {attempt}/{MAX_RETRIES}")
                totp = pyotp.TOTP(totp_secret).now()
                login_data = api.generateSession(client_code, password, totp)

                if login_data and login_data.get("status"):
                    logger.info("API authentication successful.")
                    smart_api_obj = api  # Store the authenticated object
                    # TODO: Add token fetching/caching here if needed for other calls
                    return smart_api_obj
                else:
                    error_msg = login_data.get("message", "Unknown error")
                    logger.warning(f"Login attempt {attempt} failed: {error_msg}")

            except Exception as e:
                logger.error(f"Login attempt {attempt} error: {str(e)}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        logger.error(
            f"Max login attempts ({MAX_RETRIES}) reached. Failed to create session."
        )
        return None

    except Exception as e:
        logger.error(f"API initialization failed: {str(e)}", exc_info=True)
        return None


# Add functions here later for fetching data, placing orders etc.
# Example (not used in Phase 1 file loading):
# def fetch_historical_data_live(symbol_token, interval, from_date, to_date):
#     session = get_session()
#     if not session:
#         return None
#     # ... add logic using session.getCandleData ...
