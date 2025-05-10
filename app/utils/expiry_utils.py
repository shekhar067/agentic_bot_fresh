# app/utils/expiry_utils.py
from datetime import datetime, timedelta, date as DateObject
import calendar
from typing import Dict, List, Set, Optional, Union # MODIFIED (2025-05-10): Added Set for type hinting if used

import pandas as pd

# MODIFIED (2025-05-10): Removed 'from app.simulation_engine import SimpleBacktester' to break circular import

# --- Hardcoded Holiday Lists for 2025 (Based on User Input) ---
# Ensure these are accurate and cover the exchanges/years you need.
# Consider moving these to a config file or a more dynamic source if they change often.
_HOLIDAYS_STR_NSE_2025 = [
    "26-Feb-2025", "14-Mar-2025", "31-Mar-2025", "10-Apr-2025", "14-Apr-2025",
    "18-Apr-2025", "01-May-2025", "15-Aug-2025", "27-Aug-2025", "02-Oct-2025",
    "21-Oct-2025", "22-Oct-2025", "05-Nov-2025", "25-Dec-2025"
]
HOLIDAYS_NSE_2025: List[DateObject] = [datetime.strptime(d_str, "%d-%b-%Y").date() for d_str in _HOLIDAYS_STR_NSE_2025]

_HOLIDAYS_STR_BSE_2025 = [
    "February 26, 2025", "March 14, 2025", "March 31, 2025", "April 10, 2025",
    "April 14, 2025", "April 18, 2025", "May 01, 2025", "August 15, 2025",
    "August 27, 2025", "October 02, 2025", "October 21, 2025",
    "October 22, 2025", "November 05, 2025" # Assuming Dec 25 is also a BSE holiday for 2025
]
HOLIDAYS_BSE_2025: List[DateObject] = [datetime.strptime(d_str, "%B %d, %Y").date() for d_str in _HOLIDAYS_STR_BSE_2025]
if DateObject(2025, 12, 25) not in HOLIDAYS_BSE_2025: # Example: Ensure Christmas is included if it's a standard holiday
    HOLIDAYS_BSE_2025.append(DateObject(2025, 12, 25))


# --- Expiry Day Mapping (Day of week: Monday=0, Sunday=6) ---
NSE_EXPIRY_WEEKDAY_MAP: Dict[str, int] = {
    "NIFTY": 3,       # Thursday
    "BANKNIFTY": 2,   # Wednesday
    "FINNIFTY": 1,    # Tuesday
    "MIDCPNIFTY": 0,  # Monday
    "NIFTYNEXT50": 3  # Assuming Thursday, verify if it has F&O and its expiry day
}
DEFAULT_NSE_EXPIRY_WEEKDAY: int = 3 # Default to Thursday for NSE if symbol not found

BSE_EXPIRY_WEEKDAY_MAP: Dict[str, int] = {
    "SENSEX": 4,      # Friday
    "BANKEX": 0       # Monday
}
DEFAULT_BSE_EXPIRY_WEEKDAY: int = 4 # Default to Friday for BSE

# --- Symbol Sets for Expiry Types (Ensure these are accurate for your needs) ---
# These sets define which indices have which types of expiries.
# All weekly expiring indices typically also have monthly, quarterly, etc.
NSE_WEEKLY_INDICES: Set[str] = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"} # NIFTYNEXT50 might not have weekly F&O
NSE_MONTHLY_INDICES: Set[str] = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNEXT50"}
NSE_QUARTERLY_INDICES: Set[str] = {"NIFTY", "BANKNIFTY"} # Typically March, June, Sep, Dec
NSE_HALF_YEARLY_INDICES: Set[str] = {"NIFTY"} # Typically June, Dec

BSE_WEEKLY_INDICES: Set[str] = {"SENSEX", "BANKEX"}
BSE_MONTHLY_INDICES: Set[str] = {"SENSEX", "BANKEX"}
BSE_QUARTERLY_INDICES: Set[str] = {"SENSEX"}
BSE_HALF_YEARLY_INDICES: Set[str] = {"SENSEX"}


def _get_default_holiday_list(year: int, exchange_segment: str = "NSE") -> List[DateObject]:
    """
    Returns the holiday list for a given year and exchange.
    Currently hardcoded for 2025. Extend for other years or load dynamically.
    `exchange_segment` can be "NSE", "NFO", "BSE", etc. For simplicity, using "NSE" or "BSE".
    """
    # This function should be more robust for multiple years or load from a config/file.
    if year == 2025:
        if exchange_segment.upper() in ["BSE", "BC"]: # BC for BSE Currency
            return HOLIDAYS_BSE_2025
        return HOLIDAYS_NSE_2025 # Default to NSE for NSE, NFO, NCX etc.
    # Add more years or a dynamic loading mechanism here
    # Example:
    # if year == 2024: return HOLIDAYS_NSE_2024
    # logger.warning(f"Holiday list not available for year {year} and exchange {exchange_segment}. Returning empty list.")
    return []

def get_actual_previous_trading_day(target_date: Union[datetime, DateObject], exchange_segment: str = "NSE") -> DateObject:
    """
    Finds the actual previous trading day by skipping weekends and holidays.
    Returns a DateObject.
    """
    current_date_obj = target_date.date() if isinstance(target_date, datetime) else target_date
    holiday_list = _get_default_holiday_list(current_date_obj.year, exchange_segment)
    
    while current_date_obj.weekday() >= 5 or current_date_obj in holiday_list: # Saturday=5, Sunday=6
        current_date_obj -= timedelta(days=1)
        # Re-fetch holiday list if year changes during back-stepping (unlikely for prev day but good practice)
        if current_date_obj.year != (target_date.year if isinstance(target_date, datetime) else target_date.year):
            holiday_list = _get_default_holiday_list(current_date_obj.year, exchange_segment)
            
    return current_date_obj

def get_last_specific_weekday_of_month(year: int, month: int, target_weekday: int, exchange_segment: str = "NSE") -> Optional[DateObject]:
    """
    Gets the last occurrence of a specific weekday (e.g., last Thursday) in a given month,
    ensuring it's a trading day.
    """
    if not (1 <= month <= 12): return None # Invalid month
    
    # Find the last day of the given month
    if month == 12:
        last_day_of_month = DateObject(year, month, 31)
    else:
        last_day_of_month = DateObject(year, month + 1, 1) - timedelta(days=1)

    # Iterate backwards from the last day of the month
    current_check_date = last_day_of_month
    while current_check_date.month == month:
        if current_check_date.weekday() == target_weekday:
            # Found the last target_weekday, now ensure it's a trading day
            return get_actual_previous_trading_day(current_check_date, exchange_segment)
        current_check_date -= timedelta(days=1)
    return None # Should not happen if month and weekday are valid

def get_expiry_weekday_for_symbol(symbol: str, exchange_segment: str = "NSE") -> int:
    """Gets the standard expiry weekday for a given symbol and exchange."""
    symbol_upper = symbol.upper()
    if exchange_segment.upper() in ["BSE", "BC"]:
        return BSE_EXPIRY_WEEKDAY_MAP.get(symbol_upper, DEFAULT_BSE_EXPIRY_WEEKDAY)
    # Default to NSE logic for NSE, NFO, etc.
    return NSE_EXPIRY_WEEKDAY_MAP.get(symbol_upper, DEFAULT_NSE_EXPIRY_WEEKDAY)

def get_expiry_type(symbol: str, query_date_input: Union[datetime, DateObject], exchange_segment: str = "NSE") -> str:
    """
    Determines the type of expiry (weekly, monthly, quarterly, half_yearly, none)
    for a given symbol on a query_date.
    """
    query_date = query_date_input.date() if isinstance(query_date_input, datetime) else query_date_input
    
    # 1. Check if query_date itself is a trading day. If not, it cannot be an expiry day.
    actual_trading_day_for_query = get_actual_previous_trading_day(query_date, exchange_segment)
    if actual_trading_day_for_query != query_date:
        return "none" # query_date is a holiday or weekend

    # 2. Check if query_date's weekday matches the symbol's standard expiry weekday
    symbol_expiry_weekday = get_expiry_weekday_for_symbol(symbol, exchange_segment)
    if query_date.weekday() != symbol_expiry_weekday:
        return "none" # Not the standard expiry day of the week for this symbol

    # 3. Determine if it's the last expiry of the month (for monthly/quarterly/half_yearly check)
    is_last_expiry_of_month = False
    last_expiry_day_obj_of_month = get_last_specific_weekday_of_month(query_date.year, query_date.month, symbol_expiry_weekday, exchange_segment)
    if last_expiry_day_obj_of_month and last_expiry_day_obj_of_month == query_date:
        is_last_expiry_of_month = True

    symbol_upper = symbol.upper()
    exchange_upper = exchange_segment.upper()
    
    # Check longer-term expiries first if it's the last expiry of the month
    if is_last_expiry_of_month:
        if exchange_upper in ["NSE", "NFO"]:
            if symbol_upper in NSE_HALF_YEARLY_INDICES and query_date.month in [6, 12]: return "half_yearly"
            if symbol_upper in NSE_QUARTERLY_INDICES and query_date.month in [3, 6, 9, 12]: return "quarterly"
            if symbol_upper in NSE_MONTHLY_INDICES: return "monthly"
        elif exchange_upper in ["BSE", "BC"]:
            if symbol_upper in BSE_HALF_YEARLY_INDICES and query_date.month in [6, 12]: return "half_yearly"
            if symbol_upper in BSE_QUARTERLY_INDICES and query_date.month in [3, 6, 9, 12]: return "quarterly"
            if symbol_upper in BSE_MONTHLY_INDICES: return "monthly"
            
    # If not a longer-term expiry (or not the last expiry of month for those), check for weekly
    if exchange_upper in ["NSE", "NFO"] and symbol_upper in NSE_WEEKLY_INDICES: return "weekly"
    if exchange_upper in ["BSE", "BC"] and symbol_upper in BSE_WEEKLY_INDICES: return "weekly"
    
    return "none" # Default if no specific expiry type matches

def is_expiry_day(symbol: str, current_date: Union[datetime, DateObject], exchange_segment: str = "NSE") -> bool:
    """
    Checks if the given current_date is an expiry day for the symbol.
    `current_date` should be a datetime.date or datetime.datetime object.
    `exchange_segment` helps determine holidays and expiry rules (e.g., "NSE", "NFO", "BSE").
    """
    # The parameter name is 'current_date' as per this function's definition.
    return get_expiry_type(symbol, current_date, exchange_segment) != "none"

def get_expiry_date_for_week_of(symbol: str, reference_date_input: Union[datetime, DateObject], exchange_segment: str = "NSE") -> Optional[DateObject]:
    """
    Finds the actual expiry date (e.g., Thursday for NIFTY) for the week that reference_date falls into.
    Returns a DateObject or None.
    """
    reference_date = reference_date_input.date() if isinstance(reference_date_input, datetime) else reference_date_input
    
    target_weekday = get_expiry_weekday_for_symbol(symbol, exchange_segment)
    
    # Calculate days to add to reach the target_weekday from reference_date's weekday
    days_to_offset = (target_weekday - reference_date.weekday() + 7) % 7
    potential_expiry_date = reference_date + timedelta(days=days_to_offset)
    
    # Ensure this potential expiry date is a trading day, by moving to previous trading day if it's a holiday/weekend
    actual_expiry_date = get_actual_previous_trading_day(potential_expiry_date, exchange_segment)
    
    # Sanity check: if actual_expiry_date is before the start of the week of reference_date,
    # it means the true expiry for that week was in the previous week (e.g. long holiday streak).
    # This logic might need refinement based on precise definition of "expiry for week of".
    # For now, it returns the calculated trading day.
    return actual_expiry_date


def enrich_expiry_flags(df: pd.DataFrame, symbol: str, exchange_segment: str) -> pd.DataFrame:
    """
    Enriches a DataFrame with 'is_expiry_day', 'expiry_type', and 'expiry_date' columns.
    Assumes df.index is DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex for enrich_expiry_flags.")
    
    df_enriched = df.copy()
    
    # Apply is_expiry_day
    df_enriched["is_expiry_day"] = df_enriched.index.to_series().apply(
        lambda ts: is_expiry_day(symbol, ts.date(), exchange_segment) # Pass date object
    )
    
    # Apply get_expiry_type only for rows where is_expiry_day is True
    df_enriched["expiry_type"] = None # Initialize column
    expiry_day_rows = df_enriched[df_enriched["is_expiry_day"] == True]
    if not expiry_day_rows.empty:
        df_enriched.loc[df_enriched["is_expiry_day"] == True, "expiry_type"] = \
            expiry_day_rows.index.to_series().apply(
                lambda ts: get_expiry_type(symbol, ts.date(), exchange_segment) # Pass date object
            )
            
    # Apply get_expiry_date_for_week_of
    # This will give the expiry date for the week of each row's date.
    df_enriched["week_expiry_date"] = df_enriched.index.to_series().apply(
        lambda ts: get_expiry_date_for_week_of(symbol, ts.date(), exchange_segment) # Pass date object
    )
    
    return df_enriched

if __name__ == '__main__':
    # Example Usage:
    print("--- Expiry Utils Examples ---")
    
    # Test with specific dates
    test_date_1 = datetime(2025, 10, 23) # A Thursday
    test_date_2 = datetime(2025, 10, 21) # A Tuesday (Diwali holiday)
    test_date_3 = datetime(2025, 3, 27)  # Last Thursday of March 2025 (NIFTY Monthly/Quarterly)
    test_date_4 = datetime(2025, 3, 25)  # FINNIFTY Expiry (Tuesday)

    print(f"\nTesting for NIFTY (NSE):")
    print(f"Is {test_date_1.date()} NIFTY expiry? {is_expiry_day('NIFTY', test_date_1, 'NSE')}. Type: {get_expiry_type('NIFTY', test_date_1, 'NSE')}")
    print(f"Is {test_date_2.date()} NIFTY expiry? {is_expiry_day('NIFTY', test_date_2, 'NSE')}. Type: {get_expiry_type('NIFTY', test_date_2, 'NSE')}") # Should be false due to holiday
    print(f"Is {test_date_3.date()} NIFTY expiry? {is_expiry_day('NIFTY', test_date_3, 'NSE')}. Type: {get_expiry_type('NIFTY', test_date_3, 'NSE')}")
    
    print(f"\nTesting for FINNIFTY (NSE):")
    print(f"Is {test_date_4.date()} FINNIFTY expiry? {is_expiry_day('FINNIFTY', test_date_4, 'NSE')}. Type: {get_expiry_type('FINNIFTY', test_date_4, 'NSE')}")
    
    print(f"\nTesting for BANKNIFTY (NSE) - Wednesday Expiry:")
    banknifty_expiry_wed = datetime(2025, 3, 26) # A Wednesday
    print(f"Is {banknifty_expiry_wed.date()} BANKNIFTY expiry? {is_expiry_day('BANKNIFTY', banknifty_expiry_wed, 'NSE')}. Type: {get_expiry_type('BANKNIFTY', banknifty_expiry_wed, 'NSE')}")

    print(f"\nExpiry date for week of {test_date_1.date()} for NIFTY: {get_expiry_date_for_week_of('NIFTY', test_date_1, 'NSE')}")
    # Example: a Monday, NIFTY expiry should be Thursday of that week
    monday_test = datetime(2025, 10, 20) 
    print(f"Expiry date for week of {monday_test.date()} for NIFTY: {get_expiry_date_for_week_of('NIFTY', monday_test, 'NSE')}")
    # Example: a Friday, NIFTY expiry should be Thursday of that week (already passed)
    # This might give next week's if current week's already passed, or previous if logic is strict.
    # Current logic gives the target weekday in the current week, then adjusts for holidays.
    friday_test = datetime(2025, 10, 24)
    print(f"Expiry date for week of {friday_test.date()} for NIFTY: {get_expiry_date_for_week_of('NIFTY', friday_test, 'NSE')}")


    # Test enrich_expiry_flags
    print("\n--- Enrich DataFrame Example ---")
    sample_dates = pd.to_datetime([
        '2025-03-24', '2025-03-25', '2025-03-26', '2025-03-27', '2025-03-28', # Week with FINNIFTY, BANKNIFTY, NIFTY expiries
        '2025-10-20', '2025-10-21', '2025-10-22', '2025-10-23', '2025-10-24'  # Week with holidays
    ])
    sample_df = pd.DataFrame(index=sample_dates, data={'value': range(len(sample_dates))})
    
    enriched_df_nifty = enrich_expiry_flags(sample_df, "NIFTY", "NSE")
    print("\nEnriched NIFTY DataFrame:")
    print(enriched_df_nifty[['is_expiry_day', 'expiry_type', 'week_expiry_date']])

    enriched_df_finnifty = enrich_expiry_flags(sample_df, "FINNIFTY", "NSE")
    print("\nEnriched FINNIFTY DataFrame:")
    print(enriched_df_finnifty[['is_expiry_day', 'expiry_type', 'week_expiry_date']])

