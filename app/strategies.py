# app/strategies.py

import pandas as pd
# Use absolute import
from app.config import config
import logging # Import logging

# Use a logger specific to this module
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG) # Set level to DEBUG


# EMA Crossover Potential Signal
def ema_crossover_strategy(row: pd.Series, data_history: pd.DataFrame = None) -> str:
    strategy_name = "EMA_Crossover"
    ema_fast_col = f'EMA_{config.EMA_FAST_PERIOD}'
    ema_slow_col = f'EMA_{config.EMA_SLOW_PERIOD}'

    # ✅ Safety check
    if data_history is None or not hasattr(data_history, "index"):
        logger.error(f"[{strategy_name}][{row.name}] ERROR: data_history is None or invalid")
        return 'hold'

    if ema_fast_col not in row or ema_slow_col not in row:
        return 'hold'

    try:
        current_idx = data_history.index.get_loc(row.name)
        if current_idx == 0:
            return 'hold'
        prev_row = data_history.iloc[current_idx - 1]
    except Exception as e:
        logger.error(f"[{strategy_name}][{row.name}] Index lookup error: {e}")
        return 'hold'

    if ema_fast_col not in prev_row or ema_slow_col not in prev_row:
        return 'hold'

    c_fast = row[ema_fast_col]
    c_slow = row[ema_slow_col]
    p_fast = prev_row[ema_fast_col]
    p_slow = prev_row[ema_slow_col]

    if any(pd.isna([c_fast, c_slow, p_fast, p_slow])):
        return 'hold'

    if p_fast <= p_slow and c_fast > c_slow:
        logger.debug(f"[{strategy_name}][{row.name}] Signal='buy_potential' (Crossover Up)")
        return 'buy_potential'
    elif p_fast >= p_slow and c_fast < c_slow:
        logger.debug(f"[{strategy_name}][{row.name}] Signal='sell_potential' (Crossover Down)")
        return 'sell_potential'

    return 'hold'

# RSI Potential Signal
def rsi_strategy(row: pd.Series, data_history: pd.DataFrame = None) -> str:
    strategy_name = "RSI_Basic"
    rsi_col = f'RSI_{config.INDICATOR_RSI_PERIOD}'

    if rsi_col not in row or data_history is None:
        return 'hold'

    current_idx = data_history.index.get_loc(row.name)
    if current_idx == 0:
        return 'hold'

    current_rsi = row[rsi_col]
    prev_rsi = data_history.iloc[current_idx - 1][rsi_col]

    if pd.isna(current_rsi) or pd.isna(prev_rsi):
        return 'hold'

    if current_rsi < 45 and current_rsi > prev_rsi:
        logger.debug(f"[{strategy_name}][{row.name}] Signal='buy_potential' (RSI={current_rsi:.2f}, Rising)")
        return 'buy_potential'
    elif current_rsi > 60 and current_rsi < prev_rsi:
        logger.debug(f"[{strategy_name}][{row.name}] Signal='sell_potential' (RSI={current_rsi:.2f}, Falling)")
        return 'sell_potential'

    return 'hold'

# Strategy dictionary (remains same)
strategy_functions = {
    "EMA_Crossover": ema_crossover_strategy,
    "RSI_Basic": rsi_strategy,
}