import pandas as pd
# Use absolute import
from app.config import config
import logging # Import logging

# Use a logger specific to this module
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG) # Set level to DEBUG

# === Simplified Strategy Confirmation Logic for Cleaner Signal Integration ===

def ema_crossover_strategy(row: pd.Series, data_history: pd.DataFrame = None) -> str:
    strategy_name = "EMA_Crossover"
    ema_fast_col = f'EMA_{config.EMA_FAST_PERIOD}'
    ema_slow_col = f'EMA_{config.EMA_SLOW_PERIOD}'
    if data_history is None or row.name not in data_history.index:
        return 'hold'
    current_idx = data_history.index.get_loc(row.name)
    if current_idx < 1:
        return 'hold'
    prev_row = data_history.iloc[current_idx - 1]
    if any(col not in row or pd.isna(row[col]) or pd.isna(prev_row[col]) for col in [ema_fast_col, ema_slow_col]):
        return 'hold'
    if prev_row[ema_fast_col] <= prev_row[ema_slow_col] and row[ema_fast_col] > row[ema_slow_col]:
        return 'buy_potential'
    elif prev_row[ema_fast_col] >= prev_row[ema_slow_col] and row[ema_fast_col] < row[ema_slow_col]:
        return 'sell_potential'
    return 'hold'

def supertrend_adx_strategy(row: pd.Series, data_history: pd.DataFrame = None) -> str:
    st_col = f'SUPERTd_{config.SUPERTREND_PERIOD}_{config.SUPERTREND_MULTIPLIER}'
    adx_col = f'ADX_{config.INDICATOR_ADX_PERIOD}'
    if st_col not in row or adx_col not in row:
        return 'hold'
    if pd.isna(row[st_col]) or pd.isna(row[adx_col]) or row[adx_col] < 22:
        return 'hold'
    if row[st_col] == 1:
        return 'buy_potential'
    elif row[st_col] == -1:
        return 'sell_potential'
    return 'hold'

def rsi_strategy(row: pd.Series, data_history: pd.DataFrame = None) -> str:
    rsi_col = f'RSI_{config.INDICATOR_RSI_PERIOD}'
    if data_history is None or row.name not in data_history.index:
        return 'hold'
    current_idx = data_history.index.get_loc(row.name)
    if current_idx < 3:
        return 'hold'
    rsi_vals = [data_history.iloc[current_idx - i][rsi_col] for i in range(3)] + [row[rsi_col]]
    if any(pd.isna(v) for v in rsi_vals):
        return 'hold'
    slope = (rsi_vals[3] - rsi_vals[0]) / 3
    if rsi_vals[3] < 40 and slope > 0.5:
        return 'buy_potential'
    elif rsi_vals[3] > 65 and slope < -0.5:
        return 'sell_potential'
    return 'hold'

strategy_functions = {
    "EMA_Crossover": ema_crossover_strategy,
    "SuperTrend_ADX": supertrend_adx_strategy,
    "RSI_Basic": rsi_strategy,
}
