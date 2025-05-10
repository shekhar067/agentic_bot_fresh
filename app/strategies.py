
# app/strategies.py
import pandas as pd
import pandas_ta as ta
import logging
from app.config import config # Assuming config has default values if needed
from typing import Callable, Dict, Any
from datetime import time as dt_time # For ORB
from datetime import datetime, timedelta # For ORB

# Configure logger
logger = logging.getLogger(__name__)
# Basic config if run standalone or not configured elsewhere
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)


def strategy_ema_crossover_factory(ema_short_period: int = 9, ema_long_period: int = 21, **kwargs) -> Callable[[pd.Series, pd.DataFrame], str]:
    """
    Factory for EMA Crossover strategy.
    Calculates EMAs dynamically using pandas_ta.
    """
    strategy_name = f"EMA_Crossover_{ema_short_period}_{ema_long_period}"
    required_lookback = ema_long_period + 5 # Ensure enough data for EMA calculation

    def strategy(row: pd.Series, data_history: pd.DataFrame = None) -> str:
        signal = 'hold'
        current_time_for_log = row.name if hasattr(row, 'name') else 'N/A'

        if data_history is None or len(data_history) < required_lookback:
            # logger.debug(f"[{strategy_name} @ {current_time_for_log}] Not enough data. Holding.")
            return signal

        try:
            # Calculate EMAs dynamically on a recent slice for efficiency
            history_slice = data_history.tail(required_lookback + len(data_history) - data_history.index.get_loc(row.name) if row.name in data_history.index else required_lookback)

            ema_short = ta.ema(history_slice['close'], length=ema_short_period)
            ema_long = ta.ema(history_slice['close'], length=ema_long_period)

            if ema_short is None or ema_long is None or ema_short.empty or ema_long.empty or len(ema_short) < 2 or len(ema_long) < 2:
                # logger.debug(f"[{strategy_name} @ {current_time_for_log}] EMA calculation failed or insufficient results.")
                return signal

            # Get values corresponding to the current row and previous row from the calculated series
            # Find the index of row.name in history_slice.index
            try:
                current_series_idx = ema_short.index.get_loc(row.name)
            except KeyError:
                # logger.debug(f"[{strategy_name} @ {current_time_for_log}] row.name not in indicator index. Holding.")
                return signal


            if current_series_idx < 1:
                return signal # Not enough historical indicator data

            current_ema_short = ema_short.iloc[current_series_idx]
            current_ema_long = ema_long.iloc[current_series_idx]
            prev_ema_short = ema_short.iloc[current_series_idx - 1]
            prev_ema_long = ema_long.iloc[current_series_idx - 1]

            if pd.isna(current_ema_short) or pd.isna(current_ema_long) or pd.isna(prev_ema_short) or pd.isna(prev_ema_long):
                return signal

            if prev_ema_short <= prev_ema_long and current_ema_short > current_ema_long:
                signal = "buy_potential"
            elif prev_ema_short >= prev_ema_long and current_ema_short < current_ema_long:
                signal = "sell_potential"
            
            # logger.debug(f"[{strategy_name} @ {current_time_for_log}] Prev S/L: {prev_ema_short:.2f}/{prev_ema_long:.2f}, Curr S/L: {current_ema_short:.2f}/{current_ema_long:.2f} -> Signal: {signal}")

        except Exception as e:
            logger.error(f"[{strategy_name} @ {current_time_for_log}] Error: {e}", exc_info=False)
            signal = 'hold'
        return signal
    return strategy


def create_supertrend_adx_strategy(st_period: int = 10, st_multiplier: float = 3.0, adx_period: int = 14, adx_threshold: int = 25, **kwargs) -> Callable[[pd.Series, pd.DataFrame], str]:
    strategy_name = f"SuperTrend_ADX_{st_period}_{st_multiplier:.1f}_{adx_period}_{adx_threshold}"
    required_lookback = max(st_period, adx_period) + 50 # Increased lookback for ADX stability

    def strategy(row: pd.Series, data_history: pd.DataFrame = None) -> str:
        signal = 'hold'
        current_time_for_log = row.name if hasattr(row, 'name') else 'N/A'

        required_cols = ['high', 'low', 'close']
        if data_history is None or not all(c in data_history.columns for c in required_cols) or len(data_history) < required_lookback:
            return signal

        try:
            # Take a slice of history. Ensure it's large enough for indicator calculation.
            # The exact index of 'row' in 'data_history' is needed to get the latest indicator value.
            current_data_idx = data_history.index.get_loc(row.name)
            start_slice_idx = max(0, current_data_idx - required_lookback + 1)
            history_slice = data_history.iloc[start_slice_idx : current_data_idx + 1]

            if len(history_slice) < max(st_period, adx_period) + 1: # pandas_ta might need at least length+1
                 return signal

            supertrend_df = ta.supertrend(history_slice['high'], history_slice['low'], history_slice['close'], length=st_period, multiplier=st_multiplier)
            adx_df = ta.adx(history_slice['high'], history_slice['low'], history_slice['close'], length=adx_period)

            if supertrend_df is None or adx_df is None or supertrend_df.empty or adx_df.empty:
                 return signal

            st_direction_col = f'SUPERTd_{st_period}_{st_multiplier}'
            adx_val_col = f'ADX_{adx_period}'

            if st_direction_col not in supertrend_df.columns or adx_val_col not in adx_df.columns:
                 return signal

            current_st_direction = supertrend_df[st_direction_col].iloc[-1]
            current_adx = adx_df[adx_val_col].iloc[-1]

            if pd.notna(current_st_direction) and pd.notna(current_adx) and current_adx >= adx_threshold:
                if current_st_direction == 1:
                    signal = 'buy_potential'
                elif current_st_direction == -1:
                    signal = 'sell_potential'
        except Exception as e:
            logger.error(f"[{strategy_name} @ {current_time_for_log}] Error: {e}", exc_info=False)
            signal = 'hold'
        return signal
    return strategy


def create_rsi_slope_strategy(
    rsi_period: int = 14, 
    rsi_oversold: int = 30, 
    rsi_overbought: int = 70, 
    slope_lookback: int = 3, # How many RSI periods to calculate slope
    slope_threshold_buy: float = 0.5,
    slope_threshold_sell: float = -0.5,
    **kwargs
) -> Callable[[pd.Series, pd.DataFrame], str]:
    strategy_name = f"RSI_Slope_{rsi_period}_{rsi_oversold}_{rsi_overbought}_{slope_lookback}"
    required_lookback = rsi_period + slope_lookback + 5 # For RSI calc and slope

    def strategy(row: pd.Series, data_history: pd.DataFrame = None) -> str:
        signal = 'hold'
        current_time_for_log = row.name if hasattr(row, 'name') else 'N/A'

        if data_history is None or len(data_history) < required_lookback:
            return signal

        try:
            current_data_idx = data_history.index.get_loc(row.name)
            start_slice_idx = max(0, current_data_idx - required_lookback + 1)
            history_slice = data_history.iloc[start_slice_idx : current_data_idx + 1]
            
            if len(history_slice) < rsi_period + 1: return signal

            rsi_series = ta.rsi(history_slice['close'], length=rsi_period)
            if rsi_series is None or rsi_series.empty or len(rsi_series) < slope_lookback:
                return signal

            # Get last 'slope_lookback' RSI values ending at the current point
            # The rsi_series is aligned with history_slice
            if len(rsi_series) <= slope_lookback: return signal # Not enough RSI values for slope

            # current_rsi_val = rsi_series.iloc[-1]
            # rsi_for_slope = rsi_series.iloc[-slope_lookback:] # Last N values
            
            # Ensure we have enough points for slope calculation from rsi_series up to current row
            rsi_val_current = rsi_series.iloc[-1]
            if pd.isna(rsi_val_current): return signal
            
            if len(rsi_series) >= slope_lookback:
                rsi_val_start_slope = rsi_series.iloc[-slope_lookback]
                if pd.isna(rsi_val_start_slope): return signal
                
                # Simple slope: (current_value - start_value) / (number_of_periods -1)
                # Here, number_of_periods is slope_lookback
                # (rsi_vals[last] - rsi_vals[first]) / (slope_lookback -1 if slope_lookback >1 else 1)
                if slope_lookback > 1:
                    slope = (rsi_val_current - rsi_val_start_slope) / (slope_lookback -1)
                elif slope_lookback == 1: # Slope based on 1 point is tricky, compare to previous?
                                      # For now, if slope_lookback is 1, maybe slope isn't meaningful in this way.
                                      # Original code had lookback 3 (4 vals), so slope over 3 periods.
                                      # If slope_lookback is N, it means N points. Slope uses points 0 and N-1.
                    slope = rsi_val_current - rsi_series.iloc[-2] if len(rsi_series) >=2 else 0


                if rsi_val_current < rsi_oversold and slope > slope_threshold_buy:
                    signal = 'buy_potential'
                elif rsi_val_current > rsi_overbought and slope < slope_threshold_sell:
                    signal = 'sell_potential'
        except Exception as e:
            logger.error(f"[{strategy_name} @ {current_time_for_log}] Error: {e}", exc_info=False)
            signal = 'hold'
        return signal
    return strategy


def create_bb_mean_reversion_strategy(bb_period: int = 20, bb_stddev: float = 2.0, **kwargs) -> Callable[[pd.Series, pd.DataFrame], str]:
    strategy_name = f"BB_MeanReversion_{bb_period}_{bb_stddev:.1f}"
    required_lookback = bb_period + 5

    def strategy(row: pd.Series, data_history: pd.DataFrame = None) -> str:
        signal = 'hold'
        current_time_for_log = row.name if hasattr(row, 'name') else 'N/A'

        if data_history is None or len(data_history) < required_lookback:
            return signal
        
        try:
            current_data_idx = data_history.index.get_loc(row.name)
            start_slice_idx = max(0, current_data_idx - required_lookback + 1)
            history_slice = data_history.iloc[start_slice_idx : current_data_idx + 1]

            if len(history_slice) < bb_period : return signal

            bbands_df = ta.bbands(history_slice['close'], length=bb_period, std=bb_stddev)
            if bbands_df is None or bbands_df.empty:
                return signal

            lower_band_col = f'BBL_{bb_period}_{bb_stddev:.1f}'
            upper_band_col = f'BBU_{bb_period}_{bb_stddev:.1f}'

            if lower_band_col not in bbands_df.columns or upper_band_col not in bbands_df.columns:
                return signal

            current_lower_band = bbands_df[lower_band_col].iloc[-1]
            current_upper_band = bbands_df[upper_band_col].iloc[-1]
            current_low = row['low']
            current_high = row['high']

            if pd.isna(current_lower_band) or pd.isna(current_upper_band) or pd.isna(current_low) or pd.isna(current_high):
                return signal

            if current_low <= current_lower_band:
                signal = 'buy_potential'
            elif current_high >= current_upper_band:
                signal = 'sell_potential'
        except Exception as e:
            logger.error(f"[{strategy_name} @ {current_time_for_log}] Error: {e}", exc_info=False)
            signal = 'hold'
        return signal
    return strategy


# Opening Range Breakout Strategy (patched for timezone-aware datetime handling)
def create_opening_range_breakout_strategy(orb_start_time="09:15", orb_duration_minutes=15, **kwargs):
    from datetime import datetime, timedelta

    def strategy(row: pd.Series, data_history: pd.DataFrame = None) -> str:

        ORB_START_TIME = datetime.time(9, 15)
        ORB_DURATION_MINUTES = 15
        ENTRY_BUFFER = 0.05
        STOP_LOSS_BUFFER = 0.05
        TARGET_MULTIPLIER = 1.5

        symbol = row.get("symbol")
        current_time = row.name

        # Get ORB window end datetime (same tz as row.name)
        orb_end_time = current_time.replace(
            hour=ORB_START_TIME.hour,
            minute=ORB_START_TIME.minute,
            second=0,
            microsecond=0
        ) + datetime.timedelta(minutes=ORB_DURATION_MINUTES)

        if current_time <= orb_end_time:
            return "hold"  # Still in ORB formation window

        # Get ORB High/Low from data in the first 15-min window
        orb_start_time = current_time.replace(
            hour=ORB_START_TIME.hour,
            minute=ORB_START_TIME.minute,
            second=0,
            microsecond=0
        )
        orb_mask = (data_history.index >= orb_start_time) & (data_history.index <= orb_end_time)
        orb_data = data_history.loc[orb_mask]

        if orb_data.empty:
            return "hold"

        orb_high = orb_data["high"].max()
        orb_low = orb_data["low"].min()

        price = row["close"]
        returnVal="hold"
        if price > orb_high + ENTRY_BUFFER:
            retreturnVal="buy_potential"
        elif price < orb_low - ENTRY_BUFFER:
            retreturnVal="sell_potential"
        else:
            returnVal= "hold"
            
        return returnVal

    return strategy

def create_volatility_breakout_bbs_strategy(
    bb_period: int = 20, 
    bb_stddev: float = 2.0, 
    squeeze_lookback: int = 20,
    squeeze_tolerance: float = 1.05, # e.g., 1.05 for 5% tolerance above lowest
    **kwargs
) -> Callable[[pd.Series, pd.DataFrame], str]:
    strategy_name = f"VolatilityBreakout_BBS_{bb_period}_{bb_stddev:.1f}_{squeeze_lookback}"
    required_lookback = max(bb_period, squeeze_lookback) + 5

    def strategy(row: pd.Series, data_history: pd.DataFrame = None) -> str:
        signal = 'hold'
        current_time_for_log = row.name if hasattr(row, 'name') else 'N/A'

        if data_history is None or len(data_history) < required_lookback:
            return signal

        try:
            current_data_idx = data_history.index.get_loc(row.name)
            if current_data_idx < squeeze_lookback : return signal # Need enough history for squeeze calc

            start_slice_idx = max(0, current_data_idx - required_lookback + 1) # Slice for current indicators
            history_slice_for_indicators = data_history.iloc[start_slice_idx : current_data_idx + 1]
            
            if len(history_slice_for_indicators) < bb_period : return signal

            bbands_df = ta.bbands(history_slice_for_indicators['close'], length=bb_period, std=bb_stddev)
            if bbands_df is None or bbands_df.empty: return signal

            lower_band_col = f'BBL_{bb_period}_{bb_stddev:.1f}'
            upper_band_col = f'BBU_{bb_period}_{bb_stddev:.1f}'
            bandwidth_col = f'BBB_{bb_period}_{bb_stddev:.1f}' # pandas_ta calculates bandwidth as BBB

            if not all(c in bbands_df.columns for c in [lower_band_col, upper_band_col, bandwidth_col]):
                return signal

            current_upper_band = bbands_df[upper_band_col].iloc[-1]
            current_lower_band = bbands_df[lower_band_col].iloc[-1]
            current_bandwidth = bbands_df[bandwidth_col].iloc[-1]
            
            if pd.isna(current_upper_band) or pd.isna(current_lower_band) or pd.isna(current_bandwidth):
                return signal

            # Squeeze identification based on historical bandwidth up to the *previous* bar
            # Slice for squeeze lookback (ends at previous bar relative to current row)
            squeeze_lookback_data_end_idx = current_data_idx # This means data up to previous bar
            squeeze_lookback_data_start_idx = max(0, squeeze_lookback_data_end_idx - squeeze_lookback)
            
            # We need bandwidth for these historical periods.
            # Re-calculate bbands on a slice that covers this historical lookback period for bandwidth.
            full_hist_for_squeeze_calc_end_idx = current_data_idx # Up to previous bar for defining squeeze
            full_hist_for_squeeze_calc_start_idx = max(0, full_hist_for_squeeze_calc_end_idx - (squeeze_lookback + bb_period))
            full_history_slice_for_squeeze = data_history.iloc[full_hist_for_squeeze_calc_start_idx : full_hist_for_squeeze_calc_end_idx] # Excludes current row

            if len(full_history_slice_for_squeeze) < bb_period + squeeze_lookback -1 : return signal

            historical_bbands = ta.bbands(full_history_slice_for_squeeze['close'], length=bb_period, std=bb_stddev)
            if historical_bbands is None or bandwidth_col not in historical_bbands.columns: return signal
            
            historical_bandwidth = historical_bbands[bandwidth_col]
            if len(historical_bandwidth) < squeeze_lookback : return signal # Not enough bandwidth values

            # Lowest bandwidth in the lookback period *ending at the previous bar*
            lowest_bandwidth_in_lookback = historical_bandwidth.tail(squeeze_lookback).min()
            prev_bar_bandwidth = historical_bandwidth.iloc[-1] # Bandwidth of the bar just before current `row`

            if pd.isna(lowest_bandwidth_in_lookback) or pd.isna(prev_bar_bandwidth): return signal

            in_squeeze = prev_bar_bandwidth <= lowest_bandwidth_in_lookback * squeeze_tolerance

            if in_squeeze:
                if row['close'] > current_upper_band:
                    signal = 'buy_potential'
                elif row['close'] < current_lower_band:
                    signal = 'sell_potential'
        except Exception as e:
            logger.error(f"[{strategy_name} @ {current_time_for_log}] Error: {e}", exc_info=False)
            signal = 'hold'
        return signal
    return strategy


# --- Dictionary of Strategy Factories (for tuner) ---
strategy_factories: Dict[str, Callable[..., Callable[[pd.Series, pd.DataFrame], str]]] = {
    "EMA_Crossover": strategy_ema_crossover_factory,
    "SuperTrend_ADX": create_supertrend_adx_strategy,
    "RSI_Slope": create_rsi_slope_strategy,
    "BB_MeanReversion": create_bb_mean_reversion_strategy,
    "OpeningRangeBreakout": create_opening_range_breakout_strategy,
    "VolatilityBreakout_BBS": create_volatility_breakout_bbs_strategy,
}

# --- Define tunable parameters per strategy ---
tunable_param_space: Dict[str, Dict[str, Dict[str, Any]]] = {
    "EMA_Crossover": {
        "indicator": {
            "ema_short_period": {"type": "int", "low": 5, "high": 25, "step": 1},
            "ema_long_period": {"type": "int", "low": 30, "high": 100, "step": 5}
        },
        "risk": { # Risk params are typically handled by the backtester/agent, but can be tuned per strategy context
            "sl_mult": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.1},
            "tp_mult": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.2},
            "tsl_mult": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.1, "default_disabled": True} # TSL can be optional
        }
    },
    "SuperTrend_ADX": {
        "indicator": {
            "st_period": {"type": "int", "low": 7, "high": 21, "step": 1},
            "st_multiplier": {"type": "float", "low": 1.5, "high": 4.0, "step": 0.5},
            "adx_period": {"type": "int", "low": 10, "high": 25, "step": 1},
            "adx_threshold": {"type": "int", "low": 18, "high": 30, "step": 1}
        },
        "risk": {
            "sl_mult": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.1},
            "tp_mult": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.2},
            "tsl_mult": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.1, "default_disabled": True}
        }
    },
    "RSI_Slope": {
        "indicator": {
            "rsi_period": {"type": "int", "low": 7, "high": 21, "step": 1},
            "rsi_oversold": {"type": "int", "low": 20, "high": 40, "step": 5},
            "rsi_overbought": {"type": "int", "low": 60, "high": 80, "step": 5},
            "slope_lookback": {"type": "int", "low": 2, "high": 5, "step": 1}, # Min 2 for a difference
            "slope_threshold_buy": {"type": "float", "low": 0.1, "high": 2.0, "step": 0.1},
            "slope_threshold_sell": {"type": "float", "low": -2.0, "high": -0.1, "step": 0.1}
        },
        "risk": {
            "sl_mult": {"type": "float", "low": 0.5, "high": 2.5, "step": 0.1},
            "tp_mult": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.2},
            "tsl_mult": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.1, "default_disabled": True}
        }
    },
    "BB_MeanReversion": {
        "indicator": {
            "bb_period": {"type": "int", "low": 10, "high": 30, "step": 1},
            "bb_stddev": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.1}
        },
        "risk": {
            "sl_mult": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.1}, # Tighter SL for mean reversion
            "tp_mult": {"type": "float", "low": 0.8, "high": 2.5, "step": 0.1}, # TP can be smaller
            "tsl_mult": {"type": "float", "low": 0.4, "high": 1.5, "step": 0.1, "default_disabled": True}
        }
    },
    "OpeningRangeBreakout": {
        "indicator": {
            "orb_duration_minutes": {"type": "int", "low": 5, "high": 60, "step": 5},
            "orb_start_hour": {"type": "int", "low": 9, "high": 9, "step": 1, "fixed": True, "value": 9}, # Typically fixed
            "orb_start_minute": {"type": "int", "low": 15, "high": 30, "step": 1, "fixed": False} # Allow some flexibility if needed
        },
        "risk": { # Breakouts might warrant wider SL/TP
            "sl_mult": {"type": "float", "low": 0.8, "high": 3.5, "step": 0.1},
            "tp_mult": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.2},
            "tsl_mult": {"type": "float", "low": 0.8, "high": 3.0, "step": 0.1, "default_disabled": True}
        }
    },
    "VolatilityBreakout_BBS": {
        "indicator": {
            "bb_period": {"type": "int", "low": 15, "high": 30, "step": 1},
            "bb_stddev": {"type": "float", "low": 1.8, "high": 3.0, "step": 0.1},
            "squeeze_lookback": {"type": "int", "low": 10, "high": 30, "step": 1},
            "squeeze_tolerance": {"type": "float", "low": 1.0, "high": 1.2, "step": 0.01} # e.g. 1.0 = exact, 1.05 = 5% wider
        },
        "risk": {
            "sl_mult": {"type": "float", "low": 0.8, "high": 3.0, "step": 0.1},
            "tp_mult": {"type": "float", "low": 1.5, "high": 5.0, "step": 0.2},
            "tsl_mult": {"type": "float", "low": 0.8, "high": 2.5, "step": 0.1, "default_disabled": True}
        }
    }
}

# --- For compatibility with older parts or simple non-tuned runs ---
# This dictionary holds instances created with default factory params.
# default_strategy_functions: Dict[str, Callable[[pd.Series, pd.DataFrame], str]] = {
#      name: factory() for name, factory in strategy_factories.items()
# }
default_strategy_functions: Dict[str, Callable[[pd.Series, pd.DataFrame], str]] = {
    name: factory for name, factory in strategy_factories.items()
}

# Example of how a strategy would be called by the simulation engine:
#
# 1. Optimizer (Optuna) suggests a set of params for "SuperTrend_ADX":
#    params = {"st_period": 10, "st_multiplier": 3.0, "adx_period": 14, "adx_threshold": 25, 
#              "sl_mult": 1.5, "tp_mult": 2.0, "tsl_mult": 1.0}
#
# 2. Get the factory for this strategy:
#    factory_func = strategy_factories["SuperTrend_ADX"]
#
# 3. Create the specific strategy instance with these params:
#    strategy_instance = factory_func(**params) # Unpacks dict into arguments
#
# 4. During backtesting, for each row:
#    signal = strategy_instance(row, full_data_history)
#    # sl_atr = row['atr'] * params['sl_mult'] ... etc. for risk management