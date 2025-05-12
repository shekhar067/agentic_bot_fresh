# app/feature_engine.py

import pandas as pd
import numpy as np
import logging
import pandas_ta as ta
from typing import Optional, Dict, List, Union
from datetime import datetime, date as DateObject # For type hinting

from app.config import config # Assuming this is how you import your config object
# Import expiry utils for adding expiry-related features
from app.utils.expiry_utils import get_expiry_date_for_week_of, get_expiry_type, is_expiry_day

logger = logging.getLogger(__name__)

class IndicatorCalculator:
    """
    Calculates technical indicators, market regime, volatility status,
    and expiry-related features. Outputs lowercase column names.
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = {}
        # --- Load parameters from config object ---
        self.params["sma_periods"] = getattr(config, "INDICATOR_SMA_PERIODS", (10, 20, 50))
        self.params["ema_periods"] = getattr(config, "INDICATOR_EMA_PERIODS", (9, 11, 14, 16, 21, 50))
        self.params["rsi_period"] = getattr(config, "INDICATOR_RSI_PERIOD", 14)
        self.params["atr_period"] = getattr(config, "INDICATOR_ATR_PERIOD", 14)
        self.params["bollinger_period"] = getattr(config, "INDICATOR_BBANDS_PERIOD", 20)
        self.params["bollinger_std"] = getattr(config, "INDICATOR_BBANDS_STDDEV", 2.0)

        macd_default_tuple = (
            getattr(config, "INDICATOR_MACD_FAST", 12),
            getattr(config, "INDICATOR_MACD_SLOW", 26),
            getattr(config, "INDICATOR_MACD_SIGNAL", 9)
        )
        self.params["macd_params"] = getattr(config, "INDICATOR_MACD_PARAMS", macd_default_tuple)

        self.params["vol_ma_len"] = getattr(config, "INDICATOR_VOL_MA_LEN", 20)
        self.params["vol_ma_enabled"] = getattr(config, "VOL_MA_ENABLED", True)
        self.params["adx_period"] = getattr(config, "INDICATOR_ADX_PERIOD", 14)
        self.params["adx_smoothing_period"] = getattr(config, "INDICATOR_ADX_SMOOTHING", 14) # For ADX line smoothing

        self.params["stoch_period"] = getattr(config, "INDICATOR_STOCH_PERIOD", 14)
        self.params["stoch_k_smoothing"] = getattr(config, "INDICATOR_STOCH_SMOOTHING", 3)
        self.params["stoch_d_smoothing"] = getattr(config, "INDICATOR_STOCH_D_SMOOTHING", 3)

        self.params["cci_period"] = getattr(config, "INDICATOR_CCI_PERIOD", 20)
        self.params["supertrend_length"] = getattr(config, "INDICATOR_SUPERTREND_LENGTH", 10)
        self.params["supertrend_multiplier"] = getattr(config, "INDICATOR_SUPERTREND_MULTIPLIER", 3.0)

        self.params["vwap_enabled"] = getattr(config, "VWAP_ENABLED", True)
        self.params["vwap_type"] = getattr(config, "VWAP_TYPE", 'session')

        self.params["obv_enabled"] = getattr(config, "INDICATOR_OBV_ENABLED", True)
        self.params["obv_ema_period"] = getattr(config, "INDICATOR_OBV_EMA_PERIOD", 21)
        self.params["vwma_enabled"] = getattr(config, "INDICATOR_VWMA_ENABLED", True)
        self.params["vwma_period"] = getattr(config, "INDICATOR_VWMA_PERIOD", 20)

        self.params["ichimoku_enabled"] = getattr(config, "INDICATOR_ICHIMOKU_ENABLED", True)
        self.params["ichimoku_tenkan"] = getattr(config, "INDICATOR_ICHIMOKU_TENKAN", 9)
        self.params["ichimoku_kijun"] = getattr(config, "INDICATOR_ICHIMOKU_KIJUN", 26)
        self.params["ichimoku_senkou_b"] = getattr(config, "INDICATOR_ICHIMOKU_SENKOU_B", 52)
        self.params["ichimoku_signal_period"] = getattr(config, "INDICATOR_ICHIMOKU_CHIKOU_OFFSET", 26) # Chikou offset

        self.params["mfi_enabled"] = getattr(config, "INDICATOR_MFI_ENABLED", True)
        self.params["mfi_period"] = getattr(config, "INDICATOR_MFI_PERIOD", 14)

        self.params["chaikin_osc_enabled"] = getattr(config, "INDICATOR_CHAIKIN_OSC_ENABLED", True)
        self.params["chaikin_osc_fast"] = getattr(config, "INDICATOR_CHAIKIN_OSC_FAST", 3)
        self.params["chaikin_osc_slow"] = getattr(config, "INDICATOR_CHAIKIN_OSC_SLOW", 10)

        self.params["keltner_enabled"] = getattr(config, "INDICATOR_KELTNER_ENABLED", True)
        self.params["keltner_length"] = getattr(config, "INDICATOR_KELTNER_LENGTH", 20)
        self.params["keltner_atr_length"] = getattr(config, "INDICATOR_KELTNER_ATR_LENGTH", 10)
        self.params["keltner_multiplier"] = getattr(config, "INDICATOR_KELTNER_MULTIPLIER", 2.0)
        self.params["keltner_mamode"] = getattr(config, "INDICATOR_KELTNER_MAMODE", "ema")

        self.params["donchian_enabled"] = getattr(config, "INDICATOR_DONCHIAN_ENABLED", True)
        self.params["donchian_lower_period"] = getattr(config, "INDICATOR_DONCHIAN_LOWER_PERIOD", 20)
        self.params["donchian_upper_period"] = getattr(config, "INDICATOR_DONCHIAN_UPPER_PERIOD", 20)
        
        # Parameters for Volatility Status (from config, used by _calculate_volatility_status)
        self.params["regime_atr_ma_period"] = getattr(config, "REGIME_ATR_MA_PERIOD", 20)
        self.params["regime_atr_vol_high_mult"] = getattr(config, "REGIME_ATR_VOL_HIGH_MULT", 1.2)
        self.params["regime_atr_vol_low_mult"] = getattr(config, "REGIME_ATR_VOL_LOW_MULT", 0.8)
        
        # Control for adding expiry features
        self.params["add_expiry_features"] = getattr(config, "ADD_EXPIRY_FEATURES", True)

        if params:
            self.params.update(params)

        self._validate_params()
        logger.info("IndicatorCalculator initialized with parameters.")
        logger.debug(f"Calculator Effective Params: {self.params}")

    def _validate_params(self):
        # MACD params check
        macd_p = self.params.get("macd_params")
        if not (isinstance(macd_p, (list, tuple)) and len(macd_p) == 3 and
                all(isinstance(p_val, int) and p_val > 0 for p_val in macd_p)):
            err_msg = f"Invalid MACD params: {macd_p}. Expected tuple/list of 3 positive integers."
            logger.error(err_msg) # Consider raising ValueError for critical misconfigurations
        # ADX period check
        adx_p = self.params.get("adx_period")
        if not (isinstance(adx_p, int) and adx_p > 0):
            err_msg = f"Invalid ADX period: {adx_p}. Expected positive integer."
            logger.error(err_msg)
        # List of periods check (e.g., SMA, EMA)
        for key in ["sma_periods", "ema_periods"]:
            periods = self.params.get(key, [])
            if not (isinstance(periods, (list, tuple)) and all(isinstance(p, int) and p > 0 for p in periods)):
                err_msg = f"Invalid {key}: {periods}. Expected list/tuple of positive integers."
                logger.error(err_msg)
        # Add more validations for other critical parameters as needed
        logger.debug("Parameter validation (basic checks) successful.")


    def calculate_session_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DatetimeIndex required for session VWAP. VWAP will be NaN.")
            df["vwap"] = np.nan
            return df
        try:
            df_copy = df.copy() 
            df_copy["volume"] = pd.to_numeric(df_copy["volume"], errors="coerce").fillna(0)
            typical_price = (df_copy["high"] + df_copy["low"] + df_copy["close"]) / 3
            tpv_temp_name = '_tpv_temp_vwap' # Temporary column
            df_copy[tpv_temp_name] = typical_price * df_copy["volume"]
            
            # Group by date's date object to reset VWAP daily
            vol_cumsum = df_copy.groupby(df_copy.index.date)['volume'].cumsum().replace(0, np.nan)
            tpv_cumsum = df_copy.groupby(df_copy.index.date)[tpv_temp_name].cumsum()

            df["vwap"] = tpv_cumsum / vol_cumsum
            # No need to drop tpv_temp_name from original df as it was on df_copy
        except Exception as e:
            logger.error(f"Session VWAP calculation error: {e}", exc_info=True)
            df["vwap"] = np.nan # Ensure VWAP column exists even on error
        return df

    def _calculate_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Calculating market regime...")
        adx_period = self.params.get('adx_period', 14)
        adx_col_name = f"adx_{adx_period}" # Expecting lowercase ADX column from pandas_ta output
        
        trend_thresh = self.params.get("regime_adx_threshold_trend", getattr(config, "REGIME_ADX_THRESHOLD_TREND", 25))
        range_thresh = self.params.get("regime_adx_threshold_range", getattr(config, "REGIME_ADX_THRESHOLD_RANGE", 20))

        if adx_col_name not in df.columns:
            logger.error(f"ADX column '{adx_col_name}' not found for regime detection. Regime will be 'unknown'.")
            df['regime'] = 'unknown'
            return df

        df['regime'] = 'choppy' # Default for ADX between range_thresh and trend_thresh
        df.loc[df[adx_col_name] > trend_thresh, 'regime'] = 'trending'
        df.loc[df[adx_col_name] < range_thresh, 'regime'] = 'ranging'
        logger.info(f"Market regime (trending/ranging/choppy) calculated using ADX column '{adx_col_name}'.")
        return df

    def _calculate_volatility_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a 'volatility_status' column based on ATR relative to its MA."""
        logger.debug("Calculating volatility status...")
        atr_col_name = 'atr' # Assumes ATR column is already calculated and named 'atr' (lowercase)
        
        if atr_col_name not in df.columns or df[atr_col_name].isnull().all():
            logger.error(f"ATR column '{atr_col_name}' not found or all NaN. Cannot calculate volatility status.")
            df['volatility_status'] = 'unknown'
            return df

        atr_ma_period = self.params.get("regime_atr_ma_period", 20)
        vol_high_mult = self.params.get("regime_atr_vol_high_mult", 1.2)
        vol_low_mult = self.params.get("regime_atr_vol_low_mult", 0.8)

        if not (isinstance(atr_ma_period, int) and atr_ma_period > 0):
            logger.error(f"Invalid REGIME_ATR_MA_PERIOD: {atr_ma_period}. Cannot calculate volatility status.")
            df['volatility_status'] = 'unknown'
            return df
            
        min_periods_for_sma = max(1, int(atr_ma_period * 0.8)) # Ensure enough periods for a stable MA
        df['atr_sma_temp'] = df[atr_col_name].rolling(window=atr_ma_period, min_periods=min_periods_for_sma).mean()
        
        df['volatility_status'] = 'normal' # Default
        # Apply conditions only where atr_sma_temp is not NaN
        high_vol_condition = df[atr_col_name].notna() & df['atr_sma_temp'].notna() & (df[atr_col_name] > (df['atr_sma_temp'] * vol_high_mult))
        low_vol_condition = df[atr_col_name].notna() & df['atr_sma_temp'].notna() & (df[atr_col_name] < (df['atr_sma_temp'] * vol_low_mult))
        
        df.loc[high_vol_condition, 'volatility_status'] = 'high'
        df.loc[low_vol_condition, 'volatility_status'] = 'low'
        
        #df['volatility_status'].fillna('unknown', inplace=True) # For initial NaNs in atr_sma_temp
        df['volatility_status'] = df['volatility_status'].fillna('unknown')  # Safe and compatible


        df.drop(columns=['atr_sma_temp'], inplace=True, errors='ignore') 
        logger.info(f"Volatility status (high/normal/low/unknown) calculated using ATR MA({atr_ma_period}).")
        return df

    def _add_expiry_features(self, df: pd.DataFrame, symbol: str, exchange: str) -> pd.DataFrame:
    #     logger.debug(f"Attempting to add expiry features for {symbol} on {exchange}...")
    #     if not isinstance(df.index, pd.DatetimeIndex):
    #         logger.error("DatetimeIndex required for adding expiry features. Skipping.")
    #         return df

    #     if df.empty:
    #         logger.warning("DataFrame is empty. Skipping expiry features.")
    #         return df

    #     try:
    #         # Apply expiry-related logic using date index
           
    #         # Ensure datetime index and expiry dates are tz-naive for arithmetic
    #          #Ensure both columns are tz-naive
    #         expiry_dates = pd.to_datetime(df['next_weekly_expiry_dt_temp']).dt.tz_localize(None)
    #         index_dates = pd.to_datetime(df.index).tz_localize(None)

    #         df['days_to_weekly_expiry'] = (expiry_dates - index_dates).days

    #        # df['days_to_weekly_expiry'] = (pd.to_datetime(df['next_weekly_expiry_dt_temp']) - pd.to_datetime(df.index)).dt.days

    #         df['is_expiry_day_flag'] = df.index.to_series().apply(
    #             lambda dt_idx: 1 if is_expiry_day(symbol, dt_idx, exchange) else 0
    #         )

    #         df['week_expiry_type'] = df['next_weekly_expiry_dt_temp'].apply(
    #             lambda exp_date_dt: get_expiry_type(symbol, exp_date_dt, exchange) if pd.notna(exp_date_dt) else 'none'
    #         )

    #         df.drop(columns=['next_weekly_expiry_dt_temp'], inplace=True, errors='ignore')
    #         logger.info("Expiry features (days_to_weekly_expiry, is_expiry_day_flag, week_expiry_type) added.")
    #     except Exception as e:
    #         logger.error(f"Error adding expiry features for {symbol}: {e}", exc_info=True)
    #         for col in ['days_to_weekly_expiry', 'is_expiry_day_flag', 'week_expiry_type']:
    #             if col not in df.columns:
    #                 df[col] = np.nan if 'days' in col else (0 if 'flag' in col else 'unknown')
    #     return df
    #def _add_expiry_features(self, df: pd.DataFrame, symbol: str, exchange: str) -> pd.DataFrame:
        logger.debug(f"Attempting to add expiry features for {symbol} on {exchange}...")

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DatetimeIndex required for adding expiry features. Skipping.")
            return df

        if df.empty:
            logger.warning("DataFrame is empty. Skipping expiry features.")
            return df

        try:
            # ✅ Step 1: Generate the missing column first
            df['next_weekly_expiry_dt_temp'] = df.index.to_series().apply(
                lambda ts: get_expiry_date_for_week_of(symbol, ts, exchange)
            )

            # ✅ Step 2: Proceed with your original logic
            expiry_dates = pd.to_datetime(df['next_weekly_expiry_dt_temp']).dt.tz_localize(None)
            index_dates = pd.to_datetime(df.index).tz_localize(None)

            df['days_to_weekly_expiry'] = (expiry_dates - index_dates).dt.days

            df['is_expiry_day_flag'] = df.index.to_series().apply(
                lambda dt_idx: 1 if is_expiry_day(symbol, dt_idx, exchange) else 0
            )

            df['week_expiry_type'] = df['next_weekly_expiry_dt_temp'].apply(
                lambda exp_date_dt: get_expiry_type(symbol, exp_date_dt, exchange) if pd.notna(exp_date_dt) else 'none'
            )

            df.drop(columns=['next_weekly_expiry_dt_temp'], inplace=True, errors='ignore')
            logger.info("Expiry features (days_to_weekly_expiry, is_expiry_day_flag, week_expiry_type) added.")

        except Exception as e:
            logger.error(f"Error adding expiry features for {symbol}: {e}", exc_info=True)
            for col in ['days_to_weekly_expiry', 'is_expiry_day_flag', 'week_expiry_type']:
                if col not in df.columns:
                    df[col] = np.nan if 'days' in col else (0 if 'flag' in col else 'unknown')

        return df

    def calculate_all_indicators(self, df: pd.DataFrame, symbol: str, exchange: str = "NSE") -> pd.DataFrame:
        logger.info(f"Starting all indicator calculations for {symbol} ({exchange}), DataFrame shape {df.shape}...")
        if df.empty:
            logger.warning("Input DataFrame is empty. Returning as is.")
            return df.copy()

        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        critical_ohlc_cols = ['open', 'high', 'low', 'close']
        if df[critical_ohlc_cols].isnull().all(axis=1).any():
            logger.warning("Some rows have all OHLC as NaN after coercion.")
        
        # It's generally better to drop rows with NaN in essential OHLC *before* indicator calculation
        # if strategies cannot handle them. However, some indicators might still work.
        # For robustness, let's ensure that at least 'close' is not all NaN if used by many indicators.
        # df.dropna(subset=critical_ohlc_cols, how='all', inplace=True) # Drop if ALL are NaN
        # if df.empty:
        #     logger.error("DataFrame empty after dropping rows where all OHLC are NaN.")
        #     return df.copy()


        df_out = df.copy()
        initial_columns = set(df_out.columns)

        try:
            ta_list = []
            # --- Build ta_list dynamically based on self.params ---
            # SMA
            sma_periods = self.params.get('sma_periods', [])
            if sma_periods: logger.debug(f"Cfg SMA Pds: {sma_periods}")
            for p in sma_periods:
                 if isinstance(p, int) and p > 0: ta_list.append({"kind": "sma", "length": p})
            # EMA
            ema_periods = self.params.get('ema_periods', [])
            if ema_periods: logger.debug(f"Cfg EMA Pds: {ema_periods}")
            for p in ema_periods:
                 if isinstance(p, int) and p > 0: ta_list.append({"kind": "ema", "length": p})
            # RSI
            rsi_period = self.params.get('rsi_period')
            if isinstance(rsi_period, int) and rsi_period > 0:
                logger.debug(f"Cfg RSI Pd: {rsi_period}"); ta_list.append({"kind": "rsi", "length": rsi_period})
            # ATR
            atr_period = self.params.get('atr_period')
            if isinstance(atr_period, int) and atr_period > 0:
                logger.debug(f"Cfg ATR Pd: {atr_period}"); ta_list.append({"kind": "atr", "length": atr_period, "mamode": "rma"})
            # BBands
            bb_period, bb_std = self.params.get('bollinger_period'), self.params.get('bollinger_std')
            if isinstance(bb_period,int) and bb_period>0 and isinstance(bb_std,(float,int)) and bb_std>0:
                logger.debug(f"Cfg BB: p={bb_period} std={bb_std}"); ta_list.append({"kind": "bbands", "length": bb_period, "std": bb_std, "mamode":"sma"})
            # MACD
            macd_p = self.params.get('macd_params')
            if isinstance(macd_p, (list,tuple)) and len(macd_p)==3 and all(isinstance(i,int) and i>0 for i in macd_p):
                logger.debug(f"Cfg MACD: {macd_p}"); ta_list.append({"kind":"macd", "fast":macd_p[0], "slow":macd_p[1], "signal":macd_p[2]})
            # Volume MA
            if self.params.get('vol_ma_enabled'):
                vol_ma_len = self.params.get('vol_ma_len')
                if isinstance(vol_ma_len, int) and vol_ma_len > 0:
                    logger.debug(f"Cfg VolMA: en, Len={vol_ma_len}"); ta_list.append({"kind": "sma", "close": "volume", "length": vol_ma_len, "prefix": "vol"})
            # SuperTrend
            st_len, st_mult = self.params.get('supertrend_length'), self.params.get('supertrend_multiplier')
            if isinstance(st_len,int) and st_len>0 and isinstance(st_mult,(float,int)) and st_mult>0:
                logger.debug(f"Cfg ST: len={st_len} mult={st_mult}"); ta_list.append({"kind":"supertrend", "length":st_len, "multiplier":st_mult})
            # ADX
            adx_p, adx_s = self.params.get('adx_period'), self.params.get('adx_smoothing_period')
            if isinstance(adx_p, int) and adx_p > 0:
                adx_params = {"kind": "adx", "length": adx_p}
                if isinstance(adx_s, int) and adx_s > 0 and adx_s != adx_p : # Only add if smoothing is different and valid
                    adx_params["lensig"] = adx_s # pandas_ta uses 'lensig' for ADX smoothing
                    logger.debug(f"Cfg ADX: Pd={adx_p}, Smoothing(lensig)={adx_s}")
                else:
                    logger.debug(f"Cfg ADX: Pd={adx_p} (default smoothing)")
                ta_list.append(adx_params)
            # Stochastic
            stoch_k,stoch_sk,stoch_d = self.params.get('stoch_period'),self.params.get('stoch_k_smoothing'),self.params.get('stoch_d_smoothing')
            if isinstance(stoch_k,int) and stoch_k>0 and isinstance(stoch_sk,int) and stoch_sk>0 and isinstance(stoch_d,int) and stoch_d>0:
                logger.debug(f"Cfg STOCH: k={stoch_k}, smooth_k={stoch_sk}, d={stoch_d}"); ta_list.append({"kind":"stoch", "k":stoch_k, "smooth_k":stoch_sk, "d":stoch_d})
            # CCI
            cci_len = self.params.get('cci_period')
            if isinstance(cci_len, int) and cci_len > 0:
                logger.debug(f"Cfg CCI: {cci_len}"); ta_list.append({"kind": "cci", "length": cci_len})
            # OBV
            if self.params.get("obv_enabled"):
                obv_ema_p = self.params.get('obv_ema_period')
                if isinstance(obv_ema_p, int) and obv_ema_p > 0:
                    logger.debug(f"Cfg OBV: en, EMA_Pd={obv_ema_p}"); ta_list.append({"kind": "obv", "mamode": "ema", "signal": obv_ema_p, "offset": 0, "prefix":"obv"}) # Added prefix for clarity
            # VWMA
            if self.params.get("vwma_enabled") and isinstance(self.params.get("vwma_period"),int) and self.params.get("vwma_period") > 0:
                vwma_p = self.params.get("vwma_period")
                logger.debug(f"Cfg VWMA: en, Pd={vwma_p}"); ta_list.append({"kind": "vwma", "length": vwma_p})
            # Ichimoku
            if self.params.get("ichimoku_enabled"):
                t,k,sb,sp = self.params.get('ichimoku_tenkan'),self.params.get('ichimoku_kijun'),self.params.get('ichimoku_senkou_b'),self.params.get('ichimoku_signal_period')
                if all(isinstance(i,int) and i>0 for i in [t,k,sb,sp]):
                    logger.debug(f"Cfg Ichimoku: en, T={t},K={k},SB={sb},SP={sp}"); ta_list.append({"kind":"ichimoku", "tenkan":t, "kijun":k, "senkou":sb, "signal":sp}) # 'signal' is chikou offset
            # MFI
            if self.params.get("mfi_enabled") and isinstance(self.params.get("mfi_period"),int) and self.params.get("mfi_period") > 0:
                mfi_p = self.params.get("mfi_period")
                logger.debug(f"Cfg MFI: en, Pd={mfi_p}"); ta_list.append({"kind":"mfi", "length":mfi_p})
            # Chaikin Oscillator (ADOSC)
            if self.params.get("chaikin_osc_enabled"):
                f,s = self.params.get('chaikin_osc_fast'), self.params.get('chaikin_osc_slow')
                if isinstance(f,int) and f>0 and isinstance(s,int) and s>0 and f<s : # Ensure fast < slow
                    logger.debug(f"Cfg ADOSC: en, F={f},S={s}"); ta_list.append({"kind":"adosc", "fast":f, "slow":s})
            # Keltner Channels
            if self.params.get("keltner_enabled"):
                kc_l,kc_atr_l,kc_m,kc_ma = self.params.get('keltner_length'),self.params.get('keltner_atr_length'),self.params.get('keltner_multiplier'),self.params.get('keltner_mamode')
                if isinstance(kc_l,int) and kc_l>0 and isinstance(kc_atr_l,int) and kc_atr_l>0 and isinstance(kc_m,(float,int)) and kc_m>0:
                    logger.debug(f"Cfg KC: en, L={kc_l},ATRL={kc_atr_l},M={kc_m},MA={kc_ma}"); ta_list.append({"kind":"kc", "length":kc_l, "scalar":kc_m, "mamode":kc_ma, "atr_length":kc_atr_l})
            # Donchian Channels
            if self.params.get("donchian_enabled"):
                dc_l,dc_u = self.params.get('donchian_lower_period'),self.params.get('donchian_upper_period')
                if isinstance(dc_l,int) and dc_l>0 and isinstance(dc_u,int) and dc_u>0:
                    logger.debug(f"Cfg Donchian: en, DL={dc_l},DU={dc_u}"); ta_list.append({"kind":"donchian", "lower_length":dc_l, "upper_length":dc_u})
            # --- End of ta_list building ---

            if not ta_list:
                logger.warning("No valid indicators configured in ta_list for pandas_ta strategy.")
            else:
                # Ensure volume is float for some pandas_ta indicators if it's not already
                if 'volume' in df_out.columns and pd.api.types.is_numeric_dtype(df_out['volume']) and not pd.api.types.is_float_dtype(df_out['volume']):
                    df_out['volume'] = df_out['volume'].astype(float)
                
                MyStrategy = ta.Strategy(name="DynamicIndicatorStrategy", ta=ta_list)
                logger.info(f"Applying pandas_ta strategy with {len(ta_list)} indicator configurations...")
                df_out.ta.strategy(MyStrategy)

            # Lowercase all newly added columns by pandas_ta
            new_columns_by_ta = set(df_out.columns) - initial_columns
            lowercase_map_ta = {col: col.lower() for col in new_columns_by_ta}
            df_out.rename(columns=lowercase_map_ta, inplace=True)
            logger.info(f"Converted {len(new_columns_by_ta)} pandas_ta generated columns to lowercase.")
            logger.debug(f"Resultant columns after TA and lowercasing: {df_out.columns.tolist()}")

            # --- Post-pandas_ta processing ---
            if self.params.get('vwap_enabled'):
                 if self.params.get('vwap_type') == 'session':
                     logger.info("Calculating session VWAP (output: 'vwap')...")
                     df_out = self.calculate_session_vwap(df_out) # Output 'vwap'
                 else: 
                     logger.info("Calculating cumulative VWAP (output: 'vwap')...")
                     df_out['volume_temp_for_vwap'] = pd.to_numeric(df_out['volume'], errors='coerce').fillna(0)
                     tp = (df_out['high'] + df_out['low'] + df_out['close']) / 3
                     tpv = tp * df_out['volume_temp_for_vwap']
                     df_out['vwap'] = tpv.cumsum() / df_out['volume_temp_for_vwap'].cumsum().replace(0, np.nan)
                     df_out.drop(columns=['volume_temp_for_vwap'], inplace=True, errors='ignore')

            # Specific Renaming for Volume MA (already lowercased by the step above)
            if self.params.get('vol_ma_enabled'):
                vol_ma_len_val = self.params.get("vol_ma_len")
                vol_ma_col_original_lc = f'vol_sma_{vol_ma_len_val}'
                vol_ma_target_lc = 'volume_sma'
                if vol_ma_col_original_lc in df_out.columns:
                    df_out.rename(columns={vol_ma_col_original_lc: vol_ma_target_lc}, inplace=True)
                    logger.info(f"Renamed '{vol_ma_col_original_lc}' to '{vol_ma_target_lc}'.")

            # Standardize ATR column name (already lowercased by the step above)
            atr_p_val = self.params.get('atr_period')
            potential_atr_names_lc = [f'atrr_{atr_p_val}', f'atre_{atr_p_val}', f'atr_{atr_p_val}']
            atr_target_lc = 'atr'
            atr_col_found_and_renamed = False
            for pot_atr_name in potential_atr_names_lc:
                if pot_atr_name in df_out.columns:
                    if pot_atr_name != atr_target_lc:
                        df_out.rename(columns={pot_atr_name: atr_target_lc}, inplace=True)
                        logger.info(f"Renamed '{pot_atr_name}' to '{atr_target_lc}'.")
                    else: # Already named 'atr'
                        logger.info(f"ATR column already correctly named as '{atr_target_lc}'.")
                    atr_col_found_and_renamed = True; break 
            if not atr_col_found_and_renamed: 
                logger.warning(f"Standard ATR column ('{atr_target_lc}') could not be confirmed or created from {potential_atr_names_lc}.")

            # --- Add Custom Feature Sets ---
            df_out = self._calculate_regime(df_out) 
            df_out = self._calculate_volatility_status(df_out) 
            
            if self.params.get("add_expiry_features", True):
                df_out = self._add_expiry_features(df_out, symbol, exchange)

            # --- Final NaN Dropping ---
            initial_rows_final = len(df_out)
            # Define critical columns that must not have NaNs for strategies to start
            # These are typically base indicators that many strategies might depend on.
            subset_cols_for_dropna_lc = []
            if atr_target_lc in df_out.columns and df_out[atr_target_lc].notna().any(): 
                subset_cols_for_dropna_lc.append(atr_target_lc)
            
            adx_col_lc = f"adx_{self.params.get('adx_period', 14)}" # Expected lowercase name
            if adx_col_lc in df_out.columns and df_out[adx_col_lc].notna().any(): 
                subset_cols_for_dropna_lc.append(adx_col_lc)
            
            # Regime and Volatility are usually categorical and might have 'unknown', so not typically part of subset for dropna
            # if 'regime' in df_out.columns and df_out['regime'].notna().any(): subset_cols_for_dropna_lc.append('regime')
            # if 'volatility_status' in df_out.columns and df_out['volatility_status'].notna().any(): subset_cols_for_dropna_lc.append('volatility_status')


            if subset_cols_for_dropna_lc:
                logger.info(f"Dropping rows with NaNs in any of these critical columns: {subset_cols_for_dropna_lc}")
                df_out.dropna(subset=subset_cols_for_dropna_lc, inplace=True)
                rows_dropped = initial_rows_final - len(df_out)
                if rows_dropped > 0: 
                    logger.info(f"Dropped {rows_dropped} rows due to NaNs in critical columns.")
            else:
                logger.warning("No critical subset columns with non-NaN data identified for NaN dropping, or these columns do not exist. Skipping NaN drop based on subset.")

            if df_out.empty and initial_rows_final > 0:
                logger.error("DataFrame became empty after all processing and NaN drop. Check input data quality, indicator periods, and NaN logic.")

            logger.info(f"Finished all feature engineering for {symbol}. Final DataFrame shape: {df_out.shape}")
            return df_out.copy()

        except Exception as e:
            logger.error(f"Critical error during indicator/feature calculation for {symbol}: {e}", exc_info=True)
            # In case of error, return a copy of the original df or df_out up to error point
            # This allows the pipeline to potentially continue or log the partial state.
            return df.copy() if df_out.empty or df_out.equals(df) else df_out.copy()