# app/feature_engine.py

import pandas as pd
import numpy as np
import logging
import pandas_ta as ta
from typing import Optional, Dict

# Use absolute import
from app.config import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class IndicatorCalculator:
    """Calculates technical indicators and market regime."""

    # Using parameters directly from config in __init__ now
    # DEFAULT_PARAMS = { ... } # Can be removed or kept minimal

    def __init__(self, params: Optional[Dict] = None):
        # Initialize self.params using config values
        self.params = {}
        self.params["sma_periods"] = config.INDICATOR_SMA_PERIODS
        self.params["ema_periods"] = config.INDICATOR_EMA_PERIODS
        self.params["rsi_period"] = config.INDICATOR_RSI_PERIOD
        self.params["atr_period"] = config.INDICATOR_ATR_PERIOD
        self.params["bollinger_period"] = config.INDICATOR_BBANDS_PERIOD
        self.params["bollinger_std"] = config.INDICATOR_BBANDS_STDDEV
        self.params["macd_params"] = config.INDICATOR_MACD_PARAMS
        self.params["vol_ma_len"] = config.INDICATOR_VOL_MA_LEN
        self.params["adx_period"] = config.INDICATOR_ADX_PERIOD
        self.params["stoch_period"] = config.INDICATOR_STOCH_PERIOD
        self.params["stoch_smoothing"] = config.INDICATOR_STOCH_SMOOTHING
        self.params["cci_period"] = config.INDICATOR_CCI_PERIOD
        self.params["supertrend_length"] = config.INDICATOR_SUPERTREND_LENGTH
        self.params["supertrend_multiplier"] = config.INDICATOR_SUPERTREND_MULTIPLIER
        self.params["vwap_enabled"] = config.VWAP_ENABLED
        self.params["vwap_type"] = config.VWAP_TYPE
        self.params["vol_ma_enabled"] = True # Assuming enabled if len set

        # Override with any explicitly passed params
        if params:
            self.params.update(params)

        self._validate_params()
        logger.info("IndicatorCalculator initialized.")
        logger.debug(f"Final calculator params: {self.params}")


    def _validate_params(self):
        # (Keep validation logic from previous version)
        macd_p=self.params.get("macd_params"); adx_p=self.params.get("adx_period")
        if not (isinstance(macd_p, tuple) and len(macd_p)==3 and all(isinstance(p,int) and p>0 for p in macd_p)): logger.error(f"Invalid MACD params: {macd_p}")
        if not (isinstance(adx_p, int) and adx_p>0): logger.error(f"Invalid ADX period: {adx_p}")


    def calculate_session_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        # (Keep VWAP logic from previous version)
        if not isinstance(df.index, pd.DatetimeIndex): logger.error("DatetimeIndex required for session VWAP."); df["vwap"]=np.nan; return df
        try:
            df["volume"]=pd.to_numeric(df["volume"], errors="coerce").fillna(0); tp=(df["high"]+df["low"]+df["close"])/3; tpv=(tp*df["volume"]).rename('tpv_temp')
            vol_cumsum=df.groupby(df.index.date)['volume'].cumsum().replace(0, np.nan); tpv_cumsum=df.groupby(df.index.date)[tpv.name].cumsum()
            df["vwap"]=tpv_cumsum/vol_cumsum; return df
        except Exception as e: logger.error(f"Session VWAP calculation error: {e}"); df["vwap"]=np.nan; return df


    def _calculate_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a 'regime' column based on ADX."""
        # (Keep regime logic from previous version)
        logger.debug("Calculating market regime...")
        adx_col = f"ADX_{self.params.get('adx_period', 14)}" # Use period from params
        trend_thresh = config.REGIME_ADX_THRESHOLD_TREND
        range_thresh = config.REGIME_ADX_THRESHOLD_RANGE
        if adx_col not in df.columns: logger.error(f"ADX column '{adx_col}' not found for regime detection."); df['regime']='Unknown'; return df
        is_trending = df[adx_col] > trend_thresh; is_ranging = df[adx_col] < range_thresh
        df['regime']='Ranging'; df.loc[is_trending,'regime']='Trending'
        logger.info("Market regime calculated based on ADX.")
        return df


    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculates all configured indicators including regime """
        # (Keep logic from previous version - it uses self.params)
        logger.info(f"Calculating all indicators for DataFrame shape {df.shape}...")
        if df.empty: logger.warning("Input DataFrame empty."); return df
        df_out = df.copy()
        try:
            ta_list = []
            # Build ta_list dynamically from self.params (EMA, SMA, RSI, ATR, BBands, MACD, Vol MA, SuperTrend, ADX etc.)
            # ... (keep the loop logic building ta_list as in previous correct version) ...
            ema_periods = self.params.get('ema_periods', []); logger.debug(f"Cfg EMA: {ema_periods}"); # ... (loop) ...
            for p in ema_periods:
                 if isinstance(p, int) and p > 0: ta_list.append({"kind": "ema", "length": p})
            sma_periods = self.params.get('sma_periods', []); logger.debug(f"Cfg SMA: {sma_periods}"); # ... (loop) ...
            for p in sma_periods:
                 if isinstance(p, int) and p > 0: ta_list.append({"kind": "sma", "length": p})
            rsi_period = self.params.get('rsi_period'); logger.debug(f"Cfg RSI: {rsi_period}"); # ... (check & append) ...
            if isinstance(rsi_period, int) and rsi_period > 0: ta_list.append({"kind": "rsi", "length": rsi_period})
            atr_period = self.params.get('atr_period'); logger.debug(f"Cfg ATR: {atr_period}"); # ... (check & append) ...
            if isinstance(atr_period, int) and atr_period > 0: ta_list.append({"kind": "atr", "length": atr_period})
            bb_period = self.params.get('bollinger_period'); bb_std = self.params.get('bollinger_std'); logger.debug(f"Cfg BB: p={bb_period} std={bb_std}"); # ... (check & append) ...
            if isinstance(bb_period, int) and bb_period > 0 and isinstance(bb_std, (float,int)) and bb_std > 0: ta_list.append({"kind": "bbands", "length": bb_period, "std": bb_std})
            macd_params = self.params.get('macd_params'); logger.debug(f"Cfg MACD: {macd_params}"); # ... (check & append) ...
            if isinstance(macd_params, tuple) and len(macd_params) == 3: ta_list.append({"kind": "macd", "fast": macd_params[0], "slow": macd_params[1], "signal": macd_params[2]})
            vol_ma_enabled = self.params.get('vol_ma_enabled', True); vol_ma_len = self.params.get('vol_ma_len'); logger.debug(f"Cfg VolMA: en={vol_ma_enabled} len={vol_ma_len}"); # ... (check & append) ...
            if vol_ma_enabled and isinstance(vol_ma_len, int) and vol_ma_len > 0: ta_list.append({"kind": "sma", "close": "volume", "length": vol_ma_len, "prefix": "VOL"})
            st_len = self.params.get('supertrend_length'); st_mult = self.params.get('supertrend_multiplier'); logger.debug(f"Cfg ST: len={st_len} mult={st_mult}"); # ... (check & append) ...
            if isinstance(st_len, int) and st_len > 0 and isinstance(st_mult, (float, int)) and st_mult > 0: ta_list.append({"kind": "supertrend", "length": st_len, "multiplier": st_mult})
            adx_period = self.params.get('adx_period'); logger.debug(f"Cfg ADX: {adx_period}"); # ... (check & append) ...
            if isinstance(adx_period, int) and adx_period > 0: ta_list.append({"kind": "adx", "length": adx_period})
            # Add Stochastic, CCI etc. if needed

            if not ta_list: logger.warning("No valid indicators configured."); return df_out
            MyStrategy = ta.Strategy(name="All Indicators", ta=ta_list)
            logger.info(f"Applying pandas_ta strategy with {len(ta_list)} indicators...")
            df_out.ta.strategy(MyStrategy)
            logger.info(f"Columns AFTER pandas_ta strategy: {df_out.columns.tolist()}")

            # Post-processing (VWAP, Renaming, Regime Calc, Dropna)
            if self.params.get('vwap_enabled'): # ... (VWAP logic) ...
                 if self.params.get('vwap_type') == 'session': logger.info("Calculating session VWAP..."); df_out = self.calculate_session_vwap(df_out)
                 else: logger.info("Calculating cumulative VWAP..."); tpv = ((df_out['high']+df_out['low']+df_out['close'])/3)*df_out['volume']; df_out['vwap'] = tpv.cumsum()/df_out['volume'].cumsum().replace(0, np.nan)
            vol_ma_col = f'VOL_SMA_{vol_ma_len}'; vol_ma_target='volume_sma' # ... (Vol MA rename) ...
            if vol_ma_enabled and vol_ma_col in df_out.columns: df_out.rename(columns={vol_ma_col: vol_ma_target}, inplace=True); logger.info(f"Renamed '{vol_ma_col}' to '{vol_ma_target}'.")
            atr_col_long = f'ATRr_{atr_period}'; atr_col_short = 'atr'; actual_atr_col = None # ... (ATR rename) ...
            if atr_col_short in df_out.columns: actual_atr_col = atr_col_short
            elif atr_col_long in df_out.columns: df_out.rename(columns={atr_col_long: atr_col_short}, inplace=True); actual_atr_col = atr_col_short
            else: raise ValueError("ATR Calculation failed or column not found.")
            logger.info(f"Using ATR column: '{actual_atr_col}'.")

            # Calculate Regime
            df_out = self._calculate_regime(df_out)

            # Drop NaNs
            initial_rows = len(df_out); subset_cols = [actual_atr_col] if actual_atr_col else []; adx_col_check=f"ADX_{self.params.get('adx_period', 14)}"; # ... (dropna logic) ...
            if adx_col_check in df_out.columns: subset_cols.append(adx_col_check)
            logger.info(f"Dropping NaNs based on columns: {subset_cols}")
            df_out.dropna(subset=subset_cols, inplace=True)
            rows_dropped = initial_rows - len(df_out)
            if rows_dropped > 0: logger.info(f"Dropped {rows_dropped} rows due to NaNs in required columns.")
            if df_out.empty: raise ValueError("No data remaining after dropping NaN values.")

            logger.info(f"Finished calculating indicators. Final shape: {df_out.shape}")
            return df_out.copy()

        except Exception as e:
            logger.error(f"Error during indicator calculation: {e}", exc_info=True)
            raise