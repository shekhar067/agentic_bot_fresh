# app/agentic_core.py

import pandas as pd
import logging
from typing import Dict, Tuple, Callable, Optional
from app.config import config
from app.strategies import strategy_functions

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RuleBasedAgent:
    def __init__(self):
        self.strategy_mapping: Dict[str, Optional[Tuple[str, Callable]]] = {
            "Trending": ("SuperTrend_ADX", strategy_functions.get("SuperTrend_ADX")),
            "Ranging": ("RSI_Basic", strategy_functions.get("RSI_Basic")),
            "Momentum": ("EMA_Crossover", strategy_functions.get("EMA_Crossover")),
            "Unknown": None,
        }

        self.parameter_mapping: Dict[str, Dict[str, float]] = {
            "Trending": {
                "sl_mult": getattr(config, "TREND_SL_ATR_MULT", config.DEFAULT_SL_ATR_MULT * 1.2),
                "tp_mult": getattr(config, "TREND_TP_ATR_MULT", config.DEFAULT_TP_ATR_MULT * 1.5),
            },
            "Ranging": {
                "sl_mult": getattr(config, "RANGE_SL_ATR_MULT", config.DEFAULT_SL_ATR_MULT * 0.8),
                "tp_mult": getattr(config, "RANGE_TP_ATR_MULT", config.DEFAULT_TP_ATR_MULT * 0.8),
            },
            "Momentum": {
                "sl_mult": config.DEFAULT_SL_ATR_MULT,
                "tp_mult": config.DEFAULT_TP_ATR_MULT,
            },
            "Unknown": {
                "sl_mult": config.DEFAULT_SL_ATR_MULT,
                "tp_mult": config.DEFAULT_TP_ATR_MULT,
            }
        }

        for regime, pair in self.strategy_mapping.items():
            if pair is None:
                logger.warning(f"No strategy for regime '{regime}'.")
            else:
                name, func = pair
                if not callable(func):
                    logger.error(f"Strategy '{name}' for regime '{regime}' is not callable.")
                    self.strategy_mapping[regime] = None

        logger.info("RuleBasedAgent initialized.")

    def decide(self, current_row: pd.Series, data_history: pd.DataFrame = None) -> Tuple[str, float, float, Optional[str]]:
        regime = current_row.get('regime', 'Unknown')
        strategy_entry = self.strategy_mapping.get(regime)
        params = self.parameter_mapping.get(regime, self.parameter_mapping["Unknown"])
        sl_mult = params["sl_mult"]
        tp_mult = params["tp_mult"]
        signal = "hold"
        strategy_name = None

        if strategy_entry:
            strategy_name, strategy_func = strategy_entry
            try:
                signal = strategy_func(current_row, data_history)
                if signal not in ['buy_potential', 'sell_potential', 'hold']:
                    logger.warning(f"Strategy {strategy_name} returned invalid signal: '{signal}'")
                    signal = "hold"
            except Exception as e:
                logger.error(f"Error in strategy '{strategy_name}': {e}")
                signal = "hold"
                strategy_name = None

        return signal, sl_mult, tp_mult, strategy_name
