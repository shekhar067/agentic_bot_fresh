# app/agentic_core.py

import pandas as pd
import logging
from typing import Dict, Tuple, Callable, Optional
from app.config import config
from app.strategies import strategy_factories
from datetime import datetime


from pymongo import MongoClient


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RuleBasedAgent:
    def __init__(self):
        self.strategy_mapping: Dict[str, Optional[Tuple[str, Callable]]] = {
    "Trending": ("SuperTrend_ADX", strategy_factories.get("SuperTrend_ADX")),
    # --- Use BB strategy for Ranging ---
    "Ranging": ("BB_MeanReversion", strategy_factories.get("BB_MeanReversion")),
    # --- Decide what to do with Momentum ---
    "Momentum": ("EMA_Crossover", strategy_factories.get("EMA_Crossover")), # Keep EMA? Or use another? Or None?
    "Unknown": None, # Or assign a default strategy?
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

    def decide(self, current_row: pd.Series, data_history: pd.DataFrame = None) -> Tuple[str, float, float, float, Optional[str]]:
        regime = current_row.get('regime', 'Unknown')
        symbol = current_row.get('symbol', 'nifty').lower()
        timeframe = current_row.get('timeframe', '5min')
        dt: datetime = current_row.name if isinstance(current_row.name, datetime) else datetime.now()
        
        context = {
            "day": dt.strftime("%A"),
            "session": self._infer_session(dt),
            "is_expiry": current_row.get('is_expiry', False),
            "timeframe": timeframe,
            "symbol": symbol,
            "market_condition": regime
        }

        strategy_name, best_params = RuleBasedAgent.get_top_strategy_for_context(context)
        sl_mult = best_params.get("sl_mult", config.DEFAULT_SL_ATR_MULT)
        tp_mult = best_params.get("tp_mult", config.DEFAULT_TP_ATR_MULT)
        tsl_mult = best_params.get("tsl_mult", config.DEFAULT_TSL_ATR_MULT)
        signal = "hold"

        if strategy_name and strategy_name in strategy_factories:
            strategy_func = strategy_factories[strategy_name](**best_params)
            try:
                signal = strategy_func(current_row, data_history)
                if signal not in ['buy_potential', 'sell_potential', 'hold']:
                    logger.warning(f"Strategy {strategy_name} returned invalid signal: '{signal}'")
                    signal = "hold"
            except Exception as e:
                logger.error(f"Error in strategy '{strategy_name}': {e}")
                signal = "hold"
        else:
            logger.warning(f"No valid strategy found for context: {context}")

        return signal, sl_mult, tp_mult, tsl_mult, strategy_name

    def _infer_session(self, ts: datetime) -> str:
        if ts.time() <= datetime.strptime("10:59", "%H:%M").time():
            return "Morning"
        elif ts.time() <= datetime.strptime("13:29", "%H:%M").time():
            return "Midday"
        return "Afternoon"

    def get_top_strategy_for_context(context: Dict, limit: int = 1) -> Tuple[Optional[str], Dict]:
        """
        Retrieves the top-performing strategy and its tuned parameters based on the given context,
        including market regime (trending, ranging, choppy).

        Args:
            context (Dict): {
                "day": "Monday",
                "session": "Morning",
                "is_expiry": True,
                "timeframe": "5min",
                "symbol": "nifty",
                "market_condition": "trending"
            }
            limit (int): How many top strategies to retrieve

        Returns:
            Tuple[str | None, Dict]: (strategy_name, best_params) or (None, {})
        """
        client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)

        try:
            db = client[config.MONGO_DB_NAME]
            perf_collection = db[config.MONGO_COLLECTION_BACKTEST_RESULTS]
            param_collection = db[config.MONGO_COLLECTION_TUNED_PARAMS]

            # === Build Filter Query for Performance Ranking ===
            filter_query = {
                "day": context.get("day"),
                "session": context.get("session"),
                "is_expiry": context.get("is_expiry"),
                "timeframe": context.get("timeframe"),
                "symbol": context.get("symbol"),
                "market_condition": context.get("market_condition")
            }

            filter_query = {k: v for k, v in filter_query.items() if v is not None}

            top_strats = perf_collection.find(filter_query).sort("performance_score", -1).limit(limit)
            top_strat_docs = list(top_strats)

            if not top_strat_docs:
                logger.warning(f"⚠️ No top strategies found for context: {context}")
                return None, {}

            best_doc = top_strat_docs[0]
            strategy_name = best_doc.get("strategy")

            if not strategy_name:
                logger.error("❌ Strategy field missing in top strategy document.")
                return None, {}

            # === Retrieve Best Tuned Parameters for This Context ===
            param_query = filter_query.copy()
            param_query["strategy"] = strategy_name
            param_doc = param_collection.find_one(param_query)

            best_params = param_doc.get("best_params", {}) if param_doc else {}

            logger.info(f"✅ Selected strategy: {strategy_name} | Params: {best_params} | Context: {context}")
            return strategy_name, best_params

        except Exception as e:
            logger.error(f"❌ Error in get_top_strategy_for_context: {e}", exc_info=True)
            return None, {}

        finally:
            client.close()
