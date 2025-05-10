
# app/performance_logger_mongo.py
import logging
from datetime import datetime, timezone 
from typing import Dict, Any, Optional, List 
import pandas as pd 
import numpy as np 
import sys 
from pathlib import Path 

from pymongo.database import Database
from pymongo.errors import PyMongoError

try:
    from app.config import config
    from app.mongo_manager import MongoManager 
except ImportError:
    current_dir_perf_log = Path(__file__).resolve().parent
    project_root_perf_log = current_dir_perf_log.parent
    if str(project_root_perf_log) not in sys.path:
        sys.path.insert(0, str(project_root_perf_log))
    from app.config import config
    from app.mongo_manager import MongoManager

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): 
    log_level_pl = getattr(config, "LOG_LEVEL", "INFO")
    log_format_pl = getattr(config, "LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=log_level_pl, format=log_format_pl, handlers=[logging.StreamHandler(sys.stdout)])


COLLECTION_BACKTEST_RESULTS = getattr(config, "MONGO_COLLECTION_BACKTEST_RESULTS", "strategy_backtest_results")
COLLECTION_TUNED_PARAMS = getattr(config, "MONGO_COLLECTION_TUNED_PARAMS", "strategy_tuned_parameters")


def _get_db() -> Optional[Database]:
    """Helper to get the database instance via MongoManager."""
    db = MongoManager.get_database() 
    if db is None:
        logger.error("Failed to get MongoDB database instance from MongoManager for performance logger. Logging will fail.")
    return db

def _convert_types_for_mongo(data: Any) -> Any:
    """
    Recursively converts numpy types and pandas NaT/NaN to Python natives
    for MongoDB compatibility.
    """
    if isinstance(data, dict):
        return {str(k): _convert_types_for_mongo(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_types_for_mongo(item) for item in data]
    elif isinstance(data, np.integer): 
        return int(data)
    elif isinstance(data, np.floating): 
        return float(data)
    elif isinstance(data, (np.bool_, bool)): 
         return bool(data)
    elif pd.isna(data): 
         return None
    elif isinstance(data, np.datetime64): 
        pd_ts = pd.to_datetime(data)
        if pd_ts.tzinfo is None:
            return pd_ts.tz_localize('UTC').to_pydatetime()
        return pd_ts.to_pydatetime()
    elif isinstance(data, datetime) and data.tzinfo is None: 
        return data.replace(tzinfo=timezone.utc)
    return data


def log_backtest_run_results(
    strategy_name: str,
    parameters_used: Dict[str, Any],
    performance_metrics: Dict[str, Any], 
    symbol: str,
    timeframe: str,
    market_condition: Optional[str] = None,
    session: Optional[str] = None,
    day: Optional[str] = None,
    is_expiry: Optional[bool] = None,
    performance_score: Optional[float] = None, 
    optuna_study_name: Optional[str] = None,
    optuna_trial_number: Optional[int] = None,
    run_id: Optional[str] = None, 
    custom_data: Optional[Dict[str, Any]] = None 
) -> bool:
    db = _get_db()
    if db is None:
        return False 

    collection = db[COLLECTION_BACKTEST_RESULTS]
    
    log_entry_base = {
        "strategy_name": strategy_name,
        "parameters_used": parameters_used,
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "market_condition": market_condition,
        "session": session,
        "day": day,
        "is_expiry": is_expiry,
        "performance_score": performance_score,
        "optuna_study_name": optuna_study_name,
        "optuna_trial_number": optuna_trial_number,
        "run_id": run_id,
        "logged_at": datetime.now(timezone.utc) 
    }

    log_entry = {**log_entry_base, **(performance_metrics if performance_metrics else {})}
    if custom_data:
        log_entry.update(custom_data)

    log_entry_cleaned = {k: v for k, v in log_entry.items() if v is not None}
    
    try:
        log_entry_safe = _convert_types_for_mongo(log_entry_cleaned)
    except Exception as e_convert:
        logger.error(f"Type conversion failed for backtest log: {e_convert}", exc_info=True)
        return False

    try:
        result = collection.insert_one(log_entry_safe)
        
        # MODIFIED (2025-05-09): Corrected f-string for performance_score
        score_str = f"{performance_score:.4f}" if performance_score is not None else "N/A"
        
        logger.info(
            f"Logged backtest run to '{COLLECTION_BACKTEST_RESULTS}': "
            f"Strategy={strategy_name}, Symbol={symbol}, TF={timeframe}, "
            f"Context={market_condition}/{session}/{day}, Score={score_str}, " # Use pre-formatted score_str
            f"DB_ID={result.inserted_id}"
        )
        return True
    except PyMongoError as e:
        logger.error(f"Failed to log backtest run results to MongoDB: {e}", exc_info=True)
        return False
    except Exception as e_unexpected: 
        logger.error(f"Unexpected error logging backtest run results: {e_unexpected}", exc_info=True)
        return False


def log_tuned_parameters(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    market_condition: str,
    session: str,
    day: str,
    is_expiry: bool, 
    best_parameters: Dict[str, Any],
    performance_score: float, 
    optuna_study_name: Optional[str] = None, 
    run_id: Optional[str] = None 
) -> bool:
    db = _get_db()
    if db is None:
        return False

    collection = db[COLLECTION_TUNED_PARAMS]

    context_query = {
        "strategy_name": strategy_name,
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "market_condition": market_condition,
        "session": session,
        "day": day,
        "is_expiry": bool(is_expiry), 
    }

    update_data_set = {
        **context_query,
        "best_params": best_parameters,
        "achieved_performance_score": performance_score,
        "optuna_study_name": optuna_study_name,
        "run_id": run_id,
        "last_updated_at": datetime.now(timezone.utc)
    }
    update_data_set_cleaned = {k: v for k, v in update_data_set.items() if v is not None}

    update_document = {
        "$set": update_data_set_cleaned,
        "$setOnInsert": { 
            "first_tuned_at": datetime.now(timezone.utc)
        }
    }
    
    try:
        update_document["$set"] = _convert_types_for_mongo(update_document["$set"])
        if "$setOnInsert" in update_document: 
             update_document["$setOnInsert"] = _convert_types_for_mongo(update_document["$setOnInsert"])
    except Exception as e_convert:
        logger.error(f"Type conversion failed for tuned params log: {e_convert}", exc_info=True)
        return False

    try:
        result = collection.update_one(context_query, update_document, upsert=True)
        
        log_msg_action = "No change to"
        if result.upserted_id:
            log_msg_action = "Inserted (Upserted)"
        elif result.modified_count > 0:
            log_msg_action = "Updated"
            
        context_str = f"Strategy={strategy_name}, Symbol={symbol}, TF={timeframe}, MC={market_condition}, Sess={session}, Day={day}, Exp={is_expiry}"
        # MODIFIED (2025-05-09): Ensure performance_score is float for formatting
        score_val = float(performance_score) if performance_score is not None else 0.0
        logger.info(
            f"{log_msg_action} tuned parameters in '{COLLECTION_TUNED_PARAMS}': "
            f"{context_str}, Score={score_val:.4f}"
        )
        return True
    except PyMongoError as e:
        logger.error(f"Failed to log tuned parameters to MongoDB: {e}", exc_info=True)
        return False
    except Exception as e_unexpected:
        logger.error(f"Unexpected error logging tuned parameters: {e_unexpected}", exc_info=True)
        return False


if __name__ == '__main__':
    logger.info("Testing performance_logger_mongo with MongoManager...")

    if not hasattr(config, 'MONGO_URI') or not config.MONGO_URI:
        logger.error("MONGO_URI not configured in app/config.py. Exiting example.")
        sys.exit(1)

    if not MongoManager.get_client():
        logger.error("Failed to establish initial MongoDB connection via MongoManager. Examples will likely fail.")
        sys.exit(1)

    print("\nLogging example backtest run result...")
    metrics_example1 = {"total_pnl": 120.50, "win_rate": 0.60, "sharpe_ratio": 1.2, "trades": 20, "profit_factor": 1.8}
    params_example1 = {"st_period": 10, "st_multiplier": 3.0, "adx_period": 14, "sl_mult": 1.5, "tp_mult": 2.0}
    success_br1 = log_backtest_run_results(
        strategy_name="SuperTrend_ADX", parameters_used=params_example1, performance_metrics=metrics_example1,
        symbol="NIFTY", timeframe="5min", market_condition="Trending", session="Morning", day="Monday", is_expiry=False,
        performance_score=150.75123, optuna_study_name="NIFTY_5min_Trending_Morning_Monday_False_SuperTrend_ADX",
        optuna_trial_number=5, run_id="pipeline_run_20250509_TEST001"
    )
    print(f"log_backtest_run_results success: {success_br1}")

    print("\nLogging example tuned parameters...")
    params_example2 = {"ema_short_period": 9, "ema_long_period": 21, "sl_mult": 1.0, "tp_mult": 1.5}
    success_tp1 = log_tuned_parameters(
        strategy_name="EMA_Crossover", symbol="BANKNIFTY", timeframe="15min",
        market_condition="Ranging", session="Afternoon", day="Wednesday", is_expiry=True,
        best_parameters=params_example2, performance_score=95.20345,
        optuna_study_name="BANKNIFTY_15min_Ranging_Afternoon_Wednesday_True_EMA_Crossover",
        run_id="pipeline_run_20250509_TEST001"
    )
    print(f"log_tuned_parameters success: {success_tp1}")
    
    print("\nLogging example backtest run with numpy types...")
    metrics_example3 = {
        "total_pnl": np.float64(250.75), "win_rate": np.float32(0.55), "sharpe_ratio": np.float64(1.1), 
        "trades": np.int32(25), "some_bool": np.bool_(True), "none_val": np.nan
    }
    params_example3 = {"period": np.int64(14), "level": np.int16(75)}
    success_br2 = log_backtest_run_results(
        strategy_name="RSI_Slope", parameters_used=params_example3, performance_metrics=metrics_example3,
        symbol="NIFTY", timeframe="3min", market_condition="Volatile", session="Midday", day="Tuesday", is_expiry=False,
        performance_score=np.float64(120.0), run_id="pipeline_run_20250509_TEST002"
    )
    print(f"log_backtest_run_results (with numpy types) success: {success_br2}")

    # MODIFIED (2025-05-09): Example of logging with performance_score as None
    print("\nLogging example backtest run with performance_score as None...")
    success_br3 = log_backtest_run_results(
        strategy_name="Test_Strategy_No_Score", parameters_used={"p1":1}, performance_metrics={"total_pnl": -10.0},
        symbol="NIFTY", timeframe="1min", performance_score=None, run_id="pipeline_run_20250509_TEST003"
    )
    print(f"log_backtest_run_results (score=None) success: {success_br3}")


    MongoManager.close_client()
    logger.info("Performance logger examples finished and MongoDB connection closed via MongoManager.")
