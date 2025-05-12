# app/optuna_tuner.py
import random
from sqlite3 import OperationalError as SQLiteOperationalError
import optuna
import pandas as pd
import numpy as np
import calendar
import logging
from pathlib import Path # Keep Path
from datetime import datetime
import time 
import concurrent.futures
from typing import Dict, Any, List, Optional
import sys

from app.simulation_engine import SimpleBacktester
from app.config import config
from app.performance_logger_mongo import log_tuned_parameters, log_backtest_run_results
from app.strategies import strategy_factories, tunable_param_space
from app.utils.expiry_utils import is_expiry_day

# MODIFIED (2025-05-09): Removed problematic import to prevent circular dependency
# from pipeline_manager import PROJECT_ROOT 

logger = logging.getLogger("OptunaTuner")
if not logger.hasHandlers():
    log_level_from_config = getattr(config, "LOG_LEVEL", "INFO")
    log_format_from_config = getattr(config, "LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=log_level_from_config, format=log_format_from_config, handlers=[logging.StreamHandler(sys.stdout)])

optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def filter_data_for_context(
    df: pd.DataFrame,
    day_of_week: Optional[str], 
    session: Optional[str],     
    expiry_status_filter: Optional[bool], 
    symbol: str,
    exchange: str,
    market_regime_filter: Optional[str] = None,
    volatility_status_filter: Optional[str] = None
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame index must be DatetimeIndex for context filtering. Attempting conversion.")
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.error(f"Failed to convert DataFrame index to DatetimeIndex: {e}", exc_info=True)
            raise TypeError("DataFrame index must be DatetimeIndex for context filtering.")
    df_filtered = df.copy()
    if day_of_week and day_of_week.lower() != "anyday" and day_of_week is not None:
        try:
            day_num = list(calendar.day_name).index(day_of_week)
            df_filtered = df_filtered[df_filtered.index.weekday == day_num]
        except ValueError:
            logger.warning(f"Invalid day_of_week: {day_of_week}. No day filter applied.")
    if session and session.lower() != "allday" and session is not None:
        if session == "Morning": df_filtered = df_filtered.between_time('09:15', '10:59')
        elif session == "Midday": df_filtered = df_filtered.between_time('11:00', '13:29')
        elif session == "Afternoon": df_filtered = df_filtered.between_time('13:30', '15:29')
        else: logger.warning(f"Unknown session: {session}. No session filter applied.")
    if expiry_status_filter is not None: 
        try:
            df_filtered = df_filtered[df_filtered.index.to_series().apply(
                #lambda ts: is_expiry_day(symbol=symbol, date=ts, exchange_segment=exchange) == expiry_status_filter
                lambda ts: is_expiry_day(symbol, ts, exchange) == expiry_status_filter

            )]
        except Exception as e:
            logger.error(f"Error during expiry filtering for symbol {symbol}, exchange {exchange}: {e}", exc_info=True)
            return pd.DataFrame() 
    if market_regime_filter and market_regime_filter.lower() != "anyregime" and 'regime' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['regime'] == market_regime_filter]
    if volatility_status_filter and volatility_status_filter.lower() != "anyvolatility" and 'volatility_status' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['volatility_status'] == volatility_status_filter]
    if len(df_filtered) < getattr(config, "MIN_BARS_FOR_CONTEXT_TUNING", 50): 
        logger.warning(f"Context resulted in {len(df_filtered)} bars, less than min {getattr(config, 'MIN_BARS_FOR_CONTEXT_TUNING', 50)}. Skipping study for this context.")
        return pd.DataFrame()
    return df_filtered


def run_single_study(symbol: str, market: str, segment: str, strategy_name: str, timeframe: str, 
                     context_definition: Dict[str, Any], df_full_instrument: pd.DataFrame, 
                     n_trials: int, run_id: str, run_specific_logs_dir: Path) -> None:
    study_name_parts = [
        strategy_name, symbol, timeframe,
        context_definition.get('day', 'AnyDay') or 'AnyDay', 
        context_definition.get('session', 'AllDay') or 'AllDay',
        f"Exp={str(context_definition.get('is_expiry', 'AnyExpiry'))}",
        f"Regime={context_definition.get('regime_filter', 'AnyRegime') or 'AnyRegime'}",
        f"Volatility={context_definition.get('volatility_filter', 'AnyVolatility') or 'AnyVolatility'}"
    ]
    study_name = "_".join(str(part).replace(" ", "") for part in study_name_parts)

    logger.info(f"--- Starting Optuna Study: {study_name} (Run ID: {run_id}) ---")
    logger.info(f"--- Optuna trial simulation logs will be based in: {run_specific_logs_dir} ---")

    df_context_specific = filter_data_for_context(
        df_full_instrument,
        day_of_week=context_definition.get('day'),
        session=context_definition.get('session'),
        expiry_status_filter=context_definition.get('is_expiry'),
        symbol=symbol,
        exchange=market, 
        market_regime_filter=context_definition.get('regime_filter'),
        volatility_status_filter=context_definition.get('volatility_filter')
    )

    if df_context_specific.empty:
        logger.warning(f"Skipping study '{study_name}' due to empty DataFrame after context filtering.")
        return

    param_space = tunable_param_space.get(strategy_name)
    if not param_space:
        logger.error(f"No tunable parameter space for strategy '{strategy_name}'. Skipping study '{study_name}'.")
        return

    strategy_factory = strategy_factories.get(strategy_name)
    if not strategy_factory:
        logger.error(f"No strategy factory for '{strategy_name}'. Skipping study '{study_name}'.")
        return

    def objective(trial: optuna.Trial):
        trial_params = {}
        indicator_params = {} 
        risk_params = {}      

        for p_name, p_cfg in param_space.get("indicator", {}).items():
            if p_cfg.get("fixed", False): val = p_cfg["value"]
            elif p_cfg["type"] == "float": val = trial.suggest_float(p_name, p_cfg["low"], p_cfg["high"], step=p_cfg.get("step"))
            elif p_cfg["type"] == "int": val = trial.suggest_int(p_name, p_cfg["low"], p_cfg["high"], step=p_cfg.get("step", 1))
            else: logger.warning(f"Unknown param type '{p_cfg['type']}' for {p_name}. Skipping."); continue
            trial_params[p_name] = indicator_params[p_name] = val

        for p_name, p_cfg in param_space.get("risk", {}).items():
            if p_cfg.get("fixed", False): val = p_cfg["value"]
            elif p_cfg.get("default_disabled", False) and not trial.suggest_categorical(f"enable_{p_name}", [True, False]): val = None 
            elif p_cfg["type"] == "float": val = trial.suggest_float(p_name, p_cfg["low"], p_cfg["high"], step=p_cfg.get("step"))
            elif p_cfg["type"] == "int": val = trial.suggest_int(p_name, p_cfg["low"], p_cfg["high"], step=p_cfg.get("step", 1))
            else: logger.warning(f"Unknown param type '{p_cfg['type']}' for {p_name}. Skipping."); continue
            trial_params[p_name] = risk_params[p_name] = val
        
        try:
            strategy_fn = strategy_factory(**indicator_params) 
        except Exception as e:
            logger.error(f"Study '{study_name}', Trial {trial.number}: Error creating strategy '{strategy_name}' with params {indicator_params}: {e}", exc_info=True)
            return -np.inf 

        bt = SimpleBacktester(strategy_func=strategy_fn, strategy_name=strategy_name)
        bt.default_sl_mult = risk_params.get("sl_mult", config.DEFAULT_SL_ATR_MULT) if risk_params.get("sl_mult") is not None else config.DEFAULT_SL_ATR_MULT
        bt.default_tp_mult = risk_params.get("tp_mult", config.DEFAULT_TP_ATR_MULT) if risk_params.get("tp_mult") is not None else config.DEFAULT_TP_ATR_MULT
        bt.default_tsl_mult = risk_params.get("tsl_mult") if "tsl_mult" in risk_params and risk_params["tsl_mult"] is not None else getattr(config, "DEFAULT_TSL_ATR_MULT", None)
        
        # if run_specific_logs_dir is None:
        #     run_specific_logs_dir = Path(config.LOG_DIR) / f"optuna_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        #     run_specific_logs_dir.mkdir(parents=True, exist_ok=True)
        # else:
        #     run_specific_logs_dir = Path(run_specific_logs_dir)
        #run_specific_logs_dir=run_specific_logs_dir

        trial_log_base_dir = run_specific_logs_dir / "optuna_trial_sim_logs" / study_name / f"trial_{trial.number}"
        trial_log_base_dir.mkdir(parents=True, exist_ok=True)

        try:
            result_metrics = bt.run_simulation(
                df=df_context_specific.copy(), 
                log_dir=trial_log_base_dir, 
                timeframe=f"{timeframe}_ctx_{trial.number}", 
                run_id=run_id, 
                optuna_trial_params=trial_params, 
                optuna_study_name=study_name,     
                optuna_trial_number=trial.number  
            )
            score = result_metrics.get("performance_score", -np.inf)
            if pd.isna(score) or score is None: score = -np.inf
            logger.debug(f"Study '{study_name}', Trial {trial.number} completed. Score: {score:.4f}")
            return score
        except Exception as e:
            logger.error(f"Study '{study_name}', Trial {trial.number}: Simulation or logging failed: {e}", exc_info=True)
            return -np.inf 
    # At the top of the function
    if run_specific_logs_dir is None:
        run_specific_logs_dir = Path(config.LOG_DIR) / f"optuna_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_specific_logs_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_specific_logs_dir = Path(run_specific_logs_dir)

    optuna_db_run_specific_dir = run_specific_logs_dir / "optuna_studies_db" / market / segment / symbol
    optuna_db_run_specific_dir.mkdir(parents=True, exist_ok=True)
    storage_path = f"sqlite:///{optuna_db_run_specific_dir}/{timeframe}_study_{study_name}.db" 

    try:
        study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_path, load_if_exists=True)
    except SQLiteOperationalError as e: 
        logger.warning(f"Study creation/load failed for '{study_name}' due to SQLite DB conflict: {e}. Retrying after delay.")
        time.sleep(random.uniform(1, 3)) 
        study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_path, load_if_exists=True)

    completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed_trials_count >= n_trials and study.best_value is not None and study.best_value > -np.inf :
        logger.info(f"Study '{study_name}' already has {completed_trials_count} completed trials. Best score: {study.best_value:.4f}. Skipping further optimization for now.")
    else:
        remaining_trials = n_trials - completed_trials_count
        if remaining_trials > 0:
            try:
                timeout_seconds = getattr(config, 'OPTUNA_STUDY_TIMEOUT_SECONDS_PER_STUDY', 3600) 
                study.optimize(objective, n_trials=remaining_trials, timeout=timeout_seconds, n_jobs=1) 
            except Exception as e:
                logger.error(f"Optuna study '{study_name}' failed during optimization: {e}", exc_info=True)
        else:
            logger.info(f"Study '{study_name}' already has sufficient completed trials ({completed_trials_count}).")

    if study.best_trial and study.best_value is not None and study.best_value > -np.inf:
        logger.info(f"Optuna study '{study_name}' finished. Best score: {study.best_value:.4f} with params: {study.best_trial.params}")
        log_tuned_parameters(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe, 
            market_condition=context_definition.get('regime_filter') or "AnyRegime", 
            session=context_definition.get('session') or "AllDay",        
            day=context_definition.get('day') or "AnyDay",              
            is_expiry=context_definition.get('is_expiry') if context_definition.get('is_expiry') is not None else False, 
            best_parameters=study.best_trial.params,
            performance_score=float(study.best_value), 
            optuna_study_name=study_name,
            run_id=run_id 
        )
    else:
        logger.warning(f"Optuna study '{study_name}' finished without finding a valid best trial.")


def run_contextual_tuning(
    symbol: str, market: str, segment: str, 
    timeframes: List[str], strategies: List[str], 
    n_trials_per_study: int, max_workers: int, 
    run_id: str, 
    run_specific_logs_dir: Path, 
    context_filter_override: Optional[Dict] = None 
):
    logger.info(f"Starting Contextual Tuning for: Symbol={symbol}, Market={market}, Segment={segment}, RunID={run_id}")
    logger.info(f"Run-specific base log directory for Optuna trials: {run_specific_logs_dir}")

    days_iter = [context_filter_override.get("day")] if context_filter_override and "day" in context_filter_override else ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", None]
    sessions_iter = [context_filter_override.get("session")] if context_filter_override and "session" in context_filter_override else ["Morning", "Midday", "Afternoon", None]
    expiry_flags_iter = [context_filter_override.get("is_expiry")] if context_filter_override and "is_expiry" in context_filter_override else [True, False, None]
    default_regimes = getattr(config, "MARKET_REGIMES_FOR_TUNING", ["Trending", "Ranging", "Volatile", None])
    default_volatilities = getattr(config, "VOLATILITY_STATUSES_FOR_TUNING", ["High", "Normal", "Low", None])
    market_regimes_iter = [context_filter_override.get("market_condition")] if context_filter_override and "market_condition" in context_filter_override else default_regimes
    volatility_statuses_iter = [context_filter_override.get("volatility_status")] if context_filter_override and "volatility_status" in context_filter_override else default_volatilities

    raw_data_map = {}
    for tf in timeframes:
        processed_data_dir = Path(getattr(config, "DATA_DIR_PROCESSED", "./data/datawithindicator"))
        file_path = processed_data_dir / f"{symbol.lower()}__{tf}_with_indicators.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, parse_dates=['datetime'])
                if 'datetime' not in df.columns: logger.error(f"File {file_path} missing 'datetime'. Skipping {tf}."); continue
                df.set_index('datetime', inplace=True); df.columns = df.columns.str.lower()
                required_ohlcv = ['open', 'high', 'low', 'close']
                if not all(col in df.columns for col in required_ohlcv): logger.error(f"File {file_path} missing OHLC. Skipping {tf}."); continue
                df.dropna(subset=required_ohlcv, inplace=True)
                if df.empty: logger.warning(f"Data for {tf} at {file_path} empty after NaN drop. Skipping."); continue
                raw_data_map[tf] = df
                logger.info(f"Loaded data for {tf} from {file_path}: {len(df)} rows")
            except Exception as e: logger.error(f"Failed to load/prep data for {tf} from {file_path}: {e}", exc_info=True)
        else: logger.warning(f"Data file not found for {tf}: {file_path}. Skipping timeframe.")
    if not raw_data_map: logger.error("No data loaded. Aborting contextual tuning."); return

    tasks = []
    for tf_loop, df_for_tf_loop in raw_data_map.items():
        for strategy_name_loop in strategies:
            if strategy_name_loop not in tunable_param_space or strategy_name_loop not in strategy_factories:
                logger.warning(f"Skipping strategy '{strategy_name_loop}' for {tf_loop} - not in tunable_param_space or factories.")
                continue
            for day_loop in days_iter:
                for session_loop in sessions_iter:
                    for expiry_flag_loop in expiry_flags_iter:
                        for regime_filter_loop in market_regimes_iter:
                            for vol_filter_loop in volatility_statuses_iter:
                                current_context_def = {
                                    "day": day_loop, "session": session_loop, "is_expiry": expiry_flag_loop,
                                    "regime_filter": regime_filter_loop, "volatility_filter": vol_filter_loop
                                }
                                tasks.append((
                                    symbol, market, segment, strategy_name_loop, tf_loop,
                                    current_context_def, df_for_tf_loop, n_trials_per_study,
                                    run_id, run_specific_logs_dir 
                                ))
    
    logger.info(f"Created {len(tasks)} distinct Optuna study tasks.")
    if not tasks: logger.warning("No tasks for tuning. Check configs and data."); return

    use_multiprocessing = getattr(config, "USE_MULTIPROCESSING_FOR_OPTUNA", True)
    executor_cls = concurrent.futures.ProcessPoolExecutor if use_multiprocessing else concurrent.futures.ThreadPoolExecutor
    actual_max_workers = max_workers
    if use_multiprocessing:
        import os
        cpu_count = os.cpu_count() or 1
        if max_workers > cpu_count: logger.warning(f"Requested max_workers ({max_workers}) > CPU cores ({cpu_count}). Setting to {cpu_count}."); actual_max_workers = cpu_count
    logger.info(f"Using {executor_cls.__name__} with max_workers={actual_max_workers}")

    with executor_cls(max_workers=actual_max_workers) as executor:
        futures = [executor.submit(run_single_study, *task_args) for task_args in tasks]
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            try: f.result(); logger.info(f"Completed Optuna study task {i+1}/{len(tasks)}.")
            except Exception as e: logger.error(f"Optuna study task (index {i}) failed at executor level: {e}", exc_info=True)

    logger.info("✅ All contextual Optuna studies complete.")


if __name__ == "__main__":
    logger.info("Running OptunaTuner directly for testing...")

    # MODIFIED (2025-05-09): Define PROJECT_ROOT locally for direct testing if needed
    # This assumes optuna_tuner.py is in 'app/' directory, so project root is its parent.
    PROJECT_ROOT_FOR_TEST = Path(__file__).resolve().parent.parent

    test_symbol = getattr(config, "DEFAULT_SYMBOL", "NIFTY")
    test_market = getattr(config, "DEFAULT_MARKET", "NSE")
    test_segment = getattr(config, "DEFAULT_SEGMENT", "EQUITY")
    test_timeframes = list(getattr(config, "RAW_DATA_FILES", {"5min": ""}).keys())
    if not test_timeframes: test_timeframes = ["5min"]
    available_strategies = list(strategy_factories.keys())
    test_strategies = [available_strategies[0]] if available_strategies else []
    if not test_strategies: logger.error("No strategies in strategy_factories. Cannot run test."); sys.exit(1)
    test_n_trials = getattr(config, "OPTUNA_TRIALS_PER_CONTEXT_TEST", 5)
    test_max_workers = getattr(config, "MAX_OPTUNA_WORKERS_TEST", 1)
    test_run_id = f"direct_tuner_run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    test_run_logs_dir = PROJECT_ROOT_FOR_TEST / "runs" / "direct_tuner_tests" / test_run_id / "logs"
    test_run_logs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Direct test run logs will be in: {test_run_logs_dir}")

    specific_context_to_test = {
         "day": "Monday", "session": "Morning", "is_expiry": False,
         "regime_filter": "Trending", "volatility_filter": None
    }
    logger.info(f"--- Direct Test Run Params ---")
    logger.info(f"Symbol: {test_symbol}, Market: {test_market}, Segment: {test_segment}, RunID: {test_run_id}")
    logger.info(f"Timeframes: {test_timeframes}, Strategies: {test_strategies}")
    logger.info(f"Trials/Study: {test_n_trials}, Max Workers: {test_max_workers}")
    if specific_context_to_test: logger.info(f"Specific Context Filter for Test: {specific_context_to_test}")
    
    try:
        run_contextual_tuning(
            symbol=test_symbol, market=test_market, segment=test_segment,
            timeframes=test_timeframes, strategies=test_strategies,
            n_trials_per_study=test_n_trials, max_workers=test_max_workers,
            run_id=test_run_id,
            run_specific_logs_dir=test_run_logs_dir, 
            context_filter_override=specific_context_to_test
        )
    except Exception as e: logger.critical(f"Direct run of OptunaTuner failed: {e}", exc_info=True); sys.exit(1)
    finally:
        from app.mongo_manager import MongoManager
        MongoManager.close_client()
        logger.info("MongoManager client connection closed after direct tuner test run.")
    logger.info("--- OptunaTuner direct test run finished. ---")
# # app/optuna_tuner.py
# import os
# import random
# from sqlite3 import OperationalError as SQLiteOperationalError # Keep for robust DB handling
# import optuna
# import pandas as pd
# import numpy as np
# import calendar
# import logging
# from pathlib import Path
# from datetime import datetime
# import time 
# import concurrent.futures
# from typing import Dict, Any, List, Optional
# import sys

# # Ensure SimpleBacktester is the enhanced version from Step 2
# from app.mongo_manager import MongoManager
# from app.simulation_engine import SimpleBacktester 
# from app.config import config
# from app.performance_logger_mongo import log_tuned_parameters 
# # log_backtest_run_results is called by SimpleBacktester, so not directly here.
# from app.strategies import strategy_factories, tunable_param_space
# from app.utils.expiry_utils import is_expiry_day

# logger = logging.getLogger("OptunaTuner")
# if not logger.hasHandlers(): # Standard logger setup
#     log_level_from_config = getattr(config, "LOG_LEVEL", "INFO")
#     log_format_from_config = getattr(config, "LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     logging.basicConfig(level=log_level_from_config, format=log_format_from_config, handlers=[logging.StreamHandler(sys.stdout)])

# # Optuna logging verbosity control
# optuna.logging.enable_propagation() 
# optuna.logging.disable_default_handler() 
# optuna.logging.set_verbosity(optuna.logging.WARNING)
# from pathlib import Path
# PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 'app/' → project root


# def filter_data_for_context(
#     df: pd.DataFrame,
#     day_of_week: Optional[str], 
#     session: Optional[str],     
#     expiry_status_filter: Optional[bool], 
#     symbol: str, # Already correctly taking symbol
#     exchange: str, # Already correctly taking exchange
#     market_regime_filter: Optional[str] = None,
#     volatility_status_filter: Optional[str] = None
# ) -> pd.DataFrame:
#     # This function's existing logic for filtering data seems fine.
#     # Ensure that it correctly handles the `symbol` and `exchange` for expiry filtering.
#     if not isinstance(df.index, pd.DatetimeIndex):
#         logger.error("DataFrame index must be DatetimeIndex for context filtering. Attempting conversion.")
#         try:
#             df.index = pd.to_datetime(df.index)
#             if df.index.tz is not None: # Standardize by removing timezone if present
#                 df.index = df.index.tz_localize(None)
#             df = df[~df.index.isna()] # Remove rows where index conversion failed
#         except Exception as e:
#             logger.error(f"Failed to convert DataFrame index to DatetimeIndex: {e}", exc_info=True)
#             # Raising error is good as this is critical
#             raise TypeError("DataFrame index must be DatetimeIndex for context filtering.") 
    
#     df_filtered = df.copy()
#     if day_of_week and day_of_week.lower() != "anyday" and day_of_week is not None:
#         try:
#             # Ensure day_of_week is capitalized for calendar.day_name list lookup
#             day_num = list(calendar.day_name).index(day_of_week.capitalize()) 
#             df_filtered = df_filtered[df_filtered.index.weekday == day_num]
#         except ValueError:
#             logger.warning(f"Invalid day_of_week: {day_of_week}. No day filter applied.")
    
#     # ... (your existing session, expiry_status, market_regime, volatility_status filtering logic) ...
#     # The expiry filter using `is_expiry_day(symbol, ts, exchange)` is correct.

#     min_bars = int(getattr(config, "MIN_BARS_FOR_CONTEXT_TUNING", 50)) # Cast to int
#     if len(df_filtered) < min_bars: 
#         logger.warning(f"Context (Symbol: {symbol}) resulted in {len(df_filtered)} bars, less than min {min_bars}. Skipping study.")
#         return pd.DataFrame() # Return empty DataFrame to signal skipping
#     return df_filtered

# # <<< MODIFIED (Step 1 & 2 Integration): run_single_study signature to accept execution_mode >>>
# def run_single_study(symbol: str, market: str, segment: str, strategy_name: str, timeframe: str, 
#                      context_definition: Dict[str, Any], df_full_instrument: pd.DataFrame, 
#                      n_trials: int, 
#                      run_id: str, # This is the main pipeline's RUN_ID
#                      execution_mode: str, # <<< NEW: Execution mode from pipeline_manager >>>
#                      run_specific_logs_dir: Path) -> None: # Base log dir for this Optuna run
    
#     study_name_parts = [ # Constructing a descriptive study name
#         strategy_name, symbol, timeframe,
#         (context_definition.get('day', 'AnyDay') or 'AnyDay').capitalize(), 
#         (context_definition.get('session', 'AllDay') or 'AllDay').capitalize(),
#         f"Exp={str(context_definition.get('is_expiry', 'AnyExp'))}", # Handle None for is_expiry
#         f"Regime={(context_definition.get('regime_filter', 'AnyRegime') or 'AnyRegime').capitalize()}",
#         f"Volatility={(context_definition.get('volatility_filter', 'AnyVol') or 'AnyVol').capitalize()}"
#     ]
#     study_name = "_".join(str(part).replace(" ", "") for part in study_name_parts)

#     # <<< MODIFIED (Step 1 & 2 Integration): Include execution_mode in log >>>
#     logger.info(f"--- Starting Optuna Study: {study_name} (Main Run ID: {run_id}, Mode: {execution_mode}) ---")
#     optuna_trial_logs_base = run_specific_logs_dir / "optuna_trial_sim_logs" / study_name
#     logger.info(f"--- Optuna trial simulation trace logs will be in subdirs of: {optuna_trial_logs_base} ---")

#     df_context_specific = filter_data_for_context(
#         df_full_instrument,
#         day_of_week=context_definition.get('day'),
#         session=context_definition.get('session'),
#         expiry_status_filter=context_definition.get('is_expiry'),
#         symbol=symbol, # Passed correctly
#         exchange=market, # Passed correctly
#         market_regime_filter=context_definition.get('regime_filter'),
#         volatility_status_filter=context_definition.get('volatility_filter')
#     )

#     if df_context_specific.empty:
#         logger.warning(f"Skipping Optuna study '{study_name}' due to empty DataFrame after context filtering.")
#         return

#     param_space = tunable_param_space.get(strategy_name)
#     if not param_space:
#         logger.error(f"No tunable parameter space defined for strategy '{strategy_name}'. Skipping study '{study_name}'.")
#         return

#     strategy_factory = strategy_factories.get(strategy_name)
#     if not strategy_factory:
#         logger.error(f"No strategy factory found for '{strategy_name}'. Skipping study '{study_name}'.")
#         return

#     # Define the objective function for this specific study
#     def objective(trial: optuna.Trial):
#         trial_params: Dict[str, Any] = {}
#         indicator_params: Dict[str, Any] = {} 
#         risk_params: Dict[str, Any] = {}      

#         # Suggest indicator parameters
#         for p_name, p_cfg in param_space.get("indicator", {}).items():
#             val: Any # Define val before conditional assignment
#             if p_cfg.get("fixed", False): val = p_cfg["value"]
#             elif p_cfg["type"] == "float": val = trial.suggest_float(p_name, p_cfg["low"], p_cfg["high"], step=p_cfg.get("step"))
#             elif p_cfg["type"] == "int": val = trial.suggest_int(p_name, p_cfg["low"], p_cfg["high"], step=p_cfg.get("step", 1))
#             else: 
#                 logger.warning(f"Study '{study_name}', Trial {trial.number}: Unknown param type '{p_cfg['type']}' for {p_name}. Skipping."); continue
#             trial_params[p_name] = indicator_params[p_name] = val

#         # Suggest risk parameters
#         for p_name, p_cfg in param_space.get("risk", {}).items():
#             val: Any
#             if p_cfg.get("fixed", False): val = p_cfg["value"]
#             elif p_cfg.get("default_disabled", False) and not trial.suggest_categorical(f"enable_{p_name}", [True, False]): 
#                 val = None 
#             elif p_cfg["type"] == "float": val = trial.suggest_float(p_name, p_cfg["low"], p_cfg["high"], step=p_cfg.get("step"))
#             elif p_cfg["type"] == "int": val = trial.suggest_int(p_name, p_cfg["low"], p_cfg["high"], step=p_cfg.get("step", 1))
#             else: 
#                 logger.warning(f"Study '{study_name}', Trial {trial.number}: Unknown param type '{p_cfg['type']}' for {p_name}. Skipping."); continue
#             trial_params[p_name] = risk_params[p_name] = val
        
#         try:
#             strategy_fn = strategy_factory(**indicator_params) 
#         except Exception as e_create_strat: # More descriptive variable name
#             logger.error(f"Study '{study_name}', Trial {trial.number}: Error creating strategy function for '{strategy_name}' with params {indicator_params}: {e_create_strat}", exc_info=True)
#             return -np.inf # Optuna needs a float, -np.inf signals critical failure

#         # Instantiate SimpleBacktester (it will use config.INITIAL_CAPITAL by default)
#         bt = SimpleBacktester(strategy_func=strategy_fn, strategy_name=strategy_name)
        
#         # The enhanced SimpleBacktester.run_simulation now accepts optuna_trial_params.
#         # The risk params (sl_mult, tp_mult, tsl_mult) from trial_params will be used by it if present.
#         # Setting bt.default_sl_mult here is redundant if these are correctly passed in trial_params to run_simulation.
#         # However, if your SimpleBacktester's logic directly uses these self.default_... attributes after Optuna sets them,
#         # this block can remain, but it's cleaner to pass all tuned params via optuna_trial_params.
#         # For clarity and to ensure Step 2 integration:
#         # SimpleBacktester will use `optuna_trial_params` to override its defaults if those keys exist.

#         # Define specific log directory for this trial's simulation trace logs
#         trial_sim_specific_log_dir = optuna_trial_logs_base / f"trial_{trial.number}"
#         # trial_sim_specific_log_dir.mkdir(parents=True, exist_ok=True) # SimpleBacktester's _setup_simulation_logging handles this

#         try:
#             # <<< MODIFIED (Step 2 Integration): Call the enhanced run_simulation with new parameters >>>
#             result_metrics = bt.run_simulation(
#                 df=df_context_specific.copy(), # Pass a copy to avoid modification
#                 log_dir=trial_sim_specific_log_dir, 
#                 timeframe=timeframe,  # Original timeframe (e.g., "5min")
#                 symbol=symbol,        # <<< NEW (Step 2 Integration): Pass current symbol >>>
#                 run_id=run_id,        # <<< NEW (Step 2 Integration): Pass main pipeline run_id >>>
#                 execution_mode=execution_mode, # <<< NEW (Step 2 Integration): Pass execution_mode >>>
#                 optuna_trial_params=trial_params, # Pass all suggested params for this trial
#                 optuna_study_name=study_name,     
#                 optuna_trial_number=trial.number  
#             )
#             score = result_metrics.get("performance_score", -np.inf) # performance_score is in the enhanced dict
#             if pd.isna(score) or score is None: score = -np.inf # Ensure valid float for Optuna
            
#             # Log trial completion with more info
#             logger.info( 
#                 f"Study '{study_name}', Trial {trial.number} completed. Score: {score:.4f}, PnL: {result_metrics.get('performance_metrics',{}).get('total_pnl_net','N/A')}, Trades: {result_metrics.get('performance_metrics',{}).get('trade_count','N/A')}"
#             )
#             return float(score) # Must return a float
#         except Exception as e_sim_run: # Catch errors during simulation run
#             logger.error(f"Study '{study_name}', Trial {trial.number}: Simulation run or result processing failed: {e_sim_run}", exc_info=True)
#             return -np.inf 
#     # --- End of Objective Function ---

#     # Define Optuna storage path (local SQLite DB per study, within the run-specific logs)
#     optuna_db_storage_dir = run_specific_logs_dir / "optuna_studies_db" / market / segment / symbol / timeframe
#     optuna_db_storage_dir.mkdir(parents=True, exist_ok=True)
#     # Sanitize study_name for use in filename if it contains problematic characters
#     sanitized_study_db_filename = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in study_name)
#     storage_path = f"sqlite:///{optuna_db_storage_dir}/{sanitized_study_db_filename}.db"
#     logger.info(f"Optuna study DB storage for '{study_name}': {storage_path}")

#     try:
#         study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_path, load_if_exists=True)
#     except SQLiteOperationalError as e_sql_create: # Catch specific DB errors
#         logger.warning(f"Study creation/load for '{study_name}' at '{storage_path}' failed due to SQLite DB conflict: {e_sql_create}. Retrying after small delay.")
#         time.sleep(random.uniform(0.5, 2.0)) 
#         study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_path, load_if_exists=True) # Retry

#     completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
#     #meaningful_best_value = study.best_value is not None and study.best_value > -np.inf # Check against critical failure score
#     completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
#     if not completed_trials:
#         logger.warning(f"Study '{study_name}' has no completed trials. Skipping.")
#         return

#     meaningful_best_value = study.best_value is not None and study.best_value > -np.inf

#     if completed_trials_count >= n_trials and meaningful_best_value:
#         logger.info(f"Study '{study_name}' already has {completed_trials_count} completed trials. Best score: {study.best_value:.4f}. Skipping further optimization.")
#     else:
#         remaining_trials = n_trials - completed_trials_count
#         if remaining_trials > 0:
#             logger.info(f"Study '{study_name}': {completed_trials_count}/{n_trials} trials complete. Optimizing for remaining {remaining_trials} trials.")
#             try:
#                 timeout_seconds_per_study = int(getattr(config, 'OPTUNA_STUDY_TIMEOUT_SECONDS_PER_STUDY', 3600*2)) # e.g., 2 hours
#                 study.optimize(objective, n_trials=remaining_trials, timeout=timeout_seconds_per_study, n_jobs=1) # n_jobs=1 for each study run sequentially by the worker
#             except Exception as e_optimize: # Catch errors during the optimize call itself
#                 logger.error(f"Optuna study '{study_name}' failed during .optimize() call: {e_optimize}", exc_info=True)
#         elif not meaningful_best_value:
#              logger.info(f"Study '{study_name}' has {completed_trials_count} trials but no meaningful best value yet. Will attempt optimization if remaining_trials > 0 (currently {remaining_trials}).")
#         else: # remaining_trials <= 0
#             logger.info(f"Study '{study_name}' already has sufficient completed trials ({completed_trials_count}).")

#     if study.best_trial and study.best_value is not None and study.best_value > -np.inf:
#         logger.info(f"Optuna study '{study_name}' finished. Best score: {study.best_value:.4f} with params: {study.best_trial.params}")
#         # Log tuned parameters to MongoDB
#         log_tuned_parameters(
#             strategy_name=strategy_name,
#             symbol=symbol, # Pass symbol
#             timeframe=timeframe, 
#             market_condition=context_definition.get('regime_filter') or "AnyRegime", 
#             session=context_definition.get('session') or "AllDay",        
#             day=context_definition.get('day') or "AnyDay",              
#             is_expiry=bool(context_definition.get('is_expiry', False)), # Ensure boolean
#             best_parameters=study.best_trial.params, # This is already a dict
#             performance_score=float(study.best_value), 
#             optuna_study_name=study_name,
#             run_id=run_id, # Pass main pipeline run_id
#             # <<< NEW (Step 1 & 2 Integration): Pass execution_mode >>>
#             execution_mode=execution_mode 
#             # (Ensure app.performance_logger_mongo.log_tuned_parameters accepts execution_mode)
#         )
#     else:
#         logger.warning(f"Optuna study '{study_name}' finished without finding a valid best trial (Best value: {study.best_value}). No parameters logged to DB for this study.")


# # <<< MODIFIED (Step 1 & 2 Integration): run_contextual_tuning signature to accept execution_mode >>>
# def run_contextual_tuning(
#     symbol: str, market: str, segment: str, 
#     timeframes: List[str], strategies: List[str], 
#     n_trials_per_study: int, max_workers: int, 
#     run_id: str, 
#     execution_mode: str, # <<< NEW (Step 1 Integration) >>>
#     run_specific_logs_dir: Path, # Base log dir for this specific Optuna process (e.g., runs/RUN_ID/logs)
#     context_filter_override: Optional[Dict] = None 
# ):
#     # <<< MODIFIED (Step 1 & 2 Integration): Log execution_mode >>>
#     logger.info(f"Starting Contextual Tuning for: Symbol={symbol}, Market={market}, Segment={segment}, RunID={run_id}, ExecutionMode={execution_mode}")
#     logger.info(f"Run-specific base log directory for Optuna sub-components: {run_specific_logs_dir}")

#     # Context Iterators (your existing logic is fine, ensure defaults are robust)
#     days_iter = [context_filter_override.get("day")] if context_filter_override and "day" in context_filter_override else getattr(config, "DAYS_FOR_TUNING", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", None])
#     sessions_iter = [context_filter_override.get("session")] if context_filter_override and "session" in context_filter_override else getattr(config, "SESSIONS_FOR_TUNING", ["Morning", "Midday", "Afternoon", None])
#     expiry_flags_iter = [context_filter_override.get("is_expiry")] if context_filter_override and "is_expiry" in context_filter_override else getattr(config, "EXPIRY_STATUS_FOR_TUNING", [True, False, None])
#     default_regimes = getattr(config, "MARKET_REGIMES_FOR_TUNING", ["Trending", "Ranging", "Volatile", None])
#     default_volatilities = getattr(config, "VOLATILITY_STATUSES_FOR_TUNING", ["High", "Normal", "Low", None])
    
#     market_regimes_iter = [context_filter_override.get("market_condition")] if context_filter_override and "market_condition" in context_filter_override else default_regimes
#     volatility_statuses_iter = [context_filter_override.get("volatility_status")] if context_filter_override and "volatility_status" in context_filter_override else default_volatilities

#     # Load data once per timeframe
#     raw_data_map: Dict[str, pd.DataFrame] = {}
#     for tf in timeframes:
#         # <<< MODIFIED (Integration): Use config.DATA_DIR_PROCESSED consistently >>>
#         #processed_data_base_dir = Path(getattr(config, "DATA_DIR_PROCESSED", Path(config.PROJECT_ROOT) / "data" / "datawithindicator")) # Use PROJECT_ROOT from config if needed
#         #processed_data_base_dir = Path(getattr(config, "DATA_DIR_PROCESSED", "./data/datawithindicator"))
#         processed_data_base_dir = Path(getattr(config, "DATA_DIR_PROCESSED", PROJECT_ROOT / "data" / "datawithindicator"))

#         file_path = processed_data_base_dir / f"{symbol.lower()}__{tf}_with_indicators.csv"
        
#         if file_path.exists():
#             try:
#                 # Use a robust loading function similar to run_simulation_step's load_data if needed
#                 df = pd.read_csv(file_path, index_col='datetime', parse_dates=['datetime'])
#                 if not isinstance(df.index, pd.DatetimeIndex):
#                      df.index = pd.to_datetime(df.index, errors='coerce')
#                 if df.index.tz is not None: df.index = df.index.tz_localize(None)
#                 df = df[~df.index.isna()] # Remove rows with NaT index

#                 df.columns = df.columns.str.lower()
#                 required_ohlcv = ['open', 'high', 'low', 'close'] # Add other necessary columns like 'atr_col' if known
#                 if not all(col in df.columns for col in required_ohlcv): 
#                     logger.error(f"Data file {file_path} for {tf} (Symbol {symbol}) missing OHLC. Skipping."); continue
#                 df.dropna(subset=required_ohlcv, inplace=True)

#                 if df.empty: 
#                     logger.warning(f"Data for {tf} from {file_path} (Symbol {symbol}) is empty after pre-processing. Skipping."); continue
                
#                 raw_data_map[tf] = df
#                 logger.info(f"Loaded data for {tf} (Symbol: {symbol}) from {file_path}: {len(df)} rows")
#             except Exception as e_load_data: 
#                 logger.error(f"Failed to load/process data for {tf} (Symbol: {symbol}) from {file_path}: {e_load_data}", exc_info=True)
#         else: 
#             logger.warning(f"Data file not found for {tf} (Symbol: {symbol}): {file_path}. Skipping this timeframe for Optuna.")
    
#     if not raw_data_map: 
#         logger.error(f"No data loaded for any timeframes for Symbol {symbol}. Aborting contextual tuning for this symbol."); return

#     tasks = []
#     for tf_loop, df_for_tf_loop in raw_data_map.items():
#         for strategy_name_loop in strategies:
#             if strategy_name_loop not in tunable_param_space or strategy_name_loop not in strategy_factories:
#                 logger.warning(f"Skipping strategy '{strategy_name_loop}' for {tf_loop} (Symbol {symbol}) - not in tunable_param_space or factories.")
#                 continue
#             for day_loop in days_iter:
#                 for session_loop in sessions_iter:
#                     for expiry_flag_loop in expiry_flags_iter:
#                         for regime_filter_loop in market_regimes_iter:
#                             for vol_filter_loop in volatility_statuses_iter:
#                                 current_context_def = {
#                                     "day": day_loop, "session": session_loop, "is_expiry": expiry_flag_loop,
#                                     "regime_filter": regime_filter_loop, "volatility_filter": vol_filter_loop
#                                 }
#                                 # <<< MODIFIED (Step 1 & 2 Integration): Pass execution_mode to run_single_study task args >>>
#                                 tasks.append((
#                                     symbol, market, segment, strategy_name_loop, tf_loop,
#                                     current_context_def, df_for_tf_loop, n_trials_per_study,
#                                     run_id, execution_mode, run_specific_logs_dir 
#                                 ))
    
#     logger.info(f"Created {len(tasks)} distinct Optuna study tasks for Run ID: {run_id} (Symbol: {symbol}).")
#     if not tasks: 
#         logger.warning(f"No Optuna tasks generated for Run ID: {run_id} (Symbol: {symbol}). Check context definitions, strategy lists, and available data."); return

#     # Multiprocessing/Threading for Optuna studies
#     use_multiprocessing_optuna = bool(getattr(config, "USE_MULTIPROCESSING_FOR_OPTUNA", True if int(max_workers) > 1 else False))
#     actual_max_workers = int(max_workers) # Ensure it's int
    
#     # Determine executor class based on config and requested workers
#     executor_cls = None
#     if actual_max_workers > 1:
#         if use_multiprocessing_optuna:
#             executor_cls = concurrent.futures.ProcessPoolExecutor
#             try: # Safety check for CPU count
#                 cpu_cores = os.cpu_count()
#                 if cpu_cores and actual_max_workers > cpu_cores:
#                     logger.warning(f"Requested max_workers ({actual_max_workers}) for Optuna > CPU cores ({cpu_cores}). Setting to {cpu_cores}.")
#                     actual_max_workers = cpu_cores
#             except NotImplementedError:
#                 logger.warning("Could not determine CPU count. Using requested max_workers for Optuna ProcessPool.")
#         else: # Use ThreadPool if multiprocessing explicitly disabled but workers > 1
#             executor_cls = concurrent.futures.ThreadPoolExecutor
#     else: # Single worker, run sequentially
#         actual_max_workers = 1 
#         logger.info(f"Running Optuna studies sequentially (max_workers={actual_max_workers}).")

#     if executor_cls:
#         logger.info(f"Using {executor_cls.__name__} with max_workers={actual_max_workers} for Optuna studies.")
#         with executor_cls(max_workers=actual_max_workers) as executor:
#             futures = [executor.submit(run_single_study, *task_args) for task_args in tasks]
#             for i, future_result in enumerate(concurrent.futures.as_completed(futures)):
#                 try: 
#                     future_result.result() # Check for exceptions from the study run
#                     logger.info(f"Completed Optuna study task {i+1}/{len(tasks)} (from concurrent exec).")
#                 except Exception as e_concurrent_task: 
#                     logger.error(f"An Optuna study task (index {i}) failed within the executor: {e_concurrent_task}", exc_info=True)
#     else: # Sequential execution for max_workers=1
#         for i, task_args_seq in enumerate(tasks):
#             logger.info(f"Executing Optuna study task {i+1}/{len(tasks)} sequentially...")
#             try:
#                 run_single_study(*task_args_seq)
#             except Exception as e_sequential_task:
#                 logger.error(f"Optuna study task (sequential, index {i}) failed: {e_sequential_task}", exc_info=True)

#     logger.info(f"✅ All ({len(tasks)}) contextual Optuna studies submitted/processed for Run ID: {run_id} (Symbol: {symbol}).")


# if __name__ == "__main__":
#     logger.info("Running OptunaTuner directly for a standalone test...")

#     STANDALONE_PROJECT_ROOT = Path(__file__).resolve().parent.parent

#     test_symbol = getattr(config, "DEFAULT_SYMBOL", "NIFTY")
#     test_market = getattr(config, "DEFAULT_MARKET", "NSE")
#     test_segment = getattr(config, "DEFAULT_SEGMENT", "EQUITY") 
#     test_timeframes = list(getattr(config, "RAW_DATA_FILES", {"5min": ""}).keys())
#     if not test_timeframes: test_timeframes = ["5min"] 
    
#     available_strategies = list(strategy_factories.keys())
#     test_strategies = [available_strategies[0]] if available_strategies else [] 
#     if not test_strategies: 
#         logger.error("No strategies found in strategy_factories. Cannot run test.")
#         sys.exit(1)

#     test_n_trials = int(getattr(config, "OPTUNA_TRIALS_PER_CONTEXT_TEST", 3))
#     test_max_workers = int(getattr(config, "MAX_OPTUNA_WORKERS_TEST", 1))
#     test_run_id = f"direct_tuner_test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
#     # <<< NEW (Step 1 Integration): Define execution_mode for direct test >>>
#     test_execution_mode = "direct_optuna_tuner_standalone_test"
    
#     test_base_log_dir = STANDALONE_PROJECT_ROOT / "runs" / "optuna_tuner_direct_tests" / test_run_id / "logs"
#     test_base_log_dir.mkdir(parents=True, exist_ok=True)
#     logger.info(f"Direct test run (ID: {test_run_id}) logs will be in: {test_base_log_dir}")

#     specific_context_to_test = {
#          "day": "Monday", "session": "Morning", "is_expiry": False,
#          "regime_filter": "Trending", "volatility_filter": None 
#     }
#     # specific_context_to_test = None # To test all contexts

#     logger.info(f"--- Direct Test Run Parameters ---")
#     logger.info(f"Symbol: {test_symbol}, Market: {test_market}, Segment: {test_segment}")
#     logger.info(f"Timeframes: {test_timeframes}, Strategies: {test_strategies}")
#     # <<< MODIFIED (Step 1 Integration): Log test_run_id and test_execution_mode >>>
#     logger.info(f"Run ID: {test_run_id}, Execution Mode: {test_execution_mode}")
#     logger.info(f"Trials/Study: {test_n_trials}, Max Workers: {test_max_workers}")
#     if specific_context_to_test: 
#         logger.info(f"Specific Context Filter for Test: {specific_context_to_test}")
#     else:
#         logger.info("Testing across all generated contexts.")
    
#     try:
#         # <<< MODIFIED (Step 1 & 2 Integration): Pass execution_mode to run_contextual_tuning >>>
#         run_contextual_tuning(
#             symbol=test_symbol, market=test_market, segment=test_segment,
#             timeframes=test_timeframes, strategies=test_strategies,
#             n_trials_per_study=test_n_trials, max_workers=test_max_workers,
#             run_id=test_run_id,
#             execution_mode=test_execution_mode, 
#             run_specific_logs_dir=test_base_log_dir, 
#             context_filter_override=specific_context_to_test
#         )
#     except Exception as e_main_test: 
#         logger.critical(f"Direct run of OptunaTuner failed: {e_main_test}", exc_info=True)
#         sys.exit(1)
#     finally:
#         # Ensure MongoDB client is closed if this script manages its own connection
#         # (if it uses a global MongoManager instance, that instance should handle closure on atexit from pipeline_manager)
#         MongoManager.close_client() 
#         logger.info("MongoDB client connection (if any) attempted to close after direct tuner test run.")
        
#     logger.info("--- OptunaTuner direct test run finished. ---")