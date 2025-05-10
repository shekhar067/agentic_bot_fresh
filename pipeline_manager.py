
# # # import subprocess
# # # import sys
# # # import logging
# # # from pathlib import Path
# # # import os
# # # from datetime import datetime
# # # # import time # Not strictly used, can be removed if not needed for other logic
# # # from typing import Optional, List # Added List

# # # # --- App Imports ---
# # # from app.config import config
# # # from app.optuna_tuner import run_contextual_tuning
# # # from app.strategies import strategy_factories # Used to get list of strategies

# # # # --- Directory and Logging Setup ---
# # # RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # Assuming pipeline_manager.py is in the project root.
# # # # If it's in a subdirectory like 'pipeline/', PROJECT_ROOT should be Path(__file__).resolve().parent.parent
# # # PROJECT_ROOT = Path(__file__).resolve().parent 

# # # RUNS_BASE_DIR = PROJECT_ROOT / "runs"
# # # RUN_DIR = RUNS_BASE_DIR / RUN_ID # Unique directory for this pipeline run's artifacts
# # # RUN_DATA_DIR = RUN_DIR / "data"
# # # LOGS_DIR = RUN_DIR / "logs"    # Logs specific to this pipeline run
# # # # RUN_DATA_DIR = RUN_DIR / "data" # For run-specific data artifacts, if different from global processed data
# # # RESULTS_DIR = RUN_DIR / "results" # JSON results for this pipeline run
# # # STRATEGY_RESULTS_DIR = RESULTS_DIR / "strategy_results"
# # # AGENT_RESULTS_DIR = RESULTS_DIR / "agent_results"

# # # # --- Create Directories ---
# # # for path_to_create in [RUN_DIR, LOGS_DIR, RESULTS_DIR, STRATEGY_RESULTS_DIR, AGENT_RESULTS_DIR]: # RUN_DATA_DIR removed for now
# # #     path_to_create.mkdir(parents=True, exist_ok=True)

# # # # --- Configure Logging for Pipeline Manager ---
# # # log_file_path = LOGS_DIR / f'pipeline_manager_{RUN_ID}.log'
# # # # Clear root handlers to ensure this basicConfig takes precedence for this script
# # # for handler in logging.root.handlers[:]:
# # #     logging.root.removeHandler(handler)
# # # logging.basicConfig(level=config.LOG_LEVEL if hasattr(config, "LOG_LEVEL") else logging.INFO, # Use level from config
# # #                     format=config.LOG_FORMAT if hasattr(config, "LOG_FORMAT") else '%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Use format from config
# # #                     handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
# # # logger = logging.getLogger(__name__)

# # # APP_DIR = PROJECT_ROOT / "app" # Or wherever these scripts are
# # # PIPELINE_SCRIPTS_DIR = PROJECT_ROOT / "pipeline" # If run_* scripts are here
# # # FEATURE_ENGINE_SCRIPT_PATH = PIPELINE_SCRIPTS_DIR / "run_feature_engine.py" # Example if in pipeline dir
# # # SIMULATION_SCRIPT_PATH = PIPELINE_SCRIPTS_DIR / "run_simulation_step.py" # Example if in pipeline dir


# # # # --- Helper: Run Script with Logging ---
# # # def run_script(script_path: Path, args_list: List[str] = None, timeout: Optional[int] = None, log_suffix: str = "") -> bool:
# # #     if args_list is None:
# # #         args_list = []
        
# # #     if not script_path.is_file():
# # #         logger.error(f"Script not found: {script_path}")
# # #         return False

# # #     safe_log_suffix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in log_suffix)
# # #     log_filename = f"{script_path.stem}{safe_log_suffix}.log"
# # #     script_log_file = LOGS_DIR / log_filename # Subprocess logs go into the run's LOGS_DIR

# # #     command = [sys.executable, str(script_path)] + [str(arg) for arg in args_list]
# # #     logger.info(f"Running command: {' '.join(command)}")
# # #     logger.info(f"Redirecting script output to: {script_log_file}")

# # #     process_env = os.environ.copy()
# # #     project_root_str = str(PROJECT_ROOT)
# # #     current_pythonpath = process_env.get('PYTHONPATH', '')
# # #     if project_root_str not in current_pythonpath.split(os.pathsep): # Use os.pathsep
# # #         process_env['PYTHONPATH'] = project_root_str + os.pathsep + current_pythonpath
# # #     else:
# # #         process_env['PYTHONPATH'] = current_pythonpath
# # #     logger.debug(f"Effective PYTHONPATH for subprocess: {process_env['PYTHONPATH']}")


# # #     try:
# # #         with open(script_log_file, 'w', encoding='utf-8') as f_log:
# # #             subprocess.run(command, check=True, timeout=timeout, stdout=f_log, stderr=subprocess.STDOUT, text=True, cwd=PROJECT_ROOT, env=process_env)
# # #         logger.info(f"--- Script {script_path.name} OK (Log Suffix: '{log_suffix}') ---")
# # #         return True
# # #     except subprocess.CalledProcessError as e:
# # #         logger.error(f"Script {script_path.name} FAIL (Return Code: {e.returncode}) - Check log: {script_log_file}")
# # #     except subprocess.TimeoutExpired:
# # #         logger.error(f"Script {script_path.name} TIMEOUT - Check log: {script_log_file}")
# # #     except Exception as e:
# # #         logger.error(f"Script {script_path.name} ERROR: {e}", exc_info=True)
# # #     return False

# # # # --- Main Orchestration ---
# # # def main_orchestration():
# # #     logger.info(f"--- Starting Pipeline Manager --- RUN ID: {RUN_ID} ---")
# # #     logger.info(f"Run directory: {RUN_DIR}")
# # #     logger.info(f"Project Root interpreted as: {PROJECT_ROOT}")
# # #     logger.info(f"Feature Engine Script Path: {FEATURE_ENGINE_SCRIPT_PATH}")
# # #     logger.info(f"Simulation Script Path: {SIMULATION_SCRIPT_PATH}")


# # #     timeframes_to_process = list(config.RAW_DATA_FILES.keys())
# # #     strategies_to_test = list(strategy_factories.keys()) # Assumes strategy_factories is up-to-date

# # #     # Define symbol, market, segment for this pipeline run (could be from CLI args to pipeline_manager itself)
# # #     target_symbol = getattr(config, "DEFAULT_SYMBOL", "nifty")
# # #     target_market = getattr(config, "DEFAULT_MARKET", "NSE")
# # #     target_segment = getattr(config, "DEFAULT_SEGMENT", "Index")

# # #     logger.info(f"Target Instrument: {target_symbol} ({target_market}/{target_segment})")
# # #     logger.info(f"Processing timeframes: {timeframes_to_process}")
# # #     logger.info(f"Processing strategies for independent backtests: {strategies_to_test}")
# # #     overall_success = True

# # #     # === Phase 1: Feature Generation ===
# # #     # Feature generation writes to the global config.DATA_DIR_PROCESSED
# # #     # This allows OptunaTuner and other components to find the data consistently.
# # #     config.DATA_DIR_PROCESSED.mkdir(parents=True, exist_ok=True) # Ensure global processed dir exists

# # #     for timeframe in timeframes_to_process:
# # #         logger.info(f"\n===== Phase: Feature Generation for {target_symbol} ({timeframe}) =====")
# # #         raw_file_name = config.RAW_DATA_FILES.get(timeframe)
# # #         if not raw_file_name:
# # #             logger.error(f"Missing raw data file configuration for timeframe '{timeframe}'. Skipping.")
# # #             overall_success = False
# # #             continue

# # #         raw_path = config.DATA_FOLDER / raw_file_name # Path to raw data
        
# # #         # Consistent naming for processed files
# # #         # Example: nifty__5min_with_indicators.csv
# # #         indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
# # #         indicator_output_path = config.DATA_DIR_PROCESSED / indicator_file_name

# # #         if not raw_path.exists():
# # #             logger.error(f"Raw data file not found: {raw_path}. Skipping feature generation for {timeframe}.")
# # #             overall_success = False
# # #             continue

# # #         # Arguments for run_feature_engine.py
# # #         # It needs to know the symbol and exchange for expiry features if ADD_EXPIRY_FEATURES is True
# # #         feature_args = [
# # #             "--input", str(raw_path), 
# # #             "--output", str(indicator_output_path),
# # #             "--symbol", target_symbol, # Pass symbol to feature engine
# # #             "--exchange", target_market # Pass exchange to feature engine
# # #         ]
# # #         if not run_script(FEATURE_ENGINE_SCRIPT_PATH, feature_args, timeout=300, log_suffix=f"_features_{target_symbol}_{timeframe}"):
# # #             logger.error(f"Feature generation failed for {target_symbol} - {timeframe}.")
# # #             overall_success = False
# # #             logger.critical(f"❌ Feature generation failed for {target_symbol} - {timeframe}. Aborting pipeline.")
# # #             sys.exit(1)


# # #             # No need to mark indicator_data_paths[timeframe] = None, as Optuna will check existence

# # #     # === Phase 2: Independent Strategy Backtests (Optional - depends on your workflow) ===
# # #     run_independent_backtests = True # Set to False to skip this phase
# # #     if run_independent_backtests:
# # #         logger.info(f"\n===== Phase: Independent Strategy Simulation for {target_symbol} =====")
# # #         for timeframe in timeframes_to_process:
# # #             indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
# # #             indicator_input_path = config.DATA_DIR_PROCESSED / indicator_file_name

# # #             if not indicator_input_path.is_file():
# # #                 logger.critical(f"❌ Expected file not found: {indicator_input_path}")
# # #                 sys.exit(1)
# # #                 #logger.warning(f"Skipping independent simulations for {timeframe} due to missing feature file: {indicator_input_path}")
# # #                 #continue

# # #             for strategy_name in strategies_to_test:
# # #                 logger.info(f"--- Running Independent Backtest for Strategy: {strategy_name} ({target_symbol} - {timeframe}) ---")
# # #                 result_path = STRATEGY_RESULTS_DIR / f"{strategy_name}_{target_symbol}_{timeframe}.json"
                
# # #                 sim_args = [
# # #                     "--input", str(indicator_input_path),
# # #                     "--output-json", str(result_path),
# # #                     "--log-dir", str(LOGS_DIR / "independent_sim_logs"), # Separate log dir for these
# # #                     "--strategy-name", strategy_name,
# # #                     "--symbol", target_symbol, # Pass symbol for context in backtester logging
# # #                     "--market", target_market, # Pass market for context
# # #                     # Add other necessary args for run_simulation_step.py if any
# # #                 ]
# # #                 (LOGS_DIR / "independent_sim_logs").mkdir(parents=True, exist_ok=True)

# # #                 if not run_script(SIMULATION_SCRIPT_PATH, sim_args, timeout=600, log_suffix=f"_{strategy_name}_{target_symbol}_{timeframe}"):
# # #                     logger.error(f"Independent simulation failed for {strategy_name} on {target_symbol} - {timeframe}.")
# # #                     overall_success = False
# # #                     logger.critical("❌ Feature generation failed. Pipeline cannot proceed.")
# # #                     sys.exit(1)
# # #     else:
# # #         logger.info("Skipping Phase 2: Independent Strategy Backtests as per configuration.")


# # #     # === Phase 3: Contextual Strategy Tuning (Optuna) ===
# # #     # Optuna tuner will load data from config.DATA_DIR_PROCESSED
# # #     run_optuna_phase = True # Set to False to skip
# # #     if run_optuna_phase:
# # #         logger.info(f"\n===== Phase: Contextual Strategy Tuning (Optuna) for {target_symbol} =====")
# # #         try:
# # #             # These parameters are now used by run_contextual_tuning internally via its argparse or defaults
# # #             # We call it as a function here. If run_contextual_tuning also needs symbol/market/segment,
# # #             # it should ideally take them as function arguments.
# # #             # Its current if __name__ == "__main__": uses argparse.
# # #             # For direct call, modify run_contextual_tuning to accept these or make it use config defaults.

# # #             # Assuming run_contextual_tuning is modified to accept these directly:

            
# # #             run_contextual_tuning(
# # #                 symbol=target_symbol,
# # #                 market=target_market,
# # #                 segment=target_segment,
# # #                 timeframes=timeframes_to_process,
# # #                 strategies=strategies_to_test, # Or a subset you want to tune
# # #                 n_trials_per_study=config.OPTUNA_TRIALS_PER_CONTEXT,
# # #                 max_workers=config.MAX_OPTUNA_WORKERS,
# # #                 run_id=RUN_ID,
# # #                 context_filter_override=None  # or a specific context dictionary
# # #             )
# # #             logger.info("✅ Contextual Tuning Phase completed.")
# # #         except Exception as e:
# # #             logger.error(f"❌ Contextual Tuning Phase failed: {e}", exc_info=True)
# # #             overall_success = False
# # #     else:
# # #         logger.info("Skipping Phase 3: Contextual Strategy Tuning (Optuna) as per configuration.")


# # #     # === Phase 4: Optional Agent Simulation ===
# # #     run_agent_simulation = False # DEFAULT: Skip agent run
# # #     if run_agent_simulation:
# # #         logger.info(f"\n===== Phase: Agent Simulation for {target_symbol} =====")
# # #         for timeframe in timeframes_to_process:
# # #             indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
# # #             indicator_input_path = config.DATA_DIR_PROCESSED / indicator_file_name
            
# # #             if not indicator_input_path.is_file():
# # #                 logger.warning(f"Skipping agent simulation for {timeframe} due to missing feature file: {indicator_input_path}")
# # #                 continue

# # #             logger.info(f"--- Running Simulation for Agent ({target_symbol} - {timeframe}) ---")
# # #             results_filename = f"Agent_{target_symbol}_{timeframe}.json"
# # #             results_json_path = AGENT_RESULTS_DIR / results_filename
            
# # #             agent_sim_args = [
# # #                 "--input", str(indicator_input_path),
# # #                 "--output-json", str(results_json_path),
# # #                 "--log-dir", str(LOGS_DIR / "agent_sim_logs"),
# # #                 "--symbol", target_symbol,
# # #                 "--market", target_market
# # #                 # DO NOT pass --strategy-name for agent mode
# # #             ]
# # #             (LOGS_DIR / "agent_sim_logs").mkdir(parents=True, exist_ok=True)

# # #             if not run_script(SIMULATION_SCRIPT_PATH, agent_sim_args, timeout=600, log_suffix=f"_Agent_{target_symbol}_{timeframe}"):
# # #                  logger.error(f"Agent simulation failed for {target_symbol} - {timeframe}.")
# # #                  #overall_success = False
# # #                  logger.critical("❌ Feature generation failed. Pipeline cannot proceed.")
# # #                  sys.exit(1)

# # #     else:
# # #         logger.info("Skipping Phase 4: Agent Simulation as per configuration.")

# # #     # --- Final Summary ---
# # #     logger.info("\n--- Pipeline Manager Finished ---")
# # #     if overall_success:
# # #         logger.info(f"✅ Pipeline completed. Check results in {RESULTS_DIR} and logs in {LOGS_DIR}. RUN ID: {RUN_ID}")
# # #     else:
# # #         logger.warning(f"⚠️ Pipeline completed with errors. Check script logs in {LOGS_DIR} for details. RUN ID: {RUN_ID}")


# # # # if __name__ == "__main__":
# # # #     try:
# # # #         # ✅ Temporary override for a quick test
# # # #         config.RAW_DATA_FILES = {
# # # #             "5min": "nifty_historical_data_5min.csv",
# # # #         }
# # # #         config.OPTUNA_TRIALS_PER_CONTEXT = 50
# # # #         config.MAX_OPTUNA_WORKERS = 4

# # # #         from app.strategies import strategy_factories
# # # #         # Filter to one strategy only
# # # #         filtered = {k: v for k, v in strategy_factories.items() if k == "SuperTrend_ADX"}
# # # #         # Monkey patch it globally
# # # #         import app.strategies
# # # #         app.strategies.strategy_factories = filtered

# # # #         main_orchestration()
# # # #     except Exception as e:
# # # #         logger.critical(f"Pipeline Manager CRASHED: {e}", exc_info=True)
# # # #         sys.exit(1)
# # # #     sys.exit(0)
# # # if __name__ == "__main__":
# # #     try:
# # #         # ✅ TEMPORARY OVERRIDE FOR TESTING SINGLE STRATEGY + TIMEFRAME
# # #         config.RAW_DATA_FILES = { "5min": "nifty_historical_data_5min.csv" }  # or "15min", etc.
# # #         config.OPTUNA_TRIALS_PER_CONTEXT = 5
# # #         config.MAX_OPTUNA_WORKERS = 1

# # #         from app.strategies import strategy_factories
# # #         filtered = {k: v for k, v in strategy_factories.items() if k == "SuperTrend_ADX"}  # <== put your strategy here
# # #         import app.strategies
# # #         app.strategies.strategy_factories = filtered

# # #         main_orchestration()
# # #     except Exception as e:
# # #         logger.critical(f"Pipeline Manager CRASHED: {e}", exc_info=True)
# # #         sys.exit(1)
# # #     sys.exit(0)
# # # pipeline_manager.py
# # import subprocess
# # import sys
# # import logging
# # from pathlib import Path
# # import os
# # from datetime import datetime
# # from typing import Optional, List 

# # from app.config import config
# # from app.optuna_tuner import run_contextual_tuning
# # from app.strategies import strategy_factories 

# # RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") 
# # PROJECT_ROOT = Path(__file__).resolve().parent 

# # RUNS_BASE_DIR = PROJECT_ROOT / "runs"
# # RUN_DIR = RUNS_BASE_DIR / RUN_ID 
# # LOGS_DIR = RUN_DIR / "logs" # MODIFIED (2025-05-09): This is the main log dir for the run   
# # RESULTS_DIR = RUN_DIR / "results" 
# # STRATEGY_RESULTS_DIR = RESULTS_DIR / "strategy_results"
# # AGENT_RESULTS_DIR = RESULTS_DIR / "agent_results"

# # for path_to_create in [RUN_DIR, LOGS_DIR, RESULTS_DIR, STRATEGY_RESULTS_DIR, AGENT_RESULTS_DIR]:
# #     path_to_create.mkdir(parents=True, exist_ok=True)

# # log_file_path = LOGS_DIR / f'pipeline_manager_{RUN_ID}.log'
# # for handler in logging.root.handlers[:]:
# #     logging.root.removeHandler(handler)
# # logging.basicConfig(level=config.LOG_LEVEL if hasattr(config, "LOG_LEVEL") else logging.INFO, 
# #                     format=config.LOG_FORMAT if hasattr(config, "LOG_FORMAT") else '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
# #                     handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])
# # logger = logging.getLogger(__name__)

# # APP_DIR = PROJECT_ROOT / "app" 
# # PIPELINE_SCRIPTS_DIR = PROJECT_ROOT / "pipeline" 
# # FEATURE_ENGINE_SCRIPT_PATH = PIPELINE_SCRIPTS_DIR / "run_feature_engine.py" 
# # SIMULATION_SCRIPT_PATH = PIPELINE_SCRIPTS_DIR / "run_simulation_step.py" 


# # def run_script(script_path: Path, args_list: List[str] = None, timeout: Optional[int] = None, log_suffix: str = "") -> bool:
# #     if args_list is None: args_list = []
# #     if not script_path.is_file():
# #         logger.error(f"Script not found: {script_path}")
# #         return False

# #     safe_log_suffix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in log_suffix)
# #     # MODIFIED (2025-05-09): Ensure subprocess logs go into the run-specific LOGS_DIR
# #     log_filename = f"{script_path.stem}{safe_log_suffix}.log"
# #     script_log_file = LOGS_DIR / log_filename 

# #     command = [sys.executable, str(script_path)] + [str(arg) for arg in args_list]
# #     logger.info(f"Running command: {' '.join(command)}")
# #     logger.info(f"Redirecting script output to: {script_log_file}")

# #     process_env = os.environ.copy()
# #     project_root_str = str(PROJECT_ROOT)
# #     current_pythonpath = process_env.get('PYTHONPATH', '')
# #     if project_root_str not in current_pythonpath.split(os.pathsep): 
# #         process_env['PYTHONPATH'] = project_root_str + os.pathsep + current_pythonpath
# #     logger.debug(f"Effective PYTHONPATH for subprocess: {process_env.get('PYTHONPATH', 'Not Set')}")

# #     try:
# #         with open(script_log_file, 'w', encoding='utf-8') as f_log:
# #             subprocess.run(command, check=True, timeout=timeout, stdout=f_log, stderr=subprocess.STDOUT, text=True, cwd=PROJECT_ROOT, env=process_env)
# #         logger.info(f"--- Script {script_path.name} OK (Log Suffix: '{log_suffix}') ---")
# #         return True
# #     except subprocess.CalledProcessError as e:
# #         logger.error(f"Script {script_path.name} FAIL (Return Code: {e.returncode}) - Check log: {script_log_file}")
# #     except subprocess.TimeoutExpired:
# #         logger.error(f"Script {script_path.name} TIMEOUT - Check log: {script_log_file}")
# #     except Exception as e:
# #         logger.error(f"Script {script_path.name} ERROR: {e}", exc_info=True)
# #     return False

# # def main_orchestration():
# #     logger.info(f"--- Starting Pipeline Manager --- RUN ID: {RUN_ID} ---")
# #     logger.info(f"Run directory: {RUN_DIR}")
# #     logger.info(f"Run Logs directory: {LOGS_DIR}") # MODIFIED (2025-05-09): Log the main log dir
# #     logger.info(f"Project Root interpreted as: {PROJECT_ROOT}")
# #     # ... (rest of the initial logging) ...

# #     timeframes_to_process = list(config.RAW_DATA_FILES.keys())
# #     strategies_to_test = list(strategy_factories.keys()) 

# #     target_symbol = getattr(config, "DEFAULT_SYMBOL", "nifty")
# #     target_market = getattr(config, "DEFAULT_MARKET", "NSE")
# #     target_segment = getattr(config, "DEFAULT_SEGMENT", "Index")

# #     logger.info(f"Target Instrument: {target_symbol} ({target_market}/{target_segment})")
# #     logger.info(f"Processing timeframes: {timeframes_to_process}")
# #     logger.info(f"Processing strategies for independent backtests: {strategies_to_test}")
# #     overall_success = True

# #     processed_data_dir = Path(getattr(config, "DATA_DIR_PROCESSED", PROJECT_ROOT / "data" / "datawithindicator"))
# #     processed_data_dir.mkdir(parents=True, exist_ok=True) 

# #     # === Phase 1: Feature Generation ===
# #     # (No changes needed in this phase regarding log directories)
# #     for timeframe in timeframes_to_process:
# #         logger.info(f"\n===== Phase: Feature Generation for {target_symbol} ({timeframe}) =====")
# #         # ... (feature generation logic as before) ...
# #         # Ensure run_script correctly places its logs in LOGS_DIR
# #         # (already handled by run_script's script_log_file = LOGS_DIR / log_filename)
# #         raw_file_name = config.RAW_DATA_FILES.get(timeframe)
# #         if not raw_file_name:
# #             logger.error(f"Missing raw data file configuration for timeframe '{timeframe}'. Skipping.")
# #             overall_success = False; continue
# #         data_folder_path = Path(getattr(config, "DATA_FOLDER", PROJECT_ROOT / "data" / "raw"))
# #         raw_path = data_folder_path / raw_file_name
# #         indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
# #         indicator_output_path = processed_data_dir / indicator_file_name
# #         if not raw_path.exists():
# #             logger.error(f"Raw data file not found: {raw_path}. Skipping feature generation for {timeframe}.")
# #             overall_success = False; continue
# #         feature_args = [
# #             "--input", str(raw_path), "--output", str(indicator_output_path),
# #             "--symbol", target_symbol, "--exchange", target_market 
# #         ]
# #         if not run_script(FEATURE_ENGINE_SCRIPT_PATH, feature_args, timeout=300, log_suffix=f"_features_{target_symbol}_{timeframe}"):
# #             logger.error(f"Feature generation failed for {target_symbol} - {timeframe}.")
# #             overall_success = False
# #             logger.critical(f"❌ Feature generation failed for {target_symbol} - {timeframe}. Aborting pipeline.")
# #             sys.exit(1)


# #     # === Phase 2: Independent Strategy Backtests ===
# #     run_independent_backtests = True 
# #     if run_independent_backtests:
# #         logger.info(f"\n===== Phase: Independent Strategy Simulation for {target_symbol} =====")
# #         # MODIFIED (2025-05-09): Define specific subdir for these logs within the main LOGS_DIR
# #         independent_sim_logs_dir = LOGS_DIR / "independent_sim_logs"
# #         independent_sim_logs_dir.mkdir(parents=True, exist_ok=True)

# #         for timeframe in timeframes_to_process:
# #             indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
# #             indicator_input_path = processed_data_dir / indicator_file_name

# #             if not indicator_input_path.is_file():
# #                 logger.critical(f"❌ Expected feature file not found: {indicator_input_path} for independent backtests. Aborting.")
# #                 sys.exit(1)

# #             for strategy_name in strategies_to_test:
# #                 logger.info(f"--- Running Independent Backtest for Strategy: {strategy_name} ({target_symbol} - {timeframe}) ---")
# #                 result_path = STRATEGY_RESULTS_DIR / f"{strategy_name}_{target_symbol}_{timeframe}_{RUN_ID}.json" # Add RUN_ID
                
# #                 sim_args = [
# #                     "--input", str(indicator_input_path),
# #                     "--output-json", str(result_path),
# #                     # MODIFIED (2025-05-09): Pass the specific subdir for these logs
# #                     "--log-dir", str(independent_sim_logs_dir), 
# #                     "--strategy-name", strategy_name,
# #                     "--symbol", target_symbol, 
# #                     "--market", target_market,
# #                     "--run-id", RUN_ID 
# #                 ]
# #                 # The run_script helper will place its main log (for the subprocess itself)
# #                 # into LOGS_DIR. The --log-dir argument tells run_simulation_step.py
# #                 # where SimpleBacktester should create its detailed trace log.
# #                 if not run_script(SIMULATION_SCRIPT_PATH, sim_args, timeout=600, log_suffix=f"_indep_sim_{strategy_name}_{target_symbol}_{timeframe}"):
# #                     logger.error(f"Independent simulation failed for {strategy_name} on {target_symbol} - {timeframe}.")
# #                     overall_success = False
# #                     logger.critical(f"❌ Independent simulation failed for {strategy_name}. Pipeline cannot reliably proceed to tuning.")
# #                     sys.exit(1)
# #     else:
# #         logger.info("Skipping Phase 2: Independent Strategy Backtests as per configuration.")

# #     # === Phase 3: Contextual Strategy Tuning (Optuna) ===
# #     run_optuna_phase = True 
# #     if run_optuna_phase:
# #         logger.info(f"\n===== Phase: Contextual Strategy Tuning (Optuna) for {target_symbol} =====")
# #         try:
# #             run_contextual_tuning(
# #                 symbol=target_symbol,
# #                 market=target_market,
# #                 segment=target_segment,
# #                 timeframes=timeframes_to_process,
# #                 strategies=strategies_to_test, 
# #                 n_trials_per_study=config.OPTUNA_TRIALS_PER_CONTEXT,
# #                 max_workers=config.MAX_OPTUNA_WORKERS,
# #                 run_id=RUN_ID,
# #                 # MODIFIED (2025-05-09): Pass the main run-specific LOGS_DIR to optuna_tuner
# #                 # optuna_tuner will then create its "optuna_trial_sim_logs" subdir inside this.
# #                 run_specific_logs_dir=LOGS_DIR, 
# #                 context_filter_override=None 
# #             )
# #             logger.info("✅ Contextual Tuning Phase completed.")
# #         except Exception as e:
# #             logger.error(f"❌ Contextual Strategy Tuning Phase failed: {e}", exc_info=True) # MODIFIED (2025-05-09): More specific error
# #             overall_success = False
# #             logger.critical(f"❌ Optuna tuning failed. Agent data will be incomplete. Aborting pipeline.")
# #             sys.exit(1)
# #     else:
# #         logger.info("Skipping Phase 3: Contextual Strategy Tuning (Optuna) as per configuration.")

# #     # === Phase 4: Optional Agent Simulation ===
# #     run_agent_simulation = False 
# #     if run_agent_simulation:
# #         logger.info(f"\n===== Phase: Agent Simulation for {target_symbol} =====")
# #         # MODIFIED (2025-05-09): Define specific subdir for agent logs
# #         agent_sim_logs_dir = LOGS_DIR / "agent_sim_logs"
# #         agent_sim_logs_dir.mkdir(parents=True, exist_ok=True)

# #         for timeframe in timeframes_to_process:
# #             indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
# #             indicator_input_path = processed_data_dir / indicator_file_name
            
# #             if not indicator_input_path.is_file():
# #                 logger.warning(f"Skipping agent simulation for {timeframe} due to missing feature file: {indicator_input_path}")
# #                 continue

# #             logger.info(f"--- Running Simulation for Agent ({target_symbol} - {timeframe}) ---")
# #             results_filename = f"Agent_{target_symbol}_{timeframe}_{RUN_ID}.json" 
# #             results_json_path = AGENT_RESULTS_DIR / results_filename
            
# #             agent_sim_args = [
# #                 "--input", str(indicator_input_path),
# #                 "--output-json", str(results_json_path),
# #                 # MODIFIED (2025-05-09): Pass the specific subdir for agent sim logs
# #                 "--log-dir", str(agent_sim_logs_dir),
# #                 "--symbol", target_symbol,
# #                 "--market", target_market,
# #                 "--run-id", RUN_ID
# #             ]
# #             if not run_script(SIMULATION_SCRIPT_PATH, agent_sim_args, timeout=600, log_suffix=f"_Agent_{target_symbol}_{timeframe}"):
# #                  logger.error(f"Agent simulation failed for {target_symbol} - {timeframe}.")
# #                  overall_success = False
# #                  logger.warning(f"⚠️ Agent simulation failed for {target_symbol} - {timeframe}. Continuing if other tasks exist.")
# #     else:
# #         logger.info("Skipping Phase 4: Agent Simulation as per configuration.")

# #     logger.info("\n--- Pipeline Manager Finished ---")
# #     if overall_success:
# #         logger.info(f"✅ Pipeline completed. Check results in {RESULTS_DIR} and logs in {LOGS_DIR}. RUN ID: {RUN_ID}")
# #     else:
# #         logger.warning(f"⚠️ Pipeline completed with errors. Check script logs in {LOGS_DIR} for details. RUN ID: {RUN_ID}")

# # if __name__ == "__main__":
# #     try:
# #         config.RAW_DATA_FILES = { "5min": "nifty_historical_data_5min.csv" } 
# #         config.OPTUNA_TRIALS_PER_CONTEXT = 5 
# #         config.MAX_OPTUNA_WORKERS = 1       

# #         from app.strategies import strategy_factories
# #         strategy_to_test_override = "SuperTrend_ADX" 
# #         if strategy_to_test_override in strategy_factories:
# #             filtered_strategies = {strategy_to_test_override: strategy_factories[strategy_to_test_override]}
# #             import app.strategies 
# #             app.strategies.strategy_factories = filtered_strategies
# #             logger.info(f"--- OVERRIDE: Testing only strategy: {strategy_to_test_override} ---")
# #         else:
# #             logger.warning(f"--- OVERRIDE WARNING: Strategy '{strategy_to_test_override}' not found. Testing all. ---")
# #         main_orchestration()
# #     except Exception as e:
# #         logger.critical(f"Pipeline Manager CRASHED: {e}", exc_info=True)
# #         sys.exit(1) 
# #     sys.exit(0) 
# # # pipeline_manager.py
# # import subprocess
# # import sys
# # import logging
# # from pathlib import Path
# # import os
# # from datetime import datetime
# # from typing import Optional, List 

# # # --- App Imports ---
# # from app.config import config
# # # MODIFIED (2025-05-10): Removed 'from app.optuna_tuner import run_contextual_tuning' from top level
# # from app.strategies import strategy_factories 

# # # --- Directory and Logging Setup ---
# # RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") 
# # PROJECT_ROOT = Path(__file__).resolve().parent 

# # RUNS_BASE_DIR = PROJECT_ROOT / "runs"
# # RUN_DIR = RUNS_BASE_DIR / RUN_ID 
# # LOGS_DIR = RUN_DIR / "logs"    
# # RESULTS_DIR = RUN_DIR / "results" 
# # STRATEGY_RESULTS_DIR = RESULTS_DIR / "strategy_results"
# # AGENT_RESULTS_DIR = RESULTS_DIR / "agent_results"

# # for path_to_create in [RUN_DIR, LOGS_DIR, RESULTS_DIR, STRATEGY_RESULTS_DIR, AGENT_RESULTS_DIR]:
# #     path_to_create.mkdir(parents=True, exist_ok=True)

# # log_file_path = LOGS_DIR / f'pipeline_manager_{RUN_ID}.log'
# # for handler in logging.root.handlers[:]:
# #     logging.root.removeHandler(handler)
# # logging.basicConfig(level=config.LOG_LEVEL if hasattr(config, "LOG_LEVEL") else logging.INFO, 
# #                     format=config.LOG_FORMAT if hasattr(config, "LOG_FORMAT") else '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
# #                     handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])
# # logger = logging.getLogger(__name__)

# # APP_DIR = PROJECT_ROOT / "app" 
# # PIPELINE_SCRIPTS_DIR = PROJECT_ROOT / "pipeline" 
# # FEATURE_ENGINE_SCRIPT_PATH = PIPELINE_SCRIPTS_DIR / "run_feature_engine.py" 
# # SIMULATION_SCRIPT_PATH = PIPELINE_SCRIPTS_DIR / "run_simulation_step.py" 


# # def run_script(script_path: Path, args_list: List[str] = None, timeout: Optional[int] = None, log_suffix: str = "") -> bool:
# #     if args_list is None: args_list = []
# #     if not script_path.is_file():
# #         logger.error(f"Script not found: {script_path}")
# #         return False

# #     safe_log_suffix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in log_suffix)
# #     log_filename = f"{script_path.stem}{safe_log_suffix}.log"
# #     script_log_file = LOGS_DIR / log_filename 

# #     command = [sys.executable, str(script_path)] + [str(arg) for arg in args_list]
# #     logger.info(f"Running command: {' '.join(command)}")
# #     logger.info(f"Redirecting script output to: {script_log_file}")

# #     process_env = os.environ.copy()
# #     project_root_str = str(PROJECT_ROOT)
# #     current_pythonpath = process_env.get('PYTHONPATH', '')
# #     if project_root_str not in current_pythonpath.split(os.pathsep): 
# #         process_env['PYTHONPATH'] = project_root_str + os.pathsep + current_pythonpath
# #     logger.debug(f"Effective PYTHONPATH for subprocess: {process_env.get('PYTHONPATH', 'Not Set')}")

# #     try:
# #         with open(script_log_file, 'w', encoding='utf-8') as f_log:
# #             subprocess.run(command, check=True, timeout=timeout, stdout=f_log, stderr=subprocess.STDOUT, text=True, cwd=PROJECT_ROOT, env=process_env)
# #         logger.info(f"--- Script {script_path.name} OK (Log Suffix: '{log_suffix}') ---")
# #         return True
# #     except subprocess.CalledProcessError as e:
# #         logger.error(f"Script {script_path.name} FAIL (Return Code: {e.returncode}) - Check log: {script_log_file}")
# #     except subprocess.TimeoutExpired:
# #         logger.error(f"Script {script_path.name} TIMEOUT - Check log: {script_log_file}")
# #     except Exception as e:
# #         logger.error(f"Script {script_path.name} ERROR: {e}", exc_info=True)
# #     return False

# # def main_orchestration():
# #     # MODIFIED (2025-05-10): Moved import inside the function to avoid circular import with multiprocessing
  
# #     from app.optuna_tuner import run_contextual_tuning

# #     logger.info(f"--- Starting Pipeline Manager --- RUN ID: {RUN_ID} ---")
# #     logger.info(f"Run directory: {RUN_DIR}")
# #     logger.info(f"Run Logs directory: {LOGS_DIR}") 
# #     logger.info(f"Project Root interpreted as: {PROJECT_ROOT}")
# #     logger.info(f"Feature Engine Script Path: {FEATURE_ENGINE_SCRIPT_PATH}") # For debug
# #     logger.info(f"Simulation Script Path: {SIMULATION_SCRIPT_PATH}") # For debug


# #     timeframes_to_process = list(config.RAW_DATA_FILES.keys())
# #     strategies_to_test = list(strategy_factories.keys()) 

# #     target_symbol = getattr(config, "DEFAULT_SYMBOL", "nifty")
# #     target_market = getattr(config, "DEFAULT_MARKET", "NSE")
# #     target_segment = getattr(config, "DEFAULT_SEGMENT", "Index")

# #     logger.info(f"Target Instrument: {target_symbol} ({target_market}/{target_segment})")
# #     logger.info(f"Processing timeframes: {timeframes_to_process}")
# #     logger.info(f"Processing strategies for independent backtests: {strategies_to_test}")
# #     overall_success = True

# #     processed_data_dir = Path(getattr(config, "DATA_DIR_PROCESSED", PROJECT_ROOT / "data" / "datawithindicator"))
# #     processed_data_dir.mkdir(parents=True, exist_ok=True) 

# #     # === Phase 1: Feature Generation ===
# #     for timeframe in timeframes_to_process:
# #         logger.info(f"\n===== Phase: Feature Generation for {target_symbol} ({timeframe}) =====")
# #         raw_file_name = config.RAW_DATA_FILES.get(timeframe)
# #         if not raw_file_name:
# #             logger.error(f"Missing raw data file configuration for timeframe '{timeframe}'. Skipping.")
# #             overall_success = False; continue
# #         data_folder_path = Path(getattr(config, "DATA_FOLDER", PROJECT_ROOT / "data" / "raw"))
# #         raw_path = data_folder_path / raw_file_name
# #         indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
# #         indicator_output_path = processed_data_dir / indicator_file_name
# #         if not raw_path.exists():
# #             logger.error(f"Raw data file not found: {raw_path}. Skipping feature generation for {timeframe}.")
# #             overall_success = False; continue
# #         feature_args = [
# #             "--input", str(raw_path), "--output", str(indicator_output_path),
# #             "--symbol", target_symbol, "--exchange", target_market 
# #         ]
# #         if not run_script(FEATURE_ENGINE_SCRIPT_PATH, feature_args, timeout=300, log_suffix=f"_features_{target_symbol}_{timeframe}"):
# #             logger.error(f"Feature generation failed for {target_symbol} - {timeframe}.")
# #             overall_success = False
# #             logger.critical(f"❌ Feature generation failed for {target_symbol} - {timeframe}. Aborting pipeline.")
# #             sys.exit(1)

# #     # === Phase 2: Independent Strategy Backtests ===
# #     run_independent_backtests = True 
# #     if run_independent_backtests:
# #         logger.info(f"\n===== Phase: Independent Strategy Simulation for {target_symbol} =====")
# #         independent_sim_logs_dir = LOGS_DIR / "independent_sim_logs"
# #         independent_sim_logs_dir.mkdir(parents=True, exist_ok=True)

# #         for timeframe in timeframes_to_process:
# #             indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
# #             indicator_input_path = processed_data_dir / indicator_file_name

# #             if not indicator_input_path.is_file():
# #                 logger.critical(f"❌ Expected feature file not found: {indicator_input_path} for independent backtests. Aborting.")
# #                 sys.exit(1)

# #             for strategy_name in strategies_to_test:
# #                 logger.info(f"--- Running Independent Backtest for Strategy: {strategy_name} ({target_symbol} - {timeframe}) ---")
# #                 result_path = STRATEGY_RESULTS_DIR / f"{strategy_name}_{target_symbol}_{timeframe}_{RUN_ID}.json"
                
# #                 sim_args = [
# #                     "--input", str(indicator_input_path),
# #                     "--output-json", str(result_path),
# #                     "--log-dir", str(independent_sim_logs_dir), 
# #                     "--strategy-name", strategy_name,
# #                     "--symbol", target_symbol, 
# #                     "--market", target_market,
# #                     "--run-id", RUN_ID 
# #                 ]
# #                 if not run_script(SIMULATION_SCRIPT_PATH, sim_args, timeout=600, log_suffix=f"_indep_sim_{strategy_name}_{target_symbol}_{timeframe}"):
# #                     logger.error(f"Independent simulation failed for {strategy_name} on {target_symbol} - {timeframe}.")
# #                     overall_success = False
# #                     logger.critical(f"❌ Independent simulation failed for {strategy_name}. Pipeline cannot reliably proceed to tuning.")
# #                     sys.exit(1)
# #     else:
# #         logger.info("Skipping Phase 2: Independent Strategy Backtests as per configuration.")

# #     # === Phase 3: Contextual Strategy Tuning (Optuna) ===
# #     run_optuna_phase = True 
# #     if run_optuna_phase:
# #         logger.info(f"\n===== Phase: Contextual Strategy Tuning (Optuna) for {target_symbol} =====")
# #         try:
# #             run_contextual_tuning(
# #                 symbol=target_symbol,
# #                 market=target_market,
# #                 segment=target_segment,
# #                 timeframes=timeframes_to_process,
# #                 strategies=strategies_to_test, 
# #                 n_trials_per_study=config.OPTUNA_TRIALS_PER_CONTEXT,
# #                 max_workers=config.MAX_OPTUNA_WORKERS,
# #                 run_id=RUN_ID,  
# #                 run_specific_logs_dir=LOGS_DIR, 
# #                 context_filter_override=None 
# #             )
# #             logger.info("✅ Contextual Tuning Phase completed.")
# #         except Exception as e:
# #             logger.error(f"❌ Contextual Strategy Tuning Phase failed: {e}", exc_info=True) 
# #             overall_success = False
# #             logger.critical(f"❌ Optuna tuning failed. Agent data will be incomplete. Aborting pipeline.")
# #             sys.exit(1)
# #     else:
# #         logger.info("Skipping Phase 3: Contextual Strategy Tuning (Optuna) as per configuration.")

# #     # === Phase 4: Optional Agent Simulation ===
# #     run_agent_simulation = False 
# #     if run_agent_simulation:
# #         logger.info(f"\n===== Phase: Agent Simulation for {target_symbol} =====")
# #         agent_sim_logs_dir = LOGS_DIR / "agent_sim_logs"
# #         agent_sim_logs_dir.mkdir(parents=True, exist_ok=True)

# #         for timeframe in timeframes_to_process:
# #             indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
# #             indicator_input_path = processed_data_dir / indicator_file_name
            
# #             if not indicator_input_path.is_file():
# #                 logger.warning(f"Skipping agent simulation for {timeframe} due to missing feature file: {indicator_input_path}")
# #                 continue

# #             logger.info(f"--- Running Simulation for Agent ({target_symbol} - {timeframe}) ---")
# #             results_filename = f"Agent_{target_symbol}_{timeframe}_{RUN_ID}.json" 
# #             results_json_path = AGENT_RESULTS_DIR / results_filename
            
# #             agent_sim_args = [
# #                 "--input", str(indicator_input_path),
# #                 "--output-json", str(results_json_path),
# #                 "--log-dir", str(agent_sim_logs_dir),
# #                 "--symbol", target_symbol,
# #                 "--market", target_market,
# #                 "--run-id", RUN_ID
# #             ]
# #             if not run_script(SIMULATION_SCRIPT_PATH, agent_sim_args, timeout=600, log_suffix=f"_Agent_{target_symbol}_{timeframe}"):
# #                  logger.error(f"Agent simulation failed for {target_symbol} - {timeframe}.")
# #                  overall_success = False
# #                  logger.warning(f"⚠️ Agent simulation failed for {target_symbol} - {timeframe}. Continuing if other tasks exist.")
# #     else:
# #         logger.info("Skipping Phase 4: Agent Simulation as per configuration.")

# #     logger.info("\n--- Pipeline Manager Finished ---")
# #     if overall_success:
# #         logger.info(f"✅ Pipeline completed. Check results in {RESULTS_DIR} and logs in {LOGS_DIR}. RUN ID: {RUN_ID}")
# #     else:
# #         logger.warning(f"⚠️ Pipeline completed with errors. Check script logs in {LOGS_DIR} for details. RUN ID: {RUN_ID}")


# # if __name__ == "__main__":
# #     try:
# #         # --- Ensure MongoManager connection is closed on exit ---
# #         # This is important because Optuna uses multiprocessing which might not trigger
# #         # MongoManager's own atexit if it's registered in a child process.
# #         # Registering it here in the main process is safer.
# #         from app.mongo_manager import close_mongo_connection_on_exit
# #         import atexit
# #         atexit.register(close_mongo_connection_on_exit)


# #         # --- Test overrides ---
# #         config.RAW_DATA_FILES = { "5min": "nifty_historical_data_5min.csv" } 
# #         config.OPTUNA_TRIALS_PER_CONTEXT = getattr(config, "OPTUNA_TRIALS_PER_CONTEXT_TEST", 5) # Use a test-specific config or default to 5
# #         config.MAX_OPTUNA_WORKERS = getattr(config, "MAX_OPTUNA_WORKERS_TEST", 1)       

# #         from app.strategies import strategy_factories # Re-import locally if patched globally before
# #         strategy_to_test_override = "SuperTrend_ADX" # Or None to test all
        
# #         if strategy_to_test_override and strategy_to_test_override in strategy_factories:
# #             # If you want to truly limit what Optuna sees, you might need to pass this
# #             # 'strategies_to_test' list to run_contextual_tuning, and have it use that.
# #             # Monkey-patching strategy_factories might not be effective if optuna_tuner
# #             # imports it at its own top level before the patch.
# #             # For simplicity of testing, ensure optuna_tuner uses the strategies list passed to it.
# #             strategies_to_test_for_this_run = [strategy_to_test_override]
# #             logger.info(f"--- OVERRIDE: Testing only strategy: {strategy_to_test_override} ---")
# #         else:
# #             strategies_to_test_for_this_run = list(strategy_factories.keys())
# #             if strategy_to_test_override: # Log if specified but not found
# #                  logger.warning(f"--- OVERRIDE WARNING: Strategy '{strategy_to_test_override}' not found. Testing all. ---")
        
# #         # To make the override effective for run_contextual_tuning, ensure it uses the passed 'strategies' list.
# #         # The current run_contextual_tuning in optuna_tuner_fix_expiry_arg_20250510 does use the passed list.
        
# #         # Modify strategies_to_test if override is active for main_orchestration
# #         # This is a bit of a hack; better to pass the filtered list to main_orchestration if it were a function.
# #         # For now, this will affect the global strategies_to_test used by main_orchestration.
# #         # if strategy_to_test_override and strategies_to_test_for_this_run:
# #         #     # This global patch is generally not ideal but works for this __main__ test block.
# #         #     import app.strategies
# #         #     app.strategies.strategy_factories = {
# #         #         k: strategy_factories[k] for k in strategies_to_test_for_this_run if k in strategy_factories
# #         #     }


# #         main_orchestration()
# #     except Exception as e:
# #         logger.critical(f"Pipeline Manager CRASHED: {e}", exc_info=True)
# #         sys.exit(1) 
# #     sys.exit(0) 
# # pipeline_manager.py
# import subprocess
# import sys
# import logging
# from pathlib import Path
# import os
# from datetime import datetime
# from typing import Optional, List 

# # --- App Imports ---
# from app.config import config
# # MODIFIED (2025-05-10): Moved 'from app.optuna_tuner import run_contextual_tuning' into main_orchestration
# from app.strategies import strategy_factories 

# # --- Directory and Logging Setup ---
# def generate_run_id():
#     return datetime.now().strftime("%Y%m%d_%H%M%S")
# PROJECT_ROOT = Path(__file__).resolve().parent 
# RUN_ID = generate_run_id()
# RUNS_BASE_DIR = PROJECT_ROOT / "runs"
# RUN_DIR = RUNS_BASE_DIR / RUN_ID 
# LOGS_DIR = RUN_DIR / "logs"    # This is the single main log directory for THIS run
# RESULTS_DIR = RUN_DIR / "results" 
# STRATEGY_RESULTS_DIR = RESULTS_DIR / "strategy_results"
# AGENT_RESULTS_DIR = RESULTS_DIR / "agent_results"

# for path_to_create in [RUN_DIR, LOGS_DIR, RESULTS_DIR, STRATEGY_RESULTS_DIR, AGENT_RESULTS_DIR]:
#     path_to_create.mkdir(parents=True, exist_ok=True)

# log_file_path = LOGS_DIR / f'pipeline_manager_{RUN_ID}.log'
# # Clear root handlers to ensure this basicConfig takes precedence for this script
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)
# logging.basicConfig(level=config.LOG_LEVEL if hasattr(config, "LOG_LEVEL") else logging.INFO, 
#                     format=config.LOG_FORMAT if hasattr(config, "LOG_FORMAT") else '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
#                     handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])
# logger = logging.getLogger(__name__)

# APP_DIR = PROJECT_ROOT / "app" 
# PIPELINE_SCRIPTS_DIR = PROJECT_ROOT / "pipeline" 
# FEATURE_ENGINE_SCRIPT_PATH = PIPELINE_SCRIPTS_DIR / "run_feature_engine.py" 
# SIMULATION_SCRIPT_PATH = PIPELINE_SCRIPTS_DIR / "run_simulation_step.py" 


# def run_script(script_path: Path, args_list: List[str] = None, timeout: Optional[int] = None, log_suffix: str = "") -> bool:
#     if args_list is None: args_list = []
#     if not script_path.is_file():
#         logger.error(f"Script not found: {script_path}")
#         return False

#     safe_log_suffix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in log_suffix)
#     # This log_filename is for the stdout/stderr capture of the subprocess
#     log_filename = f"{script_path.stem}{safe_log_suffix}.log"
#     script_capture_log_file = LOGS_DIR / log_filename # Goes into the main LOGS_DIR

#     command = [sys.executable, str(script_path)] + [str(arg) for arg in args_list]
#     logger.info(f"Running command: {' '.join(command)}")
#     logger.info(f"Redirecting script stdout/stderr to: {script_capture_log_file}")

#     process_env = os.environ.copy()
#     project_root_str = str(PROJECT_ROOT)
#     current_pythonpath = process_env.get('PYTHONPATH', '')
#     if project_root_str not in current_pythonpath.split(os.pathsep): 
#         process_env['PYTHONPATH'] = project_root_str + os.pathsep + current_pythonpath
    
#     # MODIFIED (2025-05-10): Set environment variables for subprocesses
#     # These tell the subprocess about the main run's context
#     process_env['PIPELINE_MAIN_RUN_ID'] = RUN_ID 
#     process_env['PIPELINE_MAIN_LOGS_DIR'] = str(LOGS_DIR.resolve()) # Pass absolute path

#     logger.debug(f"Effective PYTHONPATH for subprocess: {process_env.get('PYTHONPATH', 'Not Set')}")
#     logger.debug(f"Env PIPELINE_MAIN_RUN_ID for subprocess: {process_env['PIPELINE_MAIN_RUN_ID']}")
#     logger.debug(f"Env PIPELINE_MAIN_LOGS_DIR for subprocess: {process_env['PIPELINE_MAIN_LOGS_DIR']}")

#     try:
#         with open(script_capture_log_file, 'w', encoding='utf-8') as f_log:
#             subprocess.run(command, check=True, timeout=timeout, stdout=f_log, stderr=subprocess.STDOUT, text=True, cwd=PROJECT_ROOT, env=process_env)
#         logger.info(f"--- Script {script_path.name} OK (Log Suffix: '{log_suffix}') ---")
#         return True
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Script {script_path.name} FAIL (Return Code: {e.returncode}) - Check log: {script_capture_log_file}")
#     except subprocess.TimeoutExpired:
#         logger.error(f"Script {script_path.name} TIMEOUT - Check log: {script_capture_log_file}")
#     except Exception as e:
#         logger.error(f"Script {script_path.name} ERROR: {e}", exc_info=True)
#     return False

# def main_orchestration():
#     from app.optuna_tuner import run_contextual_tuning # Moved import here
    
#     logger.info(f"--- Starting Pipeline Manager --- RUN ID: {RUN_ID} ---")
#     logger.info(f"Run directory: {RUN_DIR}")
#     logger.info(f"Run Logs directory (all logs for this run should be here or in subdirs): {LOGS_DIR}") 
#     logger.info(f"Project Root interpreted as: {PROJECT_ROOT}")
#     logger.info(f"Feature Engine Script Path: {FEATURE_ENGINE_SCRIPT_PATH}") 
#     logger.info(f"Simulation Script Path: {SIMULATION_SCRIPT_PATH}") 


#     timeframes_to_process = list(config.RAW_DATA_FILES.keys())
#     strategies_to_test = list(strategy_factories.keys()) 

#     target_symbol = getattr(config, "DEFAULT_SYMBOL", "nifty")
#     target_market = getattr(config, "DEFAULT_MARKET", "NSE")
#     target_segment = getattr(config, "DEFAULT_SEGMENT", "Index")

#     logger.info(f"Target Instrument: {target_symbol} ({target_market}/{target_segment})")
#     logger.info(f"Processing timeframes: {timeframes_to_process}")
#     logger.info(f"Processing strategies for independent backtests: {strategies_to_test}")
#     overall_success = True

#     processed_data_dir = Path(getattr(config, "DATA_DIR_PROCESSED", PROJECT_ROOT / "data" / "datawithindicator"))
#     processed_data_dir.mkdir(parents=True, exist_ok=True) 

#     # === Phase 1: Feature Generation ===
#     for timeframe in timeframes_to_process:
#         logger.info(f"\n===== Phase: Feature Generation for {target_symbol} ({timeframe}) =====")
#         raw_file_name = config.RAW_DATA_FILES.get(timeframe)
#         if not raw_file_name:
#             logger.error(f"Missing raw data file configuration for timeframe '{timeframe}'. Skipping.")
#             overall_success = False; continue
#         data_folder_path = Path(getattr(config, "DATA_FOLDER", PROJECT_ROOT / "data" / "raw"))
#         raw_path = data_folder_path / raw_file_name
#         indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
#         indicator_output_path = processed_data_dir / indicator_file_name
#         if not raw_path.exists():
#             logger.error(f"Raw data file not found: {raw_path}. Skipping feature generation for {timeframe}.")
#             overall_success = False; continue
#         feature_args = [
#             "--input", str(raw_path), "--output", str(indicator_output_path),
#             "--symbol", target_symbol, "--exchange", target_market 
#         ]
#         if not run_script(FEATURE_ENGINE_SCRIPT_PATH, feature_args, timeout=300, log_suffix=f"_features_{target_symbol}_{timeframe}"):
#             logger.error(f"Feature generation failed for {target_symbol} - {timeframe}.")
#             overall_success = False
#             logger.critical(f"❌ Feature generation failed for {target_symbol} - {timeframe}. Aborting pipeline.")
#             sys.exit(1)

#     # === Phase 2: Independent Strategy Backtests ===
#     run_independent_backtests = True 
#     if run_independent_backtests:
#         logger.info(f"\n===== Phase: Independent Strategy Simulation for {target_symbol} =====")
#         independent_sim_trace_logs_dir = LOGS_DIR / "independent_sim_traces" 
#         independent_sim_trace_logs_dir.mkdir(parents=True, exist_ok=True)

#         for timeframe in timeframes_to_process:
#             indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
#             indicator_input_path = processed_data_dir / indicator_file_name

#             if not indicator_input_path.is_file():
#                 logger.critical(f"❌ Expected feature file not found: {indicator_input_path} for independent backtests. Aborting.")
#                 sys.exit(1)

#             for strategy_name in strategies_to_test:
#                 logger.info(f"--- Running Independent Backtest for Strategy: {strategy_name} ({target_symbol} - {timeframe}) ---")
#                 result_path = STRATEGY_RESULTS_DIR / f"{strategy_name}_{target_symbol}_{timeframe}_{RUN_ID}.json"
                
#                 sim_args = [
#                     "--input", str(indicator_input_path),
#                     "--output-json", str(result_path),
#                     "--log-dir", str(independent_sim_trace_logs_dir), 
#                     "--strategy-name", strategy_name,
#                     "--symbol", target_symbol, 
#                     "--market", target_market,
#                     "--run-id", RUN_ID 
#                 ]
#                 if not run_script(SIMULATION_SCRIPT_PATH, sim_args, timeout=600, log_suffix=f"_indep_sim_{strategy_name}_{target_symbol}_{timeframe}"):
#                     logger.error(f"Independent simulation failed for {strategy_name} on {target_symbol} - {timeframe}.")
#                     overall_success = False
#                     logger.critical(f"❌ Independent simulation failed for {strategy_name}. Pipeline cannot reliably proceed to tuning.")
#                     sys.exit(1)
#     else:
#         logger.info("Skipping Phase 2: Independent Strategy Backtests as per configuration.")

#     # === Phase 3: Contextual Strategy Tuning (Optuna) ===
#     run_optuna_phase = True 
#     if run_optuna_phase:
#         logger.info(f"\n===== Phase: Contextual Strategy Tuning (Optuna) for {target_symbol} =====")
#         try:
#             run_contextual_tuning(
#                 symbol=target_symbol,
#                 market=target_market,
#                 segment=target_segment,
#                 timeframes=timeframes_to_process,
#                 strategies=strategies_to_test, 
#                 n_trials_per_study=config.OPTUNA_TRIALS_PER_CONTEXT,
#                 max_workers=config.MAX_OPTUNA_WORKERS,
#                 run_id=RUN_ID,  
#                 run_specific_logs_dir=LOGS_DIR, 
#                 context_filter_override=None 
#             )
#             logger.info("✅ Contextual Tuning Phase completed.")
#         except Exception as e:
#             logger.error(f"❌ Contextual Strategy Tuning Phase failed: {e}", exc_info=True) 
#             overall_success = False
#             logger.critical(f"❌ Optuna tuning failed. Agent data will be incomplete. Aborting pipeline.")
#             sys.exit(1)
#     else:
#         logger.info("Skipping Phase 3: Contextual Strategy Tuning (Optuna) as per configuration.")

#     # === Phase 4: Optional Agent Simulation ===
#     run_agent_simulation = False 
#     if run_agent_simulation:
#         logger.info(f"\n===== Phase: Agent Simulation for {target_symbol} =====")
#         agent_sim_trace_logs_dir = LOGS_DIR / "agent_sim_traces" 
#         agent_sim_trace_logs_dir.mkdir(parents=True, exist_ok=True)

#         for timeframe in timeframes_to_process:
#             indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
#             indicator_input_path = processed_data_dir / indicator_file_name
            
#             if not indicator_input_path.is_file():
#                 logger.warning(f"Skipping agent simulation for {timeframe} due to missing feature file: {indicator_input_path}")
#                 continue

#             logger.info(f"--- Running Simulation for Agent ({target_symbol} - {timeframe}) ---")
#             results_filename = f"Agent_{target_symbol}_{timeframe}_{RUN_ID}.json" 
#             results_json_path = AGENT_RESULTS_DIR / results_filename
            
#             agent_sim_args = [
#                 "--input", str(indicator_input_path),
#                 "--output-json", str(results_json_path),
#                 "--log-dir", str(agent_sim_trace_logs_dir), 
#                 "--symbol", target_symbol,
#                 "--market", target_market,
#                 "--run-id", RUN_ID
#             ]
#             if not run_script(SIMULATION_SCRIPT_PATH, agent_sim_args, timeout=600, log_suffix=f"_Agent_{target_symbol}_{timeframe}"):
#                  logger.error(f"Agent simulation failed for {target_symbol} - {timeframe}.")
#                  overall_success = False
#                  logger.warning(f"⚠️ Agent simulation failed for {target_symbol} - {timeframe}. Continuing if other tasks exist.")
#     else:
#         logger.info("Skipping Phase 4: Agent Simulation as per configuration.")

#     logger.info("\n--- Pipeline Manager Finished ---")
#     if overall_success:
#         logger.info(f"✅ Pipeline completed. Check results in {RESULTS_DIR} and logs in {LOGS_DIR}. RUN ID: {RUN_ID}")
#     else:
#         logger.warning(f"⚠️ Pipeline completed with errors. Check script logs in {LOGS_DIR} for details. RUN ID: {RUN_ID}")


# if __name__ == "__main__":
#     try:
#         from app.mongo_manager import close_mongo_connection_on_exit
#         import atexit
#         atexit.register(close_mongo_connection_on_exit)

#         config.RAW_DATA_FILES = { "5min": "nifty_historical_data_5min.csv" } 
#         config.OPTUNA_TRIALS_PER_CONTEXT = getattr(config, "OPTUNA_TRIALS_PER_CONTEXT_TEST", 5) 
#         config.MAX_OPTUNA_WORKERS = getattr(config, "MAX_OPTUNA_WORKERS_TEST", 1)       

#         from app.strategies import strategy_factories 
#         strategy_to_test_override = "SuperTrend_ADX" 
        
#         # This logic for overriding strategies_to_test needs to be carefully managed.
#         # If main_orchestration is called, it uses the global strategy_factories.
#         # For testing, it's better to pass the list of strategies to main_orchestration if it were a function.
#         # For now, this patching will affect the global for the __main__ block's execution.
#         _original_strategy_factories = strategy_factories.copy() # Keep a copy
#         if strategy_to_test_override and strategy_to_test_override in strategy_factories:
#             strategy_factories.clear()
#             strategy_factories[strategy_to_test_override] = _original_strategy_factories[strategy_to_test_override]
#             logger.info(f"--- OVERRIDE: Testing only strategy: {strategy_to_test_override} ---")
#         elif strategy_to_test_override:
#              logger.warning(f"--- OVERRIDE WARNING: Strategy '{strategy_to_test_override}' not found. Testing all. ---")
        
#         main_orchestration()

#     except Exception as e:
#         logger.critical(f"Pipeline Manager CRASHED: {e}", exc_info=True)
#         sys.exit(1) 
#     finally:
#         # Restore original strategy_factories if it was patched
#         if '_original_strategy_factories' in locals() and _original_strategy_factories:
#             # This ensures that if the script is imported elsewhere later,
#             # or if other functions in this script rely on the full list, it's restored.
#             # For a simple script execution that exits, this might not be strictly necessary,
#             # but it's good practice if the module could be reused.
#             strategy_factories = _original_strategy_factories
#             logger.debug("Restored original strategy_factories.")
#     sys.exit(0) 

# pipeline_manager.py
import subprocess
import sys
import logging
from pathlib import Path
import os
from datetime import datetime
from typing import Optional, List, Any # Added Any

# --- App Imports ---
from app.config import config
# app.optuna_tuner is imported inside main_orchestration to avoid potential circular issues
# if optuna_tuner itself might (even indirectly) cause an import of pipeline_manager
from app.strategies import strategy_factories

# --- Constants and Paths (defined but not acted upon at module level) ---
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_BASE_DIR = PROJECT_ROOT / "runs"
APP_DIR = PROJECT_ROOT / "app"
PIPELINE_SCRIPTS_DIR = PROJECT_ROOT / "pipeline"
FEATURE_ENGINE_SCRIPT_PATH = PIPELINE_SCRIPTS_DIR / "run_feature_engine.py"
SIMULATION_SCRIPT_PATH = PIPELINE_SCRIPTS_DIR / "run_simulation_step.py"

# --- Global variables to store run-specific paths and ID ---
# These will be initialized by setup_run_environment()
RUN_ID_GLOBAL: Optional[str] = None
RUN_DIR_GLOBAL: Optional[Path] = None
LOGS_DIR_GLOBAL: Optional[Path] = None
RESULTS_DIR_GLOBAL: Optional[Path] = None
STRATEGY_RESULTS_DIR_GLOBAL: Optional[Path] = None
AGENT_RESULTS_DIR_GLOBAL: Optional[Path] = None

# Get a logger instance for this module
# The handlers and formatting will be set in setup_run_environment()
logger = logging.getLogger(__name__)

def generate_run_id_internal() -> str:
    """Generates a timestamp-based Run ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_run_environment():
    """
    Generates RUN_ID, creates run-specific directories, and configures
    logging for the main pipeline manager. This should be called ONLY ONCE.
    """
    global RUN_ID_GLOBAL, RUN_DIR_GLOBAL, LOGS_DIR_GLOBAL, RESULTS_DIR_GLOBAL, \
           STRATEGY_RESULTS_DIR_GLOBAL, AGENT_RESULTS_DIR_GLOBAL

    # Prevent re-initialization if called multiple times within the same process
    if RUN_ID_GLOBAL is not None:
        logger.warning(
            f"setup_run_environment called again for RUN_ID: {RUN_ID_GLOBAL}. "
            "Skipping re-initialization of run environment."
        )
        return

    RUN_ID_GLOBAL = generate_run_id_internal()
    RUN_DIR_GLOBAL = RUNS_BASE_DIR / RUN_ID_GLOBAL
    LOGS_DIR_GLOBAL = RUN_DIR_GLOBAL / "logs"
    RESULTS_DIR_GLOBAL = RUN_DIR_GLOBAL / "results"
    STRATEGY_RESULTS_DIR_GLOBAL = RESULTS_DIR_GLOBAL / "strategy_results"
    AGENT_RESULTS_DIR_GLOBAL = RESULTS_DIR_GLOBAL / "agent_results"

    paths_to_create = [
        RUN_DIR_GLOBAL, LOGS_DIR_GLOBAL, RESULTS_DIR_GLOBAL,
        STRATEGY_RESULTS_DIR_GLOBAL, AGENT_RESULTS_DIR_GLOBAL
    ]
    for path_to_create in paths_to_create:
        path_to_create.mkdir(parents=True, exist_ok=True)
        # logger.debug(f"Ensured directory exists: {path_to_create}") # Optional debug

    # Configure logging for the pipeline_manager.py script itself
    log_file_path = LOGS_DIR_GLOBAL / f'pipeline_manager_{RUN_ID_GLOBAL}.log'

    # Clear any existing handlers from the root logger to avoid conflicts or duplicates
    # This is important if other modules might have called basicConfig earlier.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Now, configure logging for this specific pipeline_manager run.
    # This setup will apply to loggers obtained via logging.getLogger() hereafter,
    # unless they have their propagate flag set to False or have their own specific handlers.
    logging.basicConfig(
        level=config.LOG_LEVEL if hasattr(config, "LOG_LEVEL") else logging.INFO,
        format=config.LOG_FORMAT if hasattr(config, "LOG_FORMAT") else '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout) # Keep console output
        ]
    )
    # The logger for *this* module (`pipeline_manager`) will now use the above config.
    logger.info(f"Run environment setup complete. RUN_ID: {RUN_ID_GLOBAL}")
    logger.info(f"Main logs for pipeline_manager will be in: {log_file_path}")
    logger.info(f"All run-specific logs will be under: {LOGS_DIR_GLOBAL}")


def run_script(script_path: Path, args_list: Optional[List[str]] = None, timeout: Optional[int] = None, log_suffix: str = "") -> bool:
    """Executes a given script as a subprocess and manages its logging."""
    if LOGS_DIR_GLOBAL is None or RUN_ID_GLOBAL is None:
        logger.critical("Run environment not set up. Call setup_run_environment() first.")
        # Optionally, could try to call setup_run_environment() here as a fallback,
        # but it's better if the main flow ensures it's called.
        return False

    if args_list is None:
        args_list = []

    if not script_path.is_file():
        logger.error(f"Script not found: {script_path}")
        return False

    safe_log_suffix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in log_suffix)
    # This log_filename is for capturing the stdout/stderr of the subprocess
    log_filename = f"{script_path.stem}{safe_log_suffix}.log"
    # All subprocess logs go into the globally defined LOGS_DIR_GLOBAL for the current run
    script_capture_log_file = LOGS_DIR_GLOBAL / log_filename

    command = [sys.executable, str(script_path)] + [str(arg) for arg in args_list]
    logger.info(f"Running command: {' '.join(command)}")
    logger.info(f"Redirecting script stdout/stderr to: {script_capture_log_file}")

    process_env = os.environ.copy()
    project_root_str = str(PROJECT_ROOT)
    current_pythonpath = process_env.get('PYTHONPATH', '')
    if project_root_str not in current_pythonpath.split(os.pathsep):
        process_env['PYTHONPATH'] = project_root_str + os.pathsep + current_pythonpath

    # Pass the main run's context to subprocesses via environment variables
    process_env['PIPELINE_MAIN_RUN_ID'] = RUN_ID_GLOBAL
    process_env['PIPELINE_MAIN_LOGS_DIR'] = str(LOGS_DIR_GLOBAL.resolve())

    logger.debug(f"Effective PYTHONPATH for subprocess: {process_env.get('PYTHONPATH', 'Not Set')}")
    logger.debug(f"Env PIPELINE_MAIN_RUN_ID for subprocess: {process_env['PIPELINE_MAIN_RUN_ID']}")
    logger.debug(f"Env PIPELINE_MAIN_LOGS_DIR for subprocess: {process_env['PIPELINE_MAIN_LOGS_DIR']}")

    try:
        with open(script_capture_log_file, 'w', encoding='utf-8') as f_log:
            subprocess.run(
                command,
                check=True,
                timeout=timeout,
                stdout=f_log,
                stderr=subprocess.STDOUT, # Redirect stderr to the same log file
                text=True,
                cwd=PROJECT_ROOT,
                env=process_env
            )
        logger.info(f"--- Script {script_path.name} OK (Log Suffix: '{log_suffix}') ---")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Script {script_path.name} FAIL (Return Code: {e.returncode}) - Check log: {script_capture_log_file}")
    except subprocess.TimeoutExpired:
        logger.error(f"Script {script_path.name} TIMEOUT - Check log: {script_capture_log_file}")
    except Exception as e:
        logger.error(f"Script {script_path.name} ERROR: {e}", exc_info=True)
    return False


def main_orchestration(strategies_to_run: Optional[List[str]] = None):
    """
    Main pipeline orchestration logic.
    Uses the globally set RUN_ID_GLOBAL and path variables.
    """
    if RUN_ID_GLOBAL is None or LOGS_DIR_GLOBAL is None:
        logger.critical("Run environment is not initialized. Aborting orchestration.")
        sys.exit("Critical error: Run environment not set up.")

    # Moved import here to ensure setup_run_environment (which configures logging)
    # has run before optuna_tuner potentially uses logging.
    from app.optuna_tuner import run_contextual_tuning

    logger.info(f"--- Starting Pipeline Manager Orchestration --- RUN ID: {RUN_ID_GLOBAL} ---")
    logger.info(f"Run directory: {RUN_DIR_GLOBAL}")
    logger.info(f"Run Logs directory: {LOGS_DIR_GLOBAL}")
    logger.info(f"Project Root interpreted as: {PROJECT_ROOT}")
    logger.info(f"Feature Engine Script Path: {FEATURE_ENGINE_SCRIPT_PATH}")
    logger.info(f"Simulation Script Path: {SIMULATION_SCRIPT_PATH}")

    timeframes_to_process = list(config.RAW_DATA_FILES.keys())

    if strategies_to_run is None:
        strategies_to_process = list(strategy_factories.keys())
    else:
        strategies_to_process = [s for s in strategies_to_run if s in strategy_factories]
        if len(strategies_to_process) != len(strategies_to_run):
            missing = set(strategies_to_run) - set(strategies_to_process)
            logger.warning(f"Requested strategies not found in factory and will be skipped: {missing}")
    if not strategies_to_process:
        logger.error("No valid strategies selected or available to process. Aborting.")
        return False


    target_symbol = getattr(config, "DEFAULT_SYMBOL", "nifty")
    target_market = getattr(config, "DEFAULT_MARKET", "NSE")
    target_segment = getattr(config, "DEFAULT_SEGMENT", "Index")

    logger.info(f"Target Instrument: {target_symbol} ({target_market}/{target_segment})")
    logger.info(f"Processing timeframes: {timeframes_to_process}")
    logger.info(f"Processing strategies: {strategies_to_process}")
    overall_success = True

    processed_data_dir = Path(getattr(config, "DATA_DIR_PROCESSED", PROJECT_ROOT / "data" / "datawithindicator"))
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # === Phase 1: Feature Generation ===
    for timeframe in timeframes_to_process:
        logger.info(f"\n===== Phase: Feature Generation for {target_symbol} ({timeframe}) =====")
        raw_file_name = config.RAW_DATA_FILES.get(timeframe)
        if not raw_file_name:
            logger.error(f"Missing raw data file configuration for timeframe '{timeframe}'. Skipping.")
            overall_success = False
            continue

        data_folder_path = Path(getattr(config, "DATA_FOLDER", PROJECT_ROOT / "data" / "raw"))
        raw_path = data_folder_path / raw_file_name
        indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
        indicator_output_path = processed_data_dir / indicator_file_name

        if not raw_path.exists():
            logger.error(f"Raw data file not found: {raw_path}. Skipping feature generation for {timeframe}.")
            overall_success = False
            continue

        feature_args = [
            "--input", str(raw_path), "--output", str(indicator_output_path),
            "--symbol", target_symbol, "--exchange", target_market
        ]
        if not run_script(FEATURE_ENGINE_SCRIPT_PATH, feature_args, timeout=300, log_suffix=f"_features_{target_symbol}_{timeframe}"):
            logger.error(f"Feature generation failed for {target_symbol} - {timeframe}.")
            logger.critical(f"❌ Feature generation failed for {target_symbol} - {timeframe}. Aborting pipeline.")
            sys.exit(1) # Critical failure

    # === Phase 2: Independent Strategy Backtests ===
    run_independent_backtests = getattr(config, "RUN_INDEPENDENT_BACKTESTS", True)
    if run_independent_backtests:
        logger.info(f"\n===== Phase: Independent Strategy Simulation for {target_symbol} =====")
        independent_sim_trace_logs_dir = LOGS_DIR_GLOBAL / "independent_sim_traces"
        independent_sim_trace_logs_dir.mkdir(parents=True, exist_ok=True)

        for timeframe in timeframes_to_process:
            indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
            indicator_input_path = processed_data_dir / indicator_file_name

            if not indicator_input_path.is_file():
                logger.critical(f"❌ Expected feature file not found: {indicator_input_path} for independent backtests. Aborting.")
                sys.exit(1) # Critical failure

            for strategy_name in strategies_to_process:
                logger.info(f"--- Running Independent Backtest for Strategy: {strategy_name} ({target_symbol} - {timeframe}) ---")
                result_path = STRATEGY_RESULTS_DIR_GLOBAL / f"{strategy_name}_{target_symbol}_{timeframe}_{RUN_ID_GLOBAL}.json"
                
                sim_args = [
                    "--input", str(indicator_input_path),
                    "--output-json", str(result_path),
                    "--log-dir", str(independent_sim_trace_logs_dir), # For SimpleBacktester's own trace log
                    "--strategy-name", strategy_name,
                    "--symbol", target_symbol,
                    "--market", target_market,
                    "--run-id", RUN_ID_GLOBAL # Pass the main RUN_ID
                ]
                if not run_script(SIMULATION_SCRIPT_PATH, sim_args, timeout=600, log_suffix=f"_indep_sim_{strategy_name}_{target_symbol}_{timeframe}"):
                    logger.error(f"Independent simulation failed for {strategy_name} on {target_symbol} - {timeframe}.")
                    logger.critical(f"❌ Independent simulation failed for {strategy_name}. Aborting pipeline.")
                    sys.exit(1) # Critical failure
    else:
        logger.info("Skipping Phase 2: Independent Strategy Backtests as per configuration.")

    # === Phase 3: Contextual Strategy Tuning (Optuna) ===
    run_optuna_phase = getattr(config, "RUN_OPTUNA_TUNING", True)
    if run_optuna_phase:
        logger.info(f"\n===== Phase: Contextual Strategy Tuning (Optuna) for {target_symbol} =====")
        try:
            run_contextual_tuning(
                symbol=target_symbol,
                market=target_market,
                segment=target_segment,
                timeframes=timeframes_to_process,
                strategies=strategies_to_process,
                n_trials_per_study=config.OPTUNA_TRIALS_PER_CONTEXT,
                max_workers=config.MAX_OPTUNA_WORKERS,
                run_id=RUN_ID_GLOBAL, # Pass the main RUN_ID
                run_specific_logs_dir=LOGS_DIR_GLOBAL, # Pass the main run's log directory
                context_filter_override=None
            )
            logger.info("✅ Contextual Tuning Phase completed.")
        except Exception as e:
            logger.error(f"❌ Contextual Strategy Tuning Phase failed: {e}", exc_info=True)
            logger.critical(f"❌ Optuna tuning failed. Aborting pipeline.")
            sys.exit(1) # Critical failure
    else:
        logger.info("Skipping Phase 3: Contextual Strategy Tuning (Optuna) as per configuration.")

    # === Phase 4: Optional Agent Simulation ===
    run_agent_simulation = getattr(config, "RUN_AGENT_SIMULATION", False)
    if run_agent_simulation:
        logger.info(f"\n===== Phase: Agent Simulation for {target_symbol} =====")
        agent_sim_trace_logs_dir = LOGS_DIR_GLOBAL / "agent_sim_traces"
        agent_sim_trace_logs_dir.mkdir(parents=True, exist_ok=True)

        for timeframe in timeframes_to_process:
            indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
            indicator_input_path = processed_data_dir / indicator_file_name
            
            if not indicator_input_path.is_file():
                logger.warning(f"Skipping agent simulation for {timeframe} due to missing feature file: {indicator_input_path}")
                continue

            logger.info(f"--- Running Simulation for Agent ({target_symbol} - {timeframe}) ---")
            results_filename = f"Agent_{target_symbol}_{timeframe}_{RUN_ID_GLOBAL}.json"
            results_json_path = AGENT_RESULTS_DIR_GLOBAL / results_filename
            
            agent_sim_args = [
                "--input", str(indicator_input_path),
                "--output-json", str(results_json_path),
                "--log-dir", str(agent_sim_trace_logs_dir),
                "--symbol", target_symbol,
                "--market", target_market,
                "--run-id", RUN_ID_GLOBAL # Pass the main RUN_ID
                # DO NOT pass --strategy-name for agent mode
            ]
            if not run_script(SIMULATION_SCRIPT_PATH, agent_sim_args, timeout=600, log_suffix=f"_Agent_{target_symbol}_{timeframe}"):
                 logger.error(f"Agent simulation failed for {target_symbol} - {timeframe}.")
                 overall_success = False # Mark as overall failure but might not be critical to abort all
                 logger.warning(f"⚠️ Agent simulation failed for {target_symbol} - {timeframe}. Continuing if other tasks exist.")
    else:
        logger.info("Skipping Phase 4: Agent Simulation as per configuration.")

    logger.info("\n--- Pipeline Manager Finished ---")
    if overall_success:
        logger.info(f"✅ Pipeline completed. Check results in {RESULTS_DIR_GLOBAL} and logs in {LOGS_DIR_GLOBAL}. RUN ID: {RUN_ID_GLOBAL}")
    else:
        logger.warning(f"⚠️ Pipeline completed with errors. Check script logs in {LOGS_DIR_GLOBAL} for details. RUN ID: {RUN_ID_GLOBAL}")
    
    return overall_success


if __name__ == "__main__":
    # --- Setup Run Environment (Call this ONCE at the beginning) ---
    setup_run_environment()
    
    # This ensures that logger is now configured and RUN_ID_GLOBAL, etc., are set.
    # The logger instance obtained by `logger = logging.getLogger(__name__)` at the top
    # will now use the handlers defined in `setup_run_environment`.

    # --- atexit registration for MongoDB connection ---
    # It's good practice to register this early.
    try:
        from app.mongo_manager import close_mongo_connection_on_exit
        import atexit
        atexit.register(close_mongo_connection_on_exit)
        logger.info("Registered MongoDB connection closer on atexit.")
    except ImportError:
        logger.warning("Could not import or register MongoDB connection closer.")

    # --- Test Overrides ---
    # These are applied after the main config is loaded and initial logging is set up.
    config.RAW_DATA_FILES = { "5min": "nifty_historical_data_5min.csv" }
    config.OPTUNA_TRIALS_PER_CONTEXT = getattr(config, "OPTUNA_TRIALS_PER_CONTEXT_TEST", 5)
    config.MAX_OPTUNA_WORKERS = getattr(config, "MAX_OPTUNA_WORKERS_TEST", 1)
    # Example: Disable independent backtests for a quick Optuna-only run
    # config.RUN_INDEPENDENT_BACKTESTS = False 
    # config.RUN_OPTUNA_TUNING = True

    # Determine which strategies to run for this specific execution
    _original_strategy_factories = strategy_factories.copy() # Keep a backup
    strategy_to_test_override_name = "SuperTrend_ADX" # Or None to test all configured in strategy_factories
    
    final_strategies_to_run_this_time: Optional[List[str]] = None
    if strategy_to_test_override_name:
        if strategy_to_test_override_name in strategy_factories:
            final_strategies_to_run_this_time = [strategy_to_test_override_name]
            logger.info(f"--- OVERRIDE: Testing only strategy: {strategy_to_test_override_name} ---")
        else:
            logger.warning(f"--- OVERRIDE WARNING: Strategy '{strategy_to_test_override_name}' not found. Will test all available strategies. ---")
            final_strategies_to_run_this_time = list(strategy_factories.keys())
    else:
        final_strategies_to_run_this_time = list(strategy_factories.keys())
        logger.info(f"--- No strategy override. Testing all available strategies: {final_strategies_to_run_this_time} ---")

    try:
        success = main_orchestration(strategies_to_run=final_strategies_to_run_this_time)
        exit_code = 0 if success else 1
    except Exception as e:
        logger.critical(f"Pipeline Manager CRASHED: {e}", exc_info=True)
        exit_code = 1
    # finally:
    #     # Restore original strategy_factories if it was patched, for potential future imports in an interactive session.
    #     # This is less critical if the script always exits, but good practice.
    #     if strategy_factories is not _original_strategy_factories : # Check if it was actually changed
    #          strategy_factories.clear()
    #          strategy_factories.update(_original_strategy_factories)
    #          logger.debug("Restored original strategy_factories.")
    #     logger.info(f"Pipeline manager exiting with code {exit_code}.")

    sys.exit(exit_code)