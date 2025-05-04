# pipeline_manager.py (in project root)

# ... (imports, RUN_ID, directory setup, logging setup remain same) ...
import subprocess
import sys
import logging
from pathlib import Path
import os
from datetime import datetime
import time
from app.config import config

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_ROOT = Path(__file__).parent
RUNS_BASE_DIR = PROJECT_ROOT / "runs"; RUN_DIR = RUNS_BASE_DIR / RUN_ID
LOGS_DIR = RUN_DIR / "logs"; RUN_DATA_DIR = RUN_DIR / "data"; RESULTS_DIR = RUN_DIR / "results"
RUN_DIR.mkdir(parents=True, exist_ok=True); LOGS_DIR.mkdir(exist_ok=True)
RUN_DATA_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)

log_file_path = LOGS_DIR / 'pipeline_manager.log'; root_logger = logging.getLogger();
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
logger = logging.getLogger(__name__)

APP_DIR = PROJECT_ROOT / "app"
FEATURE_ENGINE_SCRIPT_PATH = APP_DIR / "run_feature_engine.py"
SIMULATION_SCRIPT_PATH = APP_DIR / "run_simulation_step.py"


# --- run_script function (Keep enhanced version) ---
def run_script(script_path: Path, args_list: list = [], timeout: Optional[int] = None, log_suffix: str = "") -> bool:
    # (Keep implementation from previous step that sets PYTHONPATH and redirects output)
    if not script_path.is_file(): logger.error(f"Script not found: {script_path}"); return False
    log_filename = f"{script_path.stem}{log_suffix}.log"; script_log_file = LOGS_DIR / log_filename
    command = [sys.executable, str(script_path)] + args_list; command_str_args = [str(item) for item in command]
    logger.info(f"Running command: {' '.join(command_str_args)}"); logger.info(f"Redirecting output to: {script_log_file}")
    process_env = os.environ.copy(); process_env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + process_env.get('PYTHONPATH', '')
    logger.debug(f"Setting PYTHONPATH: {process_env['PYTHONPATH']}")
    try:
        with open(script_log_file, 'w', encoding='utf-8') as f_log:
            subprocess.run(command_str_args, check=True, timeout=timeout, stdout=f_log, stderr=subprocess.STDOUT, text=True, cwd=PROJECT_ROOT, env=process_env)
        logger.info(f"--- Script {script_path.name} OK (Suffix: '{log_suffix}') ---")
        # try: # Optional logging back
        #      with open(script_log_file, 'r', encoding='utf-8') as f_read: logger.info(f"\n=== Output from {script_log_file.name} ===\n{f_read.read()}\n===")
        # except Exception as read_e: logger.warning(f"Could not read {script_log_file}: {read_e}")
        return True
    except subprocess.CalledProcessError as e: logger.error(f"Script {script_path.name} FAIL (Suffix: '{log_suffix}', Code: {e.returncode}). Log: {script_log_file}"); return False
    except subprocess.TimeoutExpired: logger.error(f"Script {script_path.name} TIMEOUT (Suffix: '{log_suffix}'). Log: {script_log_file}"); return False
    except Exception as e: logger.error(f"Script {script_path.name} ERROR (Suffix: '{log_suffix}'): {e}", exc_info=True); return False


# --- main_orchestration function (UPDATED Simulation Step Call) ---
def main_orchestration():
    logger.info(f"--- Starting Pipeline Manager --- RUN ID: {RUN_ID} ---")
    logger.info(f"Run directory: {RUN_DIR}")
    timeframes_to_process = list(config.RAW_DATA_FILES.keys())
    logger.info(f"Processing timeframes: {timeframes_to_process}")
    overall_success = True

    for timeframe in timeframes_to_process:
        logger.info(f"\n===== Processing Timeframe: {timeframe} =====")
        log_suffix = f"_{timeframe}"
        raw_data_filename = config.RAW_DATA_FILES.get(timeframe)
        if not raw_data_filename: logger.error(f"No raw data file for '{timeframe}'. Skipping."); overall_success=False; continue
        raw_data_path = config.DATA_FOLDER / raw_data_filename
        indicator_filename = Path(raw_data_filename).stem + "_with_indicators.csv"
        indicator_data_path = RUN_DATA_DIR / indicator_filename
        results_filename = f"backtest_summary_{timeframe}.json"
        results_json_path = RESULTS_DIR / results_filename

        # --- Step 1: Calculate Features (remains same) ---
        logger.info(f"*** Phase: Feature Generation ({timeframe}) ***")
        if not raw_data_path.is_file(): logger.error(f"Raw data file not found: {raw_data_path}. Skipping."); overall_success=False; continue
        feature_args = ["--input", str(raw_data_path), "--output", str(indicator_data_path)]
        if not run_script(FEATURE_ENGINE_SCRIPT_PATH, feature_args, timeout=300, log_suffix=log_suffix):
            logger.error(f"Feature generation failed for {timeframe}. Stopping this timeframe.")
            overall_success=False; continue

        # --- Step 2: Run Simulation (UPDATED ARGS) ---
        logger.info(f"*** Phase: Simulation ({timeframe}) ***")
        if not indicator_data_path.is_file(): logger.error(f"Indicator file not found: {indicator_data_path}. Skipping."); overall_success=False; continue

        simulation_args = [
            "--input", str(indicator_data_path),
            "--output-json", str(results_json_path),
            # --- ADDED: Pass log directory ---
            "--log-dir", str(LOGS_DIR) # Pass the main log dir for this run
        ]
        if not run_script(SIMULATION_SCRIPT_PATH, simulation_args, timeout=600, log_suffix=log_suffix):
            logger.error(f"Simulation failed for {timeframe}.")
            overall_success=False; continue
        # --- End Add ---

    # ... (Final Summary remains same) ...
    logger.info("\n--- Pipeline Manager Finished ---")
    if overall_success: logger.info(f"All processed timeframes completed successfully. RUN ID: {RUN_ID}")
    else: logger.warning(f"One or more timeframes failed. Check logs in {LOGS_DIR}. RUN ID: {RUN_ID}")


if __name__ == "__main__":
    main_orchestration()