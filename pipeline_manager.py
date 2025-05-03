# orchestrator.py (Updated for Multi-Timeframe)

import subprocess
import sys
import logging
from pathlib import Path
import os
from datetime import datetime
from typing import Optional

# Use absolute imports for modules within the app package
from app.config import config

# --- Setup Run ID and Directories ---
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_ROOT = Path(__file__).parent
RUNS_BASE_DIR = PROJECT_ROOT / "runs"
RUN_DIR = RUNS_BASE_DIR / RUN_ID
LOGS_DIR = RUN_DIR / "logs"
RUN_DATA_DIR = RUN_DIR / "data"    # Stores indicator files for this run
RESULTS_DIR = RUN_DIR / "results"  # Stores backtest results for this run

# Create directories for this run
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
RUN_DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True) # Create results dir too

# --- Configure Logging for Orchestrator ---
log_file_path = LOGS_DIR / 'pipeline_manager.log'
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Define paths to the scripts
APP_DIR = PROJECT_ROOT / "app"
INDICATOR_SCRIPT_PATH = APP_DIR / "feature_engine.py"
BACKTEST_SCRIPT_PATH = APP_DIR / "run_simulation_step.py"


def run_script(script_path: Path, args_list: list = [], timeout: Optional[int] = None, log_suffix: str = "") -> bool:
    """
    Runs a python script as a subprocess, setting PYTHONPATH,
    and redirecting its output to a dedicated log file within the run's logs directory,
    adding a suffix for uniqueness (like timeframe). Also logs the output file content back.

    Args:
        script_path: Path object for the script to run.
        args_list: List of string arguments for the script.
        timeout: Optional timeout in seconds for the subprocess.
        log_suffix: String to append to the log filename (e.g., "_1min").

    Returns:
        True if the script runs successfully (exit code 0), False otherwise.
    """
    if not script_path.is_file():
        logger.error(f"Script not found: {script_path}")
        return False

    log_filename = f"{script_path.stem}{log_suffix}.log"
    script_log_file = LOGS_DIR / log_filename

    command = [sys.executable, str(script_path)] + args_list
    command_str_args = [str(item) for item in command]
    logger.info(f"Running command: {' '.join(command_str_args)}")
    logger.info(f"Redirecting output to: {script_log_file}")

    process_env = os.environ.copy()
    process_env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + process_env.get('PYTHONPATH', '')
    logger.debug(f"Setting PYTHONPATH for subprocess: {process_env['PYTHONPATH']}")

    try:
        with open(script_log_file, 'w', encoding='utf-8') as f_log:
            subprocess.run(
                command_str_args, check=True, timeout=timeout,
                stdout=f_log, stderr=subprocess.STDOUT, text=True,
                cwd=PROJECT_ROOT, env=process_env
            )
        logger.info(f"--- Script {script_path.name} finished successfully for log suffix '{log_suffix}' ---")

        # --- ADDED BACK: Log content of the script's log file ---
        try:
            with open(script_log_file, 'r', encoding='utf-8') as f_read:
                log_content = f_read.read()
                # Log only if content exists to avoid empty blocks
                if log_content.strip():
                     logger.info(f"\n=== Output from {script_path.name} ({script_log_file.name}) ===\n{log_content}\n===")
                else:
                     logger.info(f"Log file {script_log_file.name} was empty.")
        except Exception as read_e:
            logger.warning(f"Could not read output log {script_log_file}: {read_e}")
        # --- END ADDED BACK ---
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Script {script_path.name} failed with exit code {e.returncode} for log suffix '{log_suffix}'.")
        logger.error(f"Check log file for details: {script_log_file}")
        # Attempt to log error output even on failure
        try:
            with open(script_log_file, 'r', encoding='utf-8') as f_read:
                 logger.error(f"\n=== Failed Output from {script_path.name} ({script_log_file.name}) ===\n{f_read.read()}\n===")
        except Exception as read_e:
            logger.warning(f"Could not read failed output log {script_log_file}: {read_e}")
        return False
    # ... (TimeoutExpired and other Exception handling remain the same) ...
    except subprocess.TimeoutExpired:
        logger.error(f"Script {script_path.name} timed out after {timeout} seconds for log suffix '{log_suffix}'.")
        logger.error(f"Check log file for details: {script_log_file}")
        return False
    except Exception as e:
        logger.error(f"Failed to run script {script_path.name} for log suffix '{log_suffix}': {e}", exc_info=True)
        return False



# --- main_orchestration function (UPDATED FOR LOOPING) ---
def main_orchestration():
    """Runs the indicator calculation and backtesting steps for multiple timeframes."""
    logger.info(f"--- Starting Orchestration --- RUN ID: {RUN_ID} ---")
    logger.info(f"Run directory: {RUN_DIR}")

    # Define the timeframes to process based on config
    # Use the keys from the RAW_DATA_FILES dictionary in config
    timeframes_to_process = list(config.RAW_DATA_FILES.keys())
    logger.info(f"Processing timeframes: {timeframes_to_process}")

    overall_success = True

    for timeframe in timeframes_to_process:
        logger.info(f"\n===== Processing Timeframe: {timeframe} =====")

        # --- Construct filenames for this timeframe ---
        raw_data_filename = config.RAW_DATA_FILES.get(timeframe)
        if not raw_data_filename:
            logger.error(f"No raw data filename configured for timeframe '{timeframe}'. Skipping.")
            overall_success = False
            continue

        # Use Path objects for robustness
        raw_data_path = config.DATA_FOLDER / raw_data_filename
        indicator_filename = Path(raw_data_filename).stem + "_with_indicators.csv" # e.g., nifty_historical_data_5min_with_indicators.csv
        indicator_data_path = RUN_DATA_DIR / indicator_filename # Save indicator file inside the run's data dir

        # --- Step 1: Calculate Indicators for this timeframe ---
        
        logger.info(f"*** Phase: Indicator Calculation ({timeframe}) ***")
        if not raw_data_path.is_file():
            logger.error(f"Raw data file not found for {timeframe}: {raw_data_path}. Skipping.")
            overall_success = False
            continue

        indicator_args = [
            "--input", str(raw_data_path),
            "--output", str(indicator_data_path)
        ]
        log_suffix = f"_{timeframe}"  # Define log_suffix based on the timeframe
        if not run_script(INDICATOR_SCRIPT_PATH, indicator_args, timeout=300, log_suffix=log_suffix):
            logger.error(f"Indicator calculation failed for {timeframe}. Stopping processing for this timeframe.")
            overall_success = False; continue

        # --- Step 2: Run Backtest for this timeframe ---
        logger.info(f"*** Phase: Run simulator ({timeframe}) ***")
        if not indicator_data_path.is_file():
            logger.error(f"Indicator file was not created for {timeframe} at: {indicator_data_path}. Cannot run backtest.")
            overall_success = False
            continue # Skip to next timeframe
        backtest_output_json_path = RESULTS_DIR / f"backtest_summary_{timeframe}.json"
        logger.info(f"Backtest results will be saved to: {backtest_output_json_path}")
        # --- ADD LOGGING FOR BACKTESTING ---
        logger.debug(f"Running backtest with arguments: {backtest_output_json_path}")
        logger.debug(f"Indicator data path: {indicator_data_path}")
        # --- END ADD LOGGING FOR BACKTESTING ---
        # Run the backtest script with the indicator data
        backtest_args = [
            "--input", str(indicator_data_path),
            # --- ADD THIS ARGUMENT ---
            "--output-json", str(backtest_output_json_path)
        ]
        if not run_script(BACKTEST_SCRIPT_PATH, backtest_args, timeout=600,log_suffix=log_suffix):
            logger.error(f"Backtesting failed for {timeframe}.")
            overall_success = False
            # Continue to next timeframe even if one backtest fails? Or stop? Let's continue for now.
            continue

    # --- Final Summary ---
    logger.info("\n--- Orchestration Finished ---")
    if overall_success:
         logger.info(f"All processed timeframes completed successfully. RUN ID: {RUN_ID}")
    else:
         logger.warning(f"One or more timeframes failed during processing. Please check logs. RUN ID: {RUN_ID}")

    # TODO (Later Phase): Add logic here to read results saved by run_backtest.py
    # (e.g., from RESULTS_DIR) and generate a consolidated report/summary.


if __name__ == "__main__":
    main_orchestration()