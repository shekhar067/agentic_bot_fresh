# import logging
# import json
# import os
# import sys

# import pandas as pd
# from app.reporting import calculate_detailed_metrics
# from flask import Flask, request, jsonify, render_template
# from typing import Dict, List
# from pathlib import Path
# from datetime import datetime
# import threading
# from collections import defaultdict

# # --- Imports ---
# APP_DIR = Path(__file__).parent / "app"
# sys.path.insert(0, str(APP_DIR.parent))
# try:
#     from app.config import config
#     from app.data_io import load_historical_data
#     from app.feature_engine import IndicatorCalculator
#    # from app.run_simulation_step import run_and_save_agent_backtest
#     # OLD import in web_app.py
#     from app.run_simulation_step import run_simulation_task
    
#     from app.agentic_core import RuleBasedAgent
# except ImportError as e:
#     print(f"ERROR: Failed to import necessary modules: {e}", file=sys.stderr)
#     sys.exit(1)

# # --- Flask App Setup ---
# app = Flask(__name__,
#             template_folder='templates',
#             static_folder='static')


# backtest_status = {
#     "running": False,
#     "run_id": None,
#     "message": "Idle",
#     "error": None,
#     "pid": None, # Process ID of the pipeline_manager script
#     "log_file": None # Log file for the pipeline manager run
# }
# status_lock = threading.Lock()
# log_level_str = getattr(config, "LOG_LEVEL", "INFO").upper()
# log_level = getattr(logging, log_level_str, logging.INFO)
# logging.basicConfig(level=log_level)
# app.logger.setLevel(log_level)
# app.logger.propagate = True

# # --- Global Status & Lock ---
# backtest_status = {"running": False, "run_id": None, "message": "Idle", "error": None}
# status_lock = threading.Lock()

# # --- Background Pipeline Thread ---
# def run_pipeline_thread(run_id, timeframes, raw_data_map, indicator_map, results_map, log_dir, results_dir):
#     global backtest_status
#     pipeline_successful = True
#     all_tf_results_for_report = {}
#     try:
#         app.logger.info(f"Background thread started for RUN_ID: {run_id}")
#         calculator = IndicatorCalculator()
#         for tf in timeframes:
#             with status_lock:
#                 if not backtest_status["running"] or backtest_status["run_id"] != run_id:
#                     app.logger.info(f"Run {run_id} aborted during {tf}.")
#                     pipeline_successful = False
#                     break

#             app.logger.info(f"Calculating features for {tf}...")
#             raw_path = raw_data_map[tf]
#             indicator_path = indicator_map[tf]
#             if not raw_path.is_file():
#                 raise FileNotFoundError(f"Raw data not found: {raw_path}")

#             raw_df = load_historical_data(raw_path.name)
#             df_with_indicators = calculator.calculate_all_indicators(raw_df)
#             indicator_path.parent.mkdir(parents=True, exist_ok=True)
#             df_with_indicators.to_csv(indicator_path, index=True)
#             app.logger.info(f"Features saved for {tf} to {indicator_path}")

#             with status_lock:
#                 if not backtest_status["running"] or backtest_status["run_id"] != run_id:
#                     app.logger.info(f"Run {run_id} aborted after features for {tf}.")
#                     pipeline_successful = False
#                     break

#             app.logger.info(f"Running simulation for {tf}...")
#             result_json_path = results_map[tf]
#             tf_result_data = run_simulation_task(indicator_path, result_json_path, log_dir)
#             all_tf_results_for_report[tf] = tf_result_data.get("RuleBasedAgent") if tf_result_data else {"error": "Simulation failed"}
#             app.logger.info(f"Simulation done for {tf}. Results saved.")

#         if pipeline_successful and backtest_status.get("running") and backtest_status.get("run_id") == run_id:
#             app.logger.info(f"Pipeline loop finished for {run_id}")
#             final_message = f"Run {run_id} completed."
#             try:
#                 loaded_results = all_tf_results_for_report
#                 app.logger.info(f"HTML report generated (not saved to disk).")
#             except Exception as report_e:
#                 app.logger.error(f"Failed to generate report: {report_e}", exc_info=True)
#                 final_message = f"Run {run_id} completed, report generation failed."
#             with status_lock:
#                 backtest_status.update({"message": final_message, "running": False, "error": None})
#         elif not pipeline_successful:
#             app.logger.info(f"Pipeline for {run_id} did not complete fully.")
#     except Exception as e:
#         app.logger.error(f"Error in background run {run_id}: {e}", exc_info=True)
#         with status_lock:
#             backtest_status.update({"message": f"Run {run_id} failed!", "running": False, "error": str(e)})
#     finally:
#         with status_lock:
#             if backtest_status["run_id"] == run_id and backtest_status["running"]:
#                 app.logger.warning(f"BG thread {run_id} ending but status running. Resetting.")
#                 backtest_status["running"] = False
#                 if not backtest_status["error"]:
#                     backtest_status["message"] = f"Run {run_id} finished (unknown state)."

# # --- Flask Routes ---
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/start_backtest', methods=['POST'])
# def start_backtest():
#     global backtest_status
#     with status_lock:
#         if backtest_status["running"]:
#             return jsonify({"error": "Backtest already running."}), 409

#         params = request.json
#         app.logger.info(f"Received backtest request: {params}")
#         run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
#         run_dir = Path(__file__).parent / "runs" / run_id
#         run_data_dir = run_dir / "data"
#         results_dir = run_dir / "results"
#         log_dir = run_dir / "logs"

#         try:
#             run_dir.mkdir(parents=True, exist_ok=True)
#             run_data_dir.mkdir(exist_ok=True)
#             results_dir.mkdir(exist_ok=True)
#             log_dir.mkdir(exist_ok=True)
#         except OSError as e:
#             app.logger.error(f"Could not create run dirs: {e}")
#             return jsonify({"error": f"Could not create run dirs: {e}"}), 500

#         timeframes = list(config.RAW_DATA_FILES.keys())
#         if not timeframes:
#             return jsonify({"error": "No timeframes configured."}), 400

#         raw_data_map = {}
#         for tf in timeframes:
#             tf_filename = config.RAW_DATA_FILES.get(tf)
#             if not tf_filename:
#                 return jsonify({"error": f"Filename missing for {tf}."}), 400

#             tf_path = config.DATA_FOLDER / tf_filename
#             if not tf_path.is_file():
#                 return jsonify({"error": f"Raw data file not found: {tf_path}"}), 400

#             raw_data_map[tf] = tf_path

#         indicator_map = {tf: run_data_dir / (Path(config.RAW_DATA_FILES[tf]).stem + "_with_indicators.csv") for tf in timeframes}
#         results_map = {tf: results_dir / f"backtest_summary_{tf}.json" for tf in timeframes}

#         pipeline_thread = threading.Thread(target=run_pipeline_thread, args=(run_id, timeframes, raw_data_map, indicator_map, results_map, log_dir, results_dir))
#         pipeline_thread.daemon = True
#         pipeline_thread.start()

#         backtest_status.update({"running": True, "run_id": run_id, "message": f"Run {run_id} started...", "error": None})
#         return jsonify({"message": "Backtest run started.", "run_id": run_id, "auto_load": True})

# @app.route('/status')
# def get_status():
#     with status_lock:
#         return jsonify(backtest_status)

# @app.route('/stop_backtest', methods=['POST'])
# def stop_backtest():
#     global backtest_status
#     with status_lock:
#         current_run_id = backtest_status.get('run_id')
#         if not backtest_status["running"]:
#             return jsonify({"error": "No backtest running."}), 409

#         backtest_status["running"] = False
#         backtest_status["message"] = f"Stop requested by user for run {current_run_id}."
#         app.logger.info(f"Stop requested for run {current_run_id}")
#         return jsonify({"message": "Stop requested."})

# @app.route('/runs')
# def list_runs():
#     runs_dir = Path(__file__).parent / "runs"
#     app.logger.info(f"Scanning for runs in: {runs_dir}")
#     if not runs_dir.exists():
#         app.logger.warning(f"Runs directory does not exist: {runs_dir}")
#         return jsonify([])

#     try:
#         run_ids = sorted([d.name for d in runs_dir.iterdir() if d.is_dir() and len(d.name) == 15], reverse=True)
#         app.logger.info(f"Filtered run IDs: {run_ids}")
#     except Exception as e:
#         app.logger.error(f"Error scanning runs directory: {e}", exc_info=True)
#         return jsonify([])

#     return jsonify(run_ids)

# @app.route('/results_html/<run_id>')
# def view_html_report(run_id):
#     return render_template("report_template.html", run_id=run_id)

# @app.route('/results_html/<run_id>/detailed')
# def view_detailed_html_report(run_id):
#     return render_template("report_template_detailed.html", run_id=run_id)

# # @app.route('/results/<run_id>')
# # def get_results_json(run_id):
# #     results_dir = Path(__file__).parent / "runs" / run_id / "results"
# #     json_files = list(results_dir.glob("backtest_summary_*.json"))
# #     if not json_files:
# #         return jsonify({"error": f"No result files found for run ID: {run_id}"}), 404

# #     result_data = {}
# #     for f in json_files:
# #         tf = f.stem.split("_")[-1]
# #         try:
# #             with open(f) as fp:
# #                 parsed = json.load(fp)
# #                 result_data[tf] = parsed.get("RuleBasedAgent", {})
# #         except Exception as e:
# #             result_data[tf] = {"error": str(e)}

# #     return jsonify({"html_report": json.dumps(result_data)})
# @app.route('/results/<run_id>')
# def get_results_json(run_id):
#     results_dir = Path(__file__).parent / "runs" / run_id / "results"
#     json_files = list(results_dir.glob("backtest_summary_*.json"))
#     if not json_files:
#         return jsonify({"error": f"No result files found for run ID: {run_id}"}), 404

#     result_data = {}
#     summary_table = defaultdict(dict)

#     for f in json_files:
#         tf = f.stem.split("_")[-1]
#         try:
#             with open(f) as fp:
#                 parsed = json.load(fp)
#                 agent_result = parsed.get("RuleBasedAgent", {})
#                 result_data[tf] = agent_result

#                 trades = agent_result.get("trades_details", [])
#                 if trades:
#                     df = pd.DataFrame(trades)
#                     for strategy_name, group in df.groupby("StrategyName"):
#                         metrics = calculate_detailed_metrics(group.to_dict(orient="records"), tf)
#                         summary_table[tf][strategy_name] = metrics
#         except Exception as e:
#             result_data[tf] = {"error": str(e)}

#     return jsonify({
#         "html_report": json.dumps(result_data),
#         "summary_table": summary_table  # 🆕 Add this
#     })

# if __name__ == '__main__':
#     app.logger.info(f"Starting Flask server ({Path(__file__).name})...")
#     app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
# web_app.py (MODIFIED TO TRIGGER pipeline_manager.py)
#------------------------Working fine-----------------------------------
import logging
import json
import os
import sys
import time
from venv import logger
import pandas as pd # Still needed for /results endpoint potentially
from flask import Flask, request, jsonify, render_template
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import threading
import subprocess # **** ADD THIS ****
from collections import defaultdict

# --- Imports ---
# We no longer need to import most app components here if pipeline_manager handles it
APP_DIR = Path(__file__).parent / "app"
sys.path.insert(0, str(APP_DIR.parent))
try:
    from app.config import config # Keep config if needed for other things
    from app.reporting import calculate_detailed_metrics # Keep for /results endpoint
    # --- REMOVE these imports as pipeline_manager handles them ---
    # from app.data_io import load_historical_data
    # from app.feature_engine import IndicatorCalculator
    # from app.run_simulation_step import run_simulation_task # No longer called directly
    # from app.agentic_core import RuleBasedAgent
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}", file=sys.stderr)
    # Decide if Flask should exit if reporting/config fail
    # sys.exit(1)

# --- Flask App Setup ---
# ... (Flask app setup, logging setup remains same) ...
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

log_level_str = getattr(config, "LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level)
app.logger.setLevel(log_level)
app.logger.propagate = True


# --- Global Status & Lock ---
# Added 'pid' and potentially 'log_file'
backtest_status = {
    "running": False,
    "run_id": None,
    "message": "Idle",
    "error": None,
    "pid": None, # Process ID of the pipeline_manager script
    "log_file": None # Log file for the pipeline manager run
}
status_lock = threading.Lock()

# --- Background Thread to run and monitor pipeline_manager.py ---
def run_pipeline_manager_thread(run_id):
    global backtest_status
    pipeline_successful = False
    process = None
    # --- Define paths relative to this script's location ---
    project_root = Path(__file__).parent
    pipeline_script_path = project_root / "pipeline_manager.py"
    run_log_dir = project_root / "runs" / run_id / "logs"
    pipeline_manager_log = run_log_dir / f"pipeline_manager_{run_id}.log"

    # Ensure log directory exists
    try:
        run_log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
         app.logger.error(f"Failed to create log directory {run_log_dir}: {e}")
         with status_lock:
              backtest_status.update({"message": f"Run {run_id} failed! Cannot create log dir.", "running": False, "error": str(e), "pid": None, "log_file": None})
         return # Exit thread


    try:
        app.logger.info(f"Background thread starting pipeline_manager.py for RUN_ID: {run_id}")
        command = [sys.executable, str(pipeline_script_path)] # sys.executable ensures using python from same venv
        app.logger.info(f"Executing command: {' '.join(command)}")
        app.logger.info(f"Pipeline manager log file: {pipeline_manager_log}")

        # Store log file path in status
        with status_lock:
            backtest_status["log_file"] = str(pipeline_manager_log)

        # Use Popen to run in background and capture output to file
        with open(pipeline_manager_log, 'w', encoding='utf-8') as log_fp:
            process = subprocess.Popen(
                command,
                stdout=log_fp,
                stderr=subprocess.STDOUT, # Redirect stderr to same file
                cwd=project_root, # Run from project root directory
                text=True,
                # Ensure PYTHONPATH is inherited or set if needed
                env=os.environ.copy() # Inherit environment
            )

        with status_lock:
             backtest_status["pid"] = process.pid # Store process ID

        # --- Monitor the process ---
        while True:
            # Check status flag first (allows stopping)
            with status_lock:
                 should_be_running = backtest_status["running"]
                 current_thread_run_id = backtest_status["run_id"]

            if not should_be_running or current_thread_run_id != run_id:
                 app.logger.info(f"Run {run_id} aborted by status flag. Terminating process {process.pid}.")
                 try:
                     process.terminate() # Ask nicely first
                     process.wait(timeout=5) # Wait briefly
                 except subprocess.TimeoutExpired:
                     app.logger.warning(f"Process {process.pid} did not terminate gracefully, killing.")
                     process.kill() # Force kill
                 except ProcessLookupError:
                      app.logger.warning(f"Process {process.pid} already finished when trying to terminate.")
                 except Exception as term_e:
                     app.logger.error(f"Error terminating process {process.pid}: {term_e}")

                 # Update status only if this thread was the one running
                 with status_lock:
                      if backtest_status["run_id"] == run_id: # Check run_id again inside lock
                           backtest_status.update({"message": f"Run {run_id} aborted by user.", "running": False, "pid": None})
                 return # Exit thread

            # Check if process finished
            return_code = process.poll()
            if return_code is not None:
                 # Process finished
                 pipeline_successful = (return_code == 0)
                 app.logger.info(f"pipeline_manager.py for {run_id} finished with code: {return_code}. Success: {pipeline_successful}")
                 break # Exit monitoring loop

            # Wait a bit before checking again
            # Use threading.Event for interruptible sleep if needed, simple sleep is fine too
            time.sleep(3.0) # Check every 3 seconds


        # --- Update final status ---
        final_message = f"Run {run_id} completed."
        error_message = None
        if not pipeline_successful:
             final_message = f"Run {run_id} completed with errors (Code: {return_code}). Check logs."
             error_message = f"Pipeline failed with code {return_code}."

        with status_lock:
             # Ensure we are updating the status for the correct run_id,
             # in case a new run was started while this one finished.
             if backtest_status["run_id"] == run_id:
                  backtest_status.update({"message": final_message, "running": False, "error": error_message, "pid": None})


    except Exception as e:
        app.logger.error(f"Error launching/monitoring pipeline_manager.py for run {run_id}: {e}", exc_info=True)
        if process and process.poll() is None: # If process is still running despite error
             process.kill()
        with status_lock:
            if backtest_status["run_id"] == run_id: # Update status only if it's still the active run
                 backtest_status.update({"message": f"Run {run_id} failed unexpectedly!", "running": False, "error": str(e), "pid": None})

# --- Modified /start_backtest route ---
@app.route('/start_backtest', methods=['POST'])
def start_backtest():
    global backtest_status
    with status_lock:
        if backtest_status["running"]:
            return jsonify({"error": "Backtest already running."}), 409

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Start the thread targeting the new run_pipeline_manager_thread function
        pipeline_thread = threading.Thread(target=run_pipeline_manager_thread, args=(run_id,))
        pipeline_thread.daemon = True # Allows app to exit even if thread is running
        pipeline_thread.start()

        # Update status immediately
        backtest_status.update({
            "running": True,
            "run_id": run_id,
            "message": f"Run {run_id} started (Executing pipeline_manager.py)...",
            "error": None,
            "pid": None, # PID will be updated by the thread shortly
            "log_file": None # Log file path will be updated by the thread
        })
        # Return run_id so UI can potentially track it
        return jsonify({"message": "Backtest run triggered via pipeline_manager.", "run_id": run_id})


# --- Status Endpoint (Consider adding log file path) ---
@app.route('/status')
def get_status():
    with status_lock:
        # Return a copy to avoid modifying the original dict elsewhere
        return jsonify(backtest_status.copy())

# --- Stop Endpoint (Needs to handle killing subprocess) ---
@app.route('/stop_backtest', methods=['POST'])
def stop_backtest():
    global backtest_status
    stopped_pid = None
    with status_lock:
        current_run_id = backtest_status.get('run_id')
        current_pid = backtest_status.get('pid')
        if not backtest_status["running"]:
            return jsonify({"error": "No backtest running."}), 409

        app.logger.info(f"Stop requested for run {current_run_id} (PID: {current_pid})")
        # Set running flag to False - the monitoring thread will detect this and terminate/kill
        backtest_status["running"] = False
        backtest_status["message"] = f"Stop requested by user for run {current_run_id}..."
        stopped_pid = current_pid # Store pid to log outside lock

    if stopped_pid:
         app.logger.info(f"Signaled monitoring thread to stop process {stopped_pid} for run {current_run_id}.")
    return jsonify({"message": "Stop signal sent to running process."})


# --- Results Endpoint ---
# This needs to read from STRATEGY_RESULTS_DIR now
@app.route('/results/<run_id>')
def get_results_json(run_id):
    # Adjust path to look inside strategy_results
    results_dir = Path(__file__).parent / "runs" / run_id / "results" / "strategy_results"
    if not results_dir.is_dir():
         # Maybe check agent_results too? Or just report strategy results missing?
         results_dir_agent = Path(__file__).parent / "runs" / run_id / "results" / "agent_results"
         if not results_dir_agent.is_dir():
              return jsonify({"error": f"No results directories found for run ID: {run_id}"}), 404
         else: # For now, return error if strategy results are missing
              return jsonify({"error": f"Strategy results directory not found for run ID: {run_id}"}), 404


    # Find all strategy JSON files for this run
    json_files = sorted(list(results_dir.glob("*.json"))) # Sort for consistent order
    if not json_files:
        return jsonify({"error": f"No strategy result JSON files found in {results_dir}"}), 404

    # --- Process results from multiple files ---
    result_data = {} # Keyed by Timeframe
    summary_table = defaultdict(lambda: defaultdict(dict)) # TF -> Strategy -> Metrics

    for f in json_files:
        try:
            # Extract strategy name and timeframe from filename (adjust if needed)
            parts = f.stem.split('_')
            timeframe = parts[-1]
            strategy_name = "_".join(parts[:-1])

            if timeframe not in result_data:
                result_data[timeframe] = {} # Create dict for this timeframe

            with open(f) as fp:
                parsed = json.load(fp)
                # The JSON now has the strategy name as the top key
                strategy_result = parsed.get(strategy_name, {})
                result_data[timeframe][strategy_name] = strategy_result # Store under TF -> Strategy

                # Calculate detailed metrics for summary table
                trades = strategy_result.get("trades_details", [])
                if trades:
                     metrics = calculate_detailed_metrics(trades, timeframe) # Pass timeframe string
                     summary_table[timeframe][strategy_name] = metrics
                else:
                     # Store basic info even if no trades
                     summary_table[timeframe][strategy_name] = {
                         'total_pnl_points': strategy_result.get('total_pnl', 0),
                         'trade_count': strategy_result.get('trade_count', 0),
                         'win_rate': strategy_result.get('win_rate', 0),
                         # Fill others with 0 or NaN?
                         'avg_win_points': 0, 'avg_loss_points': 0, 'profit_factor': 0,
                         'max_drawdown_points': 0, 'expectancy_points': 0, 'sharpe_ratio_points': 0
                     }

        except Exception as e:
            logger.error(f"Error processing result file {f.name}: {e}", exc_info=True)
            # Handle error - maybe add an error entry to result_data?


    # --- Prepare final JSON response ---
    # The detailed report JS might need adjusting if it expects a different structure
    # Let's pass the raw result_data (TF -> Strategy -> Results)
    # And the summary_table (TF -> Strategy -> Metrics)

    return jsonify({
        # Pass the structured results
        "structured_report": result_data,
        "summary_table": summary_table
    })


# --- HTML Report Routes ---
# These might need modification depending on how the JS in them consumes data
@app.route('/runs')
def list_runs():
    runs_dir = Path(__file__).parent / "runs"
    app.logger.info(f"Scanning for runs in: {runs_dir}")
    if not runs_dir.exists():
        app.logger.warning(f"Runs directory does not exist: {runs_dir}")
        return jsonify([])
    try:
        # Basic filter: check if it looks like a date_time string and is a directory
        run_ids = sorted([d.name for d in runs_dir.iterdir() if d.is_dir() and len(d.name) == 15 and d.name.replace('_','').isdigit()], reverse=True)
        app.logger.info(f"Filtered run IDs: {run_ids}")
    except Exception as e:
        app.logger.error(f"Error scanning runs directory: {e}", exc_info=True)
        return jsonify([])
    return jsonify(run_ids)


@app.route('/results_html/<run_id>')
def view_html_report(run_id):
    # Pass run_id to template, JS in template will fetch data from /results/<run_id>
    return render_template("report_template.html", run_id=run_id)

@app.route('/results_html/<run_id>/detailed')
def view_detailed_html_report(run_id):
    # Pass run_id to template, JS in template will fetch data from /results/<run_id>
    return render_template("report_template_detailed.html", run_id=run_id)
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.logger.info(f"Starting Flask server ({Path(__file__).name})...")
    # Ensure reloader is False if using subprocesses this way, debug=False for production
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

    #------------------------Working fine ends-----------------------------------
    # web_app.py (MODIFIED TO TRIGGER pipeline_manager.py AND STREAM LOGS)

import logging
import json
import os
import sys
import pandas as pd
from flask import Flask, request, jsonify, render_template, Response # **** ADD Response ****
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import time # **** ADD time ****
import threading
import subprocess # **** ADD subprocess ****
from collections import defaultdict

# --- Imports ---
APP_DIR = Path(__file__).parent / "app"
sys.path.insert(0, str(APP_DIR.parent))
try:
    from app.config import config
    from app.reporting import calculate_detailed_metrics
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}", file=sys.stderr)

# --- Flask App Setup ---
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# ... (logging setup remains same) ...
log_level_str = getattr(config, "LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level)
app.logger.setLevel(log_level)
app.logger.propagate = True


# --- Global Status & Lock ---
backtest_status = {
    "running": False,
    "run_id": None,
    "message": "Idle",
    "error": None,
    "pid": None,
    "log_file": None
}
status_lock = threading.Lock()

# --- Background Thread to run and monitor pipeline_manager.py ---
def run_pipeline_manager_thread(run_id):
    global backtest_status
    pipeline_successful = False
    process = None
    project_root = Path(__file__).parent
    pipeline_script_path = project_root / "pipeline_manager.py"
    run_log_dir = project_root / "runs" / run_id / "logs"
    # Define the specific log file pipeline_manager will write to
    pipeline_manager_log_path = run_log_dir / f"pipeline_manager_{run_id}.log"

    # Ensure log directory exists
    try:
        run_log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
         app.logger.error(f"Failed to create log directory {run_log_dir}: {e}")
         with status_lock:
              backtest_status.update({"message": f"Run {run_id} failed! Cannot create log dir.", "running": False, "error": str(e), "pid": None, "log_file": None})
         return

    try:
        app.logger.info(f"Background thread starting pipeline_manager.py for RUN_ID: {run_id}")
        command = [sys.executable, str(pipeline_script_path)]
        app.logger.info(f"Executing command: {' '.join(command)}")
        app.logger.info(f"Pipeline manager log file: {pipeline_manager_log_path}")

        # Store log file path in status immediately so log streaming can find it
        with status_lock:
            backtest_status["log_file"] = str(pipeline_manager_log_path)

        # Use Popen to run in background and capture output to file
        # Create the log file first so streaming can start reading it
        with open(pipeline_manager_log_path, 'w', encoding='utf-8') as log_fp:
            process = subprocess.Popen(
                command,
                stdout=log_fp, # Write stdout here
                stderr=subprocess.STDOUT, # Write stderr here too
                cwd=project_root,
                text=True,
                env=os.environ.copy()
            )

        with status_lock:
             backtest_status["pid"] = process.pid # Store process ID

        # --- Monitor the process ---
        while True:
            with status_lock:
                 should_be_running = backtest_status["running"]
                 current_thread_run_id = backtest_status["run_id"]

            if not should_be_running or current_thread_run_id != run_id:
                 app.logger.info(f"Run {run_id} aborted by status flag. Terminating process {process.pid}.")
                 try:
                     process.terminate(); process.wait(timeout=5)
                 except: # Broad except for termination issues
                     process.kill()
                 with status_lock:
                      if backtest_status["run_id"] == run_id:
                           backtest_status.update({"message": f"Run {run_id} aborted by user.", "running": False, "pid": None})
                 return

            return_code = process.poll()
            if return_code is not None:
                 pipeline_successful = (return_code == 0)
                 app.logger.info(f"pipeline_manager.py for {run_id} finished with code: {return_code}. Success: {pipeline_successful}")
                 break
            time.sleep(1.0) # Check less frequently now that logs are streamed

        # --- Update final status ---
        final_message = f"Run {run_id} completed."
        error_message = None
        if not pipeline_successful:
             final_message = f"Run {run_id} completed with errors (Code: {return_code}). Check logs."
             error_message = f"Pipeline failed with code {return_code}."
        with status_lock:
             if backtest_status["run_id"] == run_id:
                  backtest_status.update({"message": final_message, "running": False, "error": error_message, "pid": None})

    except Exception as e:
        app.logger.error(f"Error launching/monitoring pipeline_manager.py for run {run_id}: {e}", exc_info=True)
        if process and process.poll() is None: process.kill()
        with status_lock:
            if backtest_status["run_id"] == run_id:
                 backtest_status.update({"message": f"Run {run_id} failed unexpectedly!", "running": False, "error": str(e), "pid": None})

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Modified /start_backtest route ---
@app.route('/start_backtest', methods=['POST'])
def start_backtest():
    global backtest_status
    with status_lock:
        if backtest_status["running"]:
            return jsonify({"error": "Backtest already running."}), 409

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Log file path will be set by the thread shortly after dirs are created
        log_file_placeholder = str(Path(__file__).parent / "runs" / run_id / "logs" / f"pipeline_manager_{run_id}.log")

        pipeline_thread = threading.Thread(target=run_pipeline_manager_thread, args=(run_id,))
        pipeline_thread.daemon = True
        pipeline_thread.start()

        backtest_status.update({
            "running": True, "run_id": run_id,
            "message": f"Run {run_id} starting (Executing pipeline_manager.py)...",
            "error": None, "pid": None,
            "log_file": log_file_placeholder # Store placeholder initially
        })
        return jsonify({"message": "Backtest run triggered via pipeline_manager.", "run_id": run_id})

# --- Status Endpoint ---
@app.route('/status')
def get_status():
    with status_lock:
        return jsonify(backtest_status.copy())

# --- Stop Endpoint ---
@app.route('/stop_backtest', methods=['POST'])
def stop_backtest():
    # ... (Stop logic remains the same - sets running flag to False) ...
    global backtest_status
    stopped_pid = None
    with status_lock:
        current_run_id = backtest_status.get('run_id')
        current_pid = backtest_status.get('pid')
        if not backtest_status["running"]:
            return jsonify({"error": "No backtest running."}), 409

        app.logger.info(f"Stop requested for run {current_run_id} (PID: {current_pid})")
        backtest_status["running"] = False
        backtest_status["message"] = f"Stop requested by user for run {current_run_id}..."
        stopped_pid = current_pid

    if stopped_pid:
         app.logger.info(f"Signaled monitoring thread to stop process {stopped_pid} for run {current_run_id}.")
    return jsonify({"message": "Stop signal sent to running process."})

# --- *** NEW: Log Streaming Endpoint *** ---
@app.route('/stream_logs/<run_id>')
def stream_logs(run_id):
    """Streams the content of the pipeline manager log file using SSE."""
    def generate_log_updates():
        log_file_path_str = None
        # Check status initially to get the log file path
        with status_lock:
             if backtest_status.get("run_id") == run_id:
                  log_file_path_str = backtest_status.get("log_file")

        if not log_file_path_str or not Path(log_file_path_str).is_file():
             yield f"data: Log file for run {run_id} not found or not ready yet.\n\n"
             return

        log_file_path = Path(log_file_path_str)
        app.logger.info(f"Starting log stream for: {log_file_path}")

        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                # Go to the end of the file initially if desired, or read from start
                # f.seek(0, os.SEEK_END) # Use this to only show new lines
                while True:
                    line = f.readline()
                    if line:
                        # Format as SSE message: data: <line_content>\n\n
                        yield f"data: {line.rstrip()}\n\n"
                    else:
                        # No new line, check if process is still running
                        with status_lock:
                             is_running = backtest_status.get("running", False)
                             current_run = backtest_status.get("run_id")
                        if not is_running and current_run == run_id:
                             app.logger.info(f"Log stream closing for completed/stopped run: {run_id}")
                             yield f"data: --- End of logs for run {run_id} ---\n\n"
                             break # Exit generator if process stopped
                        # Wait before polling file again
                        time.sleep(0.5) # Poll every 0.5 seconds
        except Exception as e:
            app.logger.error(f"Error streaming log file {log_file_path}: {e}")
            yield f"data: ERROR reading log file: {e}\n\n"
        finally:
             app.logger.info(f"Log stream stopped for {run_id}")


    # Return SSE response
    return Response(generate_log_updates(), mimetype='text/event-stream')


# --- Results Endpoint (Adjusted for strategy_results) ---
@app.route('/results/<run_id>')
def get_results_json(run_id):
    # Path to where pipeline_manager saves independent results
    results_dir = Path(__file__).parent / "runs" / run_id / "results" / "strategy_results"
    if not results_dir.is_dir():
         return jsonify({"error": f"Strategy results directory not found for run ID: {run_id}"}), 404

    json_files = sorted(list(results_dir.glob("*.json")))
    if not json_files:
        return jsonify({"error": f"No strategy result JSON files found in {results_dir}"}), 404

    result_data = {}
    summary_table = defaultdict(lambda: defaultdict(dict))

    for f in json_files:
        try:
            parts = f.stem.split('_')
            timeframe = parts[-1]
            strategy_name = "_".join(parts[:-1])
            if timeframe not in result_data: result_data[timeframe] = {}

            with open(f) as fp:
                parsed = json.load(fp)
                strategy_result = parsed.get(strategy_name, {}) # Get data using strategy name key
                result_data[timeframe][strategy_name] = strategy_result

                trades = strategy_result.get("trades_details", [])
                if trades:
                     metrics = calculate_detailed_metrics(trades, tf=f"{strategy_name}_{timeframe}") # Pass context to metrics
                     summary_table[timeframe][strategy_name] = metrics
                else: # Store defaults if no trades
                     summary_table[timeframe][strategy_name] = {
                         'total_pnl_points': strategy_result.get('total_pnl', 0),
                         'trade_count': 0, 'win_rate': 0, 'avg_win_points': 0,
                         'avg_loss_points': 0, 'profit_factor': 0, 'max_drawdown_points': 0,
                         'expectancy_points': 0, 'sharpe_ratio_points': 0
                     }
        except Exception as e:
            app.logger.error(f"Error processing result file {f.name}: {e}", exc_info=True)

    return jsonify({
        "structured_report": result_data, # TF -> Strategy -> Results
        "summary_table": summary_table   # TF -> Strategy -> Metrics
    })

# --- HTML Report Routes (No changes needed here) ---
@app.route('/runs')
# ... (list_runs remains same) ...
def list_runs():
    runs_dir = Path(__file__).parent / "runs"
    app.logger.info(f"Scanning for runs in: {runs_dir}")
    if not runs_dir.exists():
        app.logger.warning(f"Runs directory does not exist: {runs_dir}")
        return jsonify([])
    try:
        # Basic filter: check if it looks like a date_time string and is a directory
        run_ids = sorted([d.name for d in runs_dir.iterdir() if d.is_dir() and len(d.name) == 15 and d.name.replace('_','').isdigit()], reverse=True)
        app.logger.info(f"Filtered run IDs: {run_ids}")
    except Exception as e:
        app.logger.error(f"Error scanning runs directory: {e}", exc_info=True)
        return jsonify([])
    return jsonify(run_ids)

@app.route('/results_html/<run_id>')
# ... (view_html_report remains same) ...
def view_html_report(run_id):
    return render_template("report_template.html", run_id=run_id)


@app.route('/results_html/<run_id>/detailed')
# ... (view_detailed_html_report remains same) ...
def view_detailed_html_report(run_id):
    return render_template("report_template_detailed.html", run_id=run_id)


if __name__ == '__main__':
    app.logger.info(f"Starting Flask server ({Path(__file__).name})...")
    # use_reloader=False is important when using background threads/processes
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)