# import logging
# import json
# import os
# import sys
# from app.reporting import calculate_detailed_metrics
# from flask import Flask, request, jsonify, render_template, send_file, abort
# from typing import Dict, List, Optional, Any
# from pathlib import Path
# from datetime import datetime
# import traceback
# import threading
# import pandas as pd

# # --- Imports ---
# APP_DIR = Path(__file__).parent / "app"
# sys.path.insert(0, str(APP_DIR.parent))
# try:
#     from app.config import config
#     from app.data_io import load_historical_data
#     from app.feature_engine import IndicatorCalculator
#     from app.run_simulation_step import run_and_save_agent_backtest
#     from app.reporting import generate_agent_html_report # Still needed by thread
#     from app.agentic_core import RuleBasedAgent
# except ImportError as e:
#     print(f"ERROR: Failed to import necessary modules: {e}", file=sys.stderr)
#     sys.exit(1)

# # --- Flask App Setup (Corrected Folders) ---
# app = Flask(__name__,
#             template_folder='templates', # Look for templates in 'templates/' folder
#             static_folder='static')     # Look for static files in 'static/' folder

# # --- Logging Setup ---
# log_level_str = getattr(config, "LOG_LEVEL", "INFO").upper()
# log_level = getattr(logging, log_level_str, logging.INFO)
# logging.basicConfig(level=log_level)
# app.logger.setLevel(log_level)
# if not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
#      # Avoid adding handler if Flask/Gunicorn adds one
#      pass # app.logger.addHandler(logging.StreamHandler())
# app.logger.propagate = True # Let Flask manage handlers

# # --- Global Status & Lock ---
# backtest_status = {"running": False, "run_id": None, "message": "Idle", "error": None}
# status_lock = threading.Lock()

# # --- Background Pipeline Thread (Saves Report) ---
# def run_pipeline_thread(run_id, timeframes, raw_data_map, indicator_map, results_map, log_dir, results_dir):
#     # (This function remains the same as the last version - it correctly saves the report)
#     global backtest_status; pipeline_successful = True; all_tf_results_for_report = {}
#     try:
#         app.logger.info(f"Background thread started for RUN_ID: {run_id}"); calculator = IndicatorCalculator()
#         for tf in timeframes:
#             with status_lock:
#                  if not backtest_status["running"] or backtest_status["run_id"] != run_id: app.logger.info(f"Run {run_id} aborted during {tf}."); pipeline_successful = False; break
#             with status_lock: backtest_status["message"] = f"Calculating features for {tf}..."; app.logger.info(f"Calculating features for {tf}...")
#             raw_path = raw_data_map[tf]; indicator_path = indicator_map[tf]
#             if not raw_path.is_file(): raise FileNotFoundError(f"Raw data not found: {raw_path}")
#             raw_df = load_historical_data(raw_path.name); df_with_indicators = calculator.calculate_all_indicators(raw_df)
#             indicator_path.parent.mkdir(parents=True, exist_ok=True); df_with_indicators.to_csv(indicator_path, index=True); app.logger.info(f"Features saved for {tf} to {indicator_path}")
#             with status_lock:
#                  if not backtest_status["running"] or backtest_status["run_id"] != run_id: app.logger.info(f"Run {run_id} aborted after features for {tf}."); pipeline_successful = False; break
#             with status_lock: backtest_status["message"] = f"Running simulation for {tf}..."; app.logger.info(f"Running simulation for {tf}...")
#             result_json_path = results_map[tf]
#             tf_result_data = run_and_save_agent_backtest(indicator_path, result_json_path, log_dir)
#             all_tf_results_for_report[tf] = tf_result_data.get("RuleBasedAgent") if tf_result_data else {"error": "Simulation failed"}
#             app.logger.info(f"Simulation done for {tf}. Results saved.")
#         if pipeline_successful and backtest_status.get("running") and backtest_status.get("run_id") == run_id: # Check status before reporting
#             app.logger.info(f"Pipeline loop finished for {run_id}")
#             final_message = f"Run {run_id} completed."; report_path = None
#             try:
#                 # Load results again to ensure consistency
#                 loaded_results = {}
#                 for tf in timeframes:
#                      fpath = results_map[tf]
#                      if fpath.is_file():
#                          with open(fpath, 'r') as f: loaded_results[tf] = json.load(f).get("RuleBasedAgent")
#                      else: loaded_results[tf] = {"error": "Result JSON missing"}

#                 # html_report_content = generate_agent_html_report(loaded_results) # Generate HTML
#                 # report_filename = f"report_{run_id}.html"; report_path = results_dir / report_filename
#                 # report_path.write_text(html_report_content, encoding='utf-8');
#                 # app.logger.info(f"HTML report saved to {report_path}")
#                 app.logger.info(f"HTML report generated (not saved to disk).")

#             except Exception as report_e: app.logger.error(f"Failed to generate report: {report_e}", exc_info=True); final_message = f"Run {run_id} completed, report generation failed."
#             with status_lock: backtest_status["message"] = final_message; backtest_status["running"] = False; backtest_status["error"] = None
#         elif not pipeline_successful: app.logger.info(f"Pipeline for {run_id} did not complete fully (likely stopped).")
#     except Exception as e:
#         error_msg = f"Error in background run {run_id}: {e}"
#         app.logger.error(error_msg, exc_info=True)
#         with status_lock:
#             backtest_status["message"] = f"Run {run_id} failed!"
#             backtest_status["running"] = False
#             backtest_status["error"] = str(e)
#     finally:  # Ensure status reset
#         with status_lock:
#             if backtest_status["run_id"] == run_id and backtest_status["running"]:
#                 app.logger.warning(f"BG thread {run_id} ending but status running. Resetting.")
#                 backtest_status["running"] = False
#                 if not backtest_status["error"]:
#                     backtest_status["message"] = f"Run {run_id} finished (unknown state)."


# # --- Flask Routes ---
# @app.route('/')
# def index():
#     # This will now correctly look for 'index.html' in the 'templates/' folder
#     return render_template('index.html')

# @app.route('/start_backtest', methods=['POST'])
# def start_backtest():
#     # (Keep implementation - starts background thread)
#     global backtest_status
#     with status_lock:
#         if backtest_status["running"]: return jsonify({"error": "Backtest already running."}), 409
#         params = request.json; app.logger.info(f"Received backtest request: {params}")
#         run_id = datetime.now().strftime("%Y%m%d_%H%M%S"); run_dir = Path(__file__).parent / "runs" / run_id
#         run_data_dir = run_dir / "data"; results_dir = run_dir / "results"; log_dir = run_dir / "logs"
#         try: run_dir.mkdir(parents=True, exist_ok=True); run_data_dir.mkdir(exist_ok=True); results_dir.mkdir(exist_ok=True); log_dir.mkdir(exist_ok=True)
#         except OSError as e: app.logger.error(f"Could not create run dirs: {e}"); return jsonify({"error": f"Could not create run dirs: {e}"}), 500
#         timeframes = list(config.RAW_DATA_FILES.keys())
#         if not timeframes: return jsonify({"error": "No timeframes configured."}), 400
#         raw_data_map = {}
#         for tf in timeframes:
#              tf_filename = config.RAW_DATA_FILES.get(tf)
#              if not tf_filename: return jsonify({"error": f"Filename missing for {tf}."}), 400
#              tf_path = config.DATA_FOLDER / tf_filename;
#              if not tf_path.is_file(): return jsonify({"error": f"Raw data file not found: {tf_path}"}), 400
#              raw_data_map[tf] = tf_path
#         indicator_map = {tf: run_data_dir / (Path(config.RAW_DATA_FILES[tf]).stem + "_with_indicators.csv") for tf in timeframes}
#         results_map = {tf: results_dir / f"backtest_summary_{tf}.json" for tf in timeframes}
#         # Pass results_dir and log_dir to the thread function
#         pipeline_thread = threading.Thread(target=run_pipeline_thread, args=(run_id, timeframes, raw_data_map, indicator_map, results_map, log_dir, results_dir));
#         pipeline_thread.daemon = True; pipeline_thread.start()
#         backtest_status["running"] = True; backtest_status["run_id"] = run_id; backtest_status["message"] = f"Run {run_id} started..."; backtest_status["error"] = None
#         return jsonify({"message": "Backtest run started.", "run_id": run_id})


# @app.route('/status')
# def get_status():
#     with status_lock: return jsonify(backtest_status)

# @app.route('/stop_backtest', methods=['POST'])
# def stop_backtest(): # (Keep as is)
#      global backtest_status
#      with status_lock:
#          current_run_id = backtest_status.get('run_id')
#          if not backtest_status["running"]: return jsonify({"error": "No backtest running."}), 409
#          backtest_status["running"]=False; backtest_status["message"]=f"Stop requested by user for run {current_run_id}."
#          app.logger.info(f"Stop requested for run {current_run_id}")
#          return jsonify({"message": "Stop requested." })


# # --- CORRECTED /runs Endpoint Filter ---
# @app.route('/runs')
# def list_runs():
#      runs_dir = Path(__file__).parent / "runs"; app.logger.info(f"Scanning for runs in: {runs_dir}"); run_ids = []
#      if not runs_dir.exists(): app.logger.warning(f"Runs directory does not exist: {runs_dir}"); return jsonify([])
#      try:
#         found_items=list(runs_dir.iterdir()); 
#         #app.logger.info(f"Found items in runs dir: {[item.name for item in found_items]}")
#         # Corrected Filter: is directory and length is 15 (YYYYMMDD_HHMMSS)
#         run_ids=sorted([d.name for d in found_items if d.is_dir() and len(d.name)==15],reverse=True)
#         app.logger.info(f"Filtered run IDs matching pattern: {run_ids}")
#      except Exception as e: app.logger.error(f"Error scanning runs directory {runs_dir}: {e}", exc_info=True); return jsonify([])
#      return jsonify(run_ids)

# @app.route('/results_html/<run_id>')
# def view_html_report(run_id):
#     return render_template("report_template.html", run_id=run_id)
#     # run_dir = Path(__file__).parent / "runs" / run_id
#     # results_dir = run_dir / "results"
#     # json_files = list(results_dir.glob("backtest_summary_*.json"))

#     # results = {}
#     # for f in json_files:
#     #     tf = f.stem.split("_")[-1]
#     #     try:
#     #         with open(f) as fp:
#     #             data = json.load(fp)
#     #             trades = data.get("RuleBasedAgent", {}).get("trades_details", [])
#     #             metrics = calculate_detailed_metrics(trades, tf)
#     #             results[tf] = { "trades": trades, "metrics": metrics }
#     #     except Exception as e:
#     #         results[tf] = { "error": str(e) }

#     # return render_template("report_template.html",
#     #                        run_id=run_id,
#     #                        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#     #                        results=results)
# @app.route('/results_html/<run_id>/detailed')
# def view_detailed_html_report(run_id):
#     return render_template("report_template_detailed.html", run_id=run_id)
# @app.route('/results/<run_id>')
# def get_results_json(run_id):
#     results_dir = Path(__file__).parent / "runs" / run_id / "results"
#     json_files = list(results_dir.glob("backtest_summary_*.json"))
#     if not json_files:
#         return jsonify({"error": f"No result files found for run ID: {run_id}"}), 404

#     result_data = {}
#     for f in json_files:
#         tf = f.stem.split("_")[-1]
#         try:
#             with open(f) as fp:
#                 parsed = json.load(fp)
#                 result_data[tf] = parsed.get("RuleBasedAgent", {})
#         except Exception as e:
#             result_data[tf] = {"error": str(e)}

#     return jsonify({"html_report": json.dumps(result_data)})


# # --- Run Server ---
# if __name__ == '__main__':
#     app.logger.info(f"Starting Flask server ({Path(__file__).name})...")
#     # Turn off debug and reloader for stability with threading
#     # Use host='0.0.0.0' only if you need access from other devices on your network
#     app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

import logging
import json
import os
import sys
from app.reporting import calculate_detailed_metrics
from flask import Flask, request, jsonify, render_template
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import threading

# --- Imports ---
APP_DIR = Path(__file__).parent / "app"
sys.path.insert(0, str(APP_DIR.parent))
try:
    from app.config import config
    from app.data_io import load_historical_data
    from app.feature_engine import IndicatorCalculator
    from app.run_simulation_step import run_and_save_agent_backtest
    
    from app.agentic_core import RuleBasedAgent
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}", file=sys.stderr)
    sys.exit(1)

# --- Flask App Setup ---
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

log_level_str = getattr(config, "LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level)
app.logger.setLevel(log_level)
app.logger.propagate = True

# --- Global Status & Lock ---
backtest_status = {"running": False, "run_id": None, "message": "Idle", "error": None}
status_lock = threading.Lock()

# --- Background Pipeline Thread ---
def run_pipeline_thread(run_id, timeframes, raw_data_map, indicator_map, results_map, log_dir, results_dir):
    global backtest_status
    pipeline_successful = True
    all_tf_results_for_report = {}
    try:
        app.logger.info(f"Background thread started for RUN_ID: {run_id}")
        calculator = IndicatorCalculator()
        for tf in timeframes:
            with status_lock:
                if not backtest_status["running"] or backtest_status["run_id"] != run_id:
                    app.logger.info(f"Run {run_id} aborted during {tf}.")
                    pipeline_successful = False
                    break

            app.logger.info(f"Calculating features for {tf}...")
            raw_path = raw_data_map[tf]
            indicator_path = indicator_map[tf]
            if not raw_path.is_file():
                raise FileNotFoundError(f"Raw data not found: {raw_path}")

            raw_df = load_historical_data(raw_path.name)
            df_with_indicators = calculator.calculate_all_indicators(raw_df)
            indicator_path.parent.mkdir(parents=True, exist_ok=True)
            df_with_indicators.to_csv(indicator_path, index=True)
            app.logger.info(f"Features saved for {tf} to {indicator_path}")

            with status_lock:
                if not backtest_status["running"] or backtest_status["run_id"] != run_id:
                    app.logger.info(f"Run {run_id} aborted after features for {tf}.")
                    pipeline_successful = False
                    break

            app.logger.info(f"Running simulation for {tf}...")
            result_json_path = results_map[tf]
            tf_result_data = run_and_save_agent_backtest(indicator_path, result_json_path, log_dir)
            all_tf_results_for_report[tf] = tf_result_data.get("RuleBasedAgent") if tf_result_data else {"error": "Simulation failed"}
            app.logger.info(f"Simulation done for {tf}. Results saved.")

        if pipeline_successful and backtest_status.get("running") and backtest_status.get("run_id") == run_id:
            app.logger.info(f"Pipeline loop finished for {run_id}")
            final_message = f"Run {run_id} completed."
            try:
                loaded_results = all_tf_results_for_report
                app.logger.info(f"HTML report generated (not saved to disk).")
            except Exception as report_e:
                app.logger.error(f"Failed to generate report: {report_e}", exc_info=True)
                final_message = f"Run {run_id} completed, report generation failed."
            with status_lock:
                backtest_status.update({"message": final_message, "running": False, "error": None})
        elif not pipeline_successful:
            app.logger.info(f"Pipeline for {run_id} did not complete fully.")
    except Exception as e:
        app.logger.error(f"Error in background run {run_id}: {e}", exc_info=True)
        with status_lock:
            backtest_status.update({"message": f"Run {run_id} failed!", "running": False, "error": str(e)})
    finally:
        with status_lock:
            if backtest_status["run_id"] == run_id and backtest_status["running"]:
                app.logger.warning(f"BG thread {run_id} ending but status running. Resetting.")
                backtest_status["running"] = False
                if not backtest_status["error"]:
                    backtest_status["message"] = f"Run {run_id} finished (unknown state)."

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_backtest', methods=['POST'])
def start_backtest():
    global backtest_status
    with status_lock:
        if backtest_status["running"]:
            return jsonify({"error": "Backtest already running."}), 409

        params = request.json
        app.logger.info(f"Received backtest request: {params}")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(__file__).parent / "runs" / run_id
        run_data_dir = run_dir / "data"
        results_dir = run_dir / "results"
        log_dir = run_dir / "logs"

        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            run_data_dir.mkdir(exist_ok=True)
            results_dir.mkdir(exist_ok=True)
            log_dir.mkdir(exist_ok=True)
        except OSError as e:
            app.logger.error(f"Could not create run dirs: {e}")
            return jsonify({"error": f"Could not create run dirs: {e}"}), 500

        timeframes = list(config.RAW_DATA_FILES.keys())
        if not timeframes:
            return jsonify({"error": "No timeframes configured."}), 400

        raw_data_map = {}
        for tf in timeframes:
            tf_filename = config.RAW_DATA_FILES.get(tf)
            if not tf_filename:
                return jsonify({"error": f"Filename missing for {tf}."}), 400

            tf_path = config.DATA_FOLDER / tf_filename
            if not tf_path.is_file():
                return jsonify({"error": f"Raw data file not found: {tf_path}"}), 400

            raw_data_map[tf] = tf_path

        indicator_map = {tf: run_data_dir / (Path(config.RAW_DATA_FILES[tf]).stem + "_with_indicators.csv") for tf in timeframes}
        results_map = {tf: results_dir / f"backtest_summary_{tf}.json" for tf in timeframes}

        pipeline_thread = threading.Thread(target=run_pipeline_thread, args=(run_id, timeframes, raw_data_map, indicator_map, results_map, log_dir, results_dir))
        pipeline_thread.daemon = True
        pipeline_thread.start()

        backtest_status.update({"running": True, "run_id": run_id, "message": f"Run {run_id} started...", "error": None})
        return jsonify({"message": "Backtest run started.", "run_id": run_id, "auto_load": True})

@app.route('/status')
def get_status():
    with status_lock:
        return jsonify(backtest_status)

@app.route('/stop_backtest', methods=['POST'])
def stop_backtest():
    global backtest_status
    with status_lock:
        current_run_id = backtest_status.get('run_id')
        if not backtest_status["running"]:
            return jsonify({"error": "No backtest running."}), 409

        backtest_status["running"] = False
        backtest_status["message"] = f"Stop requested by user for run {current_run_id}."
        app.logger.info(f"Stop requested for run {current_run_id}")
        return jsonify({"message": "Stop requested."})

@app.route('/runs')
def list_runs():
    runs_dir = Path(__file__).parent / "runs"
    app.logger.info(f"Scanning for runs in: {runs_dir}")
    if not runs_dir.exists():
        app.logger.warning(f"Runs directory does not exist: {runs_dir}")
        return jsonify([])

    try:
        run_ids = sorted([d.name for d in runs_dir.iterdir() if d.is_dir() and len(d.name) == 15], reverse=True)
        app.logger.info(f"Filtered run IDs: {run_ids}")
    except Exception as e:
        app.logger.error(f"Error scanning runs directory: {e}", exc_info=True)
        return jsonify([])

    return jsonify(run_ids)

@app.route('/results_html/<run_id>')
def view_html_report(run_id):
    return render_template("report_template.html", run_id=run_id)

@app.route('/results_html/<run_id>/detailed')
def view_detailed_html_report(run_id):
    return render_template("report_template_detailed.html", run_id=run_id)

@app.route('/results/<run_id>')
def get_results_json(run_id):
    results_dir = Path(__file__).parent / "runs" / run_id / "results"
    json_files = list(results_dir.glob("backtest_summary_*.json"))
    if not json_files:
        return jsonify({"error": f"No result files found for run ID: {run_id}"}), 404

    result_data = {}
    for f in json_files:
        tf = f.stem.split("_")[-1]
        try:
            with open(f) as fp:
                parsed = json.load(fp)
                result_data[tf] = parsed.get("RuleBasedAgent", {})
        except Exception as e:
            result_data[tf] = {"error": str(e)}

    return jsonify({"html_report": json.dumps(result_data)})

if __name__ == '__main__':
    app.logger.info(f"Starting Flask server ({Path(__file__).name})...")
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
