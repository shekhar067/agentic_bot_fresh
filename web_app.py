# web_app.py (in project root)

import logging
import json
import os
import sys
from flask import Flask, request, jsonify, render_template # type: ignore
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import traceback
import threading
import pandas as pd

# Add app directory to path
APP_DIR = Path(__file__).parent / "app"
sys.path.insert(0, str(APP_DIR.parent))

# Use new module names for imports
from app.config import config
from app.data_io import load_historical_data
from app.feature_engine import IndicatorCalculator
from app.run_simulation_step import run_and_save_agent_backtest # Uses renamed script logic
from app.reporting import generate_agent_html_report # Import from new reporting module

# --- Flask App Setup ---
app = Flask(__name__, template_folder='.')
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)
backtest_status = {"running": False, "run_id": None, "message": "Idle", "error": None}
status_lock = threading.Lock()

# --- run_pipeline_thread (Uses new names internally) ---
def run_pipeline_thread(run_id, timeframes, raw_data_map, indicator_map, results_map):
    global backtest_status
    try:
        app.logger.info(f"Background thread started for RUN_ID: {run_id}")
        # --- Define log directory for this run ---
        run_dir = Path(__file__).parent / "runs" / run_id
        log_dir = run_dir / "logs" # Directory for detailed sim logs
        log_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists
        # --- End define ---

        calculator = IndicatorCalculator() # Initialize once

        for tf in timeframes:
            with status_lock: backtest_status["message"] = f"Calculating features for {tf}..."; app.logger.info(f"Calculating features for {tf}...")
            # ... (indicator calculation logic remains the same) ...
            raw_path = raw_data_map[tf]; indicator_path = indicator_map[tf]
            if not raw_path.is_file(): raise FileNotFoundError(f"Raw data not found: {raw_path}")
            raw_df = load_historical_data(raw_path.name); df_with_indicators = calculator.calculate_all_indicators(raw_df)
            indicator_path.parent.mkdir(parents=True, exist_ok=True); df_with_indicators.to_csv(indicator_path, index=True); app.logger.info(f"Features saved for {tf} to {indicator_path}")

            with status_lock: backtest_status["message"] = f"Running simulation for {tf}..."; app.logger.info(f"Running simulation for {tf}...")
            result_json_path = results_map[tf]

            # --- MODIFIED CALL: Pass log_dir ---
            run_and_save_agent_backtest(
                indicator_file_path=indicator_path,
                output_json_path=result_json_path,
                log_dir=log_dir # Pass the specific log directory for this run
            )
            # --- END MODIFIED CALL ---

            app.logger.info(f"Simulation done for {tf}. Results saved.")

        with status_lock: backtest_status["message"]=f"Run {run_id} completed.";backtest_status["running"]=False;backtest_status["error"]=None; app.logger.info(f"BG Thread OK: {run_id}")
    except Exception as e:
        error_msg = f"Error in background run {run_id}: {e}"; app.logger.error(error_msg, exc_info=True);
        with status_lock: backtest_status["message"]=f"Run {run_id} failed!"; backtest_status["running"]=False; backtest_status["error"]=str(e)

# --- Flask Routes (mostly same, use new names in logs/comments) ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/start_backtest', methods=['POST'])
def start_backtest():
    global backtest_status
    with status_lock:
        if backtest_status["running"]: return jsonify({"error": "Backtest already running."}), 409
        params = request.json; app.logger.info(f"Received backtest request: {params}")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S"); run_dir = Path(__file__).parent / "runs" / run_id
        run_data_dir = run_dir / "data"; results_dir = run_dir / "results" # Define results_dir here
        run_dir.mkdir(parents=True, exist_ok=True); run_data_dir.mkdir(exist_ok=True); results_dir.mkdir(exist_ok=True)
        timeframes = list(config.RAW_DATA_FILES.keys())
        raw_data_map = {tf: config.DATA_FOLDER / config.RAW_DATA_FILES[tf] for tf in timeframes}
        indicator_map = {tf: run_data_dir / (Path(config.RAW_DATA_FILES[tf]).stem + "_with_indicators.csv") for tf in timeframes}
        # Make sure results_map points to the correct results_dir
        results_map = {tf: results_dir / f"backtest_summary_{tf}.json" for tf in timeframes}
        pipeline_thread = threading.Thread(target=run_pipeline_thread, args=(run_id, timeframes, raw_data_map, indicator_map, results_map)); pipeline_thread.start()
        backtest_status["running"] = True; backtest_status["run_id"] = run_id; backtest_status["message"] = f"Run {run_id} started..."; backtest_status["error"] = None
        return jsonify({"message": "Backtest run started.", "run_id": run_id})

@app.route('/status')
def get_status():
    with status_lock:
        return jsonify(backtest_status)

@app.route('/stop_backtest', methods=['POST'])
def stop_backtest(): # (Keep as is)
    global backtest_status
    with status_lock:
        if not backtest_status["running"]: return jsonify({"error": "No backtest running."}), 409
        backtest_status["running"]=False; backtest_status["message"]="Stop requested by user."
        return jsonify({"message": "Stop requested (thread continues)." })

@app.route('/runs')
def list_runs(): # (Keep as is)
    runs_dir = Path(__file__).parent / "runs"; app.logger.info(f"Scanning: {runs_dir}"); run_ids = []
    if not runs_dir.exists(): app.logger.warning(f"Dir not exist: {runs_dir}"); return jsonify([])
    try:
        found_items=list(runs_dir.iterdir()); 
        #app.logger.info(f"Found: {[i.name for i in found_items]}")
        run_ids=sorted([d.name for d in found_items if d.is_dir() and len(d.name)==15],reverse=True)
        #app.logger.info(f"Filtered: {run_ids}")
    except Exception as e: app.logger.error(f"Error scanning: {e}"); return jsonify([])
    return jsonify(run_ids)

@app.route('/results/<run_id>')
def get_results(run_id):
    """Loads results and calls reporting module to generate HTML."""
    run_dir = Path(__file__).parent / "runs" / run_id
    results_dir = run_dir / "results"
    app.logger.info(f"--- Loading JSON results for Run ID: {run_id} from {results_dir} ---")
    if not results_dir.is_dir(): return jsonify({"error": "Results dir not found"}), 404
    json_files = list(results_dir.glob('backtest_summary_*.json'))
    timeframes_processed = sorted([p.stem.replace('backtest_summary_', '') for p in json_files])
    if not timeframes_processed: return jsonify({"error": ...}), 404

    app.logger.info(f"Found result files for timeframes: {timeframes_processed}")
    if not timeframes_processed: return jsonify({"error": "No result JSON files found"}), 404

    results_payload = {
        "run_id": run_id,
        "timeframes": {} # <<<< Initialize the 'timeframes' key HERE
    }
    all_agent_results_for_run = {}
    for timeframe in timeframes_processed:
        result_file = results_dir / f"backtest_summary_{timeframe}.json"
        if result_file.is_file():
            try:
                with open(result_file, 'r') as f:
                    agent_run_data = json.load(f)
                    agent_data = agent_run_data.get("RuleBasedAgent") # Get agent's dict
                    if agent_data:
                        # --- Populate the 'timeframes' dictionary ---
                        results_payload["timeframes"][timeframe] = agent_data # <<<< Add data HERE
                        app.logger.info(f"  Successfully loaded results for {timeframe}.")
                    else:
                        results_payload["timeframes"][timeframe] = { "error": f"Invalid data structure in {result_file.name}"} # Add error marker
                        app.logger.warning(f"  'RuleBasedAgent' key not found in {result_file}")
            except Exception as e:
                app.logger.error(f"Error loading result file {result_file}: {e}")
                results_payload["timeframes"][timeframe] = { "error": f"Failed to load results: {e}"}
        else:
             results_payload["timeframes"][timeframe] = { "error": "Result file missing."} # Add error marker

    # --- Log the final structure BEFORE returning ---
    app.logger.debug(f"Final JSON payload being sent for run {run_id}:")
    try: app.logger.debug(json.dumps(results_payload, indent=2, default=str))
    except Exception: app.logger.debug(results_payload)
    # --- End Logging ---

    # --- Return the payload ---
    return jsonify(results_payload) # Return the structured data
# --- Run Server ---
if __name__ == '__main__':
    app.logger.info("Starting Flask server (web_app.py)...")
    app.run(host='127.0.0.1', port=5000, debug=True) # debug=True is helpful for development