# web_app.py (or server.py)

import logging
import json
import os
import sys
from flask import Flask, request, jsonify, render_template # Removed send_from_directory for now
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import traceback
import threading
import pandas as pd

# --- Imports and Setup (remain the same) ---
APP_DIR = Path(__file__).parent / "app"
sys.path.insert(0, str(APP_DIR.parent))
from app.config import config
from app.data_io import load_historical_data
from app.feature_engine import IndicatorCalculator
from app.run_simulation_step import run_and_save_backtest

app = Flask(__name__, template_folder='.')
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)
backtest_status = {"running": False, "run_id": None, "message": "Idle", "error": None}
status_lock = threading.Lock()

# --- run_pipeline_thread (remains the same) ---
def run_pipeline_thread(run_id, timeframes, raw_data_map, indicator_map, results_map):
    # (Keep implementation from previous step)
    global backtest_status
    try:
        app.logger.info(f"Background thread started for RUN_ID: {run_id}")
        calculator = IndicatorCalculator(); # Initialize once
        for tf in timeframes:
            # ... (indicator calculation logic) ...
            with status_lock: backtest_status["message"] = f"Calculating indicators for {tf}..."; app.logger.info(f"Calculating indicators for {tf}...")
            raw_path = raw_data_map[tf]; indicator_path = indicator_map[tf]
            if not raw_path.is_file(): raise FileNotFoundError(f"Raw data not found for {tf}: {raw_path}")
            raw_df = load_historical_data(raw_path.name); df_with_indicators = calculator.calculate_all_indicators(raw_df)
            indicator_path.parent.mkdir(parents=True, exist_ok=True); df_with_indicators.to_csv(indicator_path, index=True); app.logger.info(f"Indicators saved for {tf} to {indicator_path}")
            # ... (backtesting logic) ...
            with status_lock: backtest_status["message"] = f"Running backtest for {tf}..."; app.logger.info(f"Running backtest for {tf}...")
            result_json_path = results_map[tf]; run_and_save_backtest(indicator_path, result_json_path); app.logger.info(f"Backtest done for {tf}. Results saved.")
        # ... (final status update) ...
        with status_lock: backtest_status["message"] = f"Run {run_id} completed successfully."; backtest_status["running"] = False; backtest_status["error"] = None; app.logger.info(f"Background thread finished successfully for RUN_ID: {run_id}")
    except Exception as e: # ... (exception handling remains same) ...
        error_msg = f"Error during background run {run_id}: {e}"; app.logger.error(error_msg, exc_info=True);
        with status_lock: backtest_status["message"] = f"Run {run_id} failed!"; backtest_status["running"] = False; backtest_status["error"] = str(e)


# --- Flask Routes ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/start_backtest', methods=['POST'])
def start_backtest():
    # (Keep implementation from previous step)
    global backtest_status
    with status_lock:
        if backtest_status["running"]: return jsonify({"error": "A backtest is already running."}), 409
        params = request.json; app.logger.info(f"Received backtest request with params: {params}")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S"); run_dir = Path(__file__).parent / "runs" / run_id
        run_data_dir = run_dir / "data"; results_dir = run_dir / "results"
        run_dir.mkdir(parents=True, exist_ok=True); run_data_dir.mkdir(exist_ok=True); results_dir.mkdir(exist_ok=True)
        timeframes = list(config.RAW_DATA_FILES.keys())
        raw_data_map = {tf: config.DATA_FOLDER / config.RAW_DATA_FILES[tf] for tf in timeframes}
        indicator_map = {tf: run_data_dir / (Path(config.RAW_DATA_FILES[tf]).stem + "_with_indicators.csv") for tf in timeframes}
        results_map = {tf: results_dir / f"backtest_summary_{tf}.json" for tf in timeframes}
        pipeline_thread = threading.Thread(target=run_pipeline_thread, args=(run_id, timeframes, raw_data_map, indicator_map, results_map)); pipeline_thread.start()
        backtest_status["running"] = True; backtest_status["run_id"] = run_id; backtest_status["message"] = f"Run {run_id} started..."; backtest_status["error"] = None
        return jsonify({"message": "Backtest run started.", "run_id": run_id})

@app.route('/status')
def get_status():
    with status_lock: return jsonify(backtest_status)

@app.route('/stop_backtest', methods=['POST'])
def stop_backtest():
     # (Keep implementation from previous step)
     global backtest_status
     with status_lock:
         if not backtest_status["running"]: return jsonify({"error": "No backtest is currently running."}), 409
         backtest_status["running"] = False; backtest_status["message"] = "Backtest stopped by user."
         # Note: This doesn't actually stop the background thread in this simple implementation
         return jsonify({"message": "Backtest stop requested (thread continues in background)." })


@app.route('/runs')
def list_runs():
     # (Keep corrected implementation from previous step)
     runs_dir = Path(__file__).parent / "runs"; app.logger.info(f"Scanning for runs in: {runs_dir}"); run_ids = []
     if not runs_dir.exists(): app.logger.warning(f"Runs directory does not exist: {runs_dir}"); return jsonify([])
     try:
        found_items = list(runs_dir.iterdir()); app.logger.info(f"Found items in runs dir: {[item.name for item in found_items]}")
        run_ids = sorted([d.name for d in found_items if d.is_dir() and len(d.name) == 15], reverse=True)
        app.logger.info(f"Filtered run IDs matching pattern: {run_ids}")
     except Exception as e: app.logger.error(f"Error scanning runs directory {runs_dir}: {e}", exc_info=True); return jsonify([])
     return jsonify(run_ids)

# --- MODIFIED /results/<run_id> Endpoint ---
@app.route('/results/<run_id>')
def get_results(run_id):
    """Loads combined results data from JSON files for a specific run ID."""
    run_dir = Path(__file__).parent / "runs" / run_id
    results_dir = run_dir / "results"
    app.logger.info(f"--- Loading results for Run ID: {run_id} from {results_dir} ---")
    if not results_dir.is_dir():
        return jsonify({"error": f"Results directory not found for run ID {run_id}"}), 404

    # Find all result JSON files for this run
    json_files = list(results_dir.glob('backtest_summary_*.json'))
    timeframes_processed = sorted([p.stem.replace('backtest_summary_', '') for p in json_files])
    app.logger.info(f"Found result files for timeframes: {timeframes_processed}")

    if not timeframes_processed:
         return jsonify({"error": f"No result JSON files found in {results_dir}"}), 404

    # Structure to hold all data for the frontend
    results_payload = {
        "run_id": run_id,
        "timeframes": {} # Use timeframe as key
    }

    for timeframe in timeframes_processed:
        result_file = results_dir / f"backtest_summary_{timeframe}.json"
        if result_file.is_file(): # Should always be true here from glob
            try:
                with open(result_file, 'r') as f:
                    # Load the dict {strategy_name: {metrics..., trades_details: [...]}}
                    tf_data = json.load(f)
                    # Reorganize slightly for easier frontend access
                    tf_metrics = {}
                    tf_trades = {}
                    for strat_name, summary in tf_data.items():
                         tf_metrics[strat_name] = {
                             'total_pnl': summary.get('total_pnl'),
                             'trade_count': summary.get('trade_count'),
                             'win_rate': summary.get('win_rate'),
                             # Add any other summary metrics here
                         }
                         tf_trades[strat_name] = summary.get('trades_details', []) # Get list of trades

                    results_payload["timeframes"][timeframe] = {
                        "metrics": tf_metrics,
                        "trades": tf_trades
                    }
                    app.logger.info(f"  Successfully loaded and structured results for {timeframe}.")

            except Exception as e:
                app.logger.error(f"Error loading or processing result file {result_file}: {e}")
                # Include error marker in payload for this timeframe
                results_payload["timeframes"][timeframe] = { "error": f"Failed to load results: {e}"}
        else:
             app.logger.warning(f"Result file {result_file} missing during iteration.")
             results_payload["timeframes"][timeframe] = { "error": "Result file missing."}


    # Return the structured JSON data
    return jsonify(results_payload)

# --- Remove generate_simple_html_report and generate_full_html_report ---
# The frontend will handle HTML generation now

# --- Run Server ---
if __name__ == '__main__':
    app.logger.info("Starting Flask server...")
    app.run(host='127.0.0.1', port=5000, debug=True)