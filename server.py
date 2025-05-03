# server.py (in project root)

import logging
import json
import os
import sys

import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from typing import Dict, List, Optional, Any # Import necessary types
from pathlib import Path
from datetime import datetime
import traceback
import threading # For running backtest in background

# Add app directory to path to allow imports
APP_DIR = Path(__file__).parent / "app"
sys.path.insert(0, str(APP_DIR.parent)) # Add project root to path

# Now import from app package
from app.config import config
from app.data_loader import load_historical_data
from app.indicator_calculator import IndicatorCalculator
from app.run_backtest import run_and_save_backtest # Import the modified function

# --- Flask App Setup ---
app = Flask(__name__, template_folder='.') # Look for templates in root
logging.basicConfig(level=logging.INFO) # Basic logging for Flask
app.logger.setLevel(logging.INFO)

# --- Global Status Tracking (Simple Example) ---
# In a real app, use a proper task queue like Celery or RQ
backtest_status = {"running": False, "run_id": None, "message": "Idle", "error": None}
status_lock = threading.Lock() # To prevent race conditions updating status

# --- Helper Function to Run Pipeline in Background ---
def run_pipeline_thread(run_id, timeframes, raw_data_map, indicator_map, results_map):
    global backtest_status
    try:
        app.logger.info(f"Background thread started for RUN_ID: {run_id}")
        calculator = IndicatorCalculator() # Initialize once

        for tf in timeframes:
            with status_lock:
                 backtest_status["message"] = f"Calculating indicators for {tf}..."
            app.logger.info(f"Calculating indicators for {tf}...")
            raw_path = raw_data_map[tf]
            indicator_path = indicator_map[tf]
            if not raw_path.is_file():
                raise FileNotFoundError(f"Raw data not found for {tf}: {raw_path}")

            raw_df = load_historical_data(raw_path.name) # Pass filename
            df_with_indicators = calculator.calculate_all_indicators(raw_df)
            indicator_path.parent.mkdir(parents=True, exist_ok=True)
            df_with_indicators.to_csv(indicator_path, index=True)
            app.logger.info(f"Indicators saved for {tf} to {indicator_path}")

            with status_lock:
                 backtest_status["message"] = f"Running backtest for {tf}..."
            app.logger.info(f"Running backtest for {tf}...")
            result_json_path = results_map[tf]
            run_and_save_backtest(indicator_path, result_json_path) # This saves the JSON
            app.logger.info(f"Backtest done for {tf}. Results saved.")

        # If loop finishes without error
        with status_lock:
             backtest_status["message"] = f"Run {run_id} completed successfully."
             backtest_status["running"] = False
             backtest_status["error"] = None
        app.logger.info(f"Background thread finished successfully for RUN_ID: {run_id}")

    except Exception as e:
        error_msg = f"Error during background run {run_id}: {e}"
        app.logger.error(error_msg, exc_info=True)
        with status_lock:
             backtest_status["message"] = f"Run {run_id} failed!"
             backtest_status["running"] = False
             backtest_status["error"] = str(e) # Store error message

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML dashboard."""
    # Renders index.html from the same directory as server.py
    return render_template('index.html')

@app.route('/start_backtest', methods=['POST'])
def start_backtest():
    """Starts a new backtest run in a background thread."""
    global backtest_status
    with status_lock:
        if backtest_status["running"]:
            return jsonify({"error": "A backtest is already running."}), 409 # Conflict

        # Get parameters from frontend request
        params = request.json
        app.logger.info(f"Received backtest request with params: {params}")
        # TODO: Add more validation for params if needed

        # Generate Run ID and Dirs
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(__file__).parent / "runs" / run_id
        run_data_dir = run_dir / "data"
        results_dir = run_dir / "results"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_data_dir.mkdir(exist_ok=True)
        results_dir.mkdir(exist_ok=True)

        # Determine timeframes to run (e.g., from params or default to all)
        # For now, let's run all configured timeframes
        timeframes = list(config.RAW_DATA_FILES.keys())

        # Prepare file path mappings to pass to the thread
        raw_data_map = {tf: config.DATA_FOLDER / config.RAW_DATA_FILES[tf] for tf in timeframes}
        indicator_map = {tf: run_data_dir / (Path(config.RAW_DATA_FILES[tf]).stem + "_with_indicators.csv") for tf in timeframes}
        results_map = {tf: results_dir / f"backtest_summary_{tf}.json" for tf in timeframes}

        # Start the pipeline in a background thread
        pipeline_thread = threading.Thread(
            target=run_pipeline_thread,
            args=(run_id, timeframes, raw_data_map, indicator_map, results_map)
        )
        pipeline_thread.start()

        # Update status immediately
        backtest_status["running"] = True
        backtest_status["run_id"] = run_id
        backtest_status["message"] = f"Run {run_id} started..."
        backtest_status["error"] = None

        return jsonify({"message": "Backtest run started.", "run_id": run_id})

@app.route('/status')
def get_status():
    """Returns the current status of the backtest runner."""
    with status_lock:
        return jsonify(backtest_status)

@app.route('/stop_backtest', methods=['POST'])
def stop_backtest():
    """Stops the currently running backtest."""
    global backtest_status
    with status_lock:
        if not backtest_status["running"]:
            return jsonify({"error": "No backtest is currently running."}), 409
        # In a real app, you would implement a way to stop the thread safely
        # For now, just set the status to stopped
        backtest_status["running"] = False
        backtest_status["message"] = "Backtest stopped."
        return jsonify({"message": "Backtest stopped."})

@app.route('/runs')
@app.route('/runs')
# In server.py

@app.route('/runs')
def list_runs():
    """Lists available run IDs by scanning the 'runs' directory."""
    runs_dir = Path(__file__).parent / "runs"
    app.logger.info(f"Scanning for runs in: {runs_dir}")
    run_ids = []
    if not runs_dir.exists():
        app.logger.warning(f"Runs directory does not exist: {runs_dir}")
        return jsonify([])

    try:
        found_items = list(runs_dir.iterdir())
        app.logger.info(f"Found items in runs dir: {[item.name for item in found_items]}")

        # --- CORRECTED Filter Logic using List Comprehension ---
        run_ids = sorted(
            [
                d.name for d in found_items
                if d.is_dir() and len(d.name) == 15 # Check if it's a directory AND has length 15
            ],
            reverse=True
        )
        # --- End Corrected Filter ---

        app.logger.info(f"Filtered run IDs matching pattern: {run_ids}") # Log the final list
    except Exception as e:
        app.logger.error(f"Error scanning runs directory {runs_dir}: {e}", exc_info=True)
        return jsonify([]) # Return empty on error

    return jsonify(run_ids)



@app.route('/results/<run_id>')
def get_results(run_id):
    """Loads and returns the combined results for a specific run ID."""
    run_dir = Path(__file__).parent / "runs" / run_id
    results_dir = run_dir / "results"
    if not results_dir.is_dir():
        return jsonify({"error": f"Results not found for run ID {run_id}"}), 404

    combined_results = {}
    timeframes_processed = sorted([p.stem.replace('backtest_summary_', '') for p in results_dir.glob('backtest_summary_*.json')])

    if not timeframes_processed:
         return jsonify({"error": f"No result files found in {results_dir}"}), 404

    all_metrics_for_run = {}
    # TODO: Add structure for plot URIs if plots were saved by backtester
    # all_plot_uris_for_run = {}
    # all_best_strategies_for_run = {} # Requires logic to determine best

    for timeframe in timeframes_processed:
        result_file = results_dir / f"backtest_summary_{timeframe}.json"
        if result_file.is_file():
            try:
                with open(result_file, 'r') as f:
                    # Load metrics per strategy for this timeframe
                    metrics_data = json.load(f)
                    # Convert to DataFrame for consistency with reporting function input type
                    metrics_df = pd.DataFrame(metrics_data).T
                    all_metrics_for_run[timeframe] = metrics_df
                    # Placeholder for plots and best strategy determination
                    # all_plot_uris_for_run[timeframe] = {}
                    # all_best_strategies_for_run[timeframe] = None
            except Exception as e:
                app.logger.error(f"Error loading result file {result_file}: {e}")
                # Optionally include error marker in results
                all_metrics_for_run[timeframe] = pd.DataFrame() # Empty DF on error

    # --- Generate Consolidated HTML Report Content ---
    # We need the actual reporting function. Let's assume it's available.
    # If generate_consolidated_html_report was part of a class, instantiate it.
    # For simplicity, let's adapt it slightly to be callable here or import it.
    # Ideally, refactor generate_consolidated_html_report into a separate module.

    # Example: Directly calling the function (requires adapting its imports/structure)
    try:
         # Assuming generate_consolidated_html_report is refactored or available
         # We won't save to file, but get the HTML string
         # Need a dummy Path object for the function signature if it expects one
         # Also need plot/best strategy data if the function requires it
         # Simplified: Generate HTML table part only for now
         report_html = generate_simple_html_report(all_metrics_for_run)
         return jsonify({"status": "complete", "html_report": report_html})
    except Exception as e:
         app.logger.error(f"Error generating report for run {run_id}: {e}", exc_info=True)
         return jsonify({"error": f"Could not generate report content."}), 500


# --- Simple HTML Generation (Replace with your full function later) ---
def generate_simple_html_report(all_metrics: Dict[str, pd.DataFrame]) -> str:
     """Generates basic HTML tables for metrics per timeframe."""
     html_content = ""
     timeframes = list(all_metrics.keys())
     if not timeframes: return "<p>No metrics data available.</p>"

     html_content += "<div class='tab-buttons'>"
     for i, tf in enumerate(timeframes):
         active_class = 'active' if i == 0 else ''
         html_content += f'<button class="tab-button {active_class}" onclick="openTab(event, \'{tf}\')">{tf}</button>'
     html_content += "</div>"

     for i, tf in enumerate(timeframes):
         active_class = 'active' if i == 0 else ''
         html_content += f'<div id="{tf}" class="tab-content {active_class}">\n<h2>Results for Timeframe: {tf}</h2>\n'
         metrics_df = all_metrics.get(tf)
         if metrics_df is not None and not metrics_df.empty:
             # Basic formatting
             metrics_to_format = {
                 'total_pnl': '{:,.2f}', 'win_rate': '{:.2f}%'
             }
             metrics_display_df = metrics_df.copy()
             for col, fmt in metrics_to_format.items():
                 if col in metrics_display_df.columns:
                     metrics_display_df[col] = metrics_display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
             if 'trade_count' in metrics_display_df.columns:
                     metrics_display_df['trade_count'] = metrics_display_df['trade_count'].astype(int)

             html_content += "<h3>Performance Summary</h3><div class=\"metric-card\">"
             html_content += metrics_display_df.to_html(classes='performance-table', border=1, justify='right')
             html_content += '</div>'
         else:
             html_content += "<p>No metrics data available for this timeframe.</p>"
         html_content += "</div>\n" # Close tab-content div

     # Add the JavaScript for tabs (simplified from your provided HTML)
     html_script = """
     <script>
         function openTab(evt, tabName) {
             var i, tabcontent, tablinks;
             tabcontent = document.getElementsByClassName("tab-content");
             for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; tabcontent[i].classList.remove("active"); }
             tablinks = document.getElementsByClassName("tab-button");
             for (i = 0; i < tablinks.length; i++) { tablinks[i].classList.remove("active"); }
             var currentTab = document.getElementById(tabName);
             if (currentTab) { currentTab.style.display = "block"; currentTab.classList.add("active"); }
             if (evt && evt.currentTarget) { evt.currentTarget.classList.add("active"); }
         }
         // Initialize first tab
         document.addEventListener('DOMContentLoaded', function() {
             var firstButton = document.querySelector('.tab-buttons button');
             if (firstButton) { firstButton.click(); }
         });
     </script>
     """
     return html_content + html_script


# --- Run Server ---
if __name__ == '__main__':
    app.logger.info("Starting Flask server...")
    # Use host='0.0.0.0' to make it accessible on your network, or '127.0.0.1' for local only
    # Use debug=True only for development, set to False for production
    app.run(host='127.0.0.1', port=5000, debug=True)