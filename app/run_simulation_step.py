# app/run_simulation_step.py

import logging
import argparse
from pathlib import Path
import pandas as pd
import json
import sys

# Use absolute imports with new names
from app.config import config
from app.simulation_engine import SimpleBacktester # Use renamed engine class
from app.agentic_core import RuleBasedAgent

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
# Keep logger level at INFO or WARNING for this script,
# the detailed logs will go to the simulation trace file
logger.setLevel(logging.INFO)


# --- FUNCTION TO RUN AND SAVE ---
# --- MODIFIED: Accept log_dir ---
# def run_and_save_agent_backtest(indicator_file_path: Path, output_json_path: Path, log_dir: Path):
#     """Loads indicator data, runs the agent-driven backtest, saves results and detailed log."""
#     logger.info(f"--- Starting Simulation Step: {indicator_file_path} -> {output_json_path} ---")
#     logger.info(f"Detailed simulation log directory: {log_dir}")

#     if not indicator_file_path.is_file(): raise FileNotFoundError(f"{indicator_file_path}")
#     if not log_dir.is_dir(): # Ensure log dir exists
#          logger.warning(f"Log directory {log_dir} not found. Attempting to create.")
#          log_dir.mkdir(parents=True, exist_ok=True)

#     try:
#         # 1. Load Data
#         logger.info(f"Loading data with indicators from {indicator_file_path}...")
#         data_with_indicators = pd.read_csv(indicator_file_path, index_col=0, parse_dates=True)
#         if not isinstance(data_with_indicators.index, pd.DatetimeIndex): raise ValueError("Index not DatetimeIndex.")
#         if 'regime' not in data_with_indicators.columns: raise ValueError("Missing 'regime' column.")
#         logger.info(f"Loaded {len(data_with_indicators)} rows. Columns: {data_with_indicators.columns.tolist()}")
#         if data_with_indicators.empty: raise ValueError("Indicator data file empty.")

#         # 2. Initialize Agent and Backtester (Simulation Engine)
#         logger.info("Initializing Agentic Core...")
#         agent = RuleBasedAgent()
#         logger.info("Initializing Simulation Engine with Agent...")
#         simulator = SimpleBacktester(agent=agent)

#         # 3. Run Simulation (Pass log_dir and timeframe)
#         logger.info("Running agent simulation...")
#         timeframe = indicator_file_path.stem.replace('_with_indicators','').split('_')[-1] # Extract timeframe string
#         results_dict = simulator.simulate_agent_run(data_with_indicators, log_dir=log_dir, timeframe=timeframe) # Pass log dir

#         if not results_dict: raise ValueError("Agent simulation did not return results.")
#         if results_dict.get("error"): raise ValueError(f"Simulation failed: {results_dict['error']}") # Check for error key

#         # 4. Prepare Summary for Saving (remains same)
#         summary_to_save = { "RuleBasedAgent": { ... } } # Keep summary prep logic
#         summary_to_save["RuleBasedAgent"] = {
#              'total_pnl': results_dict.get('total_pnl'),
#              'trade_count': results_dict.get('trade_count'),
#              'win_rate': results_dict.get('win_rate'),
#              'trades_details': results_dict.get('trades_summary_list', [])
#          }
#         # Log final structure before save
#         logger.debug(f"Structure saving to {output_json_path.name}: {summary_to_save}")

#         # 5. Save Summary to JSON (remains same)
#         output_json_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(output_json_path, 'w') as f: json.dump(summary_to_save, f, indent=4, default=str)
#         logger.info(f"Successfully ran agent simulation and saved summary to {output_json_path}")

#         # Log summary (remains same)
#         agent_summary = summary_to_save.get("RuleBasedAgent", {}); pnl=agent_summary.get('total_pnl','N/A'); trades=agent_summary.get('trade_count','N/A'); win_rate=agent_summary.get('win_rate','N/A')
#         #pnl_str=f"{pnl:.2f}" if isinstance(pnl,(int,float)) else pnl; 
#         # wr_str=f"{win_rate:.2f}%" if isinstance(win_rate,(int,float)) else wr_str = win_rate if isinstance(win_rate, str) else 'N/A'
#         pnl_str="pnl={pnl}; print(f'{pnl:.2f}' if isinstance(pnl, (int, float)) else pnl)"
#         wr_str="win_rate={win_rate}; print(f'{win_rate:.2f}%' if isinstance(win_rate, (int, float)) else win_rate if isinstance(win_rate, str) else 'N/A')"
#         logger.info(f"Agent Run Summary: PnL={pnl_str}, Trades={trades}, Win Rate={wr_str}")

#         return summary_to_save

#     except Exception as e:
#         logger.error(f"Error during simulation step: {e}", exc_info=True)
#         raise
# app/run_simulation_step.py

import logging
import argparse
from pathlib import Path
import pandas as pd
import json
import sys

# Use absolute imports with new names
from app.config import config
from app.simulation_engine import SimpleBacktester
from app.agentic_core import RuleBasedAgent

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set back to INFO for this script, DEBUG is in the engine's log

# --- FUNCTION TO RUN AND SAVE (MODIFIED CALL TO SIMULATOR) ---
def run_and_save_agent_backtest(indicator_file_path: Path, output_json_path: Path, log_dir: Path): # Added log_dir argument
    """Loads indicator data, runs the agent-driven backtest, saves results and detailed log."""
    logger.info(f"--- Starting Simulation Step: {indicator_file_path} -> {output_json_path} ---")
    logger.info(f"Detailed simulation log directory: {log_dir}") # Log received log_dir

    if not indicator_file_path.is_file(): raise FileNotFoundError(f"{indicator_file_path}")
    if not log_dir.is_dir():
        logger.warning(f"Log directory {log_dir} missing. Attempting to create.")
        log_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Load Data
        logger.info(f"Loading data with indicators from {indicator_file_path}...")
        data_with_indicators = pd.read_csv(indicator_file_path, index_col=0, parse_dates=True)
        if not isinstance(data_with_indicators.index, pd.DatetimeIndex): raise ValueError("Index not DatetimeIndex.")
        if 'regime' not in data_with_indicators.columns: raise ValueError("Missing 'regime' column.")
        logger.info(f"Loaded {len(data_with_indicators)} rows. Columns: {data_with_indicators.columns.tolist()}")
        if data_with_indicators.empty: raise ValueError("Indicator data file empty.")

        # 2. Initialize Agent and Simulator
        logger.info("Initializing Agentic Core...")
        agent = RuleBasedAgent()
        logger.info("Initializing Simulation Engine with Agent...")
        simulator = SimpleBacktester(agent=agent)

        # 3. Run Simulation (Pass log_dir and timeframe)
        logger.info("Running agent simulation...")
        # --- Extract timeframe from filename ---
        timeframe = indicator_file_path.stem.replace('_with_indicators','').split('_')[-1]
        if not timeframe: # Basic fallback if parsing fails
            timeframe = "unknown"
            logger.warning("Could not determine timeframe from filename for logging.")
        # --- END Extract ---

        # --- MODIFIED CALL: Pass log_dir and timeframe ---
        results_dict = simulator.simulate_agent_run(
            df=data_with_indicators,
            log_dir=log_dir,
            timeframe=timeframe
        )
        # --- END MODIFIED CALL ---

        if not results_dict: raise ValueError("Agent simulation did not return results.")
        if results_dict.get("error"): raise ValueError(f"Simulation failed: {results_dict['error']}")

        # 4. Prepare Summary for Saving (remains same)
        summary_to_save = { "RuleBasedAgent": { ... } } # Keep summary prep logic
        summary_to_save["RuleBasedAgent"] = {
             'total_pnl': results_dict.get('total_pnl'),
             'trade_count': results_dict.get('trade_count'),
             'win_rate': results_dict.get('win_rate'),
             'trades_details': results_dict.get('trades_summary_list', [])
         }
        logger.debug(f"Structure saving to {output_json_path.name}: {summary_to_save}")

        # 5. Save Summary to JSON (remains same)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f: json.dump(summary_to_save, f, indent=4, default=str)
        logger.info(f"Successfully ran agent simulation and saved summary to {output_json_path}")

        # Log summary (remains same)
        # ... (logging summary PnL, Trades, Win Rate) ...
        agent_summary = summary_to_save.get("RuleBasedAgent", {}); pnl=agent_summary.get('total_pnl','N/A'); trades=agent_summary.get('trade_count','N/A'); win_rate=agent_summary.get('win_rate','N/A')
        pnl_str=f"{pnl:.2f}" if isinstance(pnl,(int,float)) else pnl; wr_str=f"{win_rate:.2f}%" if isinstance(win_rate,(int,float)) else str(win_rate)
        logger.info(f"Agent Run Summary: PnL={pnl_str}, Trades={trades}, Win Rate={wr_str}")


        return summary_to_save

    except Exception as e:
        logger.error(f"Error during simulation step: {e}", exc_info=True)
        raise

# --- Main block (Accepts --log-dir) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run agent simulation step using indicator data.')
    parser.add_argument('--input', type=str, required=True, help='Input CSV with indicators+regime')
    parser.add_argument('--output-json', type=str, required=True, help='Output JSON for results')
    # --- Argument for log directory ---
    parser.add_argument('--log-dir', type=str, required=True, help='Directory to save detailed simulation log')
    args = parser.parse_args()

    indicator_path = Path(args.input)
    output_path = Path(args.output_json)
    log_dir_path = Path(args.log_dir) # Get log dir path

    try:
        # Pass log_dir_path to the function
        run_and_save_agent_backtest(indicator_path, output_path, log_dir_path)
    except Exception:
         sys.exit(1)

    parser = argparse.ArgumentParser(description='Run agent simulation step using indicator data.')
    parser.add_argument('--input', type=str, required=True, help='Input CSV with indicators+regime')
    parser.add_argument('--output-json', type=str, required=True, help='Output JSON for results')
    # --- ADDED: Argument for log directory ---
    parser.add_argument('--log-dir', type=str, required=True, help='Directory to save detailed simulation log')
    args = parser.parse_args()

    indicator_path = Path(args.input)
    output_path = Path(args.output_json)
    log_dir_path = Path(args.log_dir) # Get log dir path

    try:
        # Pass log_dir_path to the function
        run_and_save_agent_backtest(indicator_path, output_path, log_dir_path)
    except Exception:
         sys.exit(1)