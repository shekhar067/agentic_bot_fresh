
# # # # app/run_simulation_step.py

# # # import logging
# # # import argparse
# # # from pathlib import Path
# # # import pandas as pd
# # # import json
# # # import sys

# # # # Use absolute imports
# # # from app.config import config
# # # from app.simulation_engine import SimpleBacktester
# # # from app.agentic_core import RuleBasedAgent

# # # logging.basicConfig(level=logging.INFO,
# # #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# # #                     handlers=[logging.StreamHandler()])
# # # logger = logging.getLogger(__name__)
# # # logger.setLevel(logging.INFO)

# # # def run_and_save_agent_backtest(indicator_file_path: Path, output_json_path: Path, log_dir: Path):
# # #     logger.info(f"--- Starting Simulation Step: {indicator_file_path} -> {output_json_path} ---")
# # #     logger.info(f"Detailed simulation log directory: {log_dir}")

# # #     if not indicator_file_path.is_file():
# # #         raise FileNotFoundError(f"{indicator_file_path}")
# # #     if not log_dir.is_dir():
# # #         logger.warning(f"Log directory {log_dir} missing. Attempting to create.")
# # #         log_dir.mkdir(parents=True, exist_ok=True)

# # #     try:
# # #         logger.info(f"Loading data with indicators from {indicator_file_path}...")
# # #         data_with_indicators = pd.read_csv(indicator_file_path, index_col=0, parse_dates=True)
# # #         if not isinstance(data_with_indicators.index, pd.DatetimeIndex):
# # #             raise ValueError("Index not DatetimeIndex.")
# # #         if 'regime' not in data_with_indicators.columns:
# # #             raise ValueError("Missing 'regime' column.")
# # #         logger.info(f"Loaded {len(data_with_indicators)} rows. Columns: {data_with_indicators.columns.tolist()}")
# # #         if data_with_indicators.empty:
# # #             raise ValueError("Indicator data file empty.")

# # #         agent = RuleBasedAgent()
# # #         simulator = SimpleBacktester(agent=agent)

# # #         timeframe = indicator_file_path.stem.replace('_with_indicators','').split('_')[-1]
# # #         if not timeframe:
# # #             timeframe = "unknown"
# # #             logger.warning("Could not determine timeframe from filename for logging.")

# # #         results_dict = simulator.simulate_agent_run(
# # #             df=data_with_indicators,
# # #             log_dir=log_dir,
# # #             timeframe=timeframe
# # #         )

# # #         if not results_dict:
# # #             raise ValueError("Agent simulation did not return results.")
# # #         if results_dict.get("error"):
# # #             raise ValueError(f"Simulation failed: {results_dict['error']}")

# # #         summary_to_save = {
# # #             "RuleBasedAgent": {
# # #                 'total_pnl': results_dict.get('total_pnl'),
# # #                 'trade_count': results_dict.get('trade_count'),
# # #                 'win_rate': results_dict.get('win_rate'),
# # #                 'trades_details': results_dict.get('trades_details', [])
# # #             }
# # #         }

# # #         output_json_path.parent.mkdir(parents=True, exist_ok=True)
# # #         with open(output_json_path, 'w') as f:
# # #             json.dump(summary_to_save, f, indent=4, default=str)
# # #         logger.info(f"Successfully ran agent simulation and saved summary to {output_json_path}")

# # #         agent_summary = summary_to_save.get("RuleBasedAgent", {})
# # #         pnl = agent_summary.get('total_pnl','N/A')
# # #         trades = agent_summary.get('trade_count','N/A')
# # #         win_rate = agent_summary.get('win_rate','N/A')
# # #         pnl_str = f"{pnl:.2f}" if isinstance(pnl, (int, float)) else str(pnl)
# # #         wr_str = f"{win_rate:.2f}%" if isinstance(win_rate, (int, float)) else str(win_rate)
# # #         logger.info(f"Agent Run Summary: PnL={pnl_str}, Trades={trades}, Win Rate={wr_str}")

# # #         return summary_to_save

# # #     except Exception as e:
# # #         logger.error(f"Error during simulation step: {e}", exc_info=True)
# # #         raise

# # # if __name__ == "__main__":
# # #     parser = argparse.ArgumentParser(description='Run agent simulation step using indicator data.')
# # #     parser.add_argument('--input', type=str, required=True, help='Input CSV with indicators+regime')
# # #     parser.add_argument('--output-json', type=str, required=True, help='Output JSON for results')
# # #     parser.add_argument('--log-dir', type=str, required=True, help='Directory to save detailed simulation log')
# # #     args = parser.parse_args()

# # #     indicator_path = Path(args.input)
# # #     output_path = Path(args.output_json)
# # #     log_dir_path = Path(args.log_dir)

# # #     try:
# # #         run_and_save_agent_backtest(indicator_path, output_path, log_dir_path)
# # #     except Exception:
# # #         sys.exit(1)
# # # --- run_simulation_step.py modifications ---

# # # import logging
# # # import argparse
# # # from pathlib import Path
# # # import pandas as pd
# # # import json
# # # import sys
# # # from typing import Optional # Added Optional

# # # # Use absolute imports
# # # from app.config import config
# # # from app.simulation_engine import SimpleBacktester
# # # from app.agentic_core import RuleBasedAgent
# # # from app.strategies import strategy_functions # Added import

# # # logging.basicConfig(level=logging.INFO,
# # #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# # #                     handlers=[logging.StreamHandler()])
# # # logger = logging.getLogger(__name__)
# # # logger.setLevel(logging.INFO)

# # # # --- MODIFIED function signature and logic ---
# # # def run_simulation_task(indicator_file_path: Path,
# # #                           output_json_path: Path,
# # #                           log_dir: Path,
# # #                           strategy_name: Optional[str] = None): # Added strategy_name

# # #     mode = "Agent" if strategy_name is None else f"Strategy: {strategy_name}"
# # #     logger.info(f"--- Starting Simulation Step ({mode}) ---")
# # #     logger.info(f"Input: {indicator_file_path}, Output: {output_json_path}")
# # #     logger.info(f"Detailed simulation log directory: {log_dir}")

# # #     if not indicator_file_path.is_file():
# # #         raise FileNotFoundError(f"{indicator_file_path}")
# # #     # Ensure log dir exists (already done in your code)

# # #     try:
# # #         logger.info(f"Loading data with indicators from {indicator_file_path}...")
# # #         data_with_indicators = pd.read_csv(indicator_file_path, index_col=0, parse_dates=True)
# # #         # ... (data loading checks remain same) ...
# # #         # Regime column check might be skipped if running single strategy
# # #         # if 'regime' not in data_with_indicators.columns and strategy_name is None:
# # #         #     raise ValueError("Missing 'regime' column (needed for Agent mode).")

# # #         # --- MODIFIED: Initialize Simulator based on mode ---
# # #         simulator = None
# # #         results_key = None # Key to use when saving results

# # #         if strategy_name:
# # #             # Single Strategy Mode
# # #             logger.info(f"Running in SINGLE STRATEGY mode for: {strategy_name}")
# # #             strategy_func = strategy_functions.get(strategy_name)
# # #             if strategy_func is None:
# # #                 raise ValueError(f"Strategy function '{strategy_name}' not found in strategy_functions.")
# # #             # Pass strategy_func and strategy_name to SimpleBacktester
# # #             simulator = SimpleBacktester(strategy_func=strategy_func, strategy_name=strategy_name)
# # #             results_key = strategy_name # Use strategy name as the key in JSON output
# # #         else:
# # #             # Agent Mode (Original behavior)
# # #             logger.info("Running in AGENT mode.")
# # #             if 'regime' not in data_with_indicators.columns:
# # #                  raise ValueError("Missing 'regime' column (needed for Agent mode).")

# # #             agent = RuleBasedAgent() # Requires regime column
# # #             simulator = SimpleBacktester(agent=agent, strategy_name="Agent") # Pass agent and name
# # #             results_key = "RuleBasedAgent" # Original key

# # #         # --- Run Simulation ---
# # #         timeframe = indicator_file_path.stem.replace('_with_indicators','').split('_')[-1] or "unknown"
# # #         logger.info(f"Running simulation for timeframe: {timeframe}...")

# # #         # Call the renamed run_simulation method
# # #         results_dict = simulator.run_simulation(
# # #             df=data_with_indicators,
# # #             log_dir=log_dir,
# # #             timeframe=timeframe # Timeframe string is used in log file name setup
# # #         )

# # #         # ... (results checks remain same) ...
# # #         if not results_dict: raise ValueError("Simulation did not return results.")
# # #         if results_dict.get("error"): raise ValueError(f"Simulation failed: {results_dict['error']}")

# # #         # --- MODIFIED: Prepare Summary for Saving ---
# # #         # Use the determined results_key
# # #         summary_to_save = {
# # #             results_key: { # Use dynamic key
# # #                 'total_pnl': results_dict.get('total_pnl'),
# # #                 'trade_count': results_dict.get('trade_count'),
# # #                 'win_rate': results_dict.get('win_rate'),
# # #                 'trades_details': results_dict.get('trades_details', []) # Use correct key from simulator
# # #                 # Add exit reasons if simulator returns them and you want them in JSON
# # #                 # 'exit_reasons': results_dict.get('exit_reasons')
# # #             }
# # #         }
# # #         logger.debug(f"Structure saving to {output_json_path.name}: {summary_to_save}")

# # #         # --- Save Summary to JSON (remains same) ---
# # #         output_json_path.parent.mkdir(parents=True, exist_ok=True)
# # #         with open(output_json_path, 'w') as f:
# # #             json.dump(summary_to_save, f, indent=4, default=str) # Use default=str for Timestamps etc.
# # #         logger.info(f"Successfully ran simulation and saved summary to {output_json_path}")

# # #         # Log summary (using the dynamic key)
# # #         run_summary = summary_to_save.get(results_key, {})
# # #         pnl = run_summary.get('total_pnl','N/A')
# # #         trades = run_summary.get('trade_count','N/A')
# # #         win_rate = run_summary.get('win_rate','N/A')
# # #         pnl_str = f"{pnl:.2f}" if isinstance(pnl, (int, float)) else str(pnl)
# # #         wr_str = f"{win_rate:.2f}%" if isinstance(win_rate, (int, float)) else str(win_rate)
# # #         logger.info(f"Run Summary ({results_key}): PnL={pnl_str}, Trades={trades}, Win Rate={wr_str}")

# # #         return summary_to_save

# # #     except Exception as e:
# # #         logger.error(f"Error during simulation step ({mode}): {e}", exc_info=True)
# # #         raise

# # # if __name__ == "__main__":
# # #     parser = argparse.ArgumentParser(description='Run simulation step using indicator data.')
# # #     parser.add_argument('--input', type=str, required=True, help='Input CSV with indicators')
# # #     parser.add_argument('--output-json', type=str, required=True, help='Output JSON for results')
# # #     parser.add_argument('--log-dir', type=str, required=True, help='Directory to save detailed simulation log')
# # #     # --- ADDED: Optional strategy name ---
# # #     parser.add_argument('--strategy-name', type=str, required=False, default=None,
# # #                         help='Run specific strategy independently (e.g., "EMA_Crossover"). If omitted, runs RuleBasedAgent.')
# # #     args = parser.parse_args()

# # #     indicator_path = Path(args.input)
# # #     output_path = Path(args.output_json)
# # #     log_dir_path = Path(args.log_dir)

# # #     try:
# # #         # Pass strategy_name to the function
# # #         run_simulation_task(indicator_path, output_path, log_dir_path, args.strategy_name)
# # #     except Exception:
# # #          sys.exit(1)

# # # # --- End of run_simulation_step.py modifications ---

# # # app/run_simulation_step.py

# # import logging
# # import argparse
# # from pathlib import Path
# # import pandas as pd
# # import json
# # import sys
# # from typing import Optional # Added Optional

# # # Use absolute imports
# # from app.config import config
# # from app.simulation_engine import SimpleBacktester
# # from app.agentic_core import RuleBasedAgent # Keep agent import for agent mode
# # from app.strategies import strategy_functions # Added import

# # logging.basicConfig(level=logging.INFO,
# #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# #                     handlers=[logging.StreamHandler()])
# # # Configure logger for this script specifically
# # logger = logging.getLogger("RunSimStep")
# # logger.setLevel(logging.INFO) # Keep this high level, details are in simulation log


# # # --- RENAMED main function ---
# # def run_simulation_task(indicator_file_path: Path,
# #                           output_json_path: Path,
# #                           log_dir: Path,
# #                           strategy_name: Optional[str] = None): # Added strategy_name arg

# #     mode = "Agent" if strategy_name is None else f"Strategy: {strategy_name}"
# #     logger.info(f"--- Starting Simulation Step ({mode}) ---")
# #     logger.info(f"Input: {indicator_file_path.name}, Output: {output_json_path.name}")
# #     # Detailed log dir is passed to simulator, no need to log it here again maybe

# #     if not indicator_file_path.is_file():
# #         raise FileNotFoundError(f"Indicator file not found: {indicator_file_path}")
# #     if not log_dir.is_dir():
# #         logger.warning(f"Log directory {log_dir} missing. Attempting to create.")
# #         log_dir.mkdir(parents=True, exist_ok=True)

# #     try:
# #         logger.info(f"Loading data with indicators from {indicator_file_path}...")
# #         data_with_indicators = pd.read_csv(indicator_file_path, index_col=0, parse_dates=True)
# #         if not isinstance(data_with_indicators.index, pd.DatetimeIndex):
# #             raise ValueError("Index is not DatetimeIndex.")
# #         logger.info(f"Loaded {len(data_with_indicators)} rows. Columns: {data_with_indicators.columns.tolist()}")
# #         if data_with_indicators.empty:
# #             raise ValueError("Indicator data file is empty.")

# #         # --- Initialize Simulator based on mode ---
# #         simulator = None
# #         results_key = None # Key to use when saving results

# #         if strategy_name:
# #             # Single Strategy Mode
# #             logger.info(f"Running in SINGLE STRATEGY mode for: {strategy_name}")
# #             strategy_func = strategy_functions.get(strategy_name)
# #             if strategy_func is None:
# #                 raise ValueError(f"Strategy function '{strategy_name}' not found in app.strategies.strategy_functions.")
# #             # Pass strategy_func and strategy_name to SimpleBacktester
# #             simulator = SimpleBacktester(strategy_func=strategy_func, strategy_name=strategy_name)
# #             results_key = strategy_name # Use strategy name as the key in JSON output
# #         else:
# #             # Agent Mode (Original behavior)
# #             logger.info("Running in AGENT mode.")
# #             # Agent mode requires 'regime' column
# #             if 'regime' not in data_with_indicators.columns:
# #                  raise ValueError("Missing 'regime' column required for Agent mode.")

# #             agent = RuleBasedAgent() # Requires regime column
# #             # --- IMPORTANT: Modify agentic_core.py agent.decide to return tsl_mult ---
# #             # --- If not modified yet, TSL will use default from config via simulator ---
# #             simulator = SimpleBacktester(agent=agent, strategy_name="Agent") # Pass agent and name
# #             results_key = "RuleBasedAgent" # Original key

# #         # --- Run Simulation ---
# #         # Extract timeframe string from filename
# #         timeframe = indicator_file_path.stem.split('_')[-1]
# #         if not timeframe or not timeframe.replace('min','').isdigit(): # Basic check
# #              timeframe = indicator_file_path.stem # Fallback to stem if parsing fails
# #              logger.warning(f"Could not reliably determine timeframe from filename '{indicator_file_path.name}'. Using '{timeframe}'.")

# #         logger.info(f"Running simulation for timeframe: {timeframe}...")

# #         # Call the renamed run_simulation method
# #         results_dict = simulator.run_simulation(
# #             df=data_with_indicators,
# #             log_dir=log_dir, # Pass dir where simulator creates its detailed log
# #             timeframe=timeframe
# #         )

# #         # --- Results Handling ---
# #         if not results_dict:
# #             raise ValueError("Simulation did not return results.")
# #         if results_dict.get("error"):
# #             raise ValueError(f"Simulation failed: {results_dict['error']}")

# #         # --- Prepare Summary for Saving ---
# #         # Use the determined results_key
# #         summary_to_save = {
# #             results_key: { # Use dynamic key
# #                 'total_pnl': results_dict.get('total_pnl'),
# #                 'trade_count': results_dict.get('trade_count'),
# #                 'win_rate': results_dict.get('win_rate'),
# #                 'exit_reasons': results_dict.get('exit_reasons'), # Include exit counts
# #                 'trades_details': results_dict.get('trades_details', []) # Use correct key 'trades_details'
# #             }
# #         }
# #         logger.debug(f"Structure saving to {output_json_path.name}: {summary_to_save}")

# #         # --- Save Summary to JSON ---
# #         output_json_path.parent.mkdir(parents=True, exist_ok=True)
# #         with open(output_json_path, 'w') as f:
# #             # Use default=str to handle potential non-serializable types like Timestamps if they slip through
# #             json.dump(summary_to_save, f, indent=4, default=str)
# #         logger.info(f"Successfully ran simulation and saved summary to {output_json_path}")

# #         # --- Log Summary ---
# #         run_summary = summary_to_save.get(results_key, {})
# #         pnl = run_summary.get('total_pnl','N/A')
# #         trades = run_summary.get('trade_count','N/A')
# #         win_rate = run_summary.get('win_rate','N/A')
# #         exit_reasons = run_summary.get('exit_reasons', {})
# #         pnl_str = f"{pnl:.2f}" if isinstance(pnl, (int, float)) else str(pnl)
# #         wr_str = f"{win_rate:.2f}%" if isinstance(win_rate, (int, float)) else str(win_rate)
# #         logger.info(f"Run Summary ({results_key}): PnL={pnl_str}, Trades={trades}, Win Rate={wr_str}")
# #         logger.info(f"Exit Reasons ({results_key}): {exit_reasons}")


# #         return summary_to_save

# #     except Exception as e:
# #         logger.error(f"Error during simulation step ({mode}): {e}", exc_info=True)
# #         # Reraise the exception so pipeline_manager knows it failed
# #         raise

# # # --- Main block ---
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description='Run simulation step using indicator data.')
# #     parser.add_argument('--input', type=str, required=True, help='Input CSV file with indicators')
# #     parser.add_argument('--output-json', type=str, required=True, help='Output JSON file for results summary')
# #     parser.add_argument('--log-dir', type=str, required=True, help='Directory to save detailed simulation trace log')
# #     # --- ADDED: Optional strategy name ---
# #     parser.add_argument('--strategy-name', type=str, required=False, default=None,
# #                         help='If provided, runs the specified strategy independently. Otherwise, runs the RuleBasedAgent.')
# #     args = parser.parse_args()

# #     indicator_path = Path(args.input)
# #     output_path = Path(args.output_json)
# #     log_dir_path = Path(args.log_dir)

# #     try:
# #         # Pass strategy_name to the function
# #         run_simulation_task(indicator_path, output_path, log_dir_path, args.strategy_name)
# #         logger.info(f"Simulation task completed successfully for {args.strategy_name or 'Agent'}.")
# #     except Exception as main_e:
# #          # Log the error before exiting
# #          logger.error(f"Simulation task failed for {args.strategy_name or 'Agent'}. Error: {main_e}", exc_info=True)
# #          sys.exit(1) # Exit with non-zero code on failure

# # app/run_simulation_step.py

# import logging
# import argparse
# from pathlib import Path
# import pandas as pd
# import json
# import sys
# from typing import Optional # Added Optional

# # Use absolute imports
# from app.config import config
# from app.simulation_engine import SimpleBacktester
# from app.agentic_core import RuleBasedAgent # Keep agent import for agent mode
# from app.strategies import default_strategy_functions # Added import

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     handlers=[logging.StreamHandler()])
# # Configure logger for this script specifically
# logger = logging.getLogger("RunSimStep")
# logger.setLevel(logging.INFO) # Keep this high level, details are in simulation log

# import re

# def extract_timeframe_from_filename(filename: str) -> str:
#     """
#     Extracts timeframe like '5min', '15min' etc. from filenames like 'nifty__5min_with_indicators.csv'.
#     Falls back to filename stem if no pattern match.
#     """
#     match = re.search(r'__(\d+min|daily|hourly|weekly|monthly)', filename.lower())
#     if match:
#         return match.group(1)
#     else:
#         logger.warning(f"Could not reliably parse timeframe from filename '{filename}'. Using fallback: {filename}")
#         return filename.replace(".csv", "")

# # --- RENAMED main function ---
# def run_simulation_task(indicator_file_path: Path,
#                           output_json_path: Path,
#                           log_dir: Path,
#                           strategy_name: Optional[str] = None): # Added strategy_name arg

#     mode = "Agent" if strategy_name is None else f"Strategy: {strategy_name}"
#     logger.info(f"--- Starting Simulation Step ({mode}) ---")
#     logger.info(f"Input: {indicator_file_path.name}, Output: {output_json_path.name}")
#     # Detailed log dir is passed to simulator, no need to log it here again maybe

#     if not indicator_file_path.is_file():
#         raise FileNotFoundError(f"Indicator file not found: {indicator_file_path}")
#     if not log_dir.is_dir():
#         logger.warning(f"Log directory {log_dir} missing. Attempting to create.")
#         log_dir.mkdir(parents=True, exist_ok=True)

#     try:
#         logger.info(f"Loading data with indicators from {indicator_file_path}...")
#         data_with_indicators = pd.read_csv(indicator_file_path, index_col=0, parse_dates=True)
#         if not isinstance(data_with_indicators.index, pd.DatetimeIndex):
#             raise ValueError("Index is not DatetimeIndex.")
#         logger.info(f"Loaded {len(data_with_indicators)} rows. Columns: {data_with_indicators.columns.tolist()}")
#         if data_with_indicators.empty:
#             raise ValueError("Indicator data file is empty.")

#         # --- Initialize Simulator based on mode ---
#         simulator: Optional[SimpleBacktester] = None # Type hint
#         results_key = None # Key to use when saving results

#         if strategy_name:
#             # Single Strategy Mode
#             logger.info(f"Running in SINGLE STRATEGY mode for: {strategy_name}")
#             strategy_func = default_strategy_functions.get(strategy_name)
#             if strategy_func is None:
#                 raise ValueError(f"Strategy function '{strategy_name}' not found in app.strategies.strategy_functions.")
#             # Pass strategy_func and strategy_name to SimpleBacktester
#             simulator = SimpleBacktester(strategy_func=strategy_func, strategy_name=strategy_name)
#             results_key = strategy_name # Use strategy name as the key in JSON output
#         else:
#             # Agent Mode (Original behavior)
#             logger.info("Running in AGENT mode.")
#             # Agent mode requires 'regime' column
#             if 'regime' not in data_with_indicators.columns:
#                  raise ValueError("Missing 'regime' column required for Agent mode.")

#             agent = RuleBasedAgent() # Requires regime column
#             # --- IMPORTANT: Modify agentic_core.py agent.decide to return tsl_mult ---
#             # --- If agent isn't modified yet, TSL used by simulator will be the fallback ---
#             simulator = SimpleBacktester(agent=agent, strategy_name="Agent") # Pass agent and name
#             results_key = "RuleBasedAgent" # Original key

#         if simulator is None: # Should not happen based on logic, but safety check
#              raise RuntimeError("Simulator could not be initialized.")

#         # --- Run Simulation ---
#         # Extract timeframe string from filename more robustly
#         # try:
#         #      # Assumes filename format like '..._data_5min_with_indicators.csv'
#         #      timeframe = indicator_file_path.stem.split('_')[-2] # Get second to last part
#         #      if not timeframe.lower().endswith('min'): raise ValueError("Parsed part is not timeframe")
#         # except (IndexError, ValueError):
#         #      timeframe = indicator_file_path.stem # Fallback
#         #      logger.warning(f"Could not reliably parse timeframe from filename '{indicator_file_path.name}'. Using '{timeframe}'.")
#         # Robust timeframe parsing from filename
#         timeframe = extract_timeframe_from_filename(indicator_file_path.name)
#         logger.info(f"Running simulation for timeframe: {timeframe}...")

#        # logger.info(f"Running simulation for timeframe: {timeframe}...")

#         # Call the renamed run_simulation method
#         results_dict = simulator.run_simulation(
#             df=data_with_indicators.copy(), # Pass a copy to avoid modifying original df in simulator
#             log_dir=log_dir, # Dir where simulator creates its detailed log
#             timeframe=timeframe
#         )

#         # --- Results Handling ---
#         if not results_dict:
#             raise ValueError("Simulation did not return results.")
#         if results_dict.get("error"):
#             raise ValueError(f"Simulation failed: {results_dict['error']}")

#         # --- Prepare Summary for Saving ---
#         # Use the determined results_key
#         summary_to_save = {
#             results_key: { # Use dynamic key
#                 'total_pnl': results_dict.get('total_pnl'),
#                 'trade_count': results_dict.get('trade_count'),
#                 'win_rate': results_dict.get('win_rate'),
#                 'exit_reasons': results_dict.get('exit_reasons'), # Include exit counts
#                 'trades_details': results_dict.get('trades_details', [])
#             }
#         }
#         logger.debug(f"Structure saving to {output_json_path.name}: {summary_to_save}")

#         # --- Save Summary to JSON ---
#         output_json_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(output_json_path, 'w') as f:
#             # Use default=str to handle potential non-serializable types like Timestamps if they slip through
#             json.dump(summary_to_save, f, indent=4, default=str)
#         logger.info(f"Successfully ran simulation and saved summary to {output_json_path}")

#         # --- Log Summary ---
#         run_summary = summary_to_save.get(results_key, {})
#         pnl = run_summary.get('total_pnl','N/A')
#         trades = run_summary.get('trade_count','N/A')
#         win_rate = run_summary.get('win_rate','N/A')
#         exit_reasons = run_summary.get('exit_reasons', {})
#         pnl_str = f"{pnl:.2f}" if isinstance(pnl, (int, float)) else str(pnl)
#         wr_str = f"{win_rate:.2f}%" if isinstance(win_rate, (int, float)) else str(win_rate)
#         logger.info(f"Run Summary ({results_key}): PnL={pnl_str}, Trades={trades}, Win Rate={wr_str}")
#         logger.info(f"Exit Reasons ({results_key}): {exit_reasons}")

#         return summary_to_save

#     except Exception as e:
#         logger.error(f"Error during simulation step ({mode}): {e}", exc_info=True)
#         # Reraise the exception so pipeline_manager knows it failed
#         raise

# # --- Main block ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run simulation step using indicator data.')
#     parser.add_argument('--input', type=str, required=True, help='Input CSV file with indicators')
#     parser.add_argument('--output-json', type=str, required=True, help='Output JSON file for results summary')
#     parser.add_argument('--log-dir', type=str, required=True, help='Directory to save detailed simulation trace log')
#     # --- ADDED: Optional strategy name ---
#     parser.add_argument('--strategy-name', type=str, required=False, default=None,
#                         help='If provided, runs the specified strategy independently. Otherwise, runs the RuleBasedAgent.')
#     parser.add_argument('--symbol', type=str, required=False, help='Trading symbol (e.g., nifty, banknifty)')
#     parser.add_argument('--market', type=str, required=False, help='Market name or exchange (e.g., NSE, BSE)')

#     args = parser.parse_args()

#     indicator_path = Path(args.input)
#     output_path = Path(args.output_json)
#     log_dir_path = Path(args.log_dir)
#     symbol = args.symbol or 'nifty'
#     market = args.market or 'NSE'


#     try:
#         # Pass strategy_name to the function
#         run_simulation_task(indicator_path, output_path, log_dir_path, args.strategy_name)
#         logger.info(f"Simulation task completed successfully for {args.strategy_name or 'Agent'}.")
#     except Exception as main_e:
#          # Log the error before exiting
#          logger.error(f"Simulation task failed for {args.strategy_name or 'Agent'}. Error: {main_e}", exc_info=True)
#          sys.exit(1) # Exit with non-zero code on failure
# pipeline/run_simulation_step.py

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import json
import sys # For sys.exit and path manipulation
from datetime import datetime # For default run_id

# --- Add to your existing imports ---
from typing import Optional, Dict, Any 

# --- Ensure correct import paths for your app modules ---
try:
    from app.config import config
    from app.simulation_engine import SimpleBacktester
    from app.strategies import strategy_factories # For single strategy mode
    from app.agentic_core import RuleBasedAgent # For agent mode
    # Import MongoManager if performance_logger_mongo is used directly here,
    # or ensure it's closed by pipeline_manager
    from app.mongo_manager import MongoManager, close_mongo_connection_on_exit # For clean shutdown
    import atexit # To register the cleanup function

except ImportError:
    # Fallback if run directly and app module is not in PYTHONPATH
    current_dir = Path(__file__).resolve().parent.parent # Assuming this script is in 'pipeline' subdir
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from app.config import config
    from app.simulation_engine import SimpleBacktester
    from app.strategies import strategy_factories
    from app.agentic_core import RuleBasedAgent
    from app.mongo_manager import MongoManager, close_mongo_connection_on_exit
    import atexit

# --- Logger Setup (ensure it's consistent with your project) ---
logger = logging.getLogger("RunSimStep") # Specific logger for this script
if not logger.hasHandlers():
    log_level = getattr(config, "LOG_LEVEL", "INFO")
    log_format = getattr(config, "LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler(sys.stdout)])
    # If you want file logging for this script itself (distinct from simulation trace logs):
    # pipeline_manager_run_id = os.getenv("PIPELINE_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S_simstep"))
    # script_log_file = Path(config.LOG_DIR or "./logs") / f"run_simulation_step_{pipeline_manager_run_id}.log"
    # script_log_file.parent.mkdir(parents=True, exist_ok=True)
    # logger.addHandler(logging.FileHandler(script_log_file))


def load_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Loads data from CSV, ensuring datetime index."""
    if not file_path.exists():
        logger.error(f"Data file not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, parse_dates=['datetime'])
        if 'datetime' not in df.columns:
            logger.error(f"File {file_path} missing 'datetime' column.")
            return None
        df.set_index('datetime', inplace=True)
        df.columns = df.columns.str.lower() # Standardize column names
        
        required_ohlcv = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_ohlcv):
            logger.error(f"Data file {file_path} is missing one or more OHLC columns.")
            return None
        df.dropna(subset=required_ohlcv, inplace=True)
        if df.empty:
            logger.warning(f"Data at {file_path} became empty after OHLCV NaN drop.")
            return None
        logger.info(f"Successfully loaded data from {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        return None

def run_simulation_task(
    input_file_path: Path,
    output_json_path: Path,
    simulation_log_dir: Path, # Base directory for SimpleBacktester's trace logs
    strategy_name_arg: Optional[str], # From --strategy-name
    symbol_arg: Optional[str],
    market_arg: Optional[str],
    run_id_arg: Optional[str] # MODIFIED (2025-05-09): Added run_id_arg
):
    """
    Main task function to run a single simulation (either strategy or agent).
    """
    logger.info(f"--- Starting Simulation Task ---")
    logger.info(f"Input data: {input_file_path}")
    logger.info(f"Output JSON: {output_json_path}")
    logger.info(f"Simulation Log Dir: {simulation_log_dir}")
    logger.info(f"Strategy Name (arg): {strategy_name_arg}")
    logger.info(f"Symbol (arg): {symbol_arg}")
    logger.info(f"Market (arg): {market_arg}")
    logger.info(f"Run ID (arg): {run_id_arg}") # MODIFIED (2025-05-09): Log run_id

    df_instrument = load_data(input_file_path)
    if df_instrument is None or df_instrument.empty:
        logger.error("Failed to load data or data is empty. Aborting simulation task.")
        # Create a dummy error result if needed by pipeline_manager for consistent failure handling
        error_result = {"error": "Data loading failed or empty data.", "performance_score": -np.inf}
        with open(output_json_path, 'w') as f:
            json.dump(error_result, f, indent=4)
        return # Or raise an exception

    # Determine mode: Agent or Single Strategy
    backtester: Optional[SimpleBacktester] = None
    effective_strategy_name_for_run = "UnknownRun"

    if strategy_name_arg: # Single Strategy Mode
        logger.info(f"Mode: Single Strategy ('{strategy_name_arg}')")
        if strategy_name_arg not in strategy_factories:
            logger.error(f"Strategy '{strategy_name_arg}' not found in strategy_factories.")
            # Create dummy error result
            error_result = {"error": f"Strategy '{strategy_name_arg}' not found.", "performance_score": -np.inf}
            with open(output_json_path, 'w') as f:
                json.dump(error_result, f, indent=4)
            return
        
        # Get strategy factory and create instance (usually with default params for independent runs)
        # Optuna tuner would pass specific params to the factory if this script was adapted for it.
        # For now, assuming factory() gives a default instance.
        strategy_factory_func = strategy_factories[strategy_name_arg]
        try:
            # MODIFIED (2025-05-09): Pass default or configured initial params if needed by factory
            # For now, assuming factory can be called with no args for default.
            # If your factories require initial params (e.g. from config), fetch them here.
            # Example: initial_params = config.INITIAL_PARAMS.get(strategy_name_arg, {})
            # strategy_logic_func = strategy_factory_func(**initial_params)
            strategy_logic_func = strategy_factory_func() 
        except Exception as e_strat:
            logger.error(f"Error creating strategy '{strategy_name_arg}' from factory: {e_strat}", exc_info=True)
            error_result = {"error": f"Strategy creation failed for {strategy_name_arg}.", "performance_score": -np.inf}
            with open(output_json_path, 'w') as f: json.dump(error_result, f, indent=4)
            return

        backtester = SimpleBacktester(strategy_func=strategy_logic_func, strategy_name=strategy_name_arg)
        effective_strategy_name_for_run = strategy_name_arg

    else: # Agent Mode
        logger.info("Mode: Agent-driven Simulation")
        try:
            agent_instance = RuleBasedAgent() # Initialize your agent
            backtester = SimpleBacktester(agent=agent_instance, strategy_name="AgentRun") # strategy_name is for logging
            effective_strategy_name_for_run = "AgentRun" # Or agent_instance.name
        except Exception as e_agent:
            logger.error(f"Error initializing RuleBasedAgent: {e_agent}", exc_info=True)
            error_result = {"error": "Agent initialization failed.", "performance_score": -np.inf}
            with open(output_json_path, 'w') as f: json.dump(error_result, f, indent=4)
            return

    if backtester is None: # Should not happen if logic above is correct
        logger.error("Backtester could not be initialized. Aborting.")
        error_result = {"error": "Backtester init failed.", "performance_score": -np.inf}
        with open(output_json_path, 'w') as f: json.dump(error_result, f, indent=4)
        return

    # --- Run the simulation ---
    # The `timeframe` argument for run_simulation is the original data timeframe (e.g., "5min")
    # The log file name will be constructed by SimpleBacktester using this and other info.
    # `simulation_log_dir` is the specific directory for this run's trace log.
    
    # MODIFIED (2025-05-09): Extract timeframe from input_file_path for clarity,
    # or use a passed argument if available and more reliable.
    # Assuming filename format like: nifty__5min_with_indicators.csv
    try:
        parts = input_file_path.stem.split('__')
        data_timeframe = parts[1].split('_')[0] if len(parts) > 1 else "unknown_tf"
    except Exception:
        data_timeframe = "unknown_tf"
    logger.info(f"Derived data timeframe for simulation run: {data_timeframe}")


    results_dict = backtester.run_simulation(
        df=df_instrument,
        log_dir=simulation_log_dir, # This is where SimpleBacktester will create its log file
        timeframe=data_timeframe,   # Original timeframe of the data
        run_id=run_id_arg,          # MODIFIED (2025-05-09): Pass run_id
        # optuna_trial_params, optuna_study_name, optuna_trial_number are not directly relevant
        # for independent backtests or agent runs called by this script, unless this script
        # is also adapted to be the core of Optuna's objective function.
        # For now, they are None.
        optuna_trial_params=None,
        optuna_study_name=None,
        optuna_trial_number=None
    )

    # --- Process and Save Results ---
    if "error" in results_dict:
        logger.error(f"Simulation for '{effective_strategy_name_for_run}' failed: {results_dict['error']}")
        # Optionally, re-raise to make pipeline_manager aware of a hard failure
        # For now, just saving the error in JSON.
    else:
        logger.info(f"Simulation for '{effective_strategy_name_for_run}' completed.")
        logger.info(f"Performance Score: {results_dict.get('performance_score', 'N/A')}")
        logger.info(f"Total PnL: {results_dict.get('total_pnl', 'N/A')}, Trades: {results_dict.get('trade_count', 'N/A')}")

    # Add metadata to the results
    results_dict["metadata"] = {
        "run_id": run_id_arg, # MODIFIED (2025-05-09): Include run_id
        "simulation_mode": "Agent" if not strategy_name_arg else "SingleStrategy",
        "strategy_or_agent_name": effective_strategy_name_for_run,
        "input_file": str(input_file_path.name),
        "symbol": symbol_arg or getattr(config, "DEFAULT_SYMBOL", "Unknown"), # Use arg or default
        "market": market_arg or getattr(config, "DEFAULT_MARKET", "Unknown"),
        "timeframe_data": data_timeframe,
        "simulation_timestamp": datetime.now().isoformat(),
        "log_trace_file": str(Path(backtester.log_file_path).name) if backtester.log_file_path else "N/A"
    }
    # Ensure parameters used are also part of the results if it's a single strategy run
    # For agent runs, the "parameters" are more complex (agent's internal state/rules)
    if strategy_name_arg and "params_used_this_run" not in results_dict:
        # This part might need refinement based on how SimpleBacktester returns params
        # For now, adding a placeholder if not returned by run_simulation explicitly for single strategy
        results_dict["params_used_this_run"] = results_dict.get("indicator_config", "default_or_not_specified")


    try:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            # MODIFIED (2025-05-09): Custom encoder for numpy types if not handled by _convert_types_for_mongo
            # However, SimpleBacktester should ideally return Python-native types.
            # For now, assuming results_dict contains mostly Python natives.
            json.dump(results_dict, f, indent=4, default=str) # default=str for non-serializable
        logger.info(f"Results saved to {output_json_path}")
    except Exception as e:
        logger.error(f"Failed to save results JSON to {output_json_path}: {e}", exc_info=True)
        # If saving JSON fails, we might still want to signal error if results_dict had an error
        if "error" in results_dict:
             raise ValueError(f"Simulation failed: {results_dict['error']} (and JSON save also failed)")


    # MODIFIED (2025-05-09): If simulation itself had an error, raise it to fail the subprocess
    if "error" in results_dict:
        logger.error(f"Raising ValueError because simulation reported an error: {results_dict['error']}")
        raise ValueError(f"Simulation failed: {results_dict['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a backtest simulation for a strategy or agent.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input CSV data file with indicators.")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to save the output results JSON file.")
    parser.add_argument("--log-dir", type=Path, required=True, help="Directory to store the detailed simulation trace log file.")
    parser.add_argument("--strategy-name", type=str, help="Name of the strategy to run (from strategy_factories). If not provided, runs in Agent mode.")
    parser.add_argument("--symbol", type=str, help="Symbol being traded (e.g., NIFTY, BANKNIFTY). Used for metadata.")
    parser.add_argument("--market", type=str, help="Market/Exchange (e.g., NSE, BSE). Used for metadata.")
    # MODIFIED (2025-05-09): Added --run-id argument
    parser.add_argument("--run-id", type=str, help="Pipeline Run ID for linking results and logs.", default=f"sim_step_{datetime.now().strftime('%Y%m%d%H%M%S')}")

    args = parser.parse_args()

    # Register MongoDB connection cleanup
    atexit.register(close_mongo_connection_on_exit)

    try:
        run_simulation_task(
            input_file_path=args.input,
            output_json_path=args.output_json,
            simulation_log_dir=args.log_dir,
            strategy_name_arg=args.strategy_name,
            symbol_arg=args.symbol,
            market_arg=args.market,
            run_id_arg=args.run_id # MODIFIED (2025-05-09): Pass it here
        )
        logger.info("run_simulation_step.py finished successfully.")
        sys.exit(0) # Explicit success exit
    except ValueError as ve: # Catch the ValueError raised on simulation error
        logger.error(f"Simulation task failed with ValueError: {ve}")
        sys.exit(1) # Exit with error code
    except Exception as e:
        logger.error(f"An unhandled error occurred in run_simulation_step.py: {e}", exc_info=True)
        sys.exit(1) # Exit with error code
