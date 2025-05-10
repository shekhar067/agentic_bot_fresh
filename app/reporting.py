# app/reporting.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader # For HTML reports
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
import argparse # For CLI functionality
import sys # For sys.exit and sys.path

# MongoDB imports
from pymongo import MongoClient, DESCENDING, ASCENDING

# Assuming config is in app directory and accessible
try:
    from app.config import config
except ImportError:
    # Fallback for running script directly or if app module isn't found easily
    # This allows 'python app/reporting.py ...' from project root
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))
    from app.config import config


logger = logging.getLogger(__name__)
# Configure logger for this module if not already configured by a higher-level entry point
if not logger.hasHandlers():
    log_level_from_config = getattr(config, "LOG_LEVEL", "INFO")
    log_format_from_config = getattr(config, "LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=log_level_from_config, format=log_format_from_config, handlers=[logging.StreamHandler(sys.stdout)])


# --- Jinja2 Environment Setup ---
TEMPLATE_DIR_PATHS_TO_TRY = [
    Path(__file__).resolve().parent.parent / "templates", # ProjectRoot/templates
    Path(__file__).resolve().parent / "templates",        # app/templates
]
if hasattr(config, "TEMPLATES_DIR") and config.TEMPLATES_DIR and Path(config.TEMPLATES_DIR).exists():
    TEMPLATE_DIR_PATHS_TO_TRY.insert(0, Path(config.TEMPLATES_DIR)) # Prioritize config

LOADED_TEMPLATE_DIR = None
for p_dir in TEMPLATE_DIR_PATHS_TO_TRY:
    if p_dir.exists() and p_dir.is_dir():
        LOADED_TEMPLATE_DIR = p_dir
        break

if LOADED_TEMPLATE_DIR:
    env = Environment(loader=FileSystemLoader(str(LOADED_TEMPLATE_DIR)))
    logger.info(f"Jinja2 Environment loaded from: {LOADED_TEMPLATE_DIR}")
else:
    logger.error(f"Could not find templates directory. Tried: {TEMPLATE_DIR_PATHS_TO_TRY}. Ensure TEMPLATES_DIR is correctly configured or accessible.")
    class DummyEnv: # Fallback to prevent crashes
        def get_template(self, name): raise FileNotFoundError(f"Template {name} not found, Jinja2 env failed to load.")
    env = DummyEnv()

# ========= Functions for Reporting from JSON Files (Existing Logic - Kept for compatibility) =========
def load_simulation_results(json_path: Path) -> dict:
    """Loads simulation results from a JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.info(f"Successfully loaded simulation results from {json_path}")
        return results
    except FileNotFoundError:
        logger.error(f"JSON results file not found: {json_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {json_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {json_path}: {e}")
    return {}

def create_single_backtest_report_html(results: dict, output_html_path: Path, template_name: str = "report_template.html"):
    """
    Generates an HTML performance report from a single backtest's simulation results (JSON) using a Jinja2 template.
    """
    if not results:
        logger.warning("No results data to generate report. Skipping HTML generation.")
        return

    try:
        template = env.get_template(template_name)
    except Exception as e:
        logger.error(f"Failed to load template '{template_name}': {e}. Ensure template exists in {LOADED_TEMPLATE_DIR}.")
        return

    metrics = results.get("overall_metrics", {})
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    for col in metrics_df.columns:
        if not metrics_df.empty and len(metrics_df[col]) > 0:
            if metrics_df[col].dtype == 'float64' or isinstance(metrics_df[col].iloc[0], float) :
                metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int,float)) and pd.notnull(x) else x)
    metrics_html = metrics_df.to_html(classes='table table-striped table-hover table-sm', index=False, border=0)

    pnl_curve_data = results.get("pnl_curve", [])
    equity_curve_html = "<p>No P&L curve data available or data format incorrect.</p>"
    if pnl_curve_data:
        pnl_df = pd.DataFrame(pnl_curve_data)
        if not pnl_df.empty and 'timestamp' in pnl_df.columns and 'equity' in pnl_df.columns:
            try:
                pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pnl_df['timestamp'], y=pnl_df['equity'], mode='lines', name='Equity Curve'))
                fig.update_layout(title='Equity Curve', xaxis_title='Time', yaxis_title='Equity')
                equity_curve_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            except Exception as plot_e:
                logger.error(f"Error generating P&L curve plot: {plot_e}")
                equity_curve_html = f"<p>Error generating P&L plot: {plot_e}</p>"
        else:
            logger.warning("PNL curve data in JSON is missing 'timestamp' or 'equity' columns.")

    trades_log = results.get("trade_log", [])
    trades_html = "<p>No trade log data available.</p>"
    if trades_log:
        trades_df = pd.DataFrame(trades_log)
        if not trades_df.empty:
            for col_name in ['entry_price', 'exit_price', 'pnl', 'profit_loss', 'qty', 'commission', 'sl_price', 'tp_price', 'atr_at_entry']:
                if col_name in trades_df.columns:
                    trades_df[col_name] = trades_df[col_name].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) and pd.notnull(x) else x)
            trades_html = trades_df.to_html(classes='table table-striped table-hover table-sm trades-table', index=False, border=0, justify='right')

    strategy_info = results.get("strategy_info", {})
    metadata = results.get("metadata", {})
    context_params = {
        "report_title": strategy_info.get("name", metadata.get("strategy_name", "Strategy Performance Report")),
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics_table": metrics_html,
        "equity_curve_plot": equity_curve_html,
        "trades_table": trades_html,
        "strategy_name": strategy_info.get("name", metadata.get("strategy_name", "N/A")),
        "parameters": strategy_info.get("params", metadata.get("parameters_used", {})),
        "input_file": metadata.get("input_file", "N/A"),
        "symbol": metadata.get("symbol", "N/A"),
        "timeframe": metadata.get("timeframe", "N/A"),
        "run_id": metadata.get("run_id", "N/A"),
        "custom_summary": results.get("custom_summary", "")
    }

    try:
        html_content = template.render(context_params)
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML report (single backtest) generated successfully: {output_html_path}")
    except Exception as e:
        logger.error(f"Error rendering or writing HTML report for single backtest: {e}", exc_info=True)


# ========= NEW/ENHANCED Functions for MongoDB Analytics Reporting =========

def get_mongo_client() -> MongoClient:
    """Establishes and returns a MongoDB client connection."""
    try:
        client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=config.MONGO_TIMEOUT_MS)
        client.admin.command('ping') # Verify connection
        # logger.info(f"Successfully connected to MongoDB at {config.MONGO_URI_DISPLAY}") # Keep log less verbose for reporting
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB for reporting: {e}", exc_info=True)
        raise

def _format_results_to_dataframe(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Helper function to format a list of MongoDB documents into a styled DataFrame."""
    if not results_list:
        return pd.DataFrame()

    display_columns = [
        "strategy_name", "performance_score", "total_pnl", "win_rate",
        "sharpe_ratio", "sortino_ratio", "profit_factor", "max_drawdown", "total_trades",
        "symbol", "timeframe", "market_condition", "session", "day", "is_expiry",
        "parameters_used", "optuna_study_name", "optuna_trial_number" # Ensure these are logged by Optuna if needed
    ]
    
    filtered_data_for_df = []
    for doc in results_list:
        row_data = {}
        for col in display_columns:
            row_data[col] = doc.get(col) # Get value or None, defaults to None
            if col == "parameters_used" and isinstance(row_data[col], dict):
                try: 
                    row_data[col] = f"<pre>{json.dumps(row_data[col], indent=2, sort_keys=True)}</pre>"
                except TypeError: 
                    row_data[col] = str(row_data[col])
            elif row_data[col] is None: # Replace None with 'N/A' for display consistency
                 row_data[col] = 'N/A'


        filtered_data_for_df.append(row_data)

    df = pd.DataFrame(filtered_data_for_df)
    if df.empty:
        return df

    ordered_cols = [col for col in display_columns if col in df.columns]
    df = df[ordered_cols]
    
    numeric_cols_to_format = ["performance_score", "total_pnl", "win_rate", "sharpe_ratio", 
                              "sortino_ratio", "profit_factor", "max_drawdown"]
    for col in numeric_cols_to_format:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and pd.notnull(x) and x != 'N/A' else x)
    if "total_trades" in df.columns:
         df["total_trades"] = df["total_trades"].apply(lambda x: f"{int(x)}" if pd.notnull(x) and isinstance(x, (int,float)) and x != 'N/A' else x)

    return df

def get_ranked_strategies_from_db(
    client: MongoClient,
    context_filter: Dict[str, Any],
    db_name: str = config.MONGO_DB_NAME,
    collection_name: str = config.MONGO_COLLECTION_BACKTEST_RESULTS,
    sort_by: str = "performance_score",
    sort_order: int = DESCENDING, 
    limit: int = 10
) -> pd.DataFrame:
    try:
        db = client[db_name]
        collection = db[collection_name]
        logger.debug(f"Querying '{collection_name}' with filter: {context_filter}, sort: {sort_by}, order: {sort_order}, limit: {limit}")

        query = {}
        for key, value in context_filter.items():
            if value is not None:
                if isinstance(value, bool): query[key] = value
                elif isinstance(value, str) and value.lower() in ['true', 'false']: query[key] = (value.lower() == 'true')
                else: query[key] = str(value)
        
        results_cursor = collection.find(query).sort(sort_by, sort_order).limit(limit)
        results_list = list(results_cursor)
        
        return _format_results_to_dataframe(results_list)

    except Exception as e:
        logger.error(f"Error in get_ranked_strategies_from_db for context {context_filter}: {e}", exc_info=True)
        return pd.DataFrame()

def get_aggregated_strategy_performance(
    client: MongoClient,
    base_filter: Dict[str, Any], 
    db_name: str = config.MONGO_DB_NAME,
    collection_name: str = config.MONGO_COLLECTION_BACKTEST_RESULTS,
    limit: int = 10
) -> pd.DataFrame:
    try:
        db = client[db_name]
        collection = db[collection_name]
        logger.debug(f"Aggregating performance for base_filter: {base_filter}")

        match_query = {}
        for key, value in base_filter.items():
             if value is not None: match_query[key] = str(value)

        pipeline = [
            {"$match": match_query},
            {"$group": {
                "_id": "$strategy_name",
                "avg_performance_score": {"$avg": "$performance_score"},
                "total_pnl_sum": {"$sum": "$total_pnl"},
                "avg_win_rate": {"$avg": "$win_rate"},
                "avg_sharpe_ratio": {"$avg": "$sharpe_ratio"},
                "avg_profit_factor": {"$avg": "$profit_factor"},
                "distinct_contexts_count": {"$addToSet": { 
                    "market_condition": "$market_condition", "session": "$session", 
                    "day": "$day", "is_expiry": "$is_expiry"
                }},
                "total_db_entries": {"$sum": 1} 
            }},
            {"$addFields": { 
                "num_distinct_contexts": {"$size": "$distinct_contexts_count"}
            }},
            {"$sort": {"avg_performance_score": DESCENDING}},
            {"$limit": limit},
            {"$project": { 
                "strategy_name": "$_id",
                "avg_performance_score": 1,
                "total_pnl_sum": 1,
                "avg_win_rate": 1,
                "avg_sharpe_ratio": 1,
                "avg_profit_factor": 1,
                "num_distinct_contexts": 1,
                "total_db_entries": 1,
                "_id": 0
            }}
        ]
        results_cursor = collection.aggregate(pipeline)
        results_list = list(results_cursor)
        
        # Manually add strategy_name and reorder for _format_results_to_dataframe if needed,
        # or ensure display_columns in _format_results_to_dataframe handles these aggregated fields.
        # The $project stage already renames _id to strategy_name.
        # We might want a slightly different set of display_columns for aggregated results.
        
        # For aggregated data, let's prepare it slightly differently for _format_results_to_dataframe
        # or use a modified formatter. For now, we adapt its use:
        custom_display_cols_agg = [
            "strategy_name", "avg_performance_score", "total_pnl_sum", "avg_win_rate",
            "avg_sharpe_ratio", "avg_profit_factor", "num_distinct_contexts", "total_db_entries"
        ]
        
        # Re-create results_list with keys matching display_columns for _format_results_to_dataframe
        # This step is a bit redundant if _format_results_to_dataframe is flexible enough,
        # but ensures the columns match what _format_results_to_dataframe expects for formatting.
        # For this direct aggregation, we can format here or make _format_results_to_dataframe more general.
        df = pd.DataFrame(results_list)
        if not df.empty:
            # Ensure columns appear in the desired order
            ordered_cols = [col for col in custom_display_cols_agg if col in df.columns]
            df = df[ordered_cols]
            # Format numeric columns
            numeric_cols_to_format_agg = ["avg_performance_score", "total_pnl_sum", "avg_win_rate", 
                                          "avg_sharpe_ratio", "avg_profit_factor"]
            for col in numeric_cols_to_format_agg:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and pd.notnull(x) else x)
            integer_cols_agg = ["num_distinct_contexts", "total_db_entries"]
            for col in integer_cols_agg:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: f"{int(x)}" if pd.notnull(x) and isinstance(x, (int,float)) else x)
        return df


    except Exception as e:
        logger.error(f"Error in get_aggregated_strategy_performance for filter {base_filter}: {e}", exc_info=True)
        return pd.DataFrame()


def create_comprehensive_analytics_report_html(
    mongo_client: MongoClient,
    output_html_path: Path,
    report_main_title: str = "Comprehensive Strategy Analytics Report",
    template_name: str = "report_template_detailed.html", # Assumes this template can handle multiple sections
    top_n_global: int = 20,
    top_n_contextual: int = 5,
    symbols_timeframes: List[Tuple[str, str]] = None, 
    market_conditions_to_analyze: List[str] = None, 
    sessions_to_analyze: List[str] = None, 
    days_to_analyze: List[str] = None 
):
    if "DummyEnv" in str(type(env)):
        logger.error("Jinja2 environment not loaded. Cannot generate comprehensive report.")
        return
        
    # Set defaults if None
    if symbols_timeframes is None: 
        symbols_timeframes = [(config.DEFAULT_SYMBOL, tf) for tf in getattr(config, "RAW_DATA_FILES", {"5min":""}).keys()]
    if market_conditions_to_analyze is None: market_conditions_to_analyze = ["Trending", "Ranging", "Volatile", "Unknown"] # Added Unknown
    if sessions_to_analyze is None: sessions_to_analyze = ["Morning", "Midday", "Afternoon"]
    if days_to_analyze is None: days_to_analyze = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    report_sections = []

    logger.info("Generating: Global Top Performers")
    global_top_df = get_ranked_strategies_from_db(mongo_client, context_filter={}, limit=top_n_global)
    if not global_top_df.empty:
        report_sections.append({
            "section_title": f"Global Top {top_n_global} Strategy-Context Performance Entries",
            "section_content_html": global_top_df.to_html(classes='table table-striped table-hover table-sm ranked-strategy-table', index=False, border=0, justify='right', escape=False),
            "context_details_json": json.dumps({"description": f"Top {top_n_global} performance entries across all recorded contexts, sorted by performance_score."}, indent=2)
        })

    for symbol, timeframe in symbols_timeframes:
        base_context_for_instrument = {"symbol": symbol, "timeframe": timeframe}
        instrument_title_prefix = f"{symbol} {timeframe}"

        logger.info(f"Generating: Aggregated Performance for {instrument_title_prefix}")
        agg_df = get_aggregated_strategy_performance(mongo_client, base_filter=base_context_for_instrument, limit=top_n_contextual)
        if not agg_df.empty:
            report_sections.append({
                "section_title": f"Top {top_n_contextual} Strategies by Average Performance Score for {instrument_title_prefix}",
                "section_content_html": agg_df.to_html(classes='table table-striped table-hover table-sm ranked-strategy-table', index=False, border=0, justify='right', escape=False),
                "context_details_json": json.dumps({"description": f"Strategies for {instrument_title_prefix} ranked by average performance_score across various specific contexts.", "base_filter": base_context_for_instrument}, indent=2)
            })

        for mc in market_conditions_to_analyze:
            logger.info(f"Generating: Top for Market Condition '{mc}' - {instrument_title_prefix}")
            context = {**base_context_for_instrument, "market_condition": mc}
            mc_df = get_ranked_strategies_from_db(mongo_client, context_filter=context, limit=top_n_contextual)
            if not mc_df.empty:
                report_sections.append({
                    "section_title": f"Top {top_n_contextual} for {instrument_title_prefix} - Market Condition: {mc}",
                    "section_content_html": mc_df.to_html(classes='table table-striped table-hover table-sm ranked-strategy-table', index=False, border=0, justify='right', escape=False),
                    "context_details_json": json.dumps(context, indent=2)
                })
        
        for expiry_status_bool, expiry_status_label in [(True, "Expiry Day"), (False, "Non-Expiry Day")]:
            logger.info(f"Generating: Top for {expiry_status_label} - {instrument_title_prefix}")
            context = {**base_context_for_instrument, "is_expiry": expiry_status_bool}
            expiry_df = get_ranked_strategies_from_db(mongo_client, context_filter=context, limit=top_n_contextual)
            if not expiry_df.empty:
                report_sections.append({
                    "section_title": f"Top {top_n_contextual} for {instrument_title_prefix} - {expiry_status_label}",
                    "section_content_html": expiry_df.to_html(classes='table table-striped table-hover table-sm ranked-strategy-table', index=False, border=0, justify='right', escape=False),
                    "context_details_json": json.dumps(context, indent=2)
                })

        for sess in sessions_to_analyze:
            logger.info(f"Generating: Top for Session '{sess}' - {instrument_title_prefix}")
            context = {**base_context_for_instrument, "session": sess}
            sess_df = get_ranked_strategies_from_db(mongo_client, context_filter=context, limit=top_n_contextual)
            if not sess_df.empty:
                report_sections.append({
                    "section_title": f"Top {top_n_contextual} for {instrument_title_prefix} - Session: {sess}",
                    "section_content_html": sess_df.to_html(classes='table table-striped table-hover table-sm ranked-strategy-table', index=False, border=0, justify='right', escape=False),
                    "context_details_json": json.dumps(context, indent=2)
                })
        
        for day_of_week in days_to_analyze:
            logger.info(f"Generating: Top for Day '{day_of_week}' - {instrument_title_prefix}")
            context = {**base_context_for_instrument, "day": day_of_week}
            day_df = get_ranked_strategies_from_db(mongo_client, context_filter=context, limit=top_n_contextual)
            if not day_df.empty:
                report_sections.append({
                    "section_title": f"Top {top_n_contextual} for {instrument_title_prefix} - Day: {day_of_week}",
                    "section_content_html": day_df.to_html(classes='table table-striped table-hover table-sm ranked-strategy-table', index=False, border=0, justify='right', escape=False),
                    "context_details_json": json.dumps(context, indent=2)
                })

    jinja_context = {
        "report_title": report_main_title,
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "report_sections": report_sections,
    }

    try:
        template = env.get_template(template_name)
        html_content = template.render(jinja_context)
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Comprehensive analytics HTML report generated successfully: {output_html_path}")
    except Exception as e:
        logger.error(f"Error rendering or writing comprehensive analytics HTML report: {e}", exc_info=True)


# ========= Main Example / CLI Utility Functionality =========
if __name__ == '__main__':
    common_parser_cli = argparse.ArgumentParser(add_help=False) 
    common_parser_cli.add_argument("--db-name", type=str, default=config.MONGO_DB_NAME, help="MongoDB database name")
    common_parser_cli.add_argument("--results-collection", type=str, default=config.MONGO_COLLECTION_BACKTEST_RESULTS, help="MongoDB collection for backtest results")

    parser_cli = argparse.ArgumentParser(description="Reporting module for strategy performance. Can generate HTML reports or query MongoDB via CLI.")
    subparsers_cli = parser_cli.add_subparsers(dest="command", help="Available commands", required=True)

    query_parser_cli = subparsers_cli.add_parser("query", parents=[common_parser_cli], help="Query MongoDB for ranked strategies and print to console.")
    query_parser_cli.add_argument("--symbol", type=str, required=True, help="Symbol (e.g., NIFTY)")
    query_parser_cli.add_argument("--timeframe", type=str, required=True, help="Timeframe (e.g., 5min)")
    query_parser_cli.add_argument("--market-condition", type=str, help="Market condition (e.g., Trending, Ranging)")
    query_parser_cli.add_argument("--session", type=str, help="Trading session (e.g., Morning, Midday)")
    query_parser_cli.add_argument("--day", type=str, help="Day of the week (e.g., Monday)")
    query_parser_cli.add_argument("--is-expiry", type=lambda x: (str(x).lower() == 'true'), help="Is it an expiry day? (true/false)")
    query_parser_cli.add_argument("--sort-by", type=str, default="performance_score", help="Field to sort by (default: performance_score)")
    query_parser_cli.add_argument("--sort-order", type=str, choices=['asc', 'desc'], default='desc', help="Sort order: 'asc' or 'desc' (default: desc)")
    query_parser_cli.add_argument("--limit", type=int, default=10, help="Number of top results to display (default: 10)")

    report_parser_cli = subparsers_cli.add_parser("generate_report", parents=[common_parser_cli], help="Generate a comprehensive HTML analytics report from MongoDB data.")
    report_parser_cli.add_argument("--output-dir", type=str, help="Directory to save the HTML report. Defaults to a subfolder in 'runs_example_reporting'.")
    report_parser_cli.add_argument("--output-filename", type=str, default="comprehensive_analytics_report.html", help="Filename for the HTML report.")
    report_parser_cli.add_argument("--report-title", type=str, default="Comprehensive Strategy Analytics Report", help="Main title for the HTML report.")
    report_parser_cli.add_argument("--template-name", type=str, default="report_template_detailed.html", help="Jinja2 template file to use.")
    report_parser_cli.add_argument("--global-top-n", type=int, default=15, help="Number of global top performers to show.")
    report_parser_cli.add_argument("--contextual-top-n", type=int, default=5, help="Number of top performers per context to show.")
    report_parser_cli.add_argument("--symbols", type=str, help="Comma-separated list of symbols for the report (e.g., NIFTY,BANKNIFTY). Defaults to config.DEFAULT_SYMBOL.")
    report_parser_cli.add_argument("--timeframes", type=str, help="Comma-separated list of timeframes for the report (e.g., 5min,15min). Defaults to config.RAW_DATA_FILES keys.")


    args_cli = parser_cli.parse_args()

    if "DummyEnv" in str(type(env)): 
         logger.error("Jinja2 environment not loaded. Cannot generate HTML reports. Exiting.")
         sys.exit(1)

    mongo_client_instance_cli = None
    try:
        if args_cli.command == "query":
            logger.info(f"Running CLI query with context: {vars(args_cli)}")
            mongo_client_instance_cli = get_mongo_client()
            
            context_filter_cli = {
                "symbol": args_cli.symbol if args_cli.symbol else None,

                "timeframe": args_cli.timeframe,
                "market_condition": args_cli.market_condition,
                "session": args_cli.session,
                "day": args_cli.day,
                "is_expiry": args_cli.is_expiry
            }
            context_filter_cli = {k: v for k, v in context_filter_cli.items() if v is not None}
            
            sort_order_mongo_cli = DESCENDING if args_cli.sort_order == 'desc' else ASCENDING

            ranked_df_cli = get_ranked_strategies_from_db(
                client=mongo_client_instance_cli,
                context_filter=context_filter_cli,
                db_name=args_cli.db_name,
                collection_name=args_cli.results_collection,
                sort_by=args_cli.sort_by,
                sort_order=sort_order_mongo_cli,
                limit=args_cli.limit
            )

            if not ranked_df_cli.empty:
                print(f"\n--- Top {args_cli.limit} Ranked Strategies for Context: {context_filter_cli} (Sorted by {args_cli.sort_by} {args_cli.sort_order.upper()}) ---")
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 200) # Adjust width as needed
                # Clean up <pre> tags for console display
                ranked_df_display_cli = ranked_df_cli.copy()
                if 'parameters_used' in ranked_df_display_cli.columns:
                    ranked_df_display_cli['parameters_used'] = ranked_df_display_cli['parameters_used'].str.replace(r'<pre>|</pre>|&quot;', '', regex=True).str.replace(r'\s\s+', ' ', regex=True).str.strip()

                print(ranked_df_display_cli.to_string(index=False))
            else:
                print(f"\nNo strategies found matching the specified context in collection '{args_cli.results_collection}': {context_filter_cli}")

        elif args_cli.command == "generate_report":
            logger.info("Generating comprehensive HTML analytics report via CLI...")
            
            PROJECT_ROOT_EXAMPLE_CLI = Path(__file__).resolve().parent.parent
            output_dir_path_cli = Path(args_cli.output_dir) if args_cli.output_dir else PROJECT_ROOT_EXAMPLE_CLI / "runs_cli_reporting" / f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir_path_cli.mkdir(parents=True, exist_ok=True)
            output_html_file_cli = output_dir_path_cli / args_cli.output_filename

            mongo_client_instance_cli = get_mongo_client()
            
            symbols_tf_pairs = []
            default_tfs = list(getattr(config, "RAW_DATA_FILES", {"5min":""}).keys())
            
            cli_symbols = [s.strip().upper() for s in args_cli.symbols.split(',')] if args_cli.symbols else [config.DEFAULT_SYMBOL]
            cli_timeframes = [tf.strip() for tf in args_cli.timeframes.split(',')] if args_cli.timeframes else default_tfs

            for sym in cli_symbols:
                for tf_report in cli_timeframes:
                    symbols_tf_pairs.append((sym, tf_report))
            
            if not symbols_tf_pairs: # Fallback if CLI parsing results in empty
                 symbols_tf_pairs = [(config.DEFAULT_SYMBOL, default_tfs[0] if default_tfs else "5min")]


            create_comprehensive_analytics_report_html(
                mongo_client=mongo_client_instance_cli,
                output_html_path=output_html_file_cli,
                report_main_title=args_cli.report_title,
                template_name=args_cli.template_name,
                top_n_global=args_cli.global_top_n,
                top_n_contextual=args_cli.contextual_top_n,
                symbols_timeframes=symbols_tf_pairs
            )
            logger.info(f"Comprehensive analytics report generated. Check: {output_html_file_cli.resolve()}")
        
    except Exception as e:
        logger.error(f"An error occurred in reporting main (__name__ == '__main__'): {e}", exc_info=True)
        sys.exit(1) # Exit with error code if main execution fails
    finally:
        if mongo_client_instance_cli:
            mongo_client_instance_cli.close()
            logger.info("MongoDB connection (if opened by CLI main) closed.")
    
    sys.exit(0) # Explicitly exit with success code