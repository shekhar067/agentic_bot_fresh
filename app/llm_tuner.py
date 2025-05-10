# llm_tuner.py (enhanced with auto-retry and re-tune loop)

import logging
from typing import List, Dict
import pandas as pd
from pymongo import MongoClient
import openai
import os
from datetime import datetime, timedelta

# Load your config
try:
    from app.config import config
    from app.optuna_tuner import run_single_study
except ImportError:
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from app.config import config
    from app.optuna_tuner import run_single_study

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM-Tuner")

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_recent_trials(strategy: str, symbol: str, timeframe: str, limit: int = 5) -> List[Dict]:
    client = MongoClient(config.MONGO_URI)
    db = client[config.MONGO_DB_NAME]
    query = {"strategy": strategy, "symbol": symbol, "timeframe": timeframe}
    docs = list(db[config.MONGO_COLLECTION_BACKTEST_RESULTS].find(query).sort("performance_score", -1).limit(limit))
    client.close()
    return docs


def summarize_trials_for_llm(trials: List[Dict]) -> str:
    df = pd.DataFrame([{
        "params": d.get("params"),
        "win_rate": d.get("win_rate"),
        "score": d.get("performance_score"),
        "reasons": d.get("exit_reasons")
    } for d in trials])
    return df.to_markdown(index=False)


def get_llm_feedback(strategy: str, summary_markdown: str, model: str = "gpt-3.5-turbo") -> str:
    prompt = f"""
You are a trading strategy optimization expert. Here is the recent Optuna tuning summary for strategy '{strategy}':

{summary_markdown}

Evaluate the following:
1. Are the results satisfactory? (Answer: Yes or No)
2. If No, suggest better parameter ranges.
3. Should we re-run tuning? (Answer: Yes or No)
"""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a trading strategy optimization expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        tokens = response['usage']['total_tokens']
        cost = tokens / 1000 * (0.0015 + 0.002) if "gpt-3.5" in model else tokens / 1000 * (0.01 + 0.03)
        logger.info(f"LLM used {tokens} tokens | Cost: ${cost:.4f}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
        return "LLM error."


def log_llm_feedback(strategy: str, symbol: str, timeframe: str, feedback: str, retry_triggered: bool):
    client = MongoClient(config.MONGO_URI)
    db = client[config.MONGO_DB_NAME]
    db.llm_review_log.insert_one({
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "feedback": feedback,
        "retry_triggered": retry_triggered,
        "timestamp": datetime.utcnow()
    })
    client.close()


def can_retry_today(strategy: str, symbol: str, timeframe: str, limit: int = 3) -> bool:
    client = MongoClient(config.MONGO_URI)
    db = client[config.MONGO_DB_NAME]
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    count = db.llm_review_log.count_documents({
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": {"$gte": today_start},
        "retry_triggered": True
    })
    client.close()
    return count < limit


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--auto-retry", action="store_true", help="Allow auto re-tune if LLM suggests")
    parser.add_argument("--retry-limit", type=int, default=3, help="Max LLM-triggered retries per day")
    parser.add_argument("--trials", type=int, default=25, help="Trials for re-tune if triggered")
    args = parser.parse_args()

    trials = get_recent_trials(args.strategy, args.symbol, args.timeframe)
    if not trials:
        print("No trials found.")
        exit(1)

    summary = summarize_trials_for_llm(trials)
    print("\n=== Trial Summary ===\n")
    print(summary)

    feedback = get_llm_feedback(args.strategy, summary, model=args.model)
    print("\n=== LLM Feedback ===\n")
    print(feedback)

    if args.auto_retry and "re-run" in feedback.lower() and "yes" in feedback.lower():
        if can_retry_today(args.strategy, args.symbol, args.timeframe, limit=args.retry_limit):
            print("\n🔁 Re-running Optuna tuning based on LLM feedback...")
            df_path = Path(config.DATA_DIR_PROCESSED) / f"{args.symbol.lower()}__{args.timeframe}_with_indicators.csv"
            if df_path.exists():
                df = pd.read_csv(df_path, parse_dates=['datetime'])
                df.set_index("datetime", inplace=True)
                run_single_study(
                    symbol=args.symbol,
                    market=config.DEFAULT_MARKET,
                    segment=config.DEFAULT_SEGMENT,
                    strategy_name=args.strategy,
                    timeframe=args.timeframe,
                    context={},
                    df_raw=df,
                    n_trials=args.trials
                )
                log_llm_feedback(args.strategy, args.symbol, args.timeframe, feedback, retry_triggered=True)
            else:
                print(f"Feature file missing: {df_path}")
        else:
            print("🚫 Retry limit reached for today.")
            log_llm_feedback(args.strategy, args.symbol, args.timeframe, feedback, retry_triggered=False)
    else:
        log_llm_feedback(args.strategy, args.symbol, args.timeframe, feedback, retry_triggered=False)