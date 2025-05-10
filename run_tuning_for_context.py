# run_tuning_for_context.py

from app.optuna_tuner import run_contextual_tuning
from app.config import config

def main():
    run_contextual_tuning(
        symbol="nifty",
        timeframes=["5min"],
        strategies=["EMA_Crossover", "SuperTrend_ADX", "BB_MeanReversion"],  # include all relevant strategies
        context_filter={
            "day": "Friday",
            "session": "Morning",
            "is_expiry": False,
            "market_condition": "trending"
        },
        n_trials_per_study=config.OPTUNA_TRIALS_PER_CONTEXT,
        max_workers=config.MAX_OPTUNA_WORKERS
    )

if __name__ == "__main__":
    main()
