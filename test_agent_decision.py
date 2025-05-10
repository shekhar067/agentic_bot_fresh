# test_agent_decision.py

import pandas as pd
from datetime import datetime
from app.agentic_core import RuleBasedAgent

def run_test_decision():
    # Sample row (mimics 1 bar of market data + indicators + metadata)
    sample_row = pd.Series({
        'regime': 'trending',  # detected from ADX or your logic
        'symbol': 'nifty',
        'timeframe': '5min',
        'is_expiry': False,
        'open': 22000,
        'high': 22100,
        'low': 21900,
        'close': 22050,
        'volume': 100000,
    }, name=pd.Timestamp("2025-05-09 09:45:00"))

    # Optional: Add more indicators as needed (e.g., rsi_14, macd, etc.)

    # Simulated recent data history (used by strategy functions)
    history_df = pd.DataFrame([sample_row])

    agent = RuleBasedAgent()
    signal, sl, tp, tsl, strategy_name = agent.decide(sample_row, data_history=history_df)

    print(f"📈 Strategy: {strategy_name}")
    print(f"🔔 Signal: {signal}")
    print(f"🎯 SL Mult: {sl} | TP Mult: {tp} | TSL Mult: {tsl}")

if __name__ == "__main__":
    run_test_decision()
