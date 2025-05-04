import pandas as pd
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

def formatNum(value, decimals=2, suffix='', na_value='-'):
    if value is None or pd.isna(value) or not isinstance(value, (int, float)):
        return na_value
    try:
        return f"{value:,.{decimals}f}{suffix}"
    except Exception:
        return na_value

def calculate_detailed_metrics(trades: List[Dict], tf: str = "") -> Dict[str, float]:
    """Calculate PnL and trade stats from trade list."""
    if not trades:
        return {k: 0 for k in [
            'total_trades', 'win_rate', 'total_pnl_points',
            'avg_win_points', 'avg_loss_points', 'profit_factor',
            'max_drawdown_points', 'expectancy_points', 'sharpe_ratio_points'
        ]}

    try:
        df = pd.DataFrame(trades)
        df['PnL_Points'] = pd.to_numeric(df['PnL_Points'], errors='coerce')
        df['ExitTime'] = pd.to_datetime(df['ExitTime'], errors='coerce')
        df.dropna(subset=['PnL_Points', 'ExitTime'], inplace=True)

        if df.empty:
            return {k: 0 for k in [
                'total_trades', 'win_rate', 'total_pnl_points',
                'avg_win_points', 'avg_loss_points', 'profit_factor',
                'max_drawdown_points', 'expectancy_points', 'sharpe_ratio_points'
            ]}

        pnl = df['PnL_Points']
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        equity = pnl.cumsum()
        drawdown = equity.cummax() - equity

        return {
            'total_trades': len(pnl),
            'win_rate': (len(wins) / len(pnl)) * 100 if len(pnl) else 0,
            'avg_win_points': wins.mean() if not wins.empty else 0,
            'avg_loss_points': losses.mean() if not losses.empty else 0,
            'total_pnl_points': pnl.sum(),
            'profit_factor': wins.sum() / abs(losses.sum()) if losses.sum() else 1.0,
            'max_drawdown_points': drawdown.max() if not drawdown.empty else 0,
            'expectancy_points': pnl.mean(),
            'sharpe_ratio_points': pnl.mean() / pnl.std() if pnl.std() > 0 else 0
        }
    except Exception as e:
        logger.error(f"[{tf}] Error in metrics: {e}", exc_info=True)
        return {k: 0 for k in [
            'total_trades', 'win_rate', 'total_pnl_points',
            'avg_win_points', 'avg_loss_points', 'profit_factor',
            'max_drawdown_points', 'expectancy_points', 'sharpe_ratio_points'
        ]}
