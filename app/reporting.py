import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Any, Tuple
import io
import base64

# Ensure matplotlib uses a non-interactive backend
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def formatNum(value, decimals=2, suffix='', na_value='-'):
    if value is None or pd.isna(value) or not isinstance(value, (int, float)):
        return na_value
    try:
        return f"{value:,.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return na_value

def _plot_to_base64(fig: plt.Figure) -> Optional[str]:
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        logger.error(f"Failed to convert plot to base64: {e}", exc_info=True)
        plt.close(fig)
        return None

# def calculate_detailed_metrics(trades_list: List[Dict], timeframe: str = "Unknown") -> Optional[Dict]:
#     if not trades_list:
#         return {
#             'total_trades': 0, 'win_rate': 0, 'total_pnl_points': 0,
#             'avg_win_points': 0, 'avg_loss_points': 0, 'profit_factor': 1.0,
#             'max_drawdown_points': 0, 'expectancy_points': 0,
#             'sharpe_ratio_points': 0
#         }

#     try:
#         trades_df = pd.DataFrame(trades_list)
#         if 'PnL_Points' not in trades_df.columns or 'ExitTime' not in trades_df.columns:
#             logger.error(f"[{timeframe}] Trades list missing PnL_Points or ExitTime.")
#             return None

#         trades_df['PnL_Points'] = pd.to_numeric(trades_df['PnL_Points'], errors='coerce')
#         trades_df['ExitTime'] = pd.to_datetime(trades_df['ExitTime'], errors='coerce')
#         trades_df = trades_df.dropna(subset=['PnL_Points', 'ExitTime']).sort_values('ExitTime')

#         if trades_df.empty: return {'total_trades': 0}

#         pnl_points = trades_df['PnL_Points']
#         wins = pnl_points[pnl_points > 0]
#         losses = pnl_points[pnl_points <= 0]
#         total_trades = len(trades_df)

#         equity_curve_points = pnl_points.cumsum()
#         peak = equity_curve_points.cummax()
#         drawdown = peak - equity_curve_points
#         max_drawdown_points = drawdown.max() if not drawdown.empty else 0

#         total_wins = wins.sum()
#         total_losses = abs(losses.sum())
#         profit_factor = total_wins / total_losses if total_losses > 0 else np.inf if total_wins > 0 else 1.0

#         pnl_std_dev = pnl_points.std()
#         sharpe_ratio_points = (pnl_points.mean() / pnl_std_dev) if pnl_std_dev > 0 else 0

#         return {
#             'total_trades': total_trades,
#             'win_rate': (len(wins) / total_trades * 100) if total_trades > 0 else 0,
#             'avg_win_points': wins.mean() if not wins.empty else 0,
#             'avg_loss_points': losses.mean() if not losses.empty else 0,
#             'total_pnl_points': pnl_points.sum(),
#             'profit_factor': profit_factor,
#             'max_drawdown_points': max_drawdown_points,
#             'expectancy_points': pnl_points.mean() if total_trades > 0 else 0,
#             'sharpe_ratio_points': sharpe_ratio_points
#         }

#     except Exception as e:
#         logger.error(f"[{timeframe}] Error calculating detailed metrics: {e}", exc_info=True)
#         return None

def generate_agent_html_report(all_results: Dict[str, Optional[Dict]]) -> str:
    logger.info("Generating full agent HTML report content...")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Agent Backtest Report</title>
  <link rel="stylesheet" href="/static/report.css">
  <script src="/static/report.js"></script>
</head>
<body>
  <div class="container">
    <h1>Agent Backtest Report</h1>
    <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <div class="tab-buttons">
"""

    # 1. Tab Buttons
    for i, tf in enumerate(all_results):
        html += f'<button class="tab-button{" active" if i == 0 else ""}" data-tab="tab_{tf}">{tf}</button>'

    html += "</div> <!-- end tab-buttons -->\n"

    # 2. Tab Contents
    for i, (tf, result) in enumerate(all_results.items()):
        tab_id = f"tab_{tf}"
        html += f'<div id="{tab_id}" class="tab-content{" active" if i == 0 else ""}">\n'
        html += f"<h3>Timeframe: {tf}</h3>\n"

        if not result or "error" in result:
            html += f"<p style='color:red;'>Error: {result.get('error', 'No data')}</p>\n"
        else:
            trades = result.get("trades_details") or result.get("trades") or []
            metrics = calculate_detailed_metrics(trades, tf)
            #metrics = calculate_detailed_metrics(result.get("trades_details", []), tf)
            if not metrics:
                html += "<p>No metrics available.</p>"
            else:
                html += "<ul>\n"
                for key, val in metrics.items():
                    html += f"<li><b>{key.replace('_', ' ').title()}</b>: {formatNum(val)}</li>\n"
                html += "</ul>\n"

        html += "</div> <!-- end tab-content -->\n"

    html += """
  </div> <!-- end container -->
  <script>
    document.addEventListener('DOMContentLoaded', function () {
      const buttons = document.querySelectorAll('.tab-button');
      const contents = document.querySelectorAll('.tab-content');
      buttons.forEach(btn => {
        btn.addEventListener('click', function () {
          buttons.forEach(b => b.classList.remove('active'));
          contents.forEach(c => c.classList.remove('active'));
          btn.classList.add('active');
          const tabId = btn.getAttribute('data-tab');
          document.getElementById(tabId).classList.add('active');
        });
      });
    });
  </script>
</body>
</html>
"""
    return html

def calculate_detailed_metrics(trades_list: List[Dict], timeframe: str = "Unknown") -> Dict:
    if not trades_list:
        return { 'total_trades': 0, 'win_rate': 0, 'total_pnl_points': 0, 'avg_win_points': 0,
                 'avg_loss_points': 0, 'profit_factor': 1.0, 'max_drawdown_points': 0,
                 'expectancy_points': 0, 'sharpe_ratio_points': 0 }
    try:
        trades_df = pd.DataFrame(trades_list)
        trades_df['PnL_Points'] = pd.to_numeric(trades_df['PnL_Points'], errors='coerce')
        trades_df['ExitTime'] = pd.to_datetime(trades_df['ExitTime'], errors='coerce')
        trades_df.dropna(subset=['PnL_Points', 'ExitTime'], inplace=True)
        if trades_df.empty: return { 'total_trades': 0 }

        pnl = trades_df['PnL_Points']
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        equity = pnl.cumsum()
        drawdown = equity.cummax() - equity
        return {
            'total_trades': len(pnl),
            'win_rate': len(wins) / len(pnl) * 100 if len(pnl) else 0,
            'avg_win_points': wins.mean() if not wins.empty else 0,
            'avg_loss_points': losses.mean() if not losses.empty else 0,
            'total_pnl_points': pnl.sum(),
            'profit_factor': wins.sum() / abs(losses.sum()) if losses.sum() else 1,
            'max_drawdown_points': drawdown.max(),
            'expectancy_points': pnl.mean(),
            'sharpe_ratio_points': pnl.mean() / pnl.std() if pnl.std() > 0 else 0
        }
    except Exception as e:
        logger.error(f"[{timeframe}] Error in metrics: {e}")
        return {}


