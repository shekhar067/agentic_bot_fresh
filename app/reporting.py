# app/reporting.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__) # Get logger instance

# Helper function for formatting numbers in HTML
def formatNum(value, decimals=2, suffix='', na_value='-'):
     """Formats a number for HTML display."""
     if value is None or pd.isna(value) or not isinstance(value, (int, float)): return na_value
     try: return f"{value:.{decimals}f}{suffix}"
     except: return na_value # Fallback

def generate_agent_html_report(all_results: Dict[str, Optional[Dict]]) -> str:
     """Generates HTML report for the single agent across timeframes."""
     logger.info("Generating agent HTML report content...")
     html_content = ""
     timeframes = sorted(list(all_results.keys()), key=lambda x: (int(x[:-3]) if x[:-3].isdigit() else 999, x[-3:])) # Sort timeframes numerically
     if not timeframes: return "<p>No results data available for report.</p>"

     # --- Tab Buttons ---
     html_content += "<div class='tab-buttons'>"
     for i, tf in enumerate(timeframes):
         active_class = 'active' if i == 0 else ''; tf_id = str(tf).replace(' ','_') # Make ID html safe
         html_content += f'<button class="tab-button {active_class}" onclick="openTab(event, \'{tf_id}\')">{tf}</button>'
     html_content += "</div>"

     # --- Tab Content ---
     for i, tf in enumerate(timeframes):
         tf_id = str(tf).replace(' ','_'); active_class = 'active' if i == 0 else ''
         html_content += f'<div id="{tf_id}" class="tab-content {active_class}">\n<h2>Results for Timeframe: {tf}</h2>\n'
         agent_summary = all_results.get(tf)

         if agent_summary is None or agent_summary.get("error"):
              error_msg = agent_summary.get("error", "Unknown error") if agent_summary else "Unknown error."
              html_content += f"<p style='color:red;'>Error loading results: {error_msg}</p>"
              html_content += "</div>\n"; continue

         # --- Performance Summary Card ---
         html_content += "<h3>Performance Summary (Agent)</h3><div class=\"summary-card\">"
         pnl_val = agent_summary.get('total_pnl'); win_rate_val = agent_summary.get('win_rate'); trades_val = agent_summary.get('trade_count')
         pnl_str = formatNum(pnl_val); pnl_class = 'neutral' if pnl_val is None else ('profit' if pnl_val >= 0 else 'loss')
         html_content += f"<p><strong>Total PnL (Points):</strong> <span class='{pnl_class}'>{pnl_str}</span></p>"
         html_content += f"<p><strong>Total Trades:</strong> {trades_val if trades_val is not None else '-'}</p>"
         html_content += f"<p><strong>Win Rate:</strong> {formatNum(win_rate_val, 2, '%')}</p>"
         html_content += '</div>'

         # --- Trade Details Table ---
         html_content += "<h3>Trade Details (Agent)</h3>"
         trades = agent_summary.get('trades_details', [])
         if trades:
             html_content += "<div class=\"trade-table-container\">"
             try:
                 trades_df = pd.DataFrame(trades)
                 # Select and order columns
                 display_cols = ['Position', 'EntryTime', 'EntryPrice', 'ExitTime', 'ExitPrice', 'ExitReason', 'StrategyName', 'PnL_Points']
                 existing_display_cols = [col for col in display_cols if col in trades_df.columns]
                 trades_df_display = trades_df[existing_display_cols].copy()
                 # Formatting
                 for col in ['EntryPrice', 'ExitPrice', 'PnL_Points']:
                      if col in trades_df_display: trades_df_display[col] = trades_df_display[col].apply(lambda x: formatNum(x) if pd.notna(x) else '-')
                 trades_df_display.rename(columns={'PnL_Points': 'PnL (Points)', 'StrategyName': 'Strategy'}, inplace=True)
                 # Convert to HTML
                 html_content += trades_df_display.to_html(classes='performance-table trade-details', border=1, index=False, justify='right')
             except Exception as table_e: logger.error(f"Error generating HTML table for {tf}: {table_e}"); html_content += "<p>Error displaying trades.</p>"
             html_content += "</div>"
         else: html_content += "<p>No trades executed.</p>"
         html_content += "</div>\n" # Close tab-content div

     # --- JavaScript for Tabs ---
     html_script = """
     <script>
          function openTab(evt, tabName) { /* ... (keep JS as before) ... */ var i, tc, tl; tc=document.getElementsByClassName("tab-content"); for(i=0;i<tc.length;i++){tc[i].style.display="none"; tc[i].classList.remove("active");} tl=document.getElementsByClassName("tab-button"); for(i=0;i<tl.length;i++){tl[i].classList.remove("active");} var ct=document.getElementById(tabName); if(ct){ct.style.display="block"; ct.classList.add("active");} if(evt && evt.currentTarget){evt.currentTarget.classList.add("active");} else { for(i=0;i<tl.length;i++){ if(tl[i].getAttribute('onclick') && tl[i].getAttribute('onclick').includes("'" + tabName + "'")){ tl[i].classList.add("active"); break;}}} }
          // Initial tab activation might need adjustment if IDs changed
          // document.addEventListener('DOMContentLoaded', function() { /* ... */ }); // Keep initial activation
     </script>
     """
     return html_content + html_script