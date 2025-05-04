# app/agentic_core.py

import pandas as pd
import logging
from typing import Dict, Tuple, Callable, Optional
import inspect
from app.config import config
from app.strategies import strategy_functions

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RuleBasedAgent:
    def __init__(self):
        # --- MODIFIED: Store name along with function ---
        self.strategy_mapping: Dict[str, Optional[Tuple[str, Callable]]] = {
            # Regime -> (Strategy Name String, Strategy Function)
            "Trending": ("EMA_Crossover", strategy_functions.get("EMA_Crossover")),
            "Ranging":  ("RSI_Basic", strategy_functions.get("RSI_Basic")),
            "Unknown":  None,
        }
        # --- End Modification ---
        self.parameter_mapping: Dict[str, Dict[str, float]] = {
             "Trending": { "sl_mult": getattr(config, "TREND_SL_ATR_MULT", config.DEFAULT_SL_ATR_MULT * 1.2),
                           "tp_mult": getattr(config, "TREND_TP_ATR_MULT", config.DEFAULT_TP_ATR_MULT * 1.5) },
             "Ranging":  { "sl_mult": getattr(config, "RANGE_SL_ATR_MULT", config.DEFAULT_SL_ATR_MULT * 0.8),
                           "tp_mult": getattr(config, "RANGE_TP_ATR_MULT", config.DEFAULT_TP_ATR_MULT * 0.8) },
             "Unknown":  { "sl_mult": config.DEFAULT_SL_ATR_MULT, "tp_mult": config.DEFAULT_TP_ATR_MULT }
        }

        # Validation
        for regime, mapping_tuple in self.strategy_mapping.items():
            if mapping_tuple is None and regime != "Unknown":
                 logger.warning(f"No strategy mapping found for regime '{regime}'.")
            elif mapping_tuple is not None:
                 name, func = mapping_tuple
                 if not callable(func):
                      logger.error(f"Mapped strategy '{name}' for regime '{regime}' is not callable.")
                      self.strategy_mapping[regime] = None

        logger.info("RuleBasedAgent initialized.")
        # Log mapped names
        log_map = {k: v[0] if v else None for k,v in self.strategy_mapping.items()}
        logger.info(f"Strategy Mapping: {log_map}")
        logger.info(f"Parameter Mapping: {self.parameter_mapping}")
# app/agentic_core.py

# (Keep imports, logger setup, class definition, __init__ method)

    # --- MODIFIED decide method with MORE logging ---
    def decide(self, current_row: pd.Series, data_history: pd.DataFrame = None) -> Tuple[str, float, float, Optional[str]]:
        """
        Makes a decision for the current bar. Logs intermediate steps.
        Returns: Tuple[str, float, float, Optional[str]]
        """
        regime = current_row.get('regime', 'Unknown')
        strategy_mapping = self.strategy_mapping.get(regime, None)
        params = self.parameter_mapping.get(regime, self.parameter_mapping["Unknown"])
        sl_mult = params['sl_mult']
        tp_mult = params['tp_mult']

        signal = 'hold' # Default signal
        selected_strategy_name = None

        if strategy_mapping:
            selected_strategy_name, selected_strategy_func = strategy_mapping
            logger.debug(f"Agent: Regime '{regime}' mapped to Strategy '{selected_strategy_name}'") # Log mapping
            if selected_strategy_func:
                try:
                    # --- Call the strategy function ---
                   
                    if "data_history" in inspect.signature(selected_strategy_func).parameters:
                        potential_signal = selected_strategy_func(row=current_row, data_history=data_history)
                    else:
                        potential_signal = selected_strategy_func(row=current_row)

                    # --- Log the RAW signal received ---
                    logger.debug(f"Agent: Raw signal received from '{selected_strategy_name}': '{potential_signal}'")

                    # --- Logic to determine final signal ---
                    if potential_signal in ['buy_potential', 'sell_potential', 'hold']:
                         signal = potential_signal # Assign the received signal
                    else:
                         logger.warning(f"Agent: Strategy '{selected_strategy_name}' returned unexpected signal: '{potential_signal}'. Defaulting to 'hold'.")
                         signal = 'hold'

                except Exception as e:
                    logger.error(f"Agent: Error executing strategy '{selected_strategy_name}' for regime '{regime}': {e}", exc_info=False) # Log concise error
                    signal = 'hold'; selected_strategy_name = None # Reset on error
            else:
                 signal = 'hold'; selected_strategy_name = None
                 logger.debug(f"Agent: Strategy function for '{selected_strategy_name}' is invalid or None.")
        else:
            logger.debug(f"Agent: No strategy mapped for Regime='{regime}'.")

        # Log the final decision being returned by the agent
        logger.debug(f"Agent Decision: Returning Signal='{signal}', SL Mult={sl_mult:.2f}, TP Mult={tp_mult:.2f}, Strategy='{selected_strategy_name}'")
        return signal, sl_mult, tp_mult, selected_strategy_name
    # --- End decide method ---
        """
        Makes a decision for the current bar.

        Returns:
            Tuple[str, float, float, Optional[str]]:
             (signal, sl_atr_multiplier, tp_atr_multiplier, strategy_name_or_None)
             Signal can be 'buy_potential', 'sell_potential', or 'hold'.
        """
        regime = current_row.get('regime', 'Unknown')
        strategy_mapping = self.strategy_mapping.get(regime, None) # Gets tuple (name, func) or None
        params = self.parameter_mapping.get(regime, self.parameter_mapping["Unknown"])
        sl_mult = params['sl_mult']
        tp_mult = params['tp_mult']

        signal = 'hold' # Default signal
        selected_strategy_name = None # Default

        if strategy_mapping:
            selected_strategy_name, selected_strategy_func = strategy_mapping
            if selected_strategy_func: # Check if function is valid
                try:
                    # Call the strategy function (assuming it only needs current_row)
                    potential_signal = selected_strategy_func(row=current_row)
                    logger.debug(f" Raw signal from {selected_strategy_name}: {potential_signal}") # Log raw signal

                    # --- CORRECTED LOGIC ---
                    # Pass through 'buy_potential', 'sell_potential', or 'hold'
                    if potential_signal in ['buy_potential', 'sell_potential', 'hold']:
                        signal = potential_signal # Use the signal from the strategy
                    else:
                         # If strategy returns something unexpected (like 'buy' directly, which it shouldn't)
                         logger.warning(f"Strategy {selected_strategy_name} returned unexpected signal: '{potential_signal}'. Treating as 'hold'.")
                         signal = 'hold'
                    # --- END CORRECTION ---

                    # Log the final decision being returned by the agent
                    logger.debug(f"[{current_row.name}] Regime='{regime}'. Agent using '{selected_strategy_name}'. FinalAgentSignal='{signal}'. SL={sl_mult:.2f}*ATR, TP={tp_mult:.2f}*ATR")

                except Exception as e:
                    logger.error(f"Error executing strategy {selected_strategy_name} for regime {regime}: {e}", exc_info=True)
                    signal = 'hold'; selected_strategy_name = None # Reset on error
            else: # Function was not callable / None
                 signal = 'hold'; selected_strategy_name = None
                 logger.debug(f"[{current_row.name}] Strategy func invalid for regime '{regime}'. Signal='hold'.")
        else: # No strategy mapped for this regime
            logger.debug(f"[{current_row.name}] Regime='{regime}'. No strategy selected. Signal='hold'. SL={sl_mult:.2f}*ATR, TP={tp_mult:.2f}*ATR")

        # Return the CORRECT signal ('buy_potential', 'sell_potential', 'hold'), params, AND strategy name
        return signal, sl_mult, tp_mult, selected_strategy_name