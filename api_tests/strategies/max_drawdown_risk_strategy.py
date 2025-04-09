#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of a Maximum Drawdown Risk Management Strategy Overlay.

This strategy monitors the drawdown of each held asset from its recent peak.
If an asset's drawdown exceeds a predefined threshold, it generates a SELL
signal for that asset. It does NOT generate BUY signals on its own.
It's intended to be used potentially in combination with another signal
generation strategy or as a standalone risk control mechanism assuming
initial positions are established elsewhere.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# Import core components from the backtester module
from backtester import (
    Strategy,
    Order,
    Timestamp,
    HoldingsDict,
    TickerSymbol,
    BUY_ACTION,  # Although not used for generation, import for consistency
    SELL_ACTION,
    ConfigError,
    StrategyError,
)


class MaxDrawdownRiskStrategy(Strategy):
    """
    A risk management strategy that sells assets exceeding a max drawdown threshold.

    This strategy calculates the drawdown for each currently held asset based
    on its price history over a specified rolling window. If the current price
    drops below a certain percentage from the rolling high price (peak),
    a SELL order is generated to exit the position.

    This strategy *only* generates SELL signals based on drawdown risk. It
    relies on other mechanisms (manual initial holdings or another strategy)
    to establish positions.

    Strategy Parameters:
        - drawdown_window (int): The lookback period (in data points) for
                                 calculating the rolling high price (default: 60).
        - max_drawdown_pct (float): The maximum allowable drawdown percentage
                                    (e.g., 0.15 for 15%). If drawdown exceeds
                                    this, a sell signal is triggered (default: 0.15).
        - price_source (str): The price column to use for drawdown calculation
                              (e.g., 'Close', 'High'). (default: 'Close').
    """
    # Default values for parameters
    DEFAULT_DRAWDOWN_WINDOW = 60
    DEFAULT_MAX_DRAWDOWN_PCT = 0.15
    DEFAULT_PRICE_SOURCE = 'Close'

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initializes the MaxDrawdownRiskStrategy.

        Args:
            parameters (Optional[Dict[str, Any]]): Dictionary of strategy
                                                  parameters.
        """
        # >>> CORRECTED ORDER <<<
        # 1. Call super().__init__ which sets self.parameters and calls _validate_parameters
        super().__init__(parameters)

        # 2. Now extract validated parameters into instance attributes
        self.drawdown_window: int = self.parameters.get('drawdown_window', self.DEFAULT_DRAWDOWN_WINDOW)
        self.max_drawdown_pct: float = self.parameters.get('max_drawdown_pct', self.DEFAULT_MAX_DRAWDOWN_PCT)
        self.price_source: str = self.parameters.get('price_source', self.DEFAULT_PRICE_SOURCE)


        # --- Internal State ---
        # Not strictly needed here as recalculating each time based on slice
        # self.rolling_highs: Dict[TickerSymbol, float] = {}

        print(
            f"MaxDrawdownRiskStrategy initialized with: "
            f"window={self.drawdown_window}, max_dd_pct={self.max_drawdown_pct:.2%}, "
            f"price_source='{self.price_source}'"
        )
        # Note: _validate_parameters already called by super().__init__()


    def _validate_parameters(self) -> None:
        """
        Validates the strategy-specific parameters by reading from self.parameters.

        Called by the base class constructor *before* instance attributes are set.
        """
        # --- Validation logic uses self.parameters.get() ---
        drawdown_window = self.parameters.get('drawdown_window', self.DEFAULT_DRAWDOWN_WINDOW)
        if not isinstance(drawdown_window, int) or drawdown_window <= 1:
            raise ConfigError(f"'drawdown_window' must be an integer greater than 1. Got: {drawdown_window}")

        max_drawdown_pct = self.parameters.get('max_drawdown_pct', self.DEFAULT_MAX_DRAWDOWN_PCT)
        if not isinstance(max_drawdown_pct, float) or \
           not (0 < max_drawdown_pct < 1):
            raise ConfigError(f"'max_drawdown_pct' must be a float between 0 and 1 (exclusive). Got: {max_drawdown_pct}")

        price_source = self.parameters.get('price_source', self.DEFAULT_PRICE_SOURCE)
        # Example: Add more robust check if needed based on expected columns from data handler
        valid_sources = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] # Example valid columns
        if not isinstance(price_source, str) or price_source not in valid_sources:
             raise ConfigError(f"Invalid 'price_source': {price_source}. Must be one of {valid_sources} (and present in data).")


    def _calculate_current_drawdown(
        self,
        price_series: pd.Series
    ) -> Optional[float]:
        """
        Calculates the current drawdown from the rolling high.

        Args:
            price_series (pd.Series): Series of prices (e.g., 'Close') for the asset.

        Returns:
            Optional[float]: The current drawdown percentage (as a positive value,
                             e.g., 0.10 for 10% drawdown), or None if insufficient
                             data or calculation error. Returns 0.0 if price is at the peak.
        """
        # Ensure enough data points for the rolling window calculation
        if len(price_series.dropna()) < self.drawdown_window:
            return None

        try:
            # Calculate the rolling maximum price over the specified window
            rolling_high = price_series.rolling(
                window=self.drawdown_window, min_periods=self.drawdown_window # Ensure full window before calculating
            ).max()

            # Get the most recent rolling high and the current price
            current_price = price_series.iloc[-1]
            latest_rolling_high = rolling_high.iloc[-1] # Get the max over the window ending at current_dt

            if pd.isna(current_price) or pd.isna(latest_rolling_high) or latest_rolling_high <= 1e-9: # Check for NaN or zero/neg high
                 return None

            # Calculate drawdown: (Rolling High - Current Price) / Rolling High
            drawdown = (latest_rolling_high - current_price) / latest_rolling_high

            # Ensure drawdown isn't negative due to floating point issues if price equals high
            return max(0.0, drawdown)

        except IndexError:
             return None
        except Exception as e:
             # print(f"Warning: Error calculating drawdown: {e}")
             return None


    def generate_signals(
        self,
        current_dt: Timestamp,
        data_slice: pd.DataFrame,
        current_holdings: HoldingsDict,
        current_cash: float  # Unused in this strategy's logic but part of signature
    ) -> List[Order]:
        """
        Generates SELL orders for assets exceeding the maximum drawdown threshold.

        Args:
            current_dt: The current timestamp of the simulation step.
            data_slice: Historical market data up to current_dt.
            current_holdings: Current portfolio holdings.
            current_cash: Current available cash (unused).

        Returns:
            A list of SELL Order objects for the current step, or an empty list.
        """
        orders: List[Order] = []
        # Only need to check tickers currently held
        held_tickers: List[TickerSymbol] = [
            ticker for ticker, quantity in current_holdings.items() if quantity > 0
        ]

        for ticker in held_tickers:
            try:
                # Access the price series specified by the parameter
                price_series = data_slice[(self.price_source, ticker)]

                # Calculate the current drawdown for this asset
                current_drawdown = self._calculate_current_drawdown(price_series)

                # Check if drawdown calculation was successful and exceeds threshold
                if current_drawdown is not None and current_drawdown > self.max_drawdown_pct:
                    # Generate a SELL order for the entire holding
                    quantity_to_sell = current_holdings[ticker]
                    orders.append(Order(ticker=ticker, action=SELL_ACTION, quantity=quantity_to_sell))
                    # print(f"{current_dt.date()}: SELL signal (Max Drawdown {current_drawdown:.2%}) for {ticker}")

            except KeyError:
                 # This might happen if the price_source column is missing for the ticker
                 # print(f"Warning: Price data ('{self.price_source}', {ticker}) not found at {current_dt}. Skipping drawdown check.")
                 continue
            except Exception as e:
                 # Catch unexpected errors during drawdown calculation for a ticker
                 print(f"Error processing drawdown for {ticker} at {current_dt}: {e}")
                 continue # Skip this ticker on error

        # This strategy does not generate BUY signals
        return orders