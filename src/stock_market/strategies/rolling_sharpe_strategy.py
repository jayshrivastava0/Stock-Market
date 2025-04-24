#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of a Rolling Sharpe Ratio Based Trading Strategy.

This strategy selects assets based on their historical risk-adjusted returns,
quantified by the Sharpe Ratio, calculated over a rolling window. It aims
to hold assets that have demonstrated better performance relative to their
volatility.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Import core components from the backtester module
from backtester import (
    Strategy,
    Order,
    Timestamp,
    HoldingsDict,
    TickerSymbol,
    BUY_ACTION,
    SELL_ACTION,
    ConfigError,
    StrategyError,
)


class RollingSharpeStrategy(Strategy):
    """
    A strategy that invests in assets with the highest rolling Sharpe Ratio.

    Calculates the annualized Sharpe Ratio for each asset over a rolling window.
    It aims to hold the top N assets ranked by Sharpe Ratio, subject to a minimum
    Sharpe Ratio threshold. Uses equal weighting for position sizing among targets.

    Note: Calculation involves rolling standard deviation. Performance might
    be impacted by very long lookback windows or a large number of assets.

    Strategy Parameters:
        - sharpe_window (int): Lookback period (in data points) for calculating
                               rolling returns and volatility (default: 90).
        - num_holdings (int): Target number of assets to hold (default: 3).
        - min_sharpe_ratio (float): Minimum annualized Sharpe Ratio required
                                    to consider an asset for buying (default: 0.5).
        - risk_free_rate (float): The annualized risk-free rate used in the
                                  Sharpe Ratio calculation (default: 0.01 for 1%).
        - data_frequency (str): Frequency of the data ('daily', 'hourly', etc.)
                                Used to annualize the Sharpe Ratio.
                                (default: 'daily').
    """

    # Default values for parameters
    DEFAULT_SHARPE_WINDOW = 90
    DEFAULT_NUM_HOLDINGS = 3
    DEFAULT_MIN_SHARPE_RATIO = 0.5
    DEFAULT_RISK_FREE_RATE = 0.01
    DEFAULT_DATA_FREQUENCY = 'daily'


    # Annualization factors based on common data frequencies
    ANNUALIZATION_FACTORS = {
        'daily': 252,
        'hourly': 252 * 6.5, # Example assuming 6.5 trading hours/day
        'minute': 252 * 6.5 * 60,
        # Add other frequencies as needed
    }


    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initializes the RollingSharpeStrategy.

        Args:
            parameters (Optional[Dict[str, Any]]): Dictionary of strategy parameters.
        """
        # >>> CORRECTED ORDER <<<
        # 1. Call super().__init__ which sets self.parameters and calls _validate_parameters
        super().__init__(parameters)

        # 2. Extract validated parameters into instance attributes
        self.sharpe_window: int = self.parameters.get('sharpe_window', self.DEFAULT_SHARPE_WINDOW)
        self.num_holdings: int = self.parameters.get('num_holdings', self.DEFAULT_NUM_HOLDINGS)
        self.min_sharpe_ratio: float = self.parameters.get('min_sharpe_ratio', self.DEFAULT_MIN_SHARPE_RATIO)
        self.risk_free_rate: float = self.parameters.get('risk_free_rate', self.DEFAULT_RISK_FREE_RATE)
        self.data_frequency: str = self.parameters.get('data_frequency', self.DEFAULT_DATA_FREQUENCY)

        # 3. Set annualization factor based on validated frequency
        #    (Validation happens in _validate_parameters called by super().__init__)
        self.annualization_factor = self.ANNUALIZATION_FACTORS[self.data_frequency]


        # --- Internal State ---
        # No complex state needed if recalculating Sharpe each step based on slice

        print(
            f"RollingSharpeStrategy initialized with: window={self.sharpe_window}, "
            f"num_holdings={self.num_holdings}, min_sharpe={self.min_sharpe_ratio}, "
            f"rf_rate={self.risk_free_rate:.2%}, freq='{self.data_frequency}'"
        )
        # Note: _validate_parameters already called by super().__init__()


    def _validate_parameters(self) -> None:
        """
        Validates the strategy-specific parameters by reading from self.parameters.

        Called by the base class constructor *before* instance attributes are set.
        """
        # --- Validation logic uses self.parameters.get() ---
        sharpe_window = self.parameters.get('sharpe_window', self.DEFAULT_SHARPE_WINDOW)
        if not isinstance(sharpe_window, int) or sharpe_window <= 10: # Need sufficient data for std dev
            raise ConfigError(f"'sharpe_window' must be an integer > 10. Got: {sharpe_window}")

        num_holdings = self.parameters.get('num_holdings', self.DEFAULT_NUM_HOLDINGS)
        if not isinstance(num_holdings, int) or num_holdings <= 0:
            raise ConfigError(f"'num_holdings' must be a positive integer. Got: {num_holdings}")

        min_sharpe_ratio = self.parameters.get('min_sharpe_ratio', self.DEFAULT_MIN_SHARPE_RATIO)
        if not isinstance(min_sharpe_ratio, (int, float)):
            raise ConfigError(f"'min_sharpe_ratio' must be a number. Got: {min_sharpe_ratio}")

        risk_free_rate = self.parameters.get('risk_free_rate', self.DEFAULT_RISK_FREE_RATE)
        if not isinstance(risk_free_rate, (int, float)):
            raise ConfigError(f"'risk_free_rate' must be a number. Got: {risk_free_rate}")

        data_frequency = self.parameters.get('data_frequency', self.DEFAULT_DATA_FREQUENCY)
        if not isinstance(data_frequency, str):
            raise ConfigError(f"'data_frequency' must be a string. Got: {data_frequency}")
        if data_frequency not in self.ANNUALIZATION_FACTORS:
            raise ConfigError(f"Unsupported 'data_frequency': {data_frequency}. "
                              f"Supported: {list(self.ANNUALIZATION_FACTORS.keys())}")


    def _calculate_sharpe_ratio(self, price_series: pd.Series) -> Optional[float]:
        """
        Calculates the annualized Sharpe Ratio for a given price series.

        Args:
            price_series (pd.Series): Series of prices (e.g., 'Close').

        Returns:
            Optional[float]: The calculated annualized Sharpe Ratio, or None if
                             insufficient data or calculation error (e.g., zero std dev).
        """
        required_points = self.sharpe_window + 1 # Need +1 to calculate returns
        if len(price_series.dropna()) < required_points:
            return None # Not enough data

        try:
            # Calculate periodic returns (using 'Close' prices)
            periodic_returns = price_series.pct_change().dropna()

            # Select the returns within the rolling window
            window_returns = periodic_returns.iloc[-self.sharpe_window:]

            if len(window_returns) < self.sharpe_window:
                return None # Should not happen with initial check, but safety

            # Calculate mean and standard deviation of window returns
            mean_return = window_returns.mean()
            std_dev = window_returns.std()

            # Handle cases with zero standard deviation (constant price)
            if std_dev is None or pd.isna(std_dev) or std_dev <= 1e-9: # Use small epsilon for check
                # Returning None is safest to avoid division by zero / misleading signals
                return None

            # Calculate periodic risk-free rate
            periodic_rf_rate = (1 + self.risk_free_rate)**(1 / self.annualization_factor) - 1

            # Calculate Sharpe Ratio for the period
            sharpe_ratio_periodic = (mean_return - periodic_rf_rate) / std_dev

            # Annualize the Sharpe Ratio
            annualized_sharpe = sharpe_ratio_periodic * np.sqrt(self.annualization_factor)

            return annualized_sharpe

        except Exception as e:
            # Log or print the error for debugging if needed
            # print(f"Warning: Error calculating Sharpe Ratio: {e}")
            return None

    def generate_signals(
        self,
        current_dt: Timestamp,
        data_slice: pd.DataFrame,
        current_holdings: HoldingsDict,
        current_cash: float
    ) -> List[Order]:
        """
        Generates trading orders based on asset Sharpe Ratio rankings.

        Args:
            current_dt: The current timestamp of the simulation step.
            data_slice: Historical market data up to current_dt.
            current_holdings: Current portfolio holdings.
            current_cash: Current available cash.

        Returns:
            A list of Order objects for the current step.
        """
        orders: List[Order] = []
        sharpe_ratios: Dict[TickerSymbol, float] = {}
        available_tickers: pd.Index = data_slice.columns.get_level_values(1).unique()

        # 1. Calculate Sharpe Ratio for all available tickers
        for ticker in available_tickers:
            try:
                # Use 'Close' price for calculations
                price_series = data_slice[('Close', ticker)]
                sharpe = self._calculate_sharpe_ratio(price_series)

                # Store if calculation is valid and meets minimum threshold
                if sharpe is not None and not np.isnan(sharpe) and sharpe >= self.min_sharpe_ratio:
                    sharpe_ratios[ticker] = sharpe

            except KeyError:
                # print(f"Warning: 'Close' price data not found for {ticker} at {current_dt}. Skipping Sharpe calc.")
                continue
            except Exception as e:
                print(f"Error calculating Sharpe Ratio for {ticker} at {current_dt}: {e}")
                continue

        # 2. Rank tickers by Sharpe Ratio (highest first)
        ranked_tickers: List[TickerSymbol] = sorted(
            sharpe_ratios,
            key=lambda t: sharpe_ratios[t],
            reverse=True
        )

        # 3. Determine target holdings based on rank and num_holdings parameter
        target_tickers: set[TickerSymbol] = set(ranked_tickers[:self.num_holdings])
        current_held_tickers: set[TickerSymbol] = {
            ticker for ticker, quantity in current_holdings.items() if quantity > 0
        }

        # 4. Generate SELL orders for assets no longer meeting criteria
        tickers_to_sell: set[TickerSymbol] = current_held_tickers - target_tickers
        for ticker in tickers_to_sell:
            quantity_to_sell = current_holdings.get(ticker, 0)
            if quantity_to_sell > 0:
                orders.append(Order(ticker=ticker, action=SELL_ACTION, quantity=quantity_to_sell))
                # print(f"{current_dt.date()}: SELL signal (Low Sharpe/Rank) for {ticker}")


        # 5. Generate BUY orders for new target assets (using equal weight sizing)
        tickers_to_buy: set[TickerSymbol] = target_tickers - current_held_tickers

        num_target_positions = len(target_tickers)
        if num_target_positions == 0 or not tickers_to_buy:
             return orders # No targets or no new buys needed

        # Estimate total equity for weighting (using last close price)
        estimated_holdings_value = 0.0
        for ticker, quantity in current_holdings.items():
             try:
                 last_close = data_slice[('Close', ticker)].iloc[-1]
                 if not pd.isna(last_close):
                      estimated_holdings_value += float(quantity) * float(last_close)
             except (KeyError, IndexError): pass

        estimated_total_equity = current_cash + estimated_holdings_value
        target_value_per_position = estimated_total_equity / num_target_positions
        cash_for_new_buys = current_cash
        cash_per_buy = cash_for_new_buys / len(tickers_to_buy) if tickers_to_buy else 0


        buy_orders_pending: List[Tuple[TickerSymbol, int]] = []
        for ticker in tickers_to_buy:
            try:
                last_price = data_slice[('Close', ticker)].iloc[-1]
                if pd.isna(last_price) or last_price <= 1e-6: # Check for non-positive price
                    continue

                # Ideal quantity based on target value, adjusted for affordability
                ideal_quantity = int(target_value_per_position / last_price)
                affordable_quantity = int(cash_per_buy / last_price)
                quantity = min(ideal_quantity, affordable_quantity)

                if quantity > 0:
                     buy_orders_pending.append((ticker, quantity))

            except (KeyError, IndexError):
                # print(f"Warning: Price data unavailable for potential buy {ticker} at {current_dt}.")
                continue
            except Exception as e:
                print(f"Error sizing position for {ticker} at {current_dt}: {e}")
                continue

        # Add generated buy orders
        for ticker, quantity in buy_orders_pending:
             if quantity > 0:
                orders.append(Order(ticker=ticker, action=BUY_ACTION, quantity=quantity))
                # print(f"{current_dt.date()}: BUY signal (High Sharpe) for {ticker}, quantity {quantity}")

        return orders