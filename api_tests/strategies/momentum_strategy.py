#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of a Momentum Trading Strategy.

This strategy selects assets based on their past performance (momentum)
over a specified lookback period. It aims to hold a portfolio of the
top-performing assets, implicitly targeting higher growth (related to CAGR).
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


class MomentumStrategy(Strategy):
    """
    A momentum strategy that buys the top N performing assets and sells losers.

    This strategy calculates the rate of change (momentum) for each asset
    over a defined lookback period. It ranks the assets based on this momentum
    and aims to hold a fixed number of the top-ranked assets, distributing
    capital equally among them (or using a fixed quantity per asset). Assets
    that fall out of the top N are sold.

    Strategy Parameters:
        - momentum_window (int): The lookback period (in data points) for
                                 calculating momentum (default: 90).
        - num_holdings (int): The target number of assets to hold in the
                              portfolio (default: 3).
        - position_sizing (str): Method for sizing positions. Options:
                                 'equal_weight' (distributes capital equally
                                 among `num_holdings`), 'fixed_quantity'
                                 (uses `buy_quantity`). (default: 'equal_weight').
        - buy_quantity (int): The fixed number of shares to buy for each
                              position if `position_sizing` is 'fixed_quantity'
                              (default: 10). Ignored for 'equal_weight'.
        - require_positive_momentum (bool): If True, only considers assets
                                           with positive momentum for buying.
                                           (default: True).
    """
    # Default values defined here for clarity and use in validation
    DEFAULT_MOMENTUM_WINDOW = 90
    DEFAULT_NUM_HOLDINGS = 3
    DEFAULT_POSITION_SIZING = 'equal_weight'
    DEFAULT_BUY_QUANTITY = 10
    DEFAULT_REQUIRE_POSITIVE_MOMENTUM = True

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initializes the MomentumStrategy.

        Args:
            parameters (Optional[Dict[str, Any]]): Dictionary of strategy
                                                  parameters.
        """
        # >>> CORRECTED ORDER <<<
        # 1. Call super().__init__ which sets self.parameters and calls _validate_parameters
        super().__init__(parameters)

        # 2. Now extract validated parameters into instance attributes
        self.momentum_window: int = self.parameters.get('momentum_window', self.DEFAULT_MOMENTUM_WINDOW)
        self.num_holdings: int = self.parameters.get('num_holdings', self.DEFAULT_NUM_HOLDINGS)
        self.position_sizing: str = self.parameters.get('position_sizing', self.DEFAULT_POSITION_SIZING)
        self.buy_quantity: int = self.parameters.get('buy_quantity', self.DEFAULT_BUY_QUANTITY)
        self.require_positive_momentum: bool = self.parameters.get('require_positive_momentum', self.DEFAULT_REQUIRE_POSITIVE_MOMENTUM)

        # --- Internal State (if needed) ---
        # No complex state needed for this basic momentum strategy.

        # Logging initialization details (uses the now set instance attributes)
        print(
            f"MomentumStrategy initialized with: "
            f"window={self.momentum_window}, num_holdings={self.num_holdings}, "
            f"sizing='{self.position_sizing}', buy_qty={self.buy_quantity}, "
            f"req_pos_mom={self.require_positive_momentum}"
        )
        # Note: The _validate_parameters call happened inside super().__init__()


    def _validate_parameters(self) -> None:
        """
        Validates the strategy-specific parameters by reading from self.parameters.

        This method is called by the base class constructor *before* instance
        attributes like self.momentum_window are set. Therefore, it must access
        the parameters dictionary directly.
        """
        # --- Validation logic now uses self.parameters.get() ---
        momentum_window = self.parameters.get('momentum_window', self.DEFAULT_MOMENTUM_WINDOW)
        if not isinstance(momentum_window, int) or momentum_window <= 1:
            raise ConfigError(f"'momentum_window' must be an integer greater than 1. Got: {momentum_window}")

        num_holdings = self.parameters.get('num_holdings', self.DEFAULT_NUM_HOLDINGS)
        if not isinstance(num_holdings, int) or num_holdings <= 0:
            raise ConfigError(f"'num_holdings' must be a positive integer. Got: {num_holdings}")

        position_sizing = self.parameters.get('position_sizing', self.DEFAULT_POSITION_SIZING)
        if position_sizing not in ['equal_weight', 'fixed_quantity']:
            raise ConfigError(f"'position_sizing' must be 'equal_weight' or 'fixed_quantity'. Got: {position_sizing}")

        if position_sizing == 'fixed_quantity':
             buy_quantity = self.parameters.get('buy_quantity', self.DEFAULT_BUY_QUANTITY)
             if not isinstance(buy_quantity, int) or buy_quantity <= 0:
                 raise ConfigError(f"'buy_quantity' must be a positive integer when using 'fixed_quantity' sizing. Got: {buy_quantity}")

        require_positive_momentum = self.parameters.get('require_positive_momentum', self.DEFAULT_REQUIRE_POSITIVE_MOMENTUM)
        if not isinstance(require_positive_momentum, bool):
            raise ConfigError(f"'require_positive_momentum' must be a boolean. Got: {require_positive_momentum}")

        # If validation passes, parameters are considered okay for assignment in __init__

    def _calculate_momentum(self, price_series: pd.Series) -> Optional[float]:
        """
        Calculates the momentum (rate of change) for a given price series.

        Args:
            price_series (pd.Series): Series of prices (e.g., 'Close').

        Returns:
            Optional[float]: The calculated momentum value, or None if
                             insufficient data or calculation error.
        """
        if len(price_series.dropna()) < self.momentum_window:
            return None  # Not enough data

        try:
            # Using simple rate of change: (last_price / price_n_periods_ago) - 1
            # Shift avoids lookahead - uses price from window steps ago relative to *current* last price
            price_n_periods_ago = price_series.shift(self.momentum_window - 1).iloc[-1]
            last_price = price_series.iloc[-1]

            if pd.isna(last_price) or pd.isna(price_n_periods_ago) or price_n_periods_ago == 0:
                return None # Cannot calculate

            momentum = (last_price / price_n_periods_ago) - 1.0
            return momentum

        except IndexError:
             # Should be caught by length check, but defensive programming
             return None
        except Exception as e:
            # Log or print the error for debugging if needed
            # print(f"Warning: Error calculating momentum: {e}")
            return None

    def generate_signals(
        self,
        current_dt: Timestamp,
        data_slice: pd.DataFrame,
        current_holdings: HoldingsDict,
        current_cash: float
    ) -> List[Order]:
        """
        Generates trading orders based on asset momentum rankings.

        Args:
            current_dt: The current timestamp of the simulation step.
            data_slice: Historical market data up to current_dt.
            current_holdings: Current portfolio holdings.
            current_cash: Current available cash.

        Returns:
            A list of Order objects for the current step.
        """
        orders: List[Order] = []
        momentum_scores: Dict[TickerSymbol, float] = {}
        available_tickers: pd.Index = data_slice.columns.get_level_values(1).unique()

        # 1. Calculate Momentum for all available tickers
        for ticker in available_tickers:
            try:
                # Use 'Close' price for momentum calculation
                price_series = data_slice[('Close', ticker)]
                momentum = self._calculate_momentum(price_series)

                if momentum is not None:
                    # Optionally filter out assets with negative momentum
                    if not self.require_positive_momentum or momentum > 0:
                         momentum_scores[ticker] = momentum

            except KeyError:
                # print(f"Warning: 'Close' price data not found for {ticker} at {current_dt}. Skipping.")
                continue
            except Exception as e:
                # Catch unexpected errors during momentum calculation for a ticker
                print(f"Error calculating momentum for {ticker} at {current_dt}: {e}")
                continue # Skip this ticker on error

        # 2. Rank tickers by momentum (highest momentum first)
        # Ensure scores are valid floats before sorting
        valid_scores = {t: s for t, s in momentum_scores.items() if isinstance(s, (float, np.number)) and not np.isnan(s)}
        ranked_tickers: List[TickerSymbol] = sorted(
            valid_scores,
            key=lambda t: valid_scores[t],
            reverse=True
        )

        # 3. Determine target holdings based on rank
        target_tickers: set[TickerSymbol] = set(ranked_tickers[:self.num_holdings])
        current_held_tickers: set[TickerSymbol] = {
            ticker for ticker, quantity in current_holdings.items() if quantity > 0
        }

        # 4. Generate SELL orders for assets no longer in target set
        tickers_to_sell: set[TickerSymbol] = current_held_tickers - target_tickers
        for ticker in tickers_to_sell:
            quantity_to_sell = current_holdings.get(ticker, 0)
            if quantity_to_sell > 0:
                orders.append(Order(ticker=ticker, action=SELL_ACTION, quantity=quantity_to_sell))
                # print(f"{current_dt.date()}: SELL signal (Fallen Momentum) for {ticker}")


        # 5. Generate BUY orders for new target assets
        tickers_to_buy: set[TickerSymbol] = target_tickers - current_held_tickers

        # Calculate capital available for new buys (consider existing holdings value maybe? Simplification: use current cash)
        # For equal weight, calculate per-position allocation
        num_target_positions = len(target_tickers) # Use actual number targeted, could be < num_holdings if few qualify
        if num_target_positions == 0 or not tickers_to_buy: # Exit if no targets or no new buys needed
             return orders # Only process sells generated above

        # Estimate total portfolio value for weighting (crude estimate for allocation)
        # Use last close price for estimation - execution uses Open price next bar.
        estimated_holdings_value = 0.0 # Use float
        for ticker, quantity in current_holdings.items():
             try:
                 last_close = data_slice[('Close', ticker)].iloc[-1]
                 if not pd.isna(last_close):
                      estimated_holdings_value += float(quantity) * float(last_close) # Ensure float math
             except (KeyError, IndexError):
                  pass # Ignore if price is missing for held asset estimate

        estimated_total_equity = current_cash + estimated_holdings_value
        cash_for_new_buys = current_cash # Simplification: Use available cash directly

        # Determine quantity per buy order
        buy_orders_pending: List[Tuple[TickerSymbol, int]] = []
        if self.position_sizing == 'equal_weight':
            target_value_per_position = estimated_total_equity / num_target_positions
            # Allocate cash proportionally if not enough for all targets
            cash_per_buy = cash_for_new_buys / len(tickers_to_buy) # Use float division

            for ticker in tickers_to_buy:
                try:
                    # Use last close for estimation, backtester uses Open price for actual execution
                    last_price = data_slice[('Close', ticker)].iloc[-1]
                    if pd.isna(last_price) or last_price <= 1e-6: # Check for NaN and non-positive price
                        # print(f"Warning: Cannot size position for {ticker}, invalid price {last_price}")
                        continue

                    # Ideal quantity based on target value, adjusted for available cash per buy
                    ideal_quantity = int(target_value_per_position / last_price)
                    affordable_quantity = int(cash_per_buy / last_price)
                    quantity = min(ideal_quantity, affordable_quantity) # Buy based on affordability

                    if quantity > 0:
                         buy_orders_pending.append((ticker, quantity))

                except (KeyError, IndexError):
                    # print(f"Warning: Price data unavailable for potential buy {ticker} at {current_dt}.")
                    continue
                except Exception as e:
                    print(f"Error sizing position for {ticker} at {current_dt}: {e}")
                    continue

        elif self.position_sizing == 'fixed_quantity':
            # Track cash allocated to avoid overspending estimation
            estimated_cash_used = 0.0
            for ticker in tickers_to_buy:
                 try:
                     last_price = data_slice[('Close', ticker)].iloc[-1]
                     if pd.isna(last_price) or last_price <= 1e-6:
                         continue
                     estimated_cost = self.buy_quantity * last_price

                     # Check if enough *remaining estimated* cash
                     if estimated_cost <= (cash_for_new_buys - estimated_cash_used):
                         buy_orders_pending.append((ticker, self.buy_quantity))
                         estimated_cash_used += estimated_cost # Update cash used estimate
                     else:
                         # print(f"{current_dt.date()}: BUY signal for {ticker}, but potential insufficient cash (estimated).")
                         pass
                 except (KeyError, IndexError):
                     continue


        # Add generated buy orders to the list
        for ticker, quantity in buy_orders_pending:
             if quantity > 0:
                orders.append(Order(ticker=ticker, action=BUY_ACTION, quantity=quantity))
                # print(f"{current_dt.date()}: BUY signal (Top Momentum) for {ticker}, quantity {quantity}")

        return orders