# -*- coding: utf-8 -*-
"""
portfolio_manager.py

This module defines the PortfolioManager class for tracking stock portfolio value
and composition over time, handling initial holdings and subsequent changes.

Dependencies:
- pandas, numpy, typing, re
- stock_plotter: Must be importable (e.g., same directory).
(See requirements.txt for specific versions)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging
from typing import Dict, List, Tuple, Optional, Union, Set

# --- Setup Logging ---
# Get a logger for this module. Configuration should be done by the application using this module.
log = logging.getLogger(__name__)

# --- Custom Exceptions ---
class PortfolioError(Exception):
    """Base class for exceptions in this module."""
    pass

class InvalidInputError(PortfolioError, ValueError):
    """Exception raised for errors in user-provided input."""
    pass

class DataFetchError(PortfolioError, IOError):
    """Exception raised for errors during data fetching (e.g., network, API)."""
    pass

class PortfolioBuildError(PortfolioError):
    """Exception raised for errors during the portfolio build process."""
    pass


# --- Dependency Requirement ---
try:
    # Assuming StockTrendPlotter is correctly implemented and handles its own logging/errors
    from stock_plotter import StockTrendPlotter
except ImportError as e:
    log.critical("Fatal Error: Could not import StockTrendPlotter.", exc_info=True)
    log.critical("Please ensure 'stock_plotter.py' is in the same directory or accessible in PYTHONPATH.")
    # Re-raise as a PortfolioError to signal the dependency issue clearly
    raise PortfolioError("Missing required dependency: StockTrendPlotter") from e


# --- Main Class Definition ---

class PortfolioManager:
    """
    Manages a stock portfolio's value and composition over time.
    Uses StockTrendPlotter to fetch prices and calculates daily values based on holdings.
    """

    def __init__(self, initial_portfolio: Dict[str, int], time_period: str):
        """
        Initializes the PortfolioManager.

        Args:
            initial_portfolio: Dict of {TICKER: quantity} for starting holdings.
            time_period: Time period string for StockTrendPlotter (e.g., '1y').

        Raises:
            InvalidInputError: If inputs are invalid (types, empty, negative quantity).
        """
        log.info("Initializing PortfolioManager...")
        # --- Input Validation ---
        if not isinstance(initial_portfolio, dict):
            raise InvalidInputError("initial_portfolio must be a dictionary.")
        if not initial_portfolio:
            raise InvalidInputError("initial_portfolio cannot be empty.")
        if not all(isinstance(k, str) and isinstance(v, int) and v >= 0
                   for k, v in initial_portfolio.items()):
            raise InvalidInputError("Portfolio keys must be strings (tickers), "
                                    "and values must be non-negative integers (quantities).")
        if not isinstance(time_period, str) or not time_period:
             raise InvalidInputError("time_period must be a non-empty string.")

        # --- Initialization ---
        self.initial_portfolio: Dict[str, int] = {ticker.upper(): qty
                                                   for ticker, qty in initial_portfolio.items()}
        self.time_period: str = time_period
        self._changes: List[Tuple[pd.Timestamp, str, Union[int, str]]] = []
        self._all_tickers: Set[str] = set(self.initial_portfolio.keys())
        self._portfolio_data: Optional[pd.DataFrame] = None

        log.info(f"PortfolioManager initialized for period: {self.time_period}")
        log.info(f"Initial Holdings: {self.initial_portfolio}")

    def add_change(self, date: str, ticker: str, quantity_change: Union[int, str]):
        """
        Records a change (transaction) in the quantity of a specific stock.

        Args:
            date: Date string ('YYYY-MM-DD' or similar).
            ticker: Stock ticker symbol.
            quantity_change: int (absolute quantity) or str ('+X' or '-X' relative).

        Raises:
            InvalidInputError: If date, ticker, or quantity_change format is invalid.
        """
        log.debug(f"Received add_change request: date={date}, ticker={ticker}, change={quantity_change}")
        # --- Input Validation ---
        try:
            change_date: pd.Timestamp = pd.to_datetime(date).tz_localize(None)
        except ValueError as e:
            raise InvalidInputError(f"Invalid date format: '{date}'. Use 'YYYY-MM-DD' or similar.") from e
        except Exception as e:
             raise InvalidInputError(f"Could not parse date '{date}': {e}") from e

        if not isinstance(ticker, str) or not ticker.strip():
             raise InvalidInputError("Ticker must be a non-empty string.")

        ticker_upper = ticker.strip().upper()
        change_type = "absolute"

        if isinstance(quantity_change, int):
            if quantity_change < 0:
                raise InvalidInputError("Absolute quantity_change must be non-negative.")
        elif isinstance(quantity_change, str):
            if not re.match(r'^[+-]\d+$', quantity_change.strip()):
                raise InvalidInputError(f"Invalid relative quantity format: '{quantity_change}'. Use '+X' or '-X'.")
            if int(quantity_change) == 0:
                 log.warning(f"Relative change '{quantity_change}' for {ticker_upper} on {change_date.date()} is zero.")
            change_type = "relative"
        else:
            raise InvalidInputError("quantity_change must be int (absolute) or str (relative '+X' / '-X').")

        # --- Record Change ---
        self._changes.append((change_date, ticker_upper, quantity_change))
        self._all_tickers.add(ticker_upper)
        self._portfolio_data = None # Invalidate cached data

        log.info(f"Change recorded ({change_type}): {change_date.date()} | {ticker_upper} | '{quantity_change}'. Portfolio data invalidated.")


    def _fetch_all_data(self) -> Optional[pd.DataFrame]:
        """
        Fetches historical 'Close' price data for all unique tickers.

        Returns:
            DataFrame with Date index and ticker columns (prices), or None if all fail.

        Raises:
            DataFetchError: If fetching fails significantly (e.g., network issues).
                            Individual ticker failures are logged as warnings.
        """
        log.info(f"Fetching historical price data for tickers: {sorted(list(self._all_tickers))}...")
        all_data: Dict[str, pd.DataFrame] = {}
        fetch_errors = 0

        for ticker in sorted(list(self._all_tickers)):
            log.debug(f"Attempting to fetch data for {ticker} using period '{self.time_period}'")
            try:
                # Instantiate the plotter - in tests, this will trigger the mock side effect
                plotter = StockTrendPlotter(ticker, self.time_period)
                # Call fetch_data on the (potentially mocked) instance
                df_ticker = plotter.fetch_data()

                # --- Basic Validation of Fetched Data ---
                if not isinstance(df_ticker, pd.DataFrame) or df_ticker.empty:
                     log.warning(f"No data returned for {ticker}. Skipping.")
                     fetch_errors += 1
                     continue
                if 'Date' not in df_ticker.columns or 'Price' not in df_ticker.columns:
                     log.warning(f"Invalid data structure (missing Date/Price columns) for {ticker}. Skipping.")
                     fetch_errors += 1
                     continue

                df_ticker['Date'] = pd.to_datetime(df_ticker['Date']).dt.tz_localize(None) # Ensure naive datetime
                df_ticker = df_ticker.set_index('Date')
                ticker_price_df = df_ticker[['Price']].rename(columns={'Price': ticker})
                all_data[ticker] = ticker_price_df
                log.info(f"Successfully fetched data for {ticker} ({len(df_ticker)} rows)")

            # Catch specific errors expected from StockTrendPlotter or yfinance
            except (ValueError, IOError) as e: # e.g., Invalid ticker, No data found, Network issue
                 log.warning(f"Could not fetch data for {ticker}: {type(e).__name__} - {e}")
                 fetch_errors += 1
            # Catch unexpected errors during the fetch process for a single ticker
            except Exception as e:
                 log.error(f"Unexpected error fetching data for {ticker}: {type(e).__name__} - {e}", exc_info=True)
                 fetch_errors += 1
                 # Depending on severity, could raise DataFetchError here immediately

        if not all_data:
            log.error("Failed to fetch data for *any* tickers.")
            # Raise an error if absolutely no data could be retrieved
            raise DataFetchError("Unable to fetch price data for any specified tickers.")

        if fetch_errors > 0:
             log.warning(f"Completed fetching data with {fetch_errors} ticker(s) failing.")

        # --- Combine and Clean Data ---
        log.info("Combining and cleaning fetched price data...")
        try:
            combined_df = pd.concat(all_data.values(), axis=1, join='outer')
            combined_df = combined_df.ffill() # Forward fill missing values

            # Ensure index is timezone-naive DatetimeIndex
            if not isinstance(combined_df.index, pd.DatetimeIndex):
                log.error("Combined data index is not a DatetimeIndex. Build process cannot continue.")
                raise PortfolioBuildError("Internal error: Combined price data index is not DatetimeIndex.")
            if combined_df.index.tz is not None:
                log.debug("Combined price data index has timezone, converting to naive.")
                combined_df.index = combined_df.index.tz_localize(None)

        except Exception as e:
             log.exception("Error during data combination/cleaning.")
             raise PortfolioBuildError("Failed to combine or clean fetched price data.") from e

        log.info("Price data fetching and preparation complete.")
        return combined_df


    def build_portfolio(self) -> Optional[pd.DataFrame]:
        """
        Constructs the portfolio history DataFrame by applying changes to fetched prices.

        Returns:
            The calculated portfolio history DataFrame, or None if build fails.

        Raises:
            DataFetchError: If underlying data fetching fails.
            PortfolioBuildError: If the build process itself encounters critical errors.
        """
        log.info("--- Starting Portfolio Build Process ---")
        try:
            # Step 1: Fetch price data
            price_data = self._fetch_all_data() # Can raise DataFetchError
            if price_data is None or price_data.empty:
                 # _fetch_all_data should raise if it fails completely, but double-check
                 log.error("Price data fetching returned None or empty DataFrame. Cannot build.")
                 raise PortfolioBuildError("Failed to obtain valid price data.")

            # Step 2: Initialize portfolio DataFrame
            log.info("Initializing portfolio structure...")
            portfolio_df = pd.DataFrame(index=price_data.index.copy())

            # Step 3: Initialize Quantities
            log.info("Setting initial quantities...")
            available_tickers = list(price_data.columns)
            processed_tickers = set()

            for ticker in available_tickers:
                qnty_col = f"{ticker}_qnty"
                initial_qty = self.initial_portfolio.get(ticker, 0)
                portfolio_df[qnty_col] = initial_qty
                processed_tickers.add(ticker)
                if initial_qty > 0: log.debug(f"  - Set initial {ticker}: {initial_qty}")

            for ticker in self._all_tickers:
                 if ticker not in processed_tickers and ticker in available_tickers:
                     qnty_col = f"{ticker}_qnty"
                     portfolio_df[qnty_col] = 0
                     processed_tickers.add(ticker)
                     log.debug(f"  - Initialized {ticker} (added via change): 0")

            # Step 4: Apply Quantity Changes
            log.info("Applying recorded quantity changes...")
            if not self._changes:
                log.info("  No changes to apply.")
            else:
                self._changes.sort(key=lambda x: x[0]) # Sort by date

                for change_date, ticker, change_specifier in self._changes:
                    change_date_naive = change_date # Already naive from add_change
                    qnty_col = f"{ticker}_qnty"
                    log.debug(f"Processing change: {change_date_naive.date()} | {ticker} | '{change_specifier}'")

                    if qnty_col not in portfolio_df.columns:
                         log.warning(f"  Skipping change for {ticker} on {change_date_naive.date()}: No quantity column (missing price data?).")
                         continue

                    # Find quantity before the change
                    idx_before_change = portfolio_df.index[portfolio_df.index < change_date_naive]
                    if not idx_before_change.empty:
                        last_date_before = idx_before_change.max()
                        quantity_before_change = portfolio_df.loc[last_date_before, qnty_col]
                    else:
                        quantity_before_change = self.initial_portfolio.get(ticker, 0)
                        log.debug(f"  Change for {ticker} on/before first data point. Using initial qty: {quantity_before_change}")

                    # Calculate new absolute quantity
                    new_absolute_quantity: Optional[int] = None
                    change_type_log = ""

                    if isinstance(change_specifier, int):
                        new_absolute_quantity = change_specifier
                        change_type_log = "absolute"
                    elif isinstance(change_specifier, str):
                        change_type_log = "relative"
                        try:
                            relative_change_val = int(change_specifier)
                            calculated_quantity = quantity_before_change + relative_change_val
                            if calculated_quantity < 0:
                                log.warning(f"  Relative change '{change_specifier}' for {ticker} on {change_date_naive.date()} "
                                            f"yielded negative quantity ({calculated_quantity}). Setting to 0.")
                                new_absolute_quantity = 0
                            else:
                                new_absolute_quantity = calculated_quantity
                        except ValueError: # Should not happen if add_change validation works
                            log.error(f"  Internal Error: Could not parse validated relative change '{change_specifier}'. Skipping.")
                            continue

                    # Apply the change from the date onwards
                    if new_absolute_quantity is not None:
                        date_mask = portfolio_df.index >= change_date_naive
                        if date_mask.any():
                            portfolio_df.loc[date_mask, qnty_col] = new_absolute_quantity
                            log.info(f"  Applied {change_type_log}: {ticker} qty -> {new_absolute_quantity} from {change_date_naive.date()} (was {quantity_before_change})")
                        else:
                             # Check if change date is after the data range
                             if not portfolio_df.empty and change_date_naive > portfolio_df.index.max():
                                 log.warning(f"  Change date {change_date_naive.date()} for {ticker} is after last data point. Change ineffective.")
                             else:
                                 log.warning(f"  No dates found on or after {change_date_naive.date()} for {ticker}. Change ineffective.")


            # Step 5: Calculate Value Columns
            log.info("Calculating daily portfolio values...")
            portfolio_df['Total_Value'] = 0.0
            present_qnty_cols = [col for col in portfolio_df.columns if col.endswith('_qnty')]
            active_tickers_final = [col.replace('_qnty','') for col in present_qnty_cols]

            for ticker in active_tickers_final:
                if ticker not in price_data.columns:
                    log.warning(f"  Price data missing for {ticker} during value calculation. Skipping value.")
                    continue
                price_col = ticker
                qnty_col = f"{ticker}_qnty"
                value_col = f"{ticker}_value"

                # Calculate value, coercing price errors to NaN, then filling NaN results with 0
                portfolio_df[value_col] = pd.to_numeric(price_data[price_col], errors='coerce') * portfolio_df[qnty_col]
                portfolio_df[value_col] = portfolio_df[value_col].fillna(0)
                portfolio_df['Total_Value'] = portfolio_df['Total_Value'].add(portfolio_df[value_col], fill_value=0)

            log.debug("  Value calculations complete.")

            # Step 6: Final Formatting
            log.info("Finalizing DataFrame structure...")
            qnty_cols_sorted = sorted([col for col in portfolio_df.columns if col.endswith('_qnty')])
            value_cols_sorted = sorted([col for col in portfolio_df.columns if col.endswith('_value')])
            final_ordered_cols = qnty_cols_sorted + value_cols_sorted + ['Total_Value']
            portfolio_df = portfolio_df[[col for col in final_ordered_cols if col in portfolio_df.columns]]

            # --- Store Result and Return ---
            self._portfolio_data = portfolio_df
            log.info("--- Portfolio Build Process Finished Successfully ---")
            return self._portfolio_data

        except DataFetchError as e:
             log.error(f"Portfolio build failed due to data fetch error: {e}")
             self._portfolio_data = None # Ensure cache is cleared on failure
             raise # Re-raise the specific error
        except PortfolioBuildError as e:
             log.error(f"Portfolio build failed during processing: {e}")
             self._portfolio_data = None
             raise
        except Exception as e:
             log.exception("An unexpected error occurred during portfolio build.")
             self._portfolio_data = None
             # Wrap unexpected errors in a PortfolioBuildError
             raise PortfolioBuildError("Unexpected failure during portfolio build.") from e


    def get_portfolio_data(self) -> Optional[pd.DataFrame]:
        """
        Retrieves the calculated portfolio history DataFrame, building it if necessary.

        Returns:
            A *copy* of the portfolio history DataFrame, or None if build fails.

        Raises:
            DataFetchError, PortfolioBuildError: If build() is triggered and fails.
        """
        if self._portfolio_data is None:
            log.info("Portfolio data not built or invalidated. Triggering build...")
            try:
                 self.build_portfolio() # This can raise errors
            except (DataFetchError, PortfolioBuildError) as e:
                 log.error("Failed to build portfolio data when requested.")
                 # Re-raise the error caught from build_portfolio
                 raise e
            except Exception as e:
                 # Catch any other unexpected error during the build triggered here
                 log.exception("Unexpected error during implicit portfolio build.")
                 raise PortfolioBuildError("Unexpected failure during portfolio build trigger.") from e


        if self._portfolio_data is None:
             # This case should ideally be covered by exceptions raised above, but as a safeguard:
             log.error("Portfolio data is None even after build attempt.")
             return None
        else:
             log.info("Returning portfolio data...")
             return self._portfolio_data.copy()


    def get_composition(self, date: Optional[str] = None) -> Optional[pd.Series]:
        """
        Calculates portfolio composition (% value of each holding) on a specific date
        or the latest available date. Uses nearest date if exact match not found.

        Args:
            date: Target date string ('YYYY-MM-DD') or None for latest.

        Returns:
            Series {ticker: percentage} or None if data unavailable/invalid date/zero total value.

        Raises:
            InvalidInputError: If the provided date string is invalid.
            DataFetchError, PortfolioBuildError: If get_portfolio_data() triggers a build that fails.
        """
        log.info(f"Calculating portfolio composition for date: {'Latest' if date is None else date}")
        try:
            # get_portfolio_data handles build logic and can raise errors
            portfolio_df = self.get_portfolio_data()
        except (DataFetchError, PortfolioBuildError) as e:
            log.error(f"Cannot get composition because portfolio data build failed: {e}")
            return None # Or re-raise depending on desired behavior

        if portfolio_df is None or portfolio_df.empty:
            log.error("Portfolio data is not available or empty. Cannot calculate composition.")
            return None

        target_date_actual: Optional[pd.Timestamp] = None
        data_row: Optional[pd.Series] = None

        # --- Determine Target Date and Row ---
        if date:
            try:
                target_date = pd.to_datetime(date).tz_localize(None)
                log.debug(f"Parsed requested date: {target_date.date()}")

                if target_date in portfolio_df.index:
                     target_date_actual = target_date
                     data_row = portfolio_df.loc[target_date_actual]
                     log.debug(f"Exact date match found: {target_date_actual.date()}")
                else:
                    log.warning(f"Exact date '{date}' not found in portfolio data. Finding nearest.")
                    # Use get_indexer with method='nearest'
                    idx_loc = portfolio_df.index.get_indexer([target_date], method='nearest')[0]
                    if idx_loc == -1:
                         raise ValueError("Nearest index lookup failed (empty index?).") # Should be caught below
                    target_date_actual = portfolio_df.index[idx_loc]
                    data_row = portfolio_df.iloc[idx_loc]
                    log.info(f"Using nearest available date: {target_date_actual.date()}")
                    if abs((target_date_actual - target_date).days) > 3:
                        log.warning(f"  Nearest date is more than 3 days away from requested date '{date}'.")

            except ValueError as e_date:
                log.error(f"Invalid date format or value '{date}': {e_date}")
                raise InvalidInputError(f"Invalid date format or value: '{date}'") from e_date
            except Exception as e_lookup:
                 log.error(f"Error looking up date '{date}' in portfolio data: {e_lookup}", exc_info=True)
                 # Could raise a different error or return None
                 return None
        else:
            # Use the last date if no date specified
            target_date_actual = portfolio_df.index[-1]
            data_row = portfolio_df.iloc[-1]
            log.debug(f"No date specified. Using latest available date: {target_date_actual.date()}")

        if data_row is None or target_date_actual is None:
             log.error("Internal error: Could not retrieve data row for composition.")
             return None # Should not happen if logic above is correct

        # --- Calculate Composition ---
        total_value = data_row.get('Total_Value')
        log.debug(f"Total value on {target_date_actual.date()}: {total_value}")

        if total_value is None or pd.isna(total_value) or total_value <= 0:
            log.warning(f"Total portfolio value is zero, NaN, or negative ({total_value}) "
                        f"on {target_date_actual.date()}. Returning zero composition.")
            composition = pd.Series({col.replace('_value', ''): 0.0
                                     for col in portfolio_df.columns if col.endswith('_value')})
            return composition

        composition_dict = {}
        for col in portfolio_df.columns:
            if col.endswith('_value'):
                ticker = col.replace('_value', '')
                value = data_row.get(col, 0.0)
                composition_dict[ticker] = (value / total_value) * 100.0

        composition_series = pd.Series(composition_dict).sort_values(ascending=False)
        log.info(f"Composition calculated successfully for {target_date_actual.date()}.")
        return composition_series
