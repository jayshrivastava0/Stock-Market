#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Manager Module

Manages the value and composition of a stock portfolio over time,
tracking initial holdings and subsequent transactions. Utilizes the
data_handler module for efficient historical price fetching.
"""

import logging
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# --- Setup Logging ---
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler()) # Avoid duplicate handlers if app configures root logger

# --- Custom Exceptions ---
class PortfolioError(Exception):
    """Base class for exceptions specific to the portfolio_manager module."""
    pass

class InvalidInputError(PortfolioError, ValueError):
    """Exception raised for errors in user-provided input (config, changes)."""
    pass

class DataFetchError(PortfolioError, IOError):
    """Exception raised for underlying errors during data fetching via data_handler."""
    pass

class PortfolioBuildError(PortfolioError):
    """Exception raised for errors during the portfolio history build process."""
    pass

# --- Dependency ---
try:
    # Import the core fetching function and its specific exceptions
    from data_handler import (fetch_stock_data,
                              DataFetchError as DH_DataFetchError,
                              InvalidInputError as DH_InvalidInputError,
                              DataProcessingError as DH_DataProcessingError)
except ImportError as e:
    log.critical("Fatal Error: Could not import from data_handler module.", exc_info=True)
    log.critical("Please ensure 'data_handler.py' is accessible in the Python path.")
    # Re-raise as a PortfolioError to signal the dependency issue clearly
    raise PortfolioError("Missing required dependency: data_handler module") from e


# --- Constants ---
# Define standard Yahoo Finance periods for date parsing helper
VALID_YF_PERIODS: Set[str] = frozenset({
    '1d', '5d', '1mo', '3mo', '6mo',
    '1y', '2y', '5y', '10y', 'ytd', 'max'
})
# Define the price column to use from fetched data (Adjusted Close is usually preferred)
PRICE_COLUMN_NAME: str = 'Close'


# --- Main Class Definition ---

class PortfolioManager:
    """
    Manages and analyzes a stock portfolio's value and composition over time.

    Fetches historical stock prices using the data_handler module and calculates
    daily portfolio values based on specified holdings and transactions. Supports
    flexible time period specifications for analysis.
    """

    def __init__(self, initial_portfolio: Dict[str, int], time_period: str):
        """
        Initializes the PortfolioManager.

        Args:
            initial_portfolio: Dictionary mapping Ticker symbols (str) to initial
                               quantity held (int >= 0). Tickers are case-insensitive.
            time_period: The analysis period. Supports:
                         - Standard Yahoo Finance periods ('1y', '6mo', 'ytd', 'max', etc.).
                         - Relative periods ('3M', '2W', '10D').
                         - Natural language ('from YYYY[-MM]', 'this year', 'this month').

        Raises:
            InvalidInputError: If inputs are invalid (e.g., non-dict portfolio,
                               negative quantities, empty time_period).
        """
        log.info("Initializing PortfolioManager...")
        # --- Input Validation ---
        if not isinstance(initial_portfolio, dict):
            raise InvalidInputError("initial_portfolio must be a dictionary.")
        if not initial_portfolio:
            log.warning("Initial portfolio is empty. Manager initialized but will likely yield zero values.")
            # Allow empty for flexibility, but warn. Could raise InvalidInputError if strictly disallowed.
            # raise InvalidInputError("initial_portfolio cannot be empty.")
        if not all(isinstance(k, str) and k.strip() and isinstance(v, int) and v >= 0
                   for k, v in initial_portfolio.items()):
            raise InvalidInputError(
                "Portfolio keys must be non-empty strings (tickers), "
                "and values must be non-negative integers (quantities)."
            )
        if not isinstance(time_period, str) or not time_period.strip():
             raise InvalidInputError("time_period must be a non-empty string.")

        # --- Initialization ---
        # Standardize tickers to uppercase
        self.initial_portfolio: Dict[str, int] = {
            ticker.upper().strip(): qty
            for ticker, qty in initial_portfolio.items()
        }
        self.time_period: str = time_period.strip()
        # _changes stores (Timestamp, Ticker, QuantitySpecifier) tuples
        self._changes: List[Tuple[pd.Timestamp, str, Union[int, str]]] = []
        # _all_tickers tracks unique tickers involved (initial + changes)
        self._all_tickers: Set[str] = set(self.initial_portfolio.keys())
        # _portfolio_data caches the generated portfolio DataFrame
        self._portfolio_data: Optional[pd.DataFrame] = None
        # _date_range caches the determined analysis start/end dates
        self._date_range: Optional[Tuple[date, date]] = None

        log.info(f"PortfolioManager initialized for time period: '{self.time_period}'")
        log.info(f"Initial Holdings: {self.initial_portfolio if self.initial_portfolio else 'None'}")


    def add_change(self, date_str: str, ticker: str, quantity_change: Union[int, str]):
        """
        Records a change (transaction) affecting the quantity of a specific stock.

        Changes invalidate any previously calculated portfolio data cache.

        Args:
            date_str: Date of the change in 'YYYY-MM-DD' format or similar
                      parseable by pandas.to_datetime.
            ticker: Stock ticker symbol (case-insensitive).
            quantity_change: Either:
                             - An integer (>= 0) representing the *absolute* quantity held
                               from this date onwards.
                             - A string like '+X' or '-X' (e.g., '+10', '-5') representing
                               a *relative* change from the quantity held just before this date.

        Raises:
            InvalidInputError: If date format, ticker format, or quantity_change
                               format/value is invalid.
        """
        log.debug(f"Received add_change request: date='{date_str}', ticker='{ticker}', change='{quantity_change}'")

        # --- Input Validation ---
        try:
            # Parse date and ensure it's timezone-naive for internal consistency
            change_date: pd.Timestamp = pd.to_datetime(date_str).tz_localize(None)
        except ValueError as e:
            raise InvalidInputError(f"Invalid date format: '{date_str}'. Use 'YYYY-MM-DD' or similar.") from e
        except Exception as e: # Catch other potential pandas errors
             raise InvalidInputError(f"Could not parse date '{date_str}': {e}") from e

        if not isinstance(ticker, str) or not ticker.strip():
             raise InvalidInputError("Ticker must be a non-empty string.")
        ticker_upper = ticker.strip().upper()

        # Validate quantity_change format and value
        change_type = "absolute"
        if isinstance(quantity_change, int):
            if quantity_change < 0:
                raise InvalidInputError(f"Absolute quantity change for {ticker_upper} must be non-negative, got: {quantity_change}")
        elif isinstance(quantity_change, str):
            change_specifier_stripped = quantity_change.strip()
            if not re.match(r'^[+-]\d+$', change_specifier_stripped):
                raise InvalidInputError(f"Invalid relative quantity format: '{quantity_change}'. Use '+X' or '-X'.")
            try:
                relative_val = int(change_specifier_stripped)
                if relative_val == 0:
                    log.warning(f"Relative change '{quantity_change}' for {ticker_upper} on {change_date.date()} results in zero quantity change.")
            except ValueError:
                 # Should be caught by regex, but defensive check
                 raise InvalidInputError(f"Could not parse relative quantity integer from '{quantity_change}'.")
            change_type = "relative"
            quantity_change = change_specifier_stripped # Store cleaned version
        else:
            raise InvalidInputError(f"quantity_change for {ticker_upper} must be int (absolute >= 0) or str (relative '+X' / '-X').")

        # --- Record Change ---
        self._changes.append((change_date, ticker_upper, quantity_change))
        self._all_tickers.add(ticker_upper)
        # Invalidate cached data as portfolio structure/holdings have changed
        self._portfolio_data = None
        self._date_range = None # Date range might need recalculation if period is 'max'

        log.info(f"Change recorded ({change_type}): {change_date.date()} | {ticker_upper} | '{quantity_change}'. Portfolio data cache invalidated.")

    def _parse_date_range_natural_lang(self, date_input_lower: str, today: date) -> Optional[Tuple[date, date]]:
        """Parses natural language date ranges like 'this year', 'from YYYY[-MM]'."""
        log.debug(f"Attempting natural language parse: '{date_input_lower}' relative to {today}")

        if 'from' in date_input_lower:
            # Match 'from YYYY' or 'from YYYY-MM'
            match = re.match(r'.*from\s+(\d{4})(?:-(\d{1,2}))?', date_input_lower)
            if match:
                year = int(match.group(1))
                month_str = match.group(2)
                month = int(month_str) if month_str else 1
                try:
                    start_date = date(year, month, 1)
                    log.debug(f"Parsed 'from' date: {start_date}")
                    return (start_date, today) if start_date <= today else (None) # Ensure start date isn't future
                except ValueError: # Invalid date like 2020-13
                    log.warning(f"Invalid date components parsed from 'from' expression in '{date_input_lower}'")
                    return None

        elif 'this month' == date_input_lower:
            start_date = today.replace(day=1)
            log.debug(f"Parsed 'this month', start_date: {start_date}")
            return start_date, today

        elif 'this year' == date_input_lower:
            start_date = today.replace(month=1, day=1)
            log.debug(f"Parsed 'this year', start_date: {start_date}")
            return start_date, today

        return None # No match

    def _parse_date_range_relative(self, period_input: str, today: date) -> Optional[Tuple[date, date]]:
        """Parses relative time period patterns like '7M', '2W', '10D', '3Y'."""
        log.debug(f"Attempting relative period parse: '{period_input}' relative to {today}")
        match = re.match(r'^(\d+)([DdWwMmYy])$', period_input)
        if not match:
            return None

        try:
            qty = int(match.group(1))
            unit = match.group(2).lower()

            if qty <= 0:
                log.warning(f"Relative period quantity must be positive, got {qty} in '{period_input}'")
                return None

            delta_args = {}
            if unit == 'd': delta_args['days'] = qty
            elif unit == 'w': delta_args['weeks'] = qty
            elif unit == 'm': delta_args['months'] = qty
            elif unit == 'y': delta_args['years'] = qty
            else: return None # Should not happen due to regex

            start_date = today - relativedelta(**delta_args)
            log.debug(f"Parsed relative period '{period_input}', start_date: {start_date}")
            return start_date, today

        except ValueError: # int conversion error
            log.error(f"Could not parse quantity from relative period '{period_input}'")
            return None
        except Exception as e: # relativedelta calculation error
            log.error(f"Error calculating relative delta for '{period_input}': {e}", exc_info=True)
            return None

    def _determine_date_range(self) -> Tuple[date, date]:
        """
        Determines the start and end dates for data fetching based on `self.time_period`.

        Handles standard YF periods, relative periods, and natural language.
        Caches the result in `self._date_range`.

        Returns:
            A tuple (start_date, end_date) as date objects.

        Raises:
            InvalidInputError: If the time_period format is invalid or cannot be parsed.
            PortfolioError: For internal errors during date calculation.
        """
        if self._date_range:
            log.debug(f"Using cached date range: {self._date_range[0]} to {self._date_range[1]}")
            return self._date_range

        log.info(f"Determining date range for time period: '{self.time_period}'")
        today = date.today()
        period_lower = self.time_period.lower()
        start_date: Optional[date] = None
        end_date: date = today # Default end date is today

        # 1. Try Natural Language
        parsed_range = self._parse_date_range_natural_lang(period_lower, today)
        if parsed_range:
            start_date, end_date = parsed_range
            log.info(f"Using natural language date range: {start_date} to {end_date}")

        # 2. Try Standard Yahoo Finance period
        elif period_lower in VALID_YF_PERIODS:
            log.info(f"Parsing standard Yahoo Finance period: '{period_lower}'")
            try:
                if period_lower == 'ytd':
                    start_date = today.replace(month=1, day=1)
                elif period_lower == 'max':
                    # Need earliest transaction/initial holding date for 'max'
                    # For simplicity without fetching history first, use a distant past date.
                    # A better 'max' might need refinement based on actual data.
                    log.warning("'max' period uses a fixed early start date (1970-01-01). "
                                "Consider specifying 'from YYYY' for more accuracy.")
                    start_date = date(1970, 1, 1) # Or fetch earliest data point later
                else:
                    # Use relativedelta for periods like '1y', '6mo', etc.
                    match = re.match(r'^(\d+)(mo|y)$', period_lower)
                    if match:
                        qty = int(match.group(1))
                        unit = match.group(2)
                        delta_args = {'months': qty} if unit == 'mo' else {'years': qty}
                        start_date = today - relativedelta(**delta_args)
                    elif period_lower == '1d': # Special case for 1d (fetch today)
                        start_date = today
                    elif period_lower == '5d': # 5 *business* days is tricky, approximate as 7 calendar days
                         start_date = today - timedelta(days=7) # yfinance likely handles business days
                    else:
                        raise InvalidInputError(f"Unhandled standard period format: '{period_lower}'")
            except Exception as e:
                 log.error(f"Error parsing standard period '{period_lower}': {e}", exc_info=True)
                 raise PortfolioError(f"Internal error parsing standard period '{period_lower}'.") from e

        # 3. Try Relative period ('3M', '2W', etc.)
        else:
            parsed_range = self._parse_date_range_relative(self.time_period, today)
            if parsed_range:
                start_date, end_date = parsed_range
                log.info(f"Using relative date range: {start_date} to {end_date}")

        # Validation and Caching
        if start_date is None:
            # If all parsing failed
            error_msg = (
                f"Invalid time period format: '{self.time_period}'.\n"
                f"Use a standard YF period ({', '.join(VALID_YF_PERIODS)}), "
                f"a relative pattern (e.g., '7M', '10D'), "
                f"or natural language (e.g., 'from 2020', 'this year')."
            )
            log.error(error_msg)
            raise InvalidInputError(error_msg)

        if start_date > end_date:
             # Should generally not happen with parsing logic, but check
             log.warning(f"Determined start date {start_date} is after end date {end_date}. Using end date as start.")
             start_date = end_date

        self._date_range = (start_date, end_date)
        log.info(f"Determined analysis date range: {start_date} to {end_date}")
        return self._date_range


    def _fetch_all_data(self) -> pd.DataFrame:
        """
        Fetches historical price data for all relevant tickers using data_handler.

        Determines the required date range based on `self.time_period`.

        Returns:
            DataFrame with MultiIndex columns ('DataType', 'Ticker') containing
            fetched price data (e.g., 'Adj Close').

        Raises:
            InvalidInputError: If the time_period is invalid.
            DataFetchError: If data fetching fails via data_handler.
            PortfolioError: For other internal errors.
        """
        if not self._all_tickers:
            log.warning("No tickers specified in portfolio or changes. Cannot fetch data.")
            # Return an empty DataFrame with expected structure? Or raise? Let's return empty.
            return pd.DataFrame(columns=pd.MultiIndex.from_product([[],[]], names=['DataType', 'Ticker']))

        log.info(f"Fetching historical price data for tickers: {sorted(list(self._all_tickers))}")
        try:
            start_date, end_date = self._determine_date_range()
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            log.debug(f"Calling data_handler.fetch_stock_data for range {start_str} to {end_str}")

            # Use data_handler.fetch_stock_data
            price_data = fetch_stock_data(
                tickers=list(self._all_tickers),
                start_date_str=start_str,
                end_date_str=end_str,
                use_adj_close=True # Fetch Adj Close by default
            )
            log.info("Price data fetched successfully via data_handler.")
            return price_data

        except (DH_InvalidInputError, DH_DataFetchError, DH_DataProcessingError) as e:
             # Catch specific errors from data_handler
             log.error(f"Data retrieval failed via data_handler: {type(e).__name__} - {e}", exc_info=True)
             # Wrap in our DataFetchError for consistency upstream
             raise DataFetchError(f"Failed to fetch portfolio data: {e}") from e
        except InvalidInputError:
            # Raised by _determine_date_range
            raise # Re-raise directly
        except Exception as e:
            log.exception("An unexpected error occurred during data fetching preparation.")
            raise PortfolioError("Unexpected error during data fetch preparation.") from e


    def build_portfolio(self) -> pd.DataFrame:
        """
        Constructs the portfolio history DataFrame.

        Fetches necessary price data via `_fetch_all_data`, calculates daily
        holdings based on initial state and recorded changes, and computes
        daily values for each holding and the total portfolio. Caches the result.

        Returns:
            A pandas DataFrame indexed by date, containing columns for:
            - `<TICKER>_qnty`: Quantity of each ticker held on that day.
            - `<TICKER>_value`: Market value of each ticker holding on that day.
            - `Total_Value`: Sum of all holding values for that day.
            Returns an empty DataFrame if build fails or no data is available.

        Raises:
            DataFetchError: If underlying data fetching fails.
            PortfolioBuildError: If the build process encounters calculation errors.
        """
        if self._portfolio_data is not None:
            log.info("Returning cached portfolio data.")
            return self._portfolio_data.copy() # Return a copy

        log.info("--- Starting Portfolio Build Process ---")
        try:
            # Step 1: Fetch price data using the new method
            price_data = self._fetch_all_data() # Can raise DataFetchError, InvalidInputError

            if price_data.empty:
                 log.warning("Fetched price data is empty. Portfolio build will result in an empty DataFrame.")
                 # Create an empty frame with expected columns if possible? Difficult without tickers.
                 # For now, let it proceed; calculation loops will handle empty inputs.
                 # Alternatively, raise PortfolioBuildError("No price data available.")
                 pass # Allow process to continue, result will likely be empty


            # Ensure index is DatetimeIndex and timezone-naive (data_handler should ensure this, but double-check)
            if not price_data.empty:
                if not isinstance(price_data.index, pd.DatetimeIndex) or price_data.index.tz is not None:
                    log.error("Price data index is not a timezone-naive DatetimeIndex after fetch. Build cannot continue reliably.")
                    raise PortfolioBuildError("Internal error: Price data index format incorrect.")

            # Step 2: Initialize portfolio DataFrame using price data's index
            log.info("Initializing portfolio structure...")
            # Use price_data's index if available, otherwise determine range again? Safest is price data's index.
            # If price_data is empty, create an index based on determined range to handle zero-value portfolio case.
            if not price_data.empty:
                portfolio_df = pd.DataFrame(index=price_data.index.copy())
                log.debug(f"Initialized portfolio DataFrame with index from price data (length {len(portfolio_df)})")
            else:
                 # Handle case with no price data (maybe invalid tickers, market closed for range?)
                 try:
                     start_dt_idx, end_dt_idx = self._determine_date_range()
                     # Create a date range - use daily frequency for structure
                     empty_index = pd.date_range(start=start_dt_idx, end=end_dt_idx, freq='D', name='Date')
                     portfolio_df = pd.DataFrame(index=empty_index)
                     log.warning(f"Initialized portfolio DataFrame with empty daily index ({start_dt_idx} to {end_dt_idx}) due to no price data.")
                 except Exception as date_err:
                      log.error(f"Could not create index for empty portfolio build: {date_err}", exc_info=True)
                      raise PortfolioBuildError("Failed to initialize structure for empty portfolio.")


            # Step 3: Initialize Quantities based on initial portfolio
            log.info("Setting initial quantities...")
            # Get tickers for which we actually have price data columns
            # Ticker is the second level (index 1) of the MultiIndex
            available_tickers = []
            if isinstance(price_data.columns, pd.MultiIndex) and price_data.columns.nlevels > 1:
                available_tickers = price_data.columns.get_level_values(1).unique().tolist()

            processed_tickers = set()
            for ticker in self._all_tickers: # Iterate through ALL known tickers
                qnty_col = f"{ticker}_qnty"
                initial_qty = self.initial_portfolio.get(ticker, 0)
                portfolio_df[qnty_col] = initial_qty # Initialize column for all dates
                processed_tickers.add(ticker)
                if initial_qty > 0:
                    log.debug(f"  - Initialized {qnty_col} = {initial_qty}")
                elif ticker in self.initial_portfolio:
                    log.debug(f"  - Initialized {qnty_col} = 0 (from initial portfolio)")
                else:
                     log.debug(f"  - Initialized {qnty_col} = 0 (ticker added via change)")


            # Step 4: Apply Quantity Changes chronologically
            log.info(f"Applying {len(self._changes)} recorded quantity changes...")
            if self._changes:
                # Sort changes by date, then potentially by ticker or type if needed for tie-breaking (unlikely needed here)
                self._changes.sort(key=lambda x: x[0])

                for change_date, ticker, change_specifier in self._changes:
                    qnty_col = f"{ticker}_qnty"
                    log.debug(f"Processing change: {change_date.date()} | {ticker} | '{change_specifier}'")

                    if qnty_col not in portfolio_df.columns:
                         # This shouldn't happen with the new init logic, but safeguard
                         log.warning(f"  Skipping change for {ticker} on {change_date.date()}: Quantity column '{qnty_col}' not found.")
                         continue

                    # Find quantity *just before* the change date
                    # Use index slicing and get the last value before the change date
                    try:
                        # Get index locations strictly before the change date
                        relevant_indices = portfolio_df.index[portfolio_df.index < change_date]
                        if not relevant_indices.empty:
                            last_date_before = relevant_indices.max()
                            quantity_before_change = portfolio_df.loc[last_date_before, qnty_col]
                            log.debug(f"  Qty for {ticker} on {last_date_before.date()} (before change): {quantity_before_change}")
                        else:
                            # Change is on or before the first date in the index. Use initial qty.
                            quantity_before_change = self.initial_portfolio.get(ticker, 0)
                            log.debug(f"  Change date {change_date.date()} is on/before first index date. Using initial qty {quantity_before_change} as base.")
                    except KeyError:
                        # Handle cases where the last date lookup might fail (e.g., index issues)
                        log.error(f"  Error finding quantity before change for {ticker} on {change_date.date()}. Defaulting to initial 0.", exc_info=True)
                        quantity_before_change = 0 # Safe default
                    except Exception as e:
                         log.exception(f"Unexpected error determining quantity before change for {ticker}: {e}")
                         quantity_before_change = 0 # Safe default


                    # Calculate the new absolute quantity based on the change specifier
                    new_absolute_quantity: Optional[int] = None
                    change_type_log = ""

                    if isinstance(change_specifier, int):
                        # Absolute quantity set
                        new_absolute_quantity = change_specifier
                        change_type_log = "absolute set"
                    elif isinstance(change_specifier, str):
                        # Relative quantity change ('+X' or '-X')
                        change_type_log = "relative adjust"
                        try:
                            relative_change_val = int(change_specifier)
                            calculated_quantity = int(quantity_before_change) + relative_change_val
                            if calculated_quantity < 0:
                                log.warning(f"  Relative change '{change_specifier}' for {ticker} on {change_date.date()} "
                                            f"yielded negative quantity ({calculated_quantity}) from base {quantity_before_change}. Setting to 0.")
                                new_absolute_quantity = 0
                            else:
                                new_absolute_quantity = calculated_quantity
                        except ValueError: # Should not happen if add_change validation works
                            log.error(f"  Internal Error: Could not parse validated relative change '{change_specifier}' for {ticker}. Skipping change application.")
                            continue
                        except TypeError: # If quantity_before_change was NaN or unexpected type
                             log.error(f"  TypeError calculating relative change for {ticker}. Base quantity was '{quantity_before_change}'. Setting to 0.")
                             new_absolute_quantity = 0


                    # Apply the new quantity from the change date onwards
                    if new_absolute_quantity is not None:
                        # Find all dates in the index >= change_date
                        date_mask = portfolio_df.index >= change_date
                        if date_mask.any():
                            portfolio_df.loc[date_mask, qnty_col] = new_absolute_quantity
                            log.info(f"  Applied {change_type_log}: {ticker} quantity -> {new_absolute_quantity} from {change_date.date()} onwards.")
                        else:
                             # Check if change date is after the last date in our data range
                             if not portfolio_df.empty and change_date > portfolio_df.index.max():
                                 log.warning(f"  Change date {change_date.date()} for {ticker} is after the last date in the analysis range ({portfolio_df.index.max().date()}). Change is ineffective.")
                             else:
                                 log.warning(f"  No dates found in portfolio index on or after {change_date.date()} for {ticker}. Change application skipped.")

            # Step 5: Calculate Value Columns using specified price (e.g., Adj Close)
            log.info(f"Calculating daily portfolio values using '{PRICE_COLUMN_NAME}' prices...")
            portfolio_df['Total_Value'] = 0.0 # Initialize total value column

            # Use tickers present in the quantity columns for calculation
            qnty_cols = [col for col in portfolio_df.columns if col.endswith('_qnty')]
            active_tickers_in_portfolio = [col.replace('_qnty', '') for col in qnty_cols]

            for ticker in active_tickers_in_portfolio:
                value_col = f"{ticker}_value"
                qnty_col = f"{ticker}_qnty"
                # Construct the MultiIndex key for the price data
                price_col_key = (PRICE_COLUMN_NAME, ticker)

                if price_col_key not in price_data.columns:
                    log.warning(f"  Price data ('{PRICE_COLUMN_NAME}') missing for ticker {ticker}. Cannot calculate its value contribution. Setting {value_col} to 0.")
                    portfolio_df[value_col] = 0.0
                    continue # Skip to next ticker

                # Get the price series for the ticker
                price_series = price_data[price_col_key]

                # Calculate value: quantity * price
                # Ensure alignment by index. Use fillna(0) for robustness against potential NaNs in price or quantity.
                # Important: Use .multiply() method with fill_value=0 for safe multiplication if indices differ slightly
                #            or if one series has NaNs where the other has values.
                calculated_value = portfolio_df[qnty_col].multiply(price_series, fill_value=0).fillna(0)

                # Assign calculated value and add to total
                portfolio_df[value_col] = calculated_value
                portfolio_df['Total_Value'] = portfolio_df['Total_Value'].add(calculated_value, fill_value=0) # fill_value here handles potential NaNs in Total_Value itself


            log.debug("Value calculations complete.")

            # Step 6: Final Formatting (Optional but good practice)
            log.info("Finalizing DataFrame structure...")
            qnty_cols_sorted = sorted([col for col in portfolio_df.columns if col.endswith('_qnty')])
            value_cols_sorted = sorted([col for col in portfolio_df.columns if col.endswith('_value')])
            # Ensure Total_Value is the last column
            final_ordered_cols = qnty_cols_sorted + value_cols_sorted + ['Total_Value']
            # Filter to only include columns that actually exist (in case some value calcs failed)
            final_ordered_cols = [col for col in final_ordered_cols if col in portfolio_df.columns]
            portfolio_df = portfolio_df[final_ordered_cols]

            # --- Cache Result and Return Copy ---
            self._portfolio_data = portfolio_df
            log.info("--- Portfolio Build Process Finished Successfully ---")
            return self._portfolio_data.copy() # Return a copy

        except (InvalidInputError, DataFetchError) as e:
             log.error(f"Portfolio build failed due to input or data fetch error: {e}", exc_info=True)
             self._portfolio_data = None # Ensure cache is cleared on failure
             raise # Re-raise the specific error
        except PortfolioBuildError as e:
             log.error(f"Portfolio build failed during processing step: {e}", exc_info=True)
             self._portfolio_data = None
             raise
        except Exception as e:
             log.exception("An unexpected critical error occurred during portfolio build.")
             self._portfolio_data = None
             # Wrap unexpected errors in a PortfolioBuildError
             raise PortfolioBuildError("Unexpected failure during portfolio build.") from e


    def get_portfolio_data(self) -> pd.DataFrame:
        """
        Retrieves the calculated portfolio history DataFrame, building it if necessary.

        Returns:
            A *copy* of the portfolio history DataFrame (see `build_portfolio` for columns),
            or an empty DataFrame if build fails or no data exists.

        Raises:
            InvalidInputError, DataFetchError, PortfolioBuildError: If build() is triggered and fails.
        """
        if self._portfolio_data is None:
            log.info("Portfolio data not built or invalidated. Triggering build...")
            try:
                 # build_portfolio() returns a copy, so store it if successful
                 # Note: build_portfolio now handles caching internally
                 self.build_portfolio() # This builds and caches the data
            except (InvalidInputError, DataFetchError, PortfolioBuildError) as e:
                 log.error(f"Failed to build portfolio data when requested: {e}")
                 # Re-raise the error caught from build_portfolio
                 raise e
            except Exception as e:
                 # Catch any other unexpected error during the build triggered here
                 log.exception("Unexpected error during implicit portfolio build.")
                 raise PortfolioBuildError("Unexpected failure during portfolio build trigger.") from e

        # After build attempt, check cache again
        if self._portfolio_data is None:
             # Build failed or resulted in None (which it shouldn't, should raise instead)
             log.error("Portfolio data is None even after build attempt. Returning empty DataFrame.")
             # Return an empty DF consistent with build_portfolio's failure modes
             return pd.DataFrame()
        else:
             log.info("Returning portfolio data...")
             # build_portfolio now returns a copy, and caches internally. Get cached version.
             return self._portfolio_data.copy()


    def get_composition(self, date_str: Optional[str] = None) -> Optional[pd.Series]:
        """
        Calculates portfolio composition (% value of each holding) on a specific date
        or the latest available date. Uses nearest available date if exact match not found.

        Args:
            date_str: Target date string ('YYYY-MM-DD' or similar) or None for the latest
                      available date in the portfolio history.

        Returns:
            A pandas Series mapping Ticker (str) to percentage (float, 0-100),
            sorted descending by percentage. Returns None if portfolio data cannot be
            retrieved, the target date is invalid, or the DataFrame is empty.
            Returns a Series of zeros if total value on the target date is zero or negative.

        Raises:
            InvalidInputError: If the provided date string is invalid.
            DataFetchError, PortfolioBuildError: If get_portfolio_data() triggers a build that fails.
        """
        target_desc = 'Latest available date' if date_str is None else f"date '{date_str}'"
        log.info(f"Calculating portfolio composition for {target_desc}")

        try:
            # Get data, handling potential build errors
            portfolio_df = self.get_portfolio_data() # Can raise errors
        except (DataFetchError, PortfolioBuildError, InvalidInputError) as e:
             # Added InvalidInputError here as get_portfolio_data->build->fetch->determine_range can raise it
            log.error(f"Cannot get composition because portfolio data retrieval failed: {e}")
            # Depending on desired strictness, could re-raise or return None. Let's return None.
            return None

        if portfolio_df is None or portfolio_df.empty:
            log.error("Portfolio data is not available or empty. Cannot calculate composition.")
            return None

        # --- Determine Target Date and Row ---
        target_row: Optional[pd.Series] = None
        target_date_actual: Optional[pd.Timestamp] = None # The actual date used from the index

        try:
            if date_str:
                target_date_req = pd.to_datetime(date_str).tz_localize(None)
                log.debug(f"Parsed requested date: {target_date_req.date()}")

                # Use index method 'get_loc' with tolerance for nearest date lookup
                try:
                     # Find the index location closest to the requested date
                     idx_loc = portfolio_df.index.get_indexer([target_date_req], method='nearest')[0]
                     # Check if get_indexer returned -1 (shouldn't happen on non-empty index)
                     if idx_loc == -1:
                         raise IndexError("Nearest index lookup failed unexpectedly.")

                     target_date_actual = cast(pd.Timestamp, portfolio_df.index[idx_loc]) # Get the actual date from index
                     target_row = portfolio_df.iloc[idx_loc]

                     log.info(f"Using data from nearest available date: {target_date_actual.date()} "
                              f"(requested: {target_date_req.date()})")
                     # Warn if the nearest date is significantly different
                     if abs((target_date_actual - target_date_req).days) > 3:
                        log.warning(f"  Note: Nearest available date is more than 3 days "
                                    f"away from requested date '{date_str}'.")

                except IndexError:
                     # This might happen if the index is empty or other pandas issues.
                     log.error(f"Could not find nearest date for '{date_str}' in the portfolio index.")
                     return None # Cannot proceed
            else:
                # No date specified, use the latest available date
                target_date_actual = cast(pd.Timestamp, portfolio_df.index[-1])
                target_row = portfolio_df.iloc[-1]
                log.debug(f"No date specified. Using latest available date: {target_date_actual.date()}")

        except ValueError as e_date: # From pd.to_datetime
            log.error(f"Invalid date format or value provided for composition lookup: '{date_str}'. Error: {e_date}")
            raise InvalidInputError(f"Invalid date format or value for composition: '{date_str}'") from e_date
        except Exception as e_lookup:
             log.error(f"Error looking up date '{date_str if date_str else 'latest'}' in portfolio data: {e_lookup}", exc_info=True)
             return None


        if target_row is None or target_date_actual is None:
             log.error("Internal error: Could not retrieve data row for composition calculation.")
             return None # Should not happen if logic above is correct


        # --- Calculate Composition ---
        total_value = target_row.get('Total_Value')

        if total_value is None or pd.isna(total_value) or total_value <= 0:
            log.warning(f"Total portfolio value is zero, NaN, or negative ({total_value}) "
                        f"on {target_date_actual.date()}. Returning zero composition for all holdings.")
            # Create a Series of zeros for all potential value columns
            value_cols = [col for col in portfolio_df.columns if col.endswith('_value')]
            tickers = [col.replace('_value', '') for col in value_cols]
            composition_series = pd.Series(0.0, index=tickers, dtype=float)
            return composition_series

        composition_dict = {}
        # Iterate through columns ending in '_value' that are present in the target row
        for col in target_row.index:
            if col.endswith('_value'):
                ticker = col.replace('_value', '')
                value = target_row.get(col, 0.0) # Default to 0 if column somehow missing from row
                percentage = (value / total_value) * 100.0 if total_value != 0 else 0.0
                composition_dict[ticker] = percentage

        # Create Series and sort
        composition_series = pd.Series(composition_dict).sort_values(ascending=False)
        log.info(f"Composition calculated successfully for {target_date_actual.date()}.")
        return composition_series