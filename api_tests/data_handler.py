#!/usr/bin/env python3
"""
Data Handler Module

Provides robust and efficient fetching of historical stock market data
using the yfinance library. Retrieves OHLCV and Adjusted Close data for
multiple tickers simultaneously, targeting production-level reliability.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Final, Optional, Set, Union

import pandas as pd
import yfinance as yf
from pandas import DatetimeIndex

# Configure logger for this module. Relies on application-level configuration.
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# --- Constants ---
# Define standard column names expected/used (adjust if necessary)
OHLCV_COLS: Final[List[str]] = ['Open', 'High', 'Low', 'Close', 'Volume']
ADJ_CLOSE_COL: Final[str] = 'Adj Close'


# --- Custom Exceptions ---
class DataHandlerError(Exception):
    """Base exception for errors specific to the data_handler module."""
    pass


class InvalidInputError(DataHandlerError, ValueError):
    """Exception raised for invalid input parameters provided by the user."""
    pass


class DataFetchError(DataHandlerError, IOError):
    """
    Exception raised for errors during data fetching operations.
    This can include network issues, API errors, ticker not found,
    or no data being returned for the requested parameters.
    """
    pass


class DataProcessingError(DataHandlerError):
    """Exception raised for errors during the processing of fetched data."""
    pass


# --- Helper Functions ---
def _validate_date_input(date_str: str) -> date:
    """
    Validates if a string is a valid date in 'YYYY-MM-DD' format.

    Args:
        date_str: The date string to validate.

    Returns:
        The parsed date object if valid.

    Raises:
        InvalidInputError: If the date string is not a string, not in the correct format,
                           or represents an invalid date.
    """
    # --- ADD THIS CHECK ---
    if not isinstance(date_str, str):
        msg = f"Invalid date input: Expected a string, but got {type(date_str).__name__}."
        log.error(msg)
        raise InvalidInputError(msg)
    # --- END ADDED CHECK ---

    try:
        # Attempt to parse strictly in YYYY-MM-DD format
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        return parsed_date
    except ValueError as e:
        # Error messages now implicitly cover format/value issues for strings
        msg = f"Invalid date format or value for string '{date_str}'. Expected 'YYYY-MM-DD'."
        log.error(msg)
        raise InvalidInputError(msg) from e


def _clean_ticker_list(tickers: List[str]) -> List[str]:
    """
    Cleans the input ticker list: converts to uppercase, removes duplicates and empty strings.

    Args:
        tickers: The raw list of ticker symbols.

    Returns:
        A sorted list of unique, valid ticker strings.

    Raises:
        InvalidInputError: If the input is not a list or contains non-string elements,
                           or if the list is empty after cleaning.
    """
    if not isinstance(tickers, list):
        raise InvalidInputError("Input 'tickers' must be a list.")
    if not all(isinstance(t, str) for t in tickers):
        raise InvalidInputError("All elements in the 'tickers' list must be strings.")

    # Convert to uppercase, strip whitespace, filter out empty strings, remove duplicates
    cleaned_tickers: Set[str] = {t.upper().strip() for t in tickers if t and t.strip()}

    if not cleaned_tickers:
        raise InvalidInputError("The provided ticker list is empty or contains only invalid entries.")

    return sorted(list(cleaned_tickers))


# --- Core Data Fetching Function ---
def fetch_stock_data(
    tickers: List[str],
    start_date_str: str,
    end_date_str: str,
    use_adj_close: bool = True,
    yf_download_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Fetches historical OHLCV and optionally Adjusted Close data for multiple stock tickers.

    Uses `yfinance.download()` for efficient batch retrieval. Handles input validation,
    data cleaning, timezone normalization, and error reporting.

    Note on Date Range: `yfinance.download` typically excludes the `end_date`. This function
    adjusts the request to include data up to and including the provided `end_date_str`
    by requesting data up to the day *after* `end_date_str`.

    Args:
        tickers: A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
                 Case-insensitive, duplicates and whitespace are handled.
        start_date_str: Start date string in 'YYYY-MM-DD' format. Data retrieval
                        starts from this date (inclusive).
        end_date_str: End date string in 'YYYY-MM-DD' format. Data retrieval
                      includes this date (inclusive).
        use_adj_close: If True (default), uses yfinance's 'auto_adjust=True' which
                       provides Adjusted Close and adjusts OHLCV data for splits/dividends.
                       The primary price column will be 'Adj Close'.
                       If False, fetches raw OHLCV data; the primary price column is 'Close'.
        yf_download_params: Optional dictionary of parameters to pass directly
                            to `yfinance.download()`, potentially overriding defaults
                            set by this function (e.g., {'proxy': '...', 'timeout': 30}).
                            Use with caution.

    Returns:
        A pandas DataFrame containing the fetched stock data, indexed by date (timezone-naive).
        The columns form a MultiIndex:
        - Level 0: Data type ('Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume')
                   or ('Open', 'High', 'Low', 'Close', 'Volume') if use_adj_close=False.
        - Level 1: Ticker symbol (e.g., 'AAPL', 'MSFT').

    Raises:
        InvalidInputError: If input parameters (tickers, dates) are invalid.
        DataFetchError: If `yfinance` fails to download data (e.g., network error,
                        API issues, no data found for *any* ticker in the range).
                        Partial failures (some tickers missing) are logged as warnings
                        but do not raise an error by default if some data is retrieved.
        DataProcessingError: If an error occurs during internal data structuring or cleaning
                             after successful download.
    """
    log.info("Starting stock data fetch process.")
    log.debug(f"Raw input: tickers={tickers}, start='{start_date_str}', end='{end_date_str}', "
              f"adj={use_adj_close}, params={yf_download_params}")

    # --- Input Validation ---
    try:
        valid_tickers: List[str] = _clean_ticker_list(tickers)
        start_date: date = _validate_date_input(start_date_str)
        end_date: date = _validate_date_input(end_date_str)

        if start_date > end_date:
            msg = f"Start date {start_date_str} cannot be after end date {end_date_str}."
            log.error(msg)
            raise InvalidInputError(msg)

    except InvalidInputError:
        # Input validation errors already logged by helpers
        raise # Re-raise the specific InvalidInputError

    log.info(f"Validated inputs: {len(valid_tickers)} unique tickers. Range: {start_date} to {end_date}.")

    # --- Prepare yfinance Parameters ---
    # Adjust end_date for yfinance's exclusive behavior to make our end_date inclusive
    yf_end_date: date = end_date + timedelta(days=1)
    yf_end_date_str: str = yf_end_date.strftime('%Y-%m-%d')

    # Default parameters for yf.download
    default_yf_params: Dict[str, Union[str, bool, List[str], date]] = {
        "tickers": valid_tickers,
        "start": start_date,
        "end": yf_end_date_str, # Use adjusted end date string
        "auto_adjust": use_adj_close,
        "group_by": 'column',  # Results in MultiIndex columns ('Adj Close', 'AAPL')
        "progress": False,     # Suppress console progress bar
        "threads": True,       # Use multiple threads for faster download of multiple tickers
        # Consider adding: 'prepost=False' if pre/post market data is unwanted
        # Consider adding: 'timeout=30' (seconds)
    }

    # Merge user-provided params, allowing overrides
    final_yf_params = default_yf_params.copy()
    if yf_download_params and isinstance(yf_download_params, dict):
        log.debug(f"Applying user-provided yfinance parameters: {yf_download_params}")
        final_yf_params.update(yf_download_params)
        # Ensure critical parameters are not accidentally broken by user input
        final_yf_params["tickers"] = valid_tickers # Keep the validated list
        final_yf_params["group_by"] = 'column' # Enforce required grouping

    log.debug(f"Final parameters for yfinance.download: {final_yf_params}")

    # --- Data Fetching ---
    try:
        log.info(f"Requesting data from yfinance for tickers: {valid_tickers}...")
        data: pd.DataFrame = yf.download(**final_yf_params) # type: ignore # Ignore type check on dynamic kwargs
        log.info(f"yfinance download complete. Received initial DataFrame shape: {data.shape}")

    # Catching potential generic Exception from yfinance is broad, but often necessary
    # as specific yfinance exceptions aren't heavily documented/guaranteed.
    except Exception as e:
        msg = f"yfinance download failed. Error type: {type(e).__name__}, Message: {e}"
        log.error(msg, exc_info=True) # Log exception details
        raise DataFetchError(msg) from e

    # --- Post-Fetch Validation and Processing ---
    if data.empty:
        msg = (f"No data returned by yfinance for any of the tickers {valid_tickers} "
               f"in the range {start_date_str} to {end_date_str}.")
        log.error(msg)
        # Depending on requirements, this could be a warning if partial data is acceptable,
        # but often signifies a larger issue (e.g., wrong market, holidays only).
        raise DataFetchError(msg)

    # Verify primary data structure (MultiIndex columns expected)
    if not isinstance(data.columns, pd.MultiIndex):
        msg = "Downloaded data does not have the expected MultiIndex column structure."
        log.error(f"{msg} Columns found: {data.columns}")
        raise DataProcessingError(msg)

    # Check for partial failures (missing tickers)
    try:
        fetched_tickers: Set[str] = set(data.columns.get_level_values(1))
        missing_tickers: Set[str] = set(valid_tickers) - fetched_tickers
        if missing_tickers:
            log.warning(f"Data successfully fetched for {len(fetched_tickers)} tickers, "
                        f"but was NOT returned for: {sorted(list(missing_tickers))}. "
                        f"Proceeding with available data.")
        else:
             log.info(f"Data successfully fetched for all {len(fetched_tickers)} requested tickers.")

        if not fetched_tickers:
            # Should be caught by data.empty, but acts as a safeguard.
            msg = "Downloaded DataFrame is not empty but contains no recognizable ticker columns."
            log.error(msg)
            raise DataProcessingError(msg)

    except IndexError:
         msg = "Could not access ticker level (level 1) in DataFrame columns MultiIndex."
         log.error(msg, exc_info=True)
         raise DataProcessingError(msg)
    except Exception as e:
        # Catch unexpected errors during column processing
        msg = f"Unexpected error while verifying fetched ticker columns: {e}"
        log.error(msg, exc_info=True)
        raise DataProcessingError(msg) from e


    # Ensure index is DatetimeIndex and convert to timezone-naive
    try:
        if not isinstance(data.index, DatetimeIndex):
            log.warning("Index is not a DatetimeIndex. Attempting conversion.")
            data.index = pd.to_datetime(data.index)
            if not isinstance(data.index, DatetimeIndex): # Check conversion result
                 raise TypeError("Index could not be converted to DatetimeIndex.")

        if data.index.tz is not None:
            log.debug(f"Index has timezone ({data.index.tz}). Converting to timezone-naive (UTC base).")
            # Convert to UTC first to handle potential ambiguities, then remove tz info
            data.index = data.index.tz_convert('UTC').tz_localize(None)

        # Explicitly cast to DatetimeIndex after potential conversions for type safety downstream
        data.index = pd.DatetimeIndex(data.index)

    except (TypeError, ValueError) as e:
        msg = f"Failed to process or convert the DataFrame index to timezone-naive DatetimeIndex: {e}"
        log.error(msg, exc_info=True)
        raise DataProcessingError(msg) from e

    # Remove rows outside the *requested* date range (yf might return more around holidays/weekends)
    # Use the original start/end date objects for filtering
    data = data[(data.index.date >= start_date) & (data.index.date <= end_date)] # type: ignore

    if data.empty:
        log.warning("Data became empty after filtering to the exact requested date range "
                    f"({start_date_str} to {end_date_str}).")
        # Decide if this is an error or acceptable (e.g., requested range was only holidays)
        # For now, log warning and return the empty frame. Consider raising DataFetchError if stricter.


    log.info(f"Data fetch and processing complete. Final DataFrame shape: {data.shape}.")
    return data