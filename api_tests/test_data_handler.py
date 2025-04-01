# --- START OF FILE test_data_handler.py ---

import logging
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from pandas import DatetimeIndex, MultiIndex
from pandas.testing import assert_frame_equal, assert_index_equal

# Module and exceptions under test
import data_handler
from data_handler import (
    DataFetchError,
    DataHandlerError,
    DataProcessingError,
    InvalidInputError,
    fetch_stock_data,
    _validate_date_input,
    _clean_ticker_list
)

# --- Fixtures ---

@pytest.fixture
def sample_tickers() -> list[str]:
    """Provides a sample list of tickers for testing."""
    return ["AAPL", "MSFT", "GOOG"]

@pytest.fixture
def valid_start_date_str() -> str:
    """Provides a valid start date string."""
    return "2023-01-01"

@pytest.fixture
def valid_end_date_str() -> str:
    """Provides a valid end date string."""
    return "2023-01-10"

@pytest.fixture
def mock_dates(valid_start_date_str, valid_end_date_str) -> DatetimeIndex:
    """Creates a sample DatetimeIndex for mock data."""
    # Includes the end date, matching the function's inclusive behavior
    return pd.date_range(start=valid_start_date_str, end=valid_end_date_str, freq='B') # Business days

@pytest.fixture
def mock_multiindex_columns_adj() -> MultiIndex:
    """Creates sample MultiIndex columns for adjusted close data."""
    tickers = ["AAPL", "MSFT", "GOOG"]
    data_types = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    return pd.MultiIndex.from_product([data_types, tickers], names=['DataType', 'Ticker'])

@pytest.fixture
def mock_multiindex_columns_raw() -> MultiIndex:
    """Creates sample MultiIndex columns for raw OHLCV data."""
    tickers = ["AAPL", "MSFT", "GOOG"]
    data_types = ['Close', 'High', 'Low', 'Open', 'Volume']
    return pd.MultiIndex.from_product([data_types, tickers], names=['DataType', 'Ticker'])


@pytest.fixture
def mock_yf_download_success_adj(mock_dates, mock_multiindex_columns_adj) -> pd.DataFrame:
    """Creates a mock DataFrame simulating a successful yfinance download (Adj Close)."""
    df = pd.DataFrame(
        index=mock_dates,
        columns=mock_multiindex_columns_adj,
        data=100.0 # Placeholder data
    )
    # Simulate numeric types
    for level0 in df.columns.levels[0]:
        if level0 != 'Volume':
            df[level0] = df[level0].astype(float)
        else:
            df[level0] = df[level0].astype(int)
    return df


@pytest.fixture
def mock_yf_download_success_raw(mock_dates, mock_multiindex_columns_raw) -> pd.DataFrame:
    """Creates a mock DataFrame simulating a successful yfinance download (Raw OHLCV)."""
    df = pd.DataFrame(
        index=mock_dates,
        columns=mock_multiindex_columns_raw,
        data=100.0 # Placeholder data
    )
    # Simulate numeric types
    for level0 in df.columns.levels[0]:
        if level0 != 'Volume':
            df[level0] = df[level0].astype(float)
        else:
            df[level0] = df[level0].astype(int)
    return df

@pytest.fixture
def mock_yf_download_partial_fail(mock_yf_download_success_adj) -> pd.DataFrame:
    """Creates mock data where one ticker ('GOOG') is missing."""
    df = mock_yf_download_success_adj.copy()
    # Drop all columns related to GOOG
    df = df.drop(columns='GOOG', level='Ticker')
    return df

@pytest.fixture
def mock_yf_download_tz_aware(mock_yf_download_success_adj) -> pd.DataFrame:
    """Creates mock data with a timezone-aware index."""
    df = mock_yf_download_success_adj.copy()
    df.index = df.index.tz_localize('America/New_York')
    return df

# --- Tests for Helper Functions ---

# Tests for _validate_date_input
@pytest.mark.parametrize("date_str, expected_date", [
    ("2023-01-01", date(2023, 1, 1)),
    ("2024-02-29", date(2024, 2, 29)), # Leap year
    ("1999-12-31", date(1999, 12, 31)),
])
def test_validate_date_input_valid(date_str, expected_date):
    assert _validate_date_input(date_str) == expected_date

@pytest.mark.parametrize("invalid_date_str", [
    "2023-13-01",       # Invalid month
    "2023-02-29",       # Invalid day (not leap year)
    "2023/01/01",       # Wrong format
    "01-01-2023",       # Wrong format
    "20230101",         # Wrong format
    "today",            # Not a date
    "",                 # Empty string
    None,               # Not a string
    "2023-00-15",       # Invalid month 0
    "2023-04-31",       # Invalid day 31 for April
])
def test_validate_date_input_invalid(invalid_date_str):
    with pytest.raises(InvalidInputError, match="Invalid date"):
        _validate_date_input(invalid_date_str) # type: ignore # Intentionally pass invalid types

# Tests for _clean_ticker_list
@pytest.mark.parametrize("input_tickers, expected_tickers", [
    (["AAPL", "MSFT", "aapl", " msft ", ""], ["AAPL", "MSFT"]),
    (["GOOGL", "AMZN"], ["AMZN", "GOOGL"]), # Sorting check
    (["BRK.B", "BF.A"], ["BF.A", "BRK.B"]), # Tickers with dots
    ([" ^GSPC", "ES=F "], ["ES=F", "^GSPC"]), # Tickers with symbols/spaces
    ([" SINGLE "], ["SINGLE"]),
])
def test_clean_ticker_list_valid(input_tickers, expected_tickers):
    assert _clean_ticker_list(input_tickers) == expected_tickers

@pytest.mark.parametrize("invalid_input", [
    [],                 # Empty list
    ["", " "],         # List with only empty/whitespace strings
    [None],             # List with None
    [123, "AAPL"],      # List with non-string
    "AAPL, MSFT",       # String instead of list
    None,               # None instead of list
    {"AAPL": 1},        # Dict instead of list
])
def test_clean_ticker_list_invalid(invalid_input):
    # Match specific error messages if needed, otherwise general InvalidInputError
    with pytest.raises(InvalidInputError):
       _clean_ticker_list(invalid_input) # type: ignore # Intentionally pass invalid types


# --- Tests for fetch_stock_data ---

@patch('data_handler.yf.download') # Mock the external dependency
def test_fetch_success_adj_close(
    mock_download: MagicMock,
    mock_yf_download_success_adj: pd.DataFrame,
    sample_tickers: list[str],
    valid_start_date_str: str,
    valid_end_date_str: str,
):
    """Test successful fetch with use_adj_close=True."""
    mock_download.return_value = mock_yf_download_success_adj

    result_df = fetch_stock_data(
        tickers=sample_tickers,
        start_date_str=valid_start_date_str,
        end_date_str=valid_end_date_str,
        use_adj_close=True
    )

    # --- Assertions ---
    # 1. Check mock call parameters
    expected_end_date_for_yf = (date.fromisoformat(valid_end_date_str) + timedelta(days=1)).strftime('%Y-%m-%d')
    mock_download.assert_called_once()
    call_args = mock_download.call_args[1] # Keyword arguments
    assert call_args['tickers'] == sorted(sample_tickers) # Validated & sorted list
    assert call_args['start'] == date.fromisoformat(valid_start_date_str)
    assert call_args['end'] == expected_end_date_for_yf
    assert call_args['auto_adjust'] is True
    assert call_args['group_by'] == 'column'

    # 2. Check result DataFrame structure
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty
    assert isinstance(result_df.index, DatetimeIndex)
    assert result_df.index.tz is None # Should be timezone-naive
    assert isinstance(result_df.columns, MultiIndex)
    assert result_df.columns.names == ['DataType', 'Ticker']

    # 3. Check index range (inclusive)
    expected_start = datetime.strptime(valid_start_date_str, '%Y-%m-%d').date()
    expected_end = datetime.strptime(valid_end_date_str, '%Y-%m-%d').date()
    assert result_df.index.date.min() >= expected_start
    assert result_df.index.date.max() <= expected_end
    # Check if start/end dates from mock data are present (if they were business days)
    if not mock_yf_download_success_adj.empty:
        assert mock_yf_download_success_adj.index.min() == result_df.index.min()
        assert mock_yf_download_success_adj.index.max() == result_df.index.max()


    # 4. Check columns (Adj Close should be present)
    expected_tickers_set = set(t.upper() for t in sample_tickers)
    assert set(result_df.columns.get_level_values('Ticker')) == expected_tickers_set
    assert 'Adj Close' in result_df.columns.get_level_values('DataType')
    assert 'Volume' in result_df.columns.get_level_values('DataType')

    # 5. Data types (basic check)
    assert pd.api.types.is_numeric_dtype(result_df[('Adj Close', 'AAPL')])
    assert pd.api.types.is_integer_dtype(result_df[('Volume', 'MSFT')])


@patch('data_handler.yf.download')
def test_fetch_success_raw_ohlcv(
    mock_download: MagicMock,
    mock_yf_download_success_raw: pd.DataFrame,
    sample_tickers: list[str],
    valid_start_date_str: str,
    valid_end_date_str: str,
):
    """Test successful fetch with use_adj_close=False."""
    mock_download.return_value = mock_yf_download_success_raw

    result_df = fetch_stock_data(
        tickers=sample_tickers,
        start_date_str=valid_start_date_str,
        end_date_str=valid_end_date_str,
        use_adj_close=False # Key difference
    )

    # --- Assertions ---
    # 1. Check mock call parameters (auto_adjust=False)
    expected_end_date_for_yf = (date.fromisoformat(valid_end_date_str) + timedelta(days=1)).strftime('%Y-%m-%d')
    mock_download.assert_called_once()
    call_args = mock_download.call_args[1]
    assert call_args['tickers'] == sorted(sample_tickers)
    assert call_args['start'] == date.fromisoformat(valid_start_date_str)
    assert call_args['end'] == expected_end_date_for_yf
    assert call_args['auto_adjust'] is False # Check flag
    assert call_args['group_by'] == 'column'

    # 2. Check structure (similar to adj close test)
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty
    assert isinstance(result_df.index, DatetimeIndex)
    assert result_df.index.tz is None
    assert isinstance(result_df.columns, MultiIndex)

    # 3. Check columns (Adj Close should NOT be present)
    expected_tickers_set = set(t.upper() for t in sample_tickers)
    assert set(result_df.columns.get_level_values('Ticker')) == expected_tickers_set
    assert 'Adj Close' not in result_df.columns.get_level_values('DataType')
    assert 'Close' in result_df.columns.get_level_values('DataType') # Raw Close should be there
    assert 'Volume' in result_df.columns.get_level_values('DataType')


@patch('data_handler.yf.download')
def test_fetch_partial_failure_warning(
    mock_download: MagicMock,
    mock_yf_download_partial_fail: pd.DataFrame,
    sample_tickers: list[str], # Requested ['AAPL', 'MSFT', 'GOOG']
    valid_start_date_str: str,
    valid_end_date_str: str,
    caplog: pytest.LogCaptureFixture
):
    """Test scenario where yfinance doesn't return data for all tickers."""
    mock_download.return_value = mock_yf_download_partial_fail # Only has AAPL, MSFT

    with caplog.at_level(logging.WARNING, logger='data_handler'):
        result_df = fetch_stock_data(
            tickers=sample_tickers,
            start_date_str=valid_start_date_str,
            end_date_str=valid_end_date_str,
        )

    # --- Assertions ---
    # 1. Check log for warning about missing tickers
    assert "Data successfully fetched for 2 tickers" in caplog.text
    assert "but was NOT returned for: ['GOOG']" in caplog.text

    # 2. Check returned DataFrame
    assert not result_df.empty
    returned_tickers = set(result_df.columns.get_level_values('Ticker'))
    assert returned_tickers == {"AAPL", "MSFT"} # GOOG should be missing
    assert "GOOG" not in returned_tickers


@patch('data_handler.yf.download')
def test_fetch_total_failure_empty_df(
    mock_download: MagicMock,
    sample_tickers: list[str],
    valid_start_date_str: str,
    valid_end_date_str: str,
):
    """Test scenario where yfinance returns an empty DataFrame."""
    mock_download.return_value = pd.DataFrame() # Simulate empty response

    with pytest.raises(DataFetchError, match="No data returned by yfinance"):
        fetch_stock_data(
            tickers=sample_tickers,
            start_date_str=valid_start_date_str,
            end_date_str=valid_end_date_str,
        )


@patch('data_handler.yf.download')
def test_fetch_total_failure_api_exception(
    mock_download: MagicMock,
    sample_tickers: list[str],
    valid_start_date_str: str,
    valid_end_date_str: str,
):
    """Test scenario where yfinance.download() raises an exception."""
    mock_download.side_effect = ConnectionError("Simulated network failure")

    with pytest.raises(DataFetchError, match="yfinance download failed.*ConnectionError"):
        fetch_stock_data(
            tickers=sample_tickers,
            start_date_str=valid_start_date_str,
            end_date_str=valid_end_date_str,
        )


@patch('data_handler.yf.download')
def test_fetch_timezone_normalization(
    mock_download: MagicMock,
    mock_yf_download_tz_aware: pd.DataFrame, # Has tz-aware index
    sample_tickers: list[str],
    valid_start_date_str: str,
    valid_end_date_str: str,
):
    """Test that timezone-aware index from yfinance is converted to naive."""
    mock_download.return_value = mock_yf_download_tz_aware
    assert mock_yf_download_tz_aware.index.tz is not None # Precondition

    result_df = fetch_stock_data(
        tickers=sample_tickers,
        start_date_str=valid_start_date_str,
        end_date_str=valid_end_date_str,
    )

    assert not result_df.empty
    assert isinstance(result_df.index, DatetimeIndex)
    assert result_df.index.tz is None # Crucial: Postcondition


# --- Tests for Invalid Inputs to fetch_stock_data ---

# No need to mock yf.download here as validation happens first
@pytest.mark.parametrize("invalid_tickers", [
    [],
    [""],
    None,
    "AAPL",
    [1, 2, 3]
])
def test_fetch_invalid_ticker_input(invalid_tickers, valid_start_date_str, valid_end_date_str):
    with pytest.raises(InvalidInputError):
        fetch_stock_data(
            tickers=invalid_tickers, # type: ignore
            start_date_str=valid_start_date_str,
            end_date_str=valid_end_date_str,
        )

@pytest.mark.parametrize("invalid_date", ["2023-13-01", "not-a-date", None, ""])
def test_fetch_invalid_start_date_input(sample_tickers, invalid_date, valid_end_date_str):
    with pytest.raises(InvalidInputError, match="Invalid date"):
        fetch_stock_data(
            tickers=sample_tickers,
            start_date_str=invalid_date, # type: ignore
            end_date_str=valid_end_date_str,
        )

@pytest.mark.parametrize("invalid_date", ["2023-04-31", "date", None, ""])
def test_fetch_invalid_end_date_input(sample_tickers, valid_start_date_str, invalid_date):
    with pytest.raises(InvalidInputError, match="Invalid date"):
        fetch_stock_data(
            tickers=sample_tickers,
            start_date_str=valid_start_date_str,
            end_date_str=invalid_date, # type: ignore
        )

def test_fetch_start_date_after_end_date(sample_tickers, valid_start_date_str, valid_end_date_str):
    with pytest.raises(InvalidInputError, match="Start date .* cannot be after end date"):
        fetch_stock_data(
            tickers=sample_tickers,
            start_date_str=valid_end_date_str, # Start is later
            end_date_str=valid_start_date_str, # End is earlier
        )


# --- Test for Custom yfinance Parameters ---

@patch('data_handler.yf.download')
def test_fetch_with_custom_yf_params(
    mock_download: MagicMock,
    mock_yf_download_success_adj: pd.DataFrame,
    sample_tickers: list[str],
    valid_start_date_str: str,
    valid_end_date_str: str,
):
    """Test passing extra parameters to yfinance.download."""
    mock_download.return_value = mock_yf_download_success_adj
    custom_params = {'proxy': 'myproxy.com', 'timeout': 60}

    fetch_stock_data(
        tickers=sample_tickers,
        start_date_str=valid_start_date_str,
        end_date_str=valid_end_date_str,
        yf_download_params=custom_params
    )

    # Assert that the mock was called with the merged parameters
    mock_download.assert_called_once()
    call_args = mock_download.call_args[1]
    assert call_args['proxy'] == 'myproxy.com'
    assert call_args['timeout'] == 60
    # Check that core parameters were not overwritten badly
    assert call_args['tickers'] == sorted(sample_tickers)
    assert call_args['group_by'] == 'column'


# --- Tests for Potential Data Processing Errors (Harder to simulate perfectly) ---

@patch('data_handler.yf.download')
def test_fetch_non_multiindex_columns_error(
    mock_download: MagicMock,
    sample_tickers: list[str],
    valid_start_date_str: str,
    valid_end_date_str: str,
    mock_dates: DatetimeIndex
):
    """Test handling if yf.download unexpectedly returns non-MultiIndex columns."""
    # Simulate a DataFrame with simple columns
    mock_df = pd.DataFrame(index=mock_dates, columns=['AAPL', 'MSFT'], data=1.0)
    mock_download.return_value = mock_df

    with pytest.raises(DataProcessingError, match="expected MultiIndex column structure"):
        fetch_stock_data(
            tickers=sample_tickers, # Requesting 3, getting 2 simple cols
            start_date_str=valid_start_date_str,
            end_date_str=valid_end_date_str,
        )

@patch('data_handler.yf.download')
@patch('pandas.to_datetime', side_effect=TypeError("Simulated conversion failure")) # Mock conversion failure
def test_fetch_index_conversion_failure(
    mock_to_datetime: MagicMock,
    mock_download: MagicMock,
    sample_tickers: list[str],
    valid_start_date_str: str,
    valid_end_date_str: str,
):
    """Test handling if index conversion fails."""
    # Simulate data where index isn't datetime initially (e.g., plain objects/strings)
    mock_df = pd.DataFrame(
        index=['a', 'b', 'c'], # Non-datetime index
        columns=pd.MultiIndex.from_product([['Close'], ['AAPL']]),
        data=1.0
    )
    mock_download.return_value = mock_df

    with pytest.raises(DataProcessingError, match="Failed to process or convert the DataFrame index"):
         fetch_stock_data(
            tickers=sample_tickers,
            start_date_str=valid_start_date_str,
            end_date_str=valid_end_date_str,
        )
    mock_to_datetime.assert_called() # Verify our mock was hit


@patch('data_handler.yf.download')
def test_fetch_data_becomes_empty_after_filtering(
    mock_download: MagicMock,
    mock_yf_download_success_adj: pd.DataFrame,
    sample_tickers: list[str],
    caplog: pytest.LogCaptureFixture
):
    """Test case where filtering by exact date range makes the DataFrame empty."""
    # Request a range completely outside the mock data's range
    start_outside = "2024-01-01"
    end_outside = "2024-01-05"
    mock_download.return_value = mock_yf_download_success_adj # Mock data is for 2023-01

    with caplog.at_level(logging.WARNING, logger='data_handler'):
        result_df = fetch_stock_data(
            tickers=sample_tickers,
            start_date_str=start_outside,
            end_date_str=end_outside,
        )

    assert result_df.empty
    assert "Data became empty after filtering" in caplog.text
    assert f"({start_outside} to {end_outside})" in caplog.text
