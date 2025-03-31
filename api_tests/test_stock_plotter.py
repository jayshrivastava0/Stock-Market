import pytest
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from unittest.mock import patch, MagicMock # For mocking yfinance

# Import the class to be tested
from stock_plotter import StockTrendPlotter, go # Import go for type checking

# --- Fixtures ---

@pytest.fixture
def today_normalized():
    """Return today's date, normalized to midnight."""
    return datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

@pytest.fixture
def sample_dataframe():
    """Return a sample pandas DataFrame similar to yfinance output."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    return pd.DataFrame({'Date': dates, 'Close': [100.0, 101.5, 101.0]})

# --- Test Parsing Methods ---

@pytest.mark.parametrize("input_str, expected_start_func, expected_end_func", [
    ("this month", lambda t: t.replace(day=1), lambda t: t),
    ("this year", lambda t: t.replace(month=1, day=1), lambda t: t),
    ("from 2022", lambda t: datetime(2022, 1, 1), lambda t: t),
    (" from 2021-05 ", lambda t: datetime(2021, 5, 1), lambda t: t),
    ("data from 2023-11", lambda t: datetime(2023, 11, 1), lambda t: t),
])
def test_parse_date_range_valid(input_str, expected_start_func, expected_end_func, today_normalized):
    plotter = StockTrendPlotter("DUMMY", "dummy") # Ticker/period don't matter here
    expected_start = expected_start_func(today_normalized)
    expected_end = expected_end_func(today_normalized)
    # Only proceed if expected start is not in the future
    if expected_start <= expected_end:
        start, end = plotter._parse_date_range(input_str)
        assert start == expected_start
        assert end == expected_end
    else:
         pytest.skip("Skipping test where 'from' date is in the future relative to today.")


@pytest.mark.parametrize("input_str", [
    "last month",
    "from 20-01", # Invalid year format
    "from 2023-13", # Invalid month
    "from tomorrow",
    "random string",
    "",
    " from ",
])
def test_parse_date_range_invalid(input_str):
    plotter = StockTrendPlotter("DUMMY", "dummy")
    start, end = plotter._parse_date_range(input_str)
    assert start is None
    assert end is None

@pytest.mark.parametrize("input_str, delta_args", [
    ("7D", {'days': 7}),
    ("2W", {'weeks': 2}),
    ("6m", {'months': 6}), # Lowercase m
    ("1Y", {'years': 1}),
    ("10d", {'days': 10}), # Lowercase d
    ("12M", {'months': 12}), # Uppercase M
])
def test_parse_relative_period_valid(input_str, delta_args, today_normalized):
    plotter = StockTrendPlotter("DUMMY", "dummy")
    expected_start = today_normalized - relativedelta(**delta_args)
    expected_end = today_normalized
    start, end = plotter._parse_relative_period(input_str)
    assert start == expected_start
    assert end == expected_end

@pytest.mark.parametrize("input_str", [
    "D7", # Wrong order
    "2 Weeks", # Not the pattern
    "M6", # Wrong order
    "1.5Y", # No float
    "0D", # Zero quantity
    "-5M", # Negative quantity (handled by regex not matching '-')
    "10", # Missing unit
    "Y", # Missing quantity
    "",
    " ",
])
def test_parse_relative_period_invalid(input_str):
    plotter = StockTrendPlotter("DUMMY", "dummy")
    start, end = plotter._parse_relative_period(input_str)
    assert start is None
    assert end is None

# --- Test Validation Method ---

@pytest.mark.parametrize("time_period, expected_type", [
    ("1y", "period"), # Standard Yahoo period
    ("6mo", "period"),
    ("max", "period"),
    ("this year", "date_range"), # Natural language
    ("from 2022", "date_range"),
    ("7M", "date_range"), # Relative period
    ("10D", "date_range"),
])
def test_validate_period_valid(time_period, expected_type, today_normalized):
    plotter = StockTrendPlotter("DUMMY", time_period)
    start, end, period = plotter._validate_period()

    if expected_type == "period":
        assert start is None
        assert end is None
        assert period == time_period.lower()
    elif expected_type == "date_range":
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert start <= end # Start date should not be after end date
        assert end == today_normalized # End date should always be today (normalized)
        assert period is None

def test_validate_period_invalid():
    plotter = StockTrendPlotter("DUMMY", "invalid period string")
    with pytest.raises(ValueError, match="Invalid time period"):
        plotter._validate_period()

# --- Test Fetch Data (using Mocking) ---

# Patch yfinance.Ticker globally for these tests
@patch('stock_plotter.yf.Ticker') # Use the path where Ticker is *used*
def test_fetch_data_with_period(mock_yf_ticker, sample_dataframe):
    """Test fetching data using a standard yfinance period string."""
    # Configure the mock Ticker instance and its history method
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_dataframe
    mock_yf_ticker.return_value = mock_ticker_instance

    ticker = "AAPL"
    period = "1y"
    plotter = StockTrendPlotter(ticker, period)
    df = plotter.fetch_data()

    # Assertions
    mock_yf_ticker.assert_called_once_with(ticker) # Check Ticker was initialized correctly
    mock_ticker_instance.history.assert_called_once_with(period=period) # Check history called with period
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['Date', 'Price']
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df['Date'])

@patch('stock_plotter.yf.Ticker')
def test_fetch_data_with_dates(mock_yf_ticker, sample_dataframe, today_normalized):
    """Test fetching data using a calculated date range (e.g., 'from 2023')."""
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_dataframe
    mock_yf_ticker.return_value = mock_ticker_instance

    ticker = "MSFT"
    time_period = "from 2023"
    plotter = StockTrendPlotter(ticker, time_period)

    # Manually calculate expected start/end based on the period logic
    expected_start = datetime(2023, 1, 1)
    expected_end = today_normalized

    df = plotter.fetch_data()

    mock_yf_ticker.assert_called_once_with(ticker)
    # Check history was called with the correct start and end dates
    mock_ticker_instance.history.assert_called_once_with(start=expected_start, end=expected_end)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['Date', 'Price']

@patch('stock_plotter.yf.Ticker')
def test_fetch_data_empty(mock_yf_ticker):
    """Test the case where yfinance returns an empty DataFrame."""
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = pd.DataFrame() # Empty DF
    mock_yf_ticker.return_value = mock_ticker_instance

    plotter = StockTrendPlotter("NODATA", "1y")
    with pytest.raises(ValueError, match="No data found"):
        plotter.fetch_data()

@patch('stock_plotter.yf.Ticker')
def test_fetch_data_yfinance_error(mock_yf_ticker):
    """Test handling of errors during yfinance call."""
    mock_ticker_instance = MagicMock()
    # Simulate an error during the .history() call
    mock_ticker_instance.history.side_effect = Exception("Simulated yfinance network error")
    mock_yf_ticker.return_value = mock_ticker_instance

    plotter = StockTrendPlotter("ERROR", "1y")
    with pytest.raises(IOError, match="Could not fetch data"):
        plotter.fetch_data()

# --- Test Plotting (Mocking fetch_data) ---

@patch.object(StockTrendPlotter, 'fetch_data') # Mock the method within the class instance
def test_plot_trend_success(mock_fetch_data, sample_dataframe):
    """Test plot generation succeeds and returns a Figure object."""
    # Configure the mock fetch_data to return our sample data
    mock_fetch_data.return_value = sample_dataframe[['Date', 'Close']].rename(columns={'Close': 'Price'})

    ticker = "AAPL"
    period = "1mo"
    plotter = StockTrendPlotter(ticker, period)

    # We don't want the plot window to actually open during tests
    fig = plotter.plot_trend(show=False)

    # Assertions
    mock_fetch_data.assert_called_once() # Ensure fetch_data was called
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == f'{ticker} Stock Price Trend ({period})'
    assert len(fig.data) == 1 # Should have one trace (the scatter plot)
    assert isinstance(fig.data[0], go.Scatter)
    # Check if data points roughly match (optional, can be brittle)
    assert len(fig.data[0].x) == len(sample_dataframe)


@patch.object(StockTrendPlotter, 'fetch_data')
def test_plot_trend_fetch_data_fails(mock_fetch_data):
    """Test that plot_trend handles errors from fetch_data."""
    # Configure mock fetch_data to raise an error
    mock_fetch_data.side_effect = ValueError("No data found (simulated)")

    plotter = StockTrendPlotter("FAIL", "1y")

    with pytest.raises(ValueError, match="No data found"):
        plotter.plot_trend(show=False)

    mock_fetch_data.assert_called_once() # Ensure fetch_data was attempted

# --- Test Initialization ---
def test_init_invalid_ticker():
    with pytest.raises(ValueError, match="Ticker symbol must be a non-empty string"):
        StockTrendPlotter("", "1y")
    with pytest.raises(ValueError, match="Ticker symbol must be a non-empty string"):
        StockTrendPlotter(None, "1y")

def test_init_invalid_period():
    with pytest.raises(ValueError, match="Time period must be a non-empty string"):
        StockTrendPlotter("AAPL", "")
    with pytest.raises(ValueError, match="Time period must be a non-empty string"):
        StockTrendPlotter("AAPL", None)