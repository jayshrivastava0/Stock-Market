import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock, call # Import call for checking multiple calls

# Import the module and classes to test
import portfolio_manager as pm
from portfolio_manager import (
    PortfolioManager,
    InvalidInputError,
    DataFetchError,
    PortfolioBuildError,
    PortfolioError
)

# --- Fixtures ---

@pytest.fixture
def mock_stock_plotter_class():
    """Fixture to mock the StockTrendPlotter class."""
    # Patch where PortfolioManager imports StockTrendPlotter
    with patch('portfolio_manager.StockTrendPlotter', autospec=True) as mock_class:
        yield mock_class

@pytest.fixture
def sample_price_data():
    """Provides a sample DataFrame simulating fetched price data."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    data = {
        'AAPL': [150.0, 151.0, 150.5, 152.0, 153.0],
        'MSFT': [280.0, 281.0, 282.0, 280.0, 283.0],
        'GOOGL': [100.0, 101.0, 100.0, 102.0, 103.0] # Ticker added later
    }
    df = pd.DataFrame(data, index=dates)
    return df

# --- Helper Function for Mocking fetch_data ---

def create_fetch_data_side_effect(sample_data):
    """Creates a side_effect function for mock plotter's fetch_data."""
    def fetch_data_mock(self, *args, **kwargs): # 'self' here is the mock instance
        # The ticker was passed during instantiation, which the class mock captures.
        # We need the test to configure the instance mock based on the ticker.
        # This side effect will be attached to the *instance* mock.
        ticker = self.ticker # Assume ticker is stored on the instance mock by the class mock's side_effect

        # print(f"Mock fetch_data executing for ticker: {ticker}") # Debug print

        if ticker == "FAIL":
            raise ValueError(f"Simulated fetch failure for {ticker}")
        if ticker == "EMPTY":
            return pd.DataFrame(columns=['Date', 'Price']).set_index('Date')

        if ticker in sample_data.columns:
            ticker_df = sample_data[[ticker]].reset_index()
            ticker_df.columns = ['Date', 'Price']
            return ticker_df
        else:
            raise ValueError(f"No data found for {ticker} (mock)")
    return fetch_data_mock


# --- Test Initialization (__init__) ---
# (These tests remain the same as they don't involve mocking StockTrendPlotter)
def test_init_success():
    initial = {'AAPL': 10}
    period = '1y'
    pm_instance = PortfolioManager(initial, period)
    assert pm_instance.initial_portfolio == {'AAPL': 10}
    assert pm_instance.time_period == period
    assert pm_instance._all_tickers == {'AAPL'}
    assert pm_instance._changes == []
    assert pm_instance._portfolio_data is None

def test_init_uppercase_ticker():
    initial = {'aapl': 5}
    pm_instance = PortfolioManager(initial, '6m')
    assert pm_instance.initial_portfolio == {'AAPL': 5}
    assert 'AAPL' in pm_instance._all_tickers

@pytest.mark.parametrize("invalid_portfolio, error_msg_match", [
    (None, "initial_portfolio must be a dictionary"),
    ({}, "initial_portfolio cannot be empty"),
    ({'AAPL': -5}, "values must be non-negative integers"),
    ({'AAPL': 10.5}, "values must be non-negative integers"),
    ({123: 10}, "keys must be strings"),
])
def test_init_invalid_portfolio(invalid_portfolio, error_msg_match):
    with pytest.raises(InvalidInputError, match=error_msg_match):
        PortfolioManager(invalid_portfolio, '1y')

@pytest.mark.parametrize("invalid_period", [None, "", 123])
def test_init_invalid_time_period(invalid_period):
    with pytest.raises(InvalidInputError, match="time_period must be a non-empty string"):
        PortfolioManager({'AAPL': 1}, invalid_period)


# --- Test Adding Changes (add_change) ---
# (These tests remain the same)
def test_add_change_success_absolute():
    pm_instance = PortfolioManager({'AAPL': 10}, '1y')
    date_str = '2023-01-15'
    ticker = 'MSFT'
    qty = 5
    pm_instance.add_change(date_str, ticker, qty)

    assert len(pm_instance._changes) == 1
    change_tuple = pm_instance._changes[0]
    assert change_tuple[0] == pd.to_datetime(date_str) # Check date parsing
    assert change_tuple[1] == 'MSFT' # Check uppercase
    assert change_tuple[2] == qty
    assert 'MSFT' in pm_instance._all_tickers
    assert pm_instance._portfolio_data is None # Check cache invalidation

def test_add_change_success_relative_plus():
    pm_instance = PortfolioManager({'AAPL': 10}, '1y')
    pm_instance.add_change('2023-02-01', 'aapl', '+5')
    assert len(pm_instance._changes) == 1
    assert pm_instance._changes[0][1] == 'AAPL'
    assert pm_instance._changes[0][2] == '+5'
    assert pm_instance._portfolio_data is None

def test_add_change_success_relative_minus():
    pm_instance = PortfolioManager({'AAPL': 10}, '1y')
    pm_instance.add_change('2023-02-01', 'AAPL', '-2')
    assert len(pm_instance._changes) == 1
    assert pm_instance._changes[0][2] == '-2'
    assert pm_instance._portfolio_data is None

@pytest.mark.parametrize("invalid_date", ["2023/13/01", "not-a-date"])
def test_add_change_invalid_date(invalid_date):
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    with pytest.raises(InvalidInputError, match="Invalid date format"):
        pm_instance.add_change(invalid_date, 'AAPL', 5)

@pytest.mark.parametrize("invalid_qty", [-1, -100])
def test_add_change_invalid_absolute_qty(invalid_qty):
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    with pytest.raises(InvalidInputError, match="Absolute quantity_change must be non-negative"):
        pm_instance.add_change('2023-01-01', 'AAPL', invalid_qty)

@pytest.mark.parametrize("invalid_rel_qty", ["+5.5", "-abc", "10", "+", "-"])
def test_add_change_invalid_relative_qty_format(invalid_rel_qty):
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    with pytest.raises(InvalidInputError, match="Invalid relative quantity format"):
        pm_instance.add_change('2023-01-01', 'AAPL', invalid_rel_qty)

def test_add_change_invalid_qty_type():
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    with pytest.raises(InvalidInputError, match="quantity_change must be int .* or str"):
        pm_instance.add_change('2023-01-01', 'AAPL', 10.5) # Float is invalid

def test_add_change_invalid_ticker():
     pm_instance = PortfolioManager({'AAPL': 1}, '1y')
     with pytest.raises(InvalidInputError, match="Ticker must be a non-empty string"):
        pm_instance.add_change('2023-01-01', '', 5)
     with pytest.raises(InvalidInputError, match="Ticker must be a non-empty string"):
        pm_instance.add_change('2023-01-01', None, 5)


# --- Test Building Portfolio (build_portfolio & _fetch_all_data) ---

def configure_mock_plotter_class(mock_class, sample_data):
    """Helper to configure the class mock's side effect."""
    fetch_effect = create_fetch_data_side_effect(sample_data)

    def class_side_effect(ticker, time_period):
        # This function runs when StockTrendPlotter(ticker, period) is called
        # print(f"Mock StockTrendPlotter CLASS called with: ticker={ticker}, period={time_period}") # Debug
        instance_mock = MagicMock()
        instance_mock.ticker = ticker # Store the ticker on the instance mock
        # Attach the fetch_data side effect to this specific instance mock
        instance_mock.fetch_data = MagicMock(side_effect=lambda *a, **kw: fetch_effect(instance_mock, *a, **kw))
        # print(f"  Returning instance mock {id(instance_mock)} for ticker {ticker}") # Debug
        return instance_mock

    mock_class.side_effect = class_side_effect


def test_build_portfolio_no_changes(mock_stock_plotter_class, sample_price_data):
    """Test build with only initial holdings."""
    initial = {'AAPL': 10, 'MSFT': 5}
    period = '5d'
    configure_mock_plotter_class(mock_stock_plotter_class, sample_price_data)

    pm_instance = PortfolioManager(initial, period)
    df = pm_instance.build_portfolio() # This will call StockTrendPlotter(ticker, period)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert 'AAPL_qnty' in df.columns and 'MSFT_qnty' in df.columns
    assert 'AAPL_value' in df.columns and 'MSFT_value' in df.columns
    assert 'Total_Value' in df.columns

    assert (df['AAPL_qnty'] == 10).all()
    assert (df['MSFT_qnty'] == 5).all()

    expected_aapl_value_day1 = 150.0 * 10
    assert df.loc[pd.to_datetime('2023-01-01'), 'AAPL_value'] == expected_aapl_value_day1
    expected_total_day1 = (150.0 * 10) + (280.0 * 5)
    assert df.loc[pd.to_datetime('2023-01-01'), 'Total_Value'] == expected_total_day1

    # Check the class mock was called correctly
    assert mock_stock_plotter_class.call_count == len(initial)
    mock_stock_plotter_class.assert_any_call('AAPL', period)
    mock_stock_plotter_class.assert_any_call('MSFT', period)

    # Check fetch_data was called on the returned instances (via the side effect)
    # We need to check the calls made to the instances returned by the class mock side effect.
    # Accessing the mock instances directly is tricky here, but we know fetch_data should be called.
    # Let's check the call count on the *method* of the *class* mock's return_value if side_effect wasn't complex,
    # but with a function side_effect, it's harder. Instead, we trust the logic inside build_portfolio
    # calls fetch_data if the class instantiation succeeded. The successful df build implies it worked.


def test_build_portfolio_with_changes(mock_stock_plotter_class, sample_price_data):
    """Test build with initial holdings and various changes."""
    initial = {'AAPL': 10}
    period = '5d'
    configure_mock_plotter_class(mock_stock_plotter_class, sample_price_data)

    pm_instance = PortfolioManager(initial, period)
    pm_instance.add_change('2023-01-03', 'AAPL', '-3') # Relative sell (10 -> 7)
    pm_instance.add_change('2023-01-04', 'MSFT', '+20') # Add new stock (0 -> 20)
    pm_instance.add_change('2023-01-05', 'AAPL', 5) # Absolute set (7 -> 5)

    df = pm_instance.build_portfolio()

    assert df is not None
    assert 'AAPL_qnty' in df.columns and 'MSFT_qnty' in df.columns

    assert df.loc[pd.to_datetime('2023-01-01'), 'AAPL_qnty'] == 10
    assert df.loc[pd.to_datetime('2023-01-02'), 'AAPL_qnty'] == 10
    assert df.loc[pd.to_datetime('2023-01-03'), 'AAPL_qnty'] == 7
    assert df.loc[pd.to_datetime('2023-01-04'), 'AAPL_qnty'] == 7
    assert df.loc[pd.to_datetime('2023-01-05'), 'AAPL_qnty'] == 5

    assert df.loc[pd.to_datetime('2023-01-01'), 'MSFT_qnty'] == 0
    assert df.loc[pd.to_datetime('2023-01-03'), 'MSFT_qnty'] == 0
    assert df.loc[pd.to_datetime('2023-01-04'), 'MSFT_qnty'] == 20
    assert df.loc[pd.to_datetime('2023-01-05'), 'MSFT_qnty'] == 20

    # Check plotter class calls (initial + new ticker from change)
    assert mock_stock_plotter_class.call_count == 2 # AAPL, MSFT
    mock_stock_plotter_class.assert_any_call('AAPL', period)
    mock_stock_plotter_class.assert_any_call('MSFT', period)


def test_build_portfolio_relative_to_zero(mock_stock_plotter_class, sample_price_data):
    """Test relative change resulting in zero quantity."""
    initial = {'AAPL': 5}
    period = '5d'
    configure_mock_plotter_class(mock_stock_plotter_class, sample_price_data)
    pm_instance = PortfolioManager(initial, period)
    pm_instance.add_change('2023-01-03', 'AAPL', '-5') # 5 -> 0

    df = pm_instance.build_portfolio()
    assert df.loc[pd.to_datetime('2023-01-03'), 'AAPL_qnty'] == 0
    assert df.loc[pd.to_datetime('2023-01-04'), 'AAPL_qnty'] == 0
    assert df.loc[pd.to_datetime('2023-01-03'), 'AAPL_value'] == 0.0

def test_build_portfolio_relative_below_zero(mock_stock_plotter_class, sample_price_data):
    """Test relative change attempting negative quantity (should cap at 0)."""
    initial = {'AAPL': 3}
    period = '5d'
    configure_mock_plotter_class(mock_stock_plotter_class, sample_price_data)
    pm_instance = PortfolioManager(initial, period)
    pm_instance.add_change('2023-01-03', 'AAPL', '-10') # 3 -> 0 (not -7)

    df = pm_instance.build_portfolio()
    assert df.loc[pd.to_datetime('2023-01-03'), 'AAPL_qnty'] == 0
    assert df.loc[pd.to_datetime('2023-01-04'), 'AAPL_qnty'] == 0

def test_build_portfolio_change_after_data(mock_stock_plotter_class, sample_price_data):
    """Test change date after the last data point."""
    initial = {'AAPL': 10}
    period = '5d' # Data ends 2023-01-05
    configure_mock_plotter_class(mock_stock_plotter_class, sample_price_data)
    pm_instance = PortfolioManager(initial, period)
    pm_instance.add_change('2023-01-10', 'AAPL', 5) # Date after data range

    df = pm_instance.build_portfolio()
    # The change should not affect the data; quantity remains initial value
    assert (df['AAPL_qnty'] == 10).all()

def test_build_portfolio_fetch_fails_one_ticker(mock_stock_plotter_class, sample_price_data):
    """Test build when fetching fails for one ticker but succeeds for others."""
    initial = {'AAPL': 10, 'FAIL': 5} # FAIL ticker will cause fetch error
    period = '5d'
    configure_mock_plotter_class(mock_stock_plotter_class, sample_price_data)

    pm_instance = PortfolioManager(initial, period)
    df = pm_instance.build_portfolio() # Should succeed but exclude FAIL

    assert df is not None
    assert 'AAPL_qnty' in df.columns
    assert 'FAIL_qnty' not in df.columns # Should not have quantity column
    assert 'FAIL_value' not in df.columns
    assert (df['AAPL_qnty'] == 10).all()
    assert mock_stock_plotter_class.call_count == 2 # Attempted both

def test_build_portfolio_fetch_fails_all_tickers(mock_stock_plotter_class, sample_price_data):
    """Test build failure when fetching fails for all tickers."""
    initial = {'FAIL': 10} # Only has failing ticker
    period = '1y'
    configure_mock_plotter_class(mock_stock_plotter_class, sample_price_data)

    pm_instance = PortfolioManager(initial, period)
    # Expect DataFetchError because _fetch_all_data fails completely
    with pytest.raises(DataFetchError, match="Unable to fetch price data"):
        pm_instance.build_portfolio()
    assert pm_instance._portfolio_data is None # Ensure cache is None

def test_build_portfolio_fetch_returns_empty(mock_stock_plotter_class, sample_price_data):
    """Test build when fetch returns empty DataFrame for a ticker."""
    initial = {'AAPL': 10, 'EMPTY': 5}
    period = '5d'
    configure_mock_plotter_class(mock_stock_plotter_class, sample_price_data)

    pm_instance = PortfolioManager(initial, period)
    df = pm_instance.build_portfolio()

    assert df is not None
    assert 'AAPL_qnty' in df.columns
    assert 'EMPTY_qnty' not in df.columns # Should be skipped
    assert mock_stock_plotter_class.call_count == 2


# --- Test Getting Portfolio Data (get_portfolio_data) ---
# (These tests remain the same as they mock build_portfolio directly)
@patch.object(PortfolioManager, 'build_portfolio', autospec=True)
def test_get_portfolio_data_triggers_build(mock_build):
    """Test that get_portfolio_data calls build_portfolio if data is None."""
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    pm_instance._portfolio_data = None # Ensure it's None initially

    pm_instance.get_portfolio_data()
    mock_build.assert_called_once_with(pm_instance)

@patch.object(PortfolioManager, 'build_portfolio', autospec=True)
def test_get_portfolio_data_returns_copy(mock_build, sample_price_data):
    """Test that get_portfolio_data returns a copy, not the original."""
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    # Simulate data already being built
    original_df = pd.DataFrame({'Total_Value': [100, 110]})
    pm_instance._portfolio_data = original_df

    retrieved_df = pm_instance.get_portfolio_data()

    assert retrieved_df is not original_df # Check they are different objects
    assert retrieved_df.equals(original_df) # Check values are the same
    mock_build.assert_not_called() # Build should not be called if data exists

@patch.object(PortfolioManager, 'build_portfolio', side_effect=DataFetchError("Build failed"))
def test_get_portfolio_data_handles_build_failure(mock_build_fails):
    """Test get_portfolio_data when the triggered build fails."""
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    pm_instance._portfolio_data = None

    with pytest.raises(DataFetchError, match="Build failed"):
        pm_instance.get_portfolio_data()
    mock_build_fails.assert_called_once()
    assert pm_instance._portfolio_data is None # Ensure cache remains None


# --- Test Getting Composition (get_composition) ---
# (These tests remain the same as they mock get_portfolio_data)
@pytest.fixture
def built_portfolio_df(sample_price_data):
    """Provides a simple, built portfolio DataFrame for composition tests."""
    df = pd.DataFrame(index=sample_price_data.index)
    df['AAPL_qnty'] = 10
    df['MSFT_qnty'] = 5
    df['AAPL_value'] = df['AAPL_qnty'] * sample_price_data['AAPL']
    df['MSFT_value'] = df['MSFT_qnty'] * sample_price_data['MSFT']
    df['Total_Value'] = df['AAPL_value'] + df['MSFT_value']
    return df[['AAPL_qnty', 'MSFT_qnty', 'AAPL_value', 'MSFT_value', 'Total_Value']]

@patch.object(PortfolioManager, 'get_portfolio_data')
def test_get_composition_latest_date(mock_get_data, built_portfolio_df):
    """Test getting composition for the latest date."""
    mock_get_data.return_value = built_portfolio_df.copy()
    pm_instance = PortfolioManager({'AAPL': 10, 'MSFT': 5}, '5d') # Args don't matter due to mock

    composition = pm_instance.get_composition() # No date specified

    assert composition is not None
    assert isinstance(composition, pd.Series)
    assert 'AAPL' in composition.index and 'MSFT' in composition.index

    # Calculate expected for last date (2023-01-05)
    last_row = built_portfolio_df.iloc[-1]
    expected_aapl_pct = (last_row['AAPL_value'] / last_row['Total_Value']) * 100
    expected_msft_pct = (last_row['MSFT_value'] / last_row['Total_Value']) * 100

    assert composition['AAPL'] == pytest.approx(expected_aapl_pct)
    assert composition['MSFT'] == pytest.approx(expected_msft_pct)
    assert composition.sum() == pytest.approx(100.0)
    mock_get_data.assert_called_once()


@patch.object(PortfolioManager, 'get_portfolio_data')
def test_get_composition_specific_date_exact(mock_get_data, built_portfolio_df):
    """Test getting composition for a specific date that exists."""
    mock_get_data.return_value = built_portfolio_df.copy()
    pm_instance = PortfolioManager({'AAPL': 10, 'MSFT': 5}, '5d')
    target_date_str = '2023-01-03'
    target_date = pd.to_datetime(target_date_str)

    composition = pm_instance.get_composition(date=target_date_str)

    assert composition is not None
    row = built_portfolio_df.loc[target_date]
    expected_aapl_pct = (row['AAPL_value'] / row['Total_Value']) * 100
    expected_msft_pct = (row['MSFT_value'] / row['Total_Value']) * 100
    assert composition['AAPL'] == pytest.approx(expected_aapl_pct)
    assert composition['MSFT'] == pytest.approx(expected_msft_pct)

@patch.object(PortfolioManager, 'get_portfolio_data')
def test_get_composition_specific_date_nearest(mock_get_data, built_portfolio_df):
    """Test getting composition using nearest date logic."""
    mock_get_data.return_value = built_portfolio_df.copy()
    pm_instance = PortfolioManager({'AAPL': 10, 'MSFT': 5}, '5d')
    target_date_str = '2023-01-06' # Date after last data point
    nearest_date = pd.to_datetime('2023-01-05') # Expected nearest

    composition = pm_instance.get_composition(date=target_date_str)

    assert composition is not None
    row = built_portfolio_df.loc[nearest_date] # Check against the nearest date's data
    expected_aapl_pct = (row['AAPL_value'] / row['Total_Value']) * 100
    expected_msft_pct = (row['MSFT_value'] / row['Total_Value']) * 100
    assert composition['AAPL'] == pytest.approx(expected_aapl_pct)
    assert composition['MSFT'] == pytest.approx(expected_msft_pct)

def test_get_composition_invalid_date_format():
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    # No need to mock get_portfolio_data as error should happen before
    with pytest.raises(InvalidInputError, match="Invalid date format"):
        pm_instance.get_composition(date="not-a-real-date")

@patch.object(PortfolioManager, 'get_portfolio_data')
def test_get_composition_zero_total_value(mock_get_data, built_portfolio_df):
    """Test composition when total value is zero."""
    zero_value_df = built_portfolio_df.copy()
    target_date = pd.to_datetime('2023-01-03')
    zero_value_df.loc[target_date, ['AAPL_value', 'MSFT_value', 'Total_Value']] = 0.0 # Force zero value
    mock_get_data.return_value = zero_value_df
    pm_instance = PortfolioManager({'AAPL': 10, 'MSFT': 5}, '5d')

    composition = pm_instance.get_composition(date='2023-01-03')

    assert composition is not None
    assert (composition == 0.0).all() # Expect 0% for all assets
    assert list(composition.index) == ['AAPL', 'MSFT'] # Should still list assets

@patch.object(PortfolioManager, 'get_portfolio_data', return_value=None)
def test_get_composition_no_portfolio_data(mock_get_data_none):
    """Test composition when portfolio data is None."""
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    composition = pm_instance.get_composition()
    assert composition is None
    mock_get_data_none.assert_called_once()

@patch.object(PortfolioManager, 'get_portfolio_data', side_effect=DataFetchError("Failed build"))
def test_get_composition_handles_get_data_error(mock_get_data_error):
    """Test composition handles errors raised by get_portfolio_data."""
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    # Expect None because the underlying get_portfolio_data failed
    composition = pm_instance.get_composition()
    assert composition is None
    mock_get_data_error.assert_called_once()