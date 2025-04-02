import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from unittest.mock import patch, MagicMock, call
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal
from typing import Dict, List, Optional, Set, Tuple, Union, cast

# Import the module and classes/exceptions to test
import portfolio_manager as pm
from portfolio_manager import (
    PortfolioManager,
    InvalidInputError,
    DataFetchError,
    PortfolioBuildError,
    PortfolioError,
    PRICE_COLUMN_NAME # Import the constant used for price lookup
)
# Assuming data_handler exceptions might be wrapped or need checking
from data_handler import DataFetchError as DH_DataFetchError


# --- Fixtures ---

@pytest.fixture
def sample_initial_portfolio() -> Dict[str, int]:
    """Basic initial portfolio."""
    return {'AAPL': 10, 'MSFT': 5}

@pytest.fixture
def default_time_period() -> str:
    """Default time period for tests."""
    return '1y' # Example period

# --- Mock Data Fixtures for data_handler.fetch_stock_data ---

@pytest.fixture
def mock_dates() -> pd.DatetimeIndex:
    """Sample dates for mock data (timezone-naive)."""
    # Using a few business days for realism
    return pd.date_range(start='2023-01-02', end='2023-01-06', freq='B', name='Date').tz_localize(None)

@pytest.fixture
def mock_tickers() -> List[str]:
    """Tickers corresponding to mock_fetch_data."""
    return ['AAPL', 'MSFT', 'GOOG'] # Corresponds to columns in mock_fetch_data

@pytest.fixture
def mock_fetch_data_success(mock_dates, mock_tickers) -> pd.DataFrame:
    """
    Simulates a successful return DataFrame from data_handler.fetch_stock_data
    with Adj Close and Volume.
    """
    data_types = [PRICE_COLUMN_NAME, 'Volume'] # Use the imported constant
    columns = pd.MultiIndex.from_product([data_types, mock_tickers], names=['DataType', 'Ticker'])
    data = np.random.rand(len(mock_dates), len(columns)) * 100 # Random data
    df = pd.DataFrame(data, index=mock_dates, columns=columns)

    # Assign more realistic values/types
    df[(PRICE_COLUMN_NAME, 'AAPL')] = [150.0, 151.0, 150.5, 152.0, 153.0]
    df[(PRICE_COLUMN_NAME, 'MSFT')] = [280.0, 281.0, 282.0, 280.0, 283.0]
    df[(PRICE_COLUMN_NAME, 'GOOG')] = [100.0, 101.0, 100.0, 102.0, 103.0]
    df[('Volume', 'AAPL')] = [1e6, 1.1e6, 1.05e6, 1.2e6, 1.3e6]
    df[('Volume', 'MSFT')] = [5e5, 5.1e5, 5.2e5, 5.0e5, 5.3e5]
    df[('Volume', 'GOOG')] = [8e5, 8.1e5, 8.0e5, 8.2e5, 8.3e5]

    # Ensure Volume is integer
    df['Volume'] = df['Volume'].astype(int)
    return df

@pytest.fixture
def mock_fetch_data_partial_fail(mock_fetch_data_success) -> pd.DataFrame:
    """Simulates data_handler returning data for only some tickers."""
    df = mock_fetch_data_success.copy()
    # Drop all columns related to GOOG
    df = df.drop(columns='GOOG', level='Ticker')
    return df

@pytest.fixture
def mock_fetch_data_empty() -> pd.DataFrame:
    """Simulates data_handler returning an empty DataFrame."""
    return pd.DataFrame(columns=pd.MultiIndex.from_product([[],[]], names=['DataType', 'Ticker']))


# --- Test Initialization (__init__) ---

def test_init_success(sample_initial_portfolio, default_time_period):
    pm_instance = PortfolioManager(sample_initial_portfolio, default_time_period)
    assert pm_instance.initial_portfolio == sample_initial_portfolio
    assert pm_instance.time_period == default_time_period
    assert pm_instance._all_tickers == set(sample_initial_portfolio.keys())
    assert pm_instance._changes == []
    assert pm_instance._portfolio_data is None
    assert pm_instance._date_range is None # Date range not calculated yet

def test_init_uppercase_ticker():
    initial = {'aapl': 5}
    pm_instance = PortfolioManager(initial, '6m')
    assert pm_instance.initial_portfolio == {'AAPL': 5}
    assert 'AAPL' in pm_instance._all_tickers

# REMOVED: Test for empty initial portfolio failure (now allowed)
# def test_init_invalid_portfolio_empty(): ...

@pytest.mark.parametrize("invalid_portfolio, error_msg_match", [
    (None, "initial_portfolio must be a dictionary"),
    # Case removed: ({}, "initial_portfolio cannot be empty"),
    ({'AAPL': -5}, "values must be non-negative integers"),
    ({'AAPL': 10.5}, "values must be non-negative integers"),
    # Updated match based on actual code
    ({123: 10}, "Portfolio keys must be non-empty strings.*values must be non-negative integers"),
])
def test_init_invalid_portfolio(invalid_portfolio, error_msg_match):
    with pytest.raises(InvalidInputError, match=error_msg_match):
        PortfolioManager(invalid_portfolio, '1y')

@pytest.mark.parametrize("invalid_period", [None, "", "   ", 123])
def test_init_invalid_time_period(invalid_period):
    with pytest.raises(InvalidInputError, match="time_period must be a non-empty string"):
        PortfolioManager({'AAPL': 1}, invalid_period)


# --- Test Adding Changes (add_change) ---

def test_add_change_success_absolute(sample_initial_portfolio, default_time_period):
    pm_instance = PortfolioManager(sample_initial_portfolio, default_time_period)
    date_str = '2023-01-15'
    ticker = 'GOOG' # New ticker
    qty = 5
    pm_instance.add_change(date_str, ticker, qty)

    assert len(pm_instance._changes) == 1
    change_tuple = pm_instance._changes[0]
    assert change_tuple[0] == pd.to_datetime(date_str).tz_localize(None) # Check tz naive
    assert change_tuple[1] == 'GOOG' # Check uppercase
    assert change_tuple[2] == qty
    assert 'GOOG' in pm_instance._all_tickers
    assert pm_instance._all_tickers == {'AAPL', 'MSFT', 'GOOG'}
    assert pm_instance._portfolio_data is None # Check cache invalidation
    assert pm_instance._date_range is None # Check date range cache invalidation

def test_add_change_success_relative_plus(sample_initial_portfolio, default_time_period):
    pm_instance = PortfolioManager(sample_initial_portfolio, default_time_period)
    pm_instance.add_change('2023-02-01', 'aapl', '+5') # Case-insensitive ticker
    assert len(pm_instance._changes) == 1
    assert pm_instance._changes[0][1] == 'AAPL' # Stored uppercase
    assert pm_instance._changes[0][2] == '+5' # Stored cleaned format
    assert pm_instance._portfolio_data is None

def test_add_change_success_relative_minus(sample_initial_portfolio, default_time_period):
    pm_instance = PortfolioManager(sample_initial_portfolio, default_time_period)
    pm_instance.add_change('2023-02-01', 'MSFT', ' -2 ') # With spaces
    assert len(pm_instance._changes) == 1
    assert pm_instance._changes[0][1] == 'MSFT'
    assert pm_instance._changes[0][2] == '-2' # Check cleaning
    assert pm_instance._portfolio_data is None

@pytest.mark.parametrize("invalid_date", ["2023/13/01", "not-a-date"])
def test_add_change_invalid_date(invalid_date, sample_initial_portfolio):
    pm_instance = PortfolioManager(sample_initial_portfolio, '1y')
    with pytest.raises(InvalidInputError, match="Invalid date format"):
        pm_instance.add_change(invalid_date, 'AAPL', 5)

@pytest.mark.parametrize("invalid_qty", [-1, -100])
def test_add_change_invalid_absolute_qty(invalid_qty, sample_initial_portfolio):
    pm_instance = PortfolioManager(sample_initial_portfolio, '1y')
    # Update match pattern
    with pytest.raises(InvalidInputError, match="Absolute quantity change.*must be non-negative"):
        pm_instance.add_change('2023-01-01', 'AAPL', invalid_qty)

@pytest.mark.parametrize("invalid_rel_qty", ["+5.5", "-abc", "10", "+", "-"])
def test_add_change_invalid_relative_qty_format(invalid_rel_qty, sample_initial_portfolio):
    pm_instance = PortfolioManager(sample_initial_portfolio, '1y')
    with pytest.raises(InvalidInputError, match="Invalid relative quantity format"):
        pm_instance.add_change('2023-01-01', 'AAPL', invalid_rel_qty)

def test_add_change_invalid_qty_type(sample_initial_portfolio):
    pm_instance = PortfolioManager(sample_initial_portfolio, '1y')
    # Update match pattern
    with pytest.raises(InvalidInputError, match="quantity_change.*must be int.*or str"):
        pm_instance.add_change('2023-01-01', 'AAPL', 10.5) # Float is invalid

def test_add_change_invalid_ticker(sample_initial_portfolio):
     pm_instance = PortfolioManager(sample_initial_portfolio, '1y')
     with pytest.raises(InvalidInputError, match="Ticker must be a non-empty string"):
        pm_instance.add_change('2023-01-01', '', 5)
     with pytest.raises(InvalidInputError, match="Ticker must be a non-empty string"):
        pm_instance.add_change('2023-01-01', None, 5)


# --- Test _determine_date_range (Internal Method - Optional but useful) ---
# Note: Requires knowledge of internal implementation, less ideal for pure black-box testing

@pytest.mark.parametrize("time_period, expected_start_partial", [
    ('1y', date.today() - relativedelta(years=1)),
    ('6mo', date.today() - relativedelta(months=6)),
    ('ytd', date.today().replace(month=1, day=1)),
    ('this year', date.today().replace(month=1, day=1)),
    ('this month', date.today().replace(day=1)),
    ('3M', date.today() - relativedelta(months=3)),
    ('10D', date.today() - relativedelta(days=10)),
    ('from 2022', date(2022, 1, 1)),
    ('from 2021-07', date(2021, 7, 1)),
    ('max', date(1970, 1, 1)), # Based on current implementation
])
def test_determine_date_range_parsing(time_period, expected_start_partial):
    # Only test specific periods, no external calls needed here
    pm_instance = PortfolioManager({'AAPL': 1}, time_period)
    start_date, end_date = pm_instance._determine_date_range()
    assert end_date == date.today()
    # Allow slight difference for calculation variances if comparing relativedelta directly
    if isinstance(expected_start_partial, date):
        assert start_date == expected_start_partial
    else: # Handle relativedelta comparison if needed (though fixture provides date)
        pass

def test_determine_date_range_invalid_period():
    pm_instance = PortfolioManager({'AAPL': 1}, "invalid period format")
    with pytest.raises(InvalidInputError, match="Invalid time period format"):
        pm_instance._determine_date_range()

# --- Test Building Portfolio (build_portfolio & _fetch_all_data) ---

# Use patch where fetch_stock_data is imported/used inside portfolio_manager
@patch('portfolio_manager.fetch_stock_data', autospec=True)
def test_build_portfolio_no_changes(mock_fetch, mock_fetch_data_success, sample_initial_portfolio):
    """Test build with only initial holdings, using mocked fetch_stock_data."""
    mock_fetch.return_value = mock_fetch_data_success # Configure mock return
    period = '1y' # Used for date range determination
    expected_tickers = list(sample_initial_portfolio.keys())

    pm_instance = PortfolioManager(sample_initial_portfolio, period)
    df_portfolio = pm_instance.build_portfolio()

    # --- Assertions ---
    # 1. Check mock call
    mock_fetch.assert_called_once()
    call_args = mock_fetch.call_args[1] # Keyword args
    assert sorted(call_args['tickers']) == sorted(expected_tickers)
    assert call_args['use_adj_close'] is True # Default should be True
    # Check dates (need to know what _determine_date_range calculates for '1y')
    expected_start = (date.today() - relativedelta(years=1)).strftime('%Y-%m-%d')
    expected_end = date.today().strftime('%Y-%m-%d')
    assert call_args['start_date_str'] == expected_start
    assert call_args['end_date_str'] == expected_end

    # 2. Check built DataFrame structure
    assert df_portfolio is not None
    assert isinstance(df_portfolio, pd.DataFrame)
    assert 'AAPL_qnty' in df_portfolio.columns and 'MSFT_qnty' in df_portfolio.columns
    assert 'AAPL_value' in df_portfolio.columns and 'MSFT_value' in df_portfolio.columns
    assert 'GOOG_qnty' not in df_portfolio.columns # GOOG not in initial portfolio
    assert 'Total_Value' in df_portfolio.columns
    # Check index aligns with mock data (subset of dates might be used depending on period)
    assert_index_equal(df_portfolio.index, mock_fetch_data_success.index, check_names=False)

    # 3. Check quantity columns
    assert (df_portfolio['AAPL_qnty'] == 10).all()
    assert (df_portfolio['MSFT_qnty'] == 5).all()

    # 4. Check value calculations based on PRICE_COLUMN_NAME ('Adj Close')
    aapl_price_col = (PRICE_COLUMN_NAME, 'AAPL')
    msft_price_col = (PRICE_COLUMN_NAME, 'MSFT')
    # Example: check first day's values
    first_date = mock_fetch_data_success.index[0]
    expected_aapl_value_day1 = mock_fetch_data_success.loc[first_date, aapl_price_col] * 10
    expected_msft_value_day1 = mock_fetch_data_success.loc[first_date, msft_price_col] * 5
    expected_total_day1 = expected_aapl_value_day1 + expected_msft_value_day1

    assert df_portfolio.loc[first_date, 'AAPL_value'] == pytest.approx(expected_aapl_value_day1)
    assert df_portfolio.loc[first_date, 'MSFT_value'] == pytest.approx(expected_msft_value_day1)
    assert df_portfolio.loc[first_date, 'Total_Value'] == pytest.approx(expected_total_day1)


@patch('portfolio_manager.fetch_stock_data', autospec=True)
def test_build_portfolio_with_changes(mock_fetch, mock_fetch_data_success):
    """Test build with initial holdings and various changes."""
    mock_fetch.return_value = mock_fetch_data_success
    initial = {'AAPL': 10}
    period = '1y' # Covers date range '2023-01-02' to '2023-01-06'
    pm_instance = PortfolioManager(initial, period)

    # Add changes within the mock data date range
    pm_instance.add_change('2023-01-04', 'AAPL', '-3') # Relative sell (10 -> 7)
    pm_instance.add_change('2023-01-05', 'MSFT', '+20') # Add new stock (0 -> 20)
    pm_instance.add_change('2023-01-06', 'AAPL', 5) # Absolute set (7 -> 5)

    df = pm_instance.build_portfolio()

    # Check fetch call (includes all involved tickers)
    mock_fetch.assert_called_once()
    call_args = mock_fetch.call_args[1]
    assert sorted(call_args['tickers']) == sorted(['AAPL', 'MSFT'])

    # Check DataFrame structure and values
    assert df is not None
    assert 'AAPL_qnty' in df.columns and 'MSFT_qnty' in df.columns

    # Check quantity propagation based on change dates (index from mock_fetch_data_success)
    assert df.loc['2023-01-02', 'AAPL_qnty'] == 10
    assert df.loc['2023-01-03', 'AAPL_qnty'] == 10
    assert df.loc['2023-01-04', 'AAPL_qnty'] == 7
    assert df.loc['2023-01-05', 'AAPL_qnty'] == 7
    assert df.loc['2023-01-06', 'AAPL_qnty'] == 5

    assert df.loc['2023-01-02', 'MSFT_qnty'] == 0
    assert df.loc['2023-01-04', 'MSFT_qnty'] == 0
    assert df.loc['2023-01-05', 'MSFT_qnty'] == 20
    assert df.loc['2023-01-06', 'MSFT_qnty'] == 20

    # Check value on a specific date after changes
    date_check = pd.Timestamp('2023-01-05', tz=None)
    aapl_price = mock_fetch_data_success.loc[date_check, (PRICE_COLUMN_NAME, 'AAPL')]
    msft_price = mock_fetch_data_success.loc[date_check, (PRICE_COLUMN_NAME, 'MSFT')]
    expected_aapl_val = 7 * aapl_price
    expected_msft_val = 20 * msft_price
    expected_total = expected_aapl_val + expected_msft_val

    assert df.loc[date_check, 'AAPL_value'] == pytest.approx(expected_aapl_val)
    assert df.loc[date_check, 'MSFT_value'] == pytest.approx(expected_msft_val)
    assert df.loc[date_check, 'Total_Value'] == pytest.approx(expected_total)


@patch('portfolio_manager.fetch_stock_data', autospec=True)
def test_build_portfolio_relative_below_zero(mock_fetch, mock_fetch_data_success):
    """Test relative change attempting negative quantity (should cap at 0)."""
    mock_fetch.return_value = mock_fetch_data_success
    initial = {'AAPL': 3}
    period = '1y'
    pm_instance = PortfolioManager(initial, period)
    pm_instance.add_change('2023-01-04', 'AAPL', '-10') # 3 -> 0 (not -7)

    df = pm_instance.build_portfolio()
    assert df.loc[pd.to_datetime('2023-01-04'), 'AAPL_qnty'] == 0
    assert df.loc[pd.to_datetime('2023-01-05'), 'AAPL_qnty'] == 0
    assert df.loc[pd.to_datetime('2023-01-06'), 'AAPL_qnty'] == 0
    assert (df['AAPL_value'] >= 0).all() # Value should not be negative


@patch('portfolio_manager.fetch_stock_data', autospec=True)
def test_build_portfolio_fetch_fails_one_ticker(mock_fetch, mock_fetch_data_partial_fail):
    """Test build when fetch only returns data for some tickers."""
    mock_fetch.return_value = mock_fetch_data_partial_fail # GOOG data missing
    initial = {'AAPL': 10, 'MSFT': 5, 'GOOG': 2} # Initially hold GOOG
    period = '1y'
    pm_instance = PortfolioManager(initial, period)

    df = pm_instance.build_portfolio()

    mock_fetch.assert_called_once() # Fetch was attempted for all 3
    assert sorted(mock_fetch.call_args[1]['tickers']) == sorted(['AAPL', 'GOOG', 'MSFT'])

    # Check that successful tickers are present, failed one is handled gracefully
    assert df is not None
    assert 'AAPL_qnty' in df.columns and 'MSFT_qnty' in df.columns
    assert 'AAPL_value' in df.columns and 'MSFT_value' in df.columns
    assert 'GOOG_qnty' in df.columns # Quantity column should still exist (initialized)
    assert 'GOOG_value' in df.columns # Value column exists but should be 0

    assert (df['AAPL_qnty'] == 10).all()
    assert (df['MSFT_qnty'] == 5).all()
    assert (df['GOOG_qnty'] == 2).all() # Qty remains initial as no changes applied
    assert (df['GOOG_value'] == 0.0).all() # Value is 0 as price data was missing

    # Total value should only include AAPL and MSFT
    date_check = pd.Timestamp('2023-01-03')
    expected_total = df.loc[date_check, 'AAPL_value'] + df.loc[date_check, 'MSFT_value']
    assert df.loc[date_check, 'Total_Value'] == pytest.approx(expected_total)


@patch('portfolio_manager.fetch_stock_data', side_effect=DH_DataFetchError("Simulated fetch failure"))
def test_build_portfolio_fetch_fails_all(mock_fetch_fails):
    """Test build failure when data_handler.fetch_stock_data raises error."""
    initial = {'AAPL': 10}
    period = '1y'
    pm_instance = PortfolioManager(initial, period)

    # Expect DataFetchError (raised by build_portfolio wrapping the DH error)
    with pytest.raises(DataFetchError, match="Failed to fetch portfolio data: Simulated fetch failure"):
        pm_instance.build_portfolio()

    mock_fetch_fails.assert_called_once()
    assert pm_instance._portfolio_data is None # Ensure cache is None


@patch('portfolio_manager.fetch_stock_data', return_value=pd.DataFrame()) # Return empty DF
def test_build_portfolio_fetch_returns_empty_df(mock_fetch_empty):
    """Test build when data_handler returns an empty DataFrame."""
    initial = {'AAPL': 10}
    period = '1y'
    pm_instance = PortfolioManager(initial, period)

    # Build should proceed but result in a DataFrame with quantities but zero values
    df = pm_instance.build_portfolio()

    mock_fetch_empty.assert_called_once()
    assert df is not None
    # Index might be based on date range if fetch is empty
    # Check expected columns exist
    assert 'AAPL_qnty' in df.columns
    assert 'AAPL_value' in df.columns
    assert 'Total_Value' in df.columns

    # Quantities should be present, values should be zero
    assert not df['AAPL_qnty'].empty
    assert (df['AAPL_qnty'] == 10).all() # Check initial quantity is populated
    assert (df['AAPL_value'] == 0).all()
    assert (df['Total_Value'] == 0).all()


# --- Test Getting Portfolio Data (get_portfolio_data) ---

@patch.object(PortfolioManager, 'build_portfolio', autospec=True)
def test_get_portfolio_data_triggers_build(mock_build):
    """Test get_portfolio_data calls build_portfolio if data is None."""
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    pm_instance._portfolio_data = None # Ensure cache is None initially

    # Define the DataFrame build_portfolio should return AND cache
    mock_return_df = pd.DataFrame({'A': [1]})

    # --- Use side_effect to simulate caching ---
    def build_side_effect(self_arg): # self_arg will be pm_instance
        # Simulate the real method caching the data
        self_arg._portfolio_data = mock_return_df.copy()
        # Simulate the real method returning a copy
        return mock_return_df.copy()

    mock_build.side_effect = build_side_effect
    # --- End side_effect setup ---

    # Call the method under test
    result = pm_instance.get_portfolio_data()

    # Assertions
    mock_build.assert_called_once_with(pm_instance)
    # Check the result is what the mock should have returned
    assert_frame_equal(result, mock_return_df)
    assert not result.empty
    # Check that the cache *was* set by the side_effect
    assert pm_instance._portfolio_data is not None
    assert_frame_equal(pm_instance._portfolio_data, mock_return_df)



@patch.object(PortfolioManager, 'build_portfolio', autospec=True)
def test_get_portfolio_data_returns_copy(mock_build):
    """Test get_portfolio_data returns a copy, not the original cached df."""
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    # Simulate data already being built and cached
    original_df = pd.DataFrame({'Total_Value': [100, 110]})
    pm_instance._portfolio_data = original_df

    retrieved_df = pm_instance.get_portfolio_data()

    assert retrieved_df is not original_df # Check they are different objects in memory
    assert_frame_equal(retrieved_df, original_df) # Check values are the same
    mock_build.assert_not_called() # Build should not be called if data exists


@patch.object(PortfolioManager, 'build_portfolio', side_effect=DataFetchError("Build failed"))
def test_get_portfolio_data_handles_build_failure(mock_build_fails):
    """Test get_portfolio_data raises error when the triggered build fails."""
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    pm_instance._portfolio_data = None

    with pytest.raises(DataFetchError, match="Build failed"):
        pm_instance.get_portfolio_data()
    mock_build_fails.assert_called_once()
    assert pm_instance._portfolio_data is None # Ensure cache remains None


# --- Test Getting Composition (get_composition) ---

@pytest.fixture
def built_portfolio_df(mock_fetch_data_success):
    """Provides a simple, built portfolio DataFrame for composition tests, based on mock fetched data."""
    # Rebuild a simple portfolio based on the mock fetched data
    pm = PortfolioManager({'AAPL': 10, 'MSFT': 5}, '5d') # Period matches mock dates

    # Use the mock data directly to simulate the build output
    price_data = mock_fetch_data_success
    df = pd.DataFrame(index=price_data.index)
    df['AAPL_qnty'] = 10
    df['MSFT_qnty'] = 5
    df['AAPL_value'] = df['AAPL_qnty'] * price_data[(PRICE_COLUMN_NAME, 'AAPL')]
    df['MSFT_value'] = df['MSFT_qnty'] * price_data[(PRICE_COLUMN_NAME, 'MSFT')]
    df['Total_Value'] = df['AAPL_value'].fillna(0) + df['MSFT_value'].fillna(0)
    # Select only relevant columns
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

    # Calculate expected for last date from the built_portfolio_df fixture
    last_row = built_portfolio_df.iloc[-1]
    expected_aapl_pct = (last_row['AAPL_value'] / last_row['Total_Value']) * 100 if last_row['Total_Value'] else 0
    expected_msft_pct = (last_row['MSFT_value'] / last_row['Total_Value']) * 100 if last_row['Total_Value'] else 0

    assert composition['AAPL'] == pytest.approx(expected_aapl_pct)
    assert composition['MSFT'] == pytest.approx(expected_msft_pct)
    assert composition.sum() == pytest.approx(100.0)
    mock_get_data.assert_called_once()


@patch.object(PortfolioManager, 'get_portfolio_data')
def test_get_composition_specific_date_exact(mock_get_data, built_portfolio_df):
    """Test getting composition for a specific date that exists."""
    mock_get_data.return_value = built_portfolio_df.copy()
    pm_instance = PortfolioManager({'AAPL': 10, 'MSFT': 5}, '5d')
    target_date_str = '2023-01-04' # Date within the fixture index
    target_date = pd.to_datetime(target_date_str)

    # FIX: Use date_str=
    composition = pm_instance.get_composition(date_str=target_date_str)

    assert composition is not None
    row = built_portfolio_df.loc[target_date]
    expected_aapl_pct = (row['AAPL_value'] / row['Total_Value']) * 100 if row['Total_Value'] else 0
    expected_msft_pct = (row['MSFT_value'] / row['Total_Value']) * 100 if row['Total_Value'] else 0
    assert composition['AAPL'] == pytest.approx(expected_aapl_pct)
    assert composition['MSFT'] == pytest.approx(expected_msft_pct)


@patch.object(PortfolioManager, 'get_portfolio_data')
def test_get_composition_specific_date_nearest(mock_get_data, built_portfolio_df):
    """Test getting composition using nearest date logic."""
    mock_get_data.return_value = built_portfolio_df.copy()
    pm_instance = PortfolioManager({'AAPL': 10, 'MSFT': 5}, '5d')
    target_date_str = '2023-01-07' # Date after last data point in fixture
    nearest_date = pd.to_datetime('2023-01-06') # Expected nearest date in fixture

    # FIX: Use date_str=
    composition = pm_instance.get_composition(date_str=target_date_str)

    assert composition is not None
    row = built_portfolio_df.loc[nearest_date] # Check against the nearest date's data
    expected_aapl_pct = (row['AAPL_value'] / row['Total_Value']) * 100 if row['Total_Value'] else 0
    expected_msft_pct = (row['MSFT_value'] / row['Total_Value']) * 100 if row['Total_Value'] else 0
    assert composition['AAPL'] == pytest.approx(expected_aapl_pct)
    assert composition['MSFT'] == pytest.approx(expected_msft_pct)


def test_get_composition_invalid_date_format():
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    # FIX: Use date_str=
    with pytest.raises(InvalidInputError, match="Invalid date format"):
        pm_instance.get_composition(date_str="not-a-real-date")


@patch.object(PortfolioManager, 'get_portfolio_data')
def test_get_composition_zero_total_value(mock_get_data, built_portfolio_df):
    """Test composition when total value is zero on the target date."""
    zero_value_df = built_portfolio_df.copy()
    target_date_str = '2023-01-04'
    target_date = pd.to_datetime(target_date_str)
    # Force zero value for the specific date
    zero_value_df.loc[target_date, ['AAPL_value', 'MSFT_value', 'Total_Value']] = 0.0
    mock_get_data.return_value = zero_value_df
    pm_instance = PortfolioManager({'AAPL': 10, 'MSFT': 5}, '5d')

    # FIX: Use date_str=
    composition = pm_instance.get_composition(date_str=target_date_str)

    assert composition is not None
    assert isinstance(composition, pd.Series)
    assert (composition == 0.0).all() # Expect 0% for all assets
    # Check that the index still contains the expected tickers
    assert set(composition.index) == {'AAPL', 'MSFT'}


@patch.object(PortfolioManager, 'get_portfolio_data', return_value=None)
def test_get_composition_no_portfolio_data(mock_get_data_none):
    """Test composition returns None when get_portfolio_data returns None."""
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    composition = pm_instance.get_composition()
    assert composition is None
    mock_get_data_none.assert_called_once()


@patch.object(PortfolioManager, 'get_portfolio_data', side_effect=DataFetchError("Failed build"))
def test_get_composition_handles_get_data_error(mock_get_data_error):
    """Test composition returns None if get_portfolio_data raises an error."""
    pm_instance = PortfolioManager({'AAPL': 1}, '1y')
    # Expect None because the underlying get_portfolio_data failed
    composition = pm_instance.get_composition()
    assert composition is None
    mock_get_data_error.assert_called_once()