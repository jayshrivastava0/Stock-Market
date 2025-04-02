#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# test_backtester.py
"""
Test cases for the Backtester class and its components
This file contains unit tests for the Backtester class and its components.
It includes tests for initialization, data handling, order validation, and the main run method.
"""


import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Tuple, Literal
from unittest.mock import MagicMock, patch
import logging 
import re

# Imports from the module being tested
from backtester import (
    Backtester, Strategy, Order, Trade, BasicCommission, CommissionModel,
    # Exceptions
    ConfigError, DataError, StrategyError, InsufficientFundsError, InsufficientHoldingsError, BacktesterError,
    # Constants, Type Aliases if needed for setup
    BUY_ACTION, SELL_ACTION, CASH_COL, HOLDINGS_VALUE_COL, TOTAL_EQUITY_COL,
    Timestamp, TickerSymbol, HoldingsDict,
    DEFAULT_EXECUTION_PRICE_COL, DEFAULT_VALUATION_PRICE_COL,
    OPEN, CLOSE, HIGH, LOW, VOLUME 
)
# Assume data_handler exceptions might be raised indirectly via fetch_func
from data_handler import (
    DataFetchError as DH_DataFetchError,
    InvalidInputError as DH_InvalidInputError,
    DataProcessingError as DH_DataProcessingError
)

# --- Constants for Tests ---
TEST_TICKERS: List[TickerSymbol] = sorted(['TST1', 'TST2'])
TEST_START_DATE_STR: str = '2023-01-01'
TEST_END_DATE_STR: str = '2023-01-10'
TEST_START_DATE: date = date(2023, 1, 1)
TEST_END_DATE: date = date(2023, 1, 10)
INITIAL_CAPITAL: float = 100000.0

# --- Mock Strategy Implementations --- (Keep mocks as before)
class MockStrategyStaticOrders(Strategy):
    def __init__(self, orders_by_date: Dict[Timestamp, List[Order]], parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.orders_by_date = orders_by_date
    def generate_signals(self, current_dt: Timestamp, data_slice: pd.DataFrame,
                         current_holdings: HoldingsDict, current_cash: float) -> List[Order]:
        return self.orders_by_date.get(current_dt, [])

class MockStrategyDynamic(Strategy):
    def __init__(self, buy_threshold: float = 100.0, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.buy_threshold = buy_threshold
        self.bought = set()
    def generate_signals(self, current_dt: Timestamp, data_slice: pd.DataFrame,
                         current_holdings: HoldingsDict, current_cash: float) -> List[Order]:
        signals = []
        for ticker in TEST_TICKERS:
            if ticker in current_holdings or ticker in self.bought: continue
            price_col = (DEFAULT_VALUATION_PRICE_COL, ticker)
            if price_col in data_slice.columns:
                current_price = data_slice.loc[current_dt, price_col]
                if not pd.isna(current_price) and current_price > self.buy_threshold:
                    estimated_cost = 10 * current_price
                    if current_cash > estimated_cost:
                         signals.append(Order(ticker=ticker, action=BUY_ACTION, quantity=10))
                         self.bought.add(ticker)
        return signals

class MockStrategyError(Strategy):
    def generate_signals(self, current_dt: Timestamp, data_slice: pd.DataFrame,
                         current_holdings: HoldingsDict, current_cash: float) -> List[Order]:
        if current_dt >= pd.Timestamp('2023-01-05'):
             raise StrategyError("Deliberate strategy failure on or after 2023-01-05")
        return []

# --- Fixtures ---
@pytest.fixture
def valid_backtester_params() -> Dict[str, Any]:
    return {
        "initial_capital": INITIAL_CAPITAL, "tickers": TEST_TICKERS,
        "strategy_class": MockStrategyStaticOrders, "start_date": TEST_START_DATE_STR,
        "end_date": TEST_END_DATE_STR, "strategy_params": {"orders_by_date": {}},
        "commission_model": BasicCommission(fixed_fee=1.0, percent_fee=0.001),
        "data_handler_params": None, "execution_price_col": OPEN,
        "valuation_price_col": CLOSE, "data_fetch_func": MagicMock()
    }

@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    try:
        dates = pd.date_range(TEST_START_DATE_STR, TEST_END_DATE_STR, freq='B')
    except ValueError:
        dates = pd.date_range(TEST_START_DATE_STR, TEST_END_DATE_STR, freq='D')
        dates = dates[dates.dayofweek < 5]

    if dates.empty:
        start_dt_force = pd.Timestamp(TEST_START_DATE_STR)
        if start_dt_force.dayofweek >= 5:
            start_dt_force = start_dt_force + pd.offsets.BDay(1)
        dates = pd.date_range(start=start_dt_force, periods=7, freq='B')


    num_dates = len(dates)
    columns = pd.MultiIndex.from_product(
        [[OPEN, HIGH, LOW, CLOSE, VOLUME], TEST_TICKERS],
        names=['OHLCV', 'Ticker']
    )
    data = pd.DataFrame(np.nan, index=dates, columns=columns)

    # --- Deterministic First Day ---
    first_day_idx = dates[0]
    tst1_open_d1 = 100.0
    tst1_close_d1 = 101.0
    tst2_open_d1 = 150.0
    tst2_close_d1 = 151.0
    data.loc[first_day_idx, (OPEN, 'TST1')] = tst1_open_d1
    data.loc[first_day_idx, (CLOSE, 'TST1')] = tst1_close_d1
    data.loc[first_day_idx, (HIGH, 'TST1')] = tst1_close_d1 + 1.0
    data.loc[first_day_idx, (LOW, 'TST1')] = tst1_open_d1 - 1.0
    data.loc[first_day_idx, (VOLUME, 'TST1')] = 50000
    data.loc[first_day_idx, (OPEN, 'TST2')] = tst2_open_d1
    data.loc[first_day_idx, (CLOSE, 'TST2')] = tst2_close_d1
    data.loc[first_day_idx, (HIGH, 'TST2')] = tst2_close_d1 + 1.0
    data.loc[first_day_idx, (LOW, 'TST2')] = tst2_open_d1 - 1.0
    data.loc[first_day_idx, (VOLUME, 'TST2')] = 75000

    # --- Fill remaining days with predictable data ---
    if num_dates > 1:
        for i in range(1, num_dates):
            prev_idx = dates[i-1]
            curr_idx = dates[i]
            for ticker in TEST_TICKERS:
                data.loc[curr_idx, (OPEN, ticker)] = data.loc[prev_idx, (CLOSE, ticker)] # Open at prev close
                data.loc[curr_idx, (CLOSE, ticker)] = data.loc[curr_idx, (OPEN, ticker)] + np.random.uniform(-1, 1) # Slight random walk close
                data.loc[curr_idx, (HIGH, ticker)] = max(data.loc[curr_idx, (OPEN, ticker)], data.loc[curr_idx, (CLOSE, ticker)]) + 0.5
                data.loc[curr_idx, (LOW, ticker)] = min(data.loc[curr_idx, (OPEN, ticker)], data.loc[curr_idx, (CLOSE, ticker)]) - 0.5
                data.loc[curr_idx, (VOLUME, ticker)] = int(np.random.randint(10000, 200000))

    data.loc[:, (LOW, slice(None))] = data.loc[:, (LOW, slice(None))].clip(lower=0.01)
    data.ffill(inplace=True)
    data.bfill(inplace=True)

    return data.round(2)


@pytest.fixture
def mock_fetch_success(sample_market_data: pd.DataFrame) -> MagicMock:
    return MagicMock(return_value=sample_market_data)

@pytest.fixture
def mock_fetch_empty() -> MagicMock:
    return MagicMock(return_value=pd.DataFrame())

@pytest.fixture
def mock_fetch_data_error() -> MagicMock:
    return MagicMock(side_effect=DH_DataFetchError("Simulated yfinance download failure"))

# --- Test Classes ---

class TestBacktesterInit:
    """Tests for Backtester initialization and configuration validation."""

    def test_valid_initialization(self, valid_backtester_params: Dict[str, Any], mock_fetch_success: MagicMock):
        params = valid_backtester_params.copy()
        params["data_fetch_func"] = mock_fetch_success
        params["strategy_params"] = {"orders_by_date": {}}
        try:
            bt = Backtester(**params)
            assert bt.initial_capital == INITIAL_CAPITAL
            assert bt.tickers == TEST_TICKERS
            assert isinstance(bt.strategy, MockStrategyStaticOrders)
        except Exception as e:
            pytest.fail(f"Valid initialization raised unexpected exception: {e}")

    @pytest.mark.parametrize("param_name, invalid_value, expected_error", [
        ("initial_capital", 0, ConfigError),
        ("initial_capital", -100, ConfigError),
        ("tickers", [], ConfigError),
        ("tickers", ["   "], ConfigError),
        ("tickers", ["TST1", None], ConfigError),
        ("tickers", [123], ConfigError),
        ("strategy_class", Strategy, ConfigError),
        ("strategy_class", dict, ConfigError),
        ("end_date", None, ConfigError),
        ("commission_model", object(), ConfigError),
        ("execution_price_col", "InvalidCol", ConfigError),
        ("valuation_price_col", "BadCol", ConfigError),
        ("data_fetch_func", 123, ConfigError),
    ])
    def test_invalid_params_raises_config_error(self, valid_backtester_params: Dict[str, Any], param_name: Literal['initial_capital'] | Literal['tickers'] | Literal['strategy_class'] | Literal['end_date'] | Literal['commission_model'] | Literal['execution_price_col'] | Literal['valuation_price_col'] | Literal['data_fetch_func'], invalid_value: Any, expected_error: Any):
        params = valid_backtester_params.copy()
        params[param_name] = invalid_value
        if param_name == "strategy_class" and not isinstance(invalid_value, type): pass
        elif param_name == "strategy_class" and invalid_value == Strategy: pass

        with pytest.raises(expected_error):
            Backtester(**params)

    def test_start_date_after_end_date_init(self, valid_backtester_params: Dict[str, Any], mock_fetch_success: MagicMock):
        params = valid_backtester_params.copy()
        params["start_date"] = "2023-01-10"
        params["end_date"] = "2023-01-01"
        params["data_fetch_func"] = mock_fetch_success
        bt = Backtester(**params)
        with pytest.raises(ConfigError, match="Start date .* cannot be after end date"):
             bt._prepare_run()

    def test_invalid_date_formats_values_init(self, valid_backtester_params: Dict[str, Any], mock_fetch_success: MagicMock):
         params_bad_format = valid_backtester_params.copy()
         params_bad_format["start_date"] = "2023/01/01"
         params_bad_format["data_fetch_func"] = mock_fetch_success
         bt_bad_format = Backtester(**params_bad_format)
         with pytest.raises(ConfigError, match="Invalid date format"):
              bt_bad_format._prepare_run()

         params_bad_value = valid_backtester_params.copy()
         params_bad_value["start_date"] = "2023-13-01"
         params_bad_value["data_fetch_func"] = mock_fetch_success
         bt_bad_value = Backtester(**params_bad_value)
         with pytest.raises(ConfigError, match="Invalid date value"):
              bt_bad_value._prepare_run()


    def test_strategy_init_error(self, valid_backtester_params: Dict[str, Any]):
        class StrategyRaisesInitError(Strategy):
            def __init__(self, parameters: Optional[Dict[str, Any]] = None):
                raise ValueError("Failed to init strategy")
            def generate_signals(self, *args) -> List[Order]: return []
        params = valid_backtester_params.copy()
        params["strategy_class"] = StrategyRaisesInitError
        with pytest.raises(ConfigError, match="Failed to initialize strategy"):
            Backtester(**params)

class TestBasicCommission:
    @pytest.mark.parametrize("fixed, percent, quantity, price, expected", [
        (0.0, 0.0, 100, 50.0, 0.0),(1.0, 0.0, 100, 50.0, 1.0),
        (0.0, 0.001, 100, 50.0, 5.0),(1.5, 0.001, 100, 50.0, 6.5),
        (1.0, 0.001, 10, 0.5, 1.005),(0.0, 0.0, 0, 100.0, 0.0),
        (0.0, 0.0, 100, 0.0, 0.0),])
    def test_commission_calculation(self, fixed: float, percent: float, quantity: int, price: float, expected: float):
        commission_model = BasicCommission(fixed_fee=fixed, percent_fee=percent)
        assert commission_model.calculate(quantity, price) == pytest.approx(expected)
    @pytest.mark.parametrize("fixed, percent", [(-1.0, 0.0), (0.0, -0.001)])
    def test_invalid_commission_params(self, fixed: float, percent: float):
        with pytest.raises(ConfigError): BasicCommission(fixed_fee=fixed, percent_fee=percent)

class TestOrderValidation:
    def test_valid_order(self):
        try:
            Order(ticker="AAPL", action=BUY_ACTION, quantity=10)
            Order(ticker="MSFT", action=SELL_ACTION, quantity=1)
        except Exception as e: pytest.fail(f"Valid Order raised exception: {e}")
    @pytest.mark.parametrize("ticker, action, quantity", [
        ("", BUY_ACTION, 10), ("AAPL", "HOLD", 10),("AAPL", BUY_ACTION, 0),
        ("AAPL", SELL_ACTION, -5),(None, BUY_ACTION, 10)])
    def test_invalid_order_raises_config_error(self, ticker: Optional[str], action: str, quantity: int):
        with pytest.raises(ConfigError): Order(ticker=ticker, action=action, quantity=quantity)

class TestBacktesterDataHandling:
    def test_fetch_success_and_validation(self, valid_backtester_params: Dict[str, Any], mock_fetch_success: MagicMock, sample_market_data: pd.DataFrame):
        params = valid_backtester_params.copy()
        params["data_fetch_func"] = mock_fetch_success
        bt = Backtester(**params)
        try:
            bt._fetch_data()
        except DataError as e: pytest.fail(f"_fetch_data failed on valid mock data: {e}")
        assert bt._market_data is not None
        pd.testing.assert_frame_equal(bt._market_data, sample_market_data)
        assert bt._actual_start_date == sample_market_data.index.min().date()
        assert bt._actual_end_date == sample_market_data.index.max().date()

    def test_fetch_returns_empty_data(self, valid_backtester_params: Dict[str, Any], mock_fetch_empty: MagicMock):
        params = valid_backtester_params.copy()
        params["data_fetch_func"] = mock_fetch_empty
        bt = Backtester(**params)
        with pytest.raises(DataError, match="No market data returned"): bt._fetch_data()

    def test_fetch_raises_data_fetch_error(self, valid_backtester_params: Dict[str, Any], mock_fetch_data_error: MagicMock):
        params = valid_backtester_params.copy()
        params["data_fetch_func"] = mock_fetch_data_error
        bt = Backtester(**params)
        with pytest.raises(DataError, match="Failed to retrieve or process market data"): bt._fetch_data()

    def test_fetch_invalid_index_type(self, valid_backtester_params: Dict[str, Any], sample_market_data: pd.DataFrame):
        invalid_data = sample_market_data.copy()
        invalid_data.index = range(len(invalid_data))
        mock_fetch_invalid = MagicMock(return_value=invalid_data)
        params = valid_backtester_params.copy()
        params["data_fetch_func"] = mock_fetch_invalid
        bt = Backtester(**params)
        with pytest.raises(DataError, match="Market data index is not a DatetimeIndex"): bt._fetch_data()

    def test_fetch_invalid_columns(self, valid_backtester_params: Dict[str, Any], sample_market_data: pd.DataFrame):
        invalid_data = sample_market_data.copy()
        invalid_data.columns = [f"Col_{i}" for i in range(len(invalid_data.columns))]
        mock_fetch_invalid = MagicMock(return_value=invalid_data)
        params = valid_backtester_params.copy()
        params["data_fetch_func"] = mock_fetch_invalid
        bt = Backtester(**params)
        with pytest.raises(DataError, match="Market data columns have unexpected format"): bt._fetch_data()

    def test_fetch_missing_required_column_type(self, valid_backtester_params: Dict[str, Any], sample_market_data: pd.DataFrame):
        invalid_data = sample_market_data.copy()
        invalid_data = invalid_data.drop(columns=OPEN, level=0)
        mock_fetch_invalid = MagicMock(return_value=invalid_data)
        params = valid_backtester_params.copy()
        params["data_fetch_func"] = mock_fetch_invalid
        bt = Backtester(**params)
        with pytest.raises(DataError, match=r"Market data is missing essential column types:.*\['Open'\]"): bt._fetch_data()

    def test_fetch_missing_price_column_for_ticker(self, valid_backtester_params: Dict[str, Any], sample_market_data: pd.DataFrame):
        invalid_data = sample_market_data.copy()
        if (OPEN, 'TST1') in invalid_data.columns: invalid_data = invalid_data.drop(columns=[(OPEN, 'TST1')])
        else: pytest.fail(f"Column ('{OPEN}', 'TST1') not found in sample data for removal.")
        mock_fetch_invalid = MagicMock(return_value=invalid_data)
        params = valid_backtester_params.copy()
        params["data_fetch_func"] = mock_fetch_invalid
        bt = Backtester(**params)
        with pytest.raises(DataError, match=r"Market data incomplete\. Missing required price columns"): bt._fetch_data()


TST1_BUY_QTY = 100
TST1_BUY_COMMISSION_RATE = 0.001
TST1_BUY_COMMISSION_FIXED = 1.0

class TestBacktesterRun:
    """Tests the main simulation loop (run method) and its interactions."""

    @pytest.fixture
    def backtester_ready_to_run(self, valid_backtester_params: Dict[str, Any], mock_fetch_success: MagicMock, sample_market_data: pd.DataFrame):
        params = valid_backtester_params.copy()
        params["data_fetch_func"] = mock_fetch_success
        if params["strategy_class"] == MockStrategyStaticOrders: params["strategy_params"] = {"orders_by_date": {}}
        bt = Backtester(**params)
        return bt, sample_market_data

    def _calculate_buy_cost(self, data: pd.DataFrame, ticker: str, quantity: int, commission_model: CommissionModel, exec_col: str=OPEN) -> Tuple[float, float, float, Timestamp]:
        first_day = data.index[0]
        price = data.loc[first_day, (exec_col, ticker)]
        commission = commission_model.calculate(quantity, price)
        cost = quantity * price + commission
        return cost, price, commission, first_day

    def test_run_simple_buy_order(self, backtester_ready_to_run: Tuple[Backtester, pd.DataFrame]):
        bt, data = backtester_ready_to_run
        cost, price, commission, first_day = self._calculate_buy_cost(data, 'TST1', TST1_BUY_QTY, bt.commission_model)
        buy_order = Order(ticker='TST1', action=BUY_ACTION, quantity=TST1_BUY_QTY)
        bt.strategy = MockStrategyStaticOrders(orders_by_date={first_day: [buy_order]})
        bt.run()
        # Assertions
        assert len(bt.trades) == 1; trade = bt.trades[0]
        assert trade.timestamp == first_day; assert trade.ticker == 'TST1'; assert trade.action == BUY_ACTION
        assert trade.quantity == TST1_BUY_QTY; assert trade.price == pytest.approx(price)
        assert trade.commission == pytest.approx(commission); assert trade.cost_proceeds == pytest.approx(-cost)
        assert trade.cash_after == pytest.approx(INITIAL_CAPITAL - cost)
        assert trade.holdings_after == {'TST1': TST1_BUY_QTY}
        # Final State
        assert bt._cash == pytest.approx(INITIAL_CAPITAL - cost); assert bt._holdings == {'TST1': TST1_BUY_QTY}
        # Equity Curve
        equity_curve = bt.equity_curve; assert equity_curve is not None; assert not equity_curve.isnull().values.any()
        first_day_close = data.loc[first_day, (CLOSE, 'TST1')]
        assert equity_curve.loc[first_day, CASH_COL] == pytest.approx(INITIAL_CAPITAL - cost)
        assert equity_curve.loc[first_day, HOLDINGS_VALUE_COL] == pytest.approx(TST1_BUY_QTY * first_day_close)
        assert equity_curve.loc[first_day, TOTAL_EQUITY_COL] == pytest.approx((INITIAL_CAPITAL - cost) + (TST1_BUY_QTY * first_day_close) )
        last_day = data.index[-1]; last_day_close = data.loc[last_day, (CLOSE, 'TST1')]
        assert equity_curve.loc[last_day, TOTAL_EQUITY_COL] == pytest.approx((INITIAL_CAPITAL - cost) + (TST1_BUY_QTY * last_day_close) )

    def test_run_buy_and_sell_order(self, backtester_ready_to_run: Tuple[Backtester, pd.DataFrame]):
        bt, data = backtester_ready_to_run
        cost_buy, _, _, day1 = self._calculate_buy_cost(data, 'TST1', TST1_BUY_QTY, bt.commission_model)
        cash_after_buy = INITIAL_CAPITAL - cost_buy; day3 = data.index[2]
        sell_qty = 50; buy_order = Order(ticker='TST1', action=BUY_ACTION, quantity=TST1_BUY_QTY)
        sell_order = Order(ticker='TST1', action=SELL_ACTION, quantity=sell_qty)
        bt.strategy = MockStrategyStaticOrders(orders_by_date={ day1: [buy_order], day3: [sell_order] })
        bt.run()
        # Assertions
        assert len(bt.trades) == 2; sell_trade = bt.trades[1]
        sell_price = data.loc[day3, (OPEN, 'TST1')]
        commission_sell = bt.commission_model.calculate(sell_qty, sell_price)
        proceeds_sell = sell_qty * sell_price - commission_sell
        cash_after_sell = cash_after_buy + proceeds_sell
        holdings_after_sell = TST1_BUY_QTY - sell_qty
        assert sell_trade.timestamp == day3; assert sell_trade.action == SELL_ACTION
        assert sell_trade.quantity == sell_qty; assert sell_trade.price == pytest.approx(sell_price)
        assert sell_trade.commission == pytest.approx(commission_sell)
        assert sell_trade.cost_proceeds == pytest.approx(proceeds_sell)
        assert sell_trade.cash_after == pytest.approx(cash_after_sell)
        assert sell_trade.holdings_after == {'TST1': holdings_after_sell}
        # Final State
        assert bt._cash == pytest.approx(cash_after_sell)
        assert bt._holdings == {'TST1': holdings_after_sell}

    def test_run_insufficient_funds(self, backtester_ready_to_run: Tuple[Backtester, pd.DataFrame], caplog: pytest.LogCaptureFixture):
        bt, data = backtester_ready_to_run; first_trading_day = data.index[0]
        buy_order = Order(ticker='TST2', action=BUY_ACTION, quantity=10000) # Too much
        bt.strategy = MockStrategyStaticOrders(orders_by_date={first_trading_day: [buy_order]})
        caplog.set_level(logging.WARNING); bt.run()
        # Assertions
        assert len(bt.trades) == 0; assert bt._cash == pytest.approx(INITIAL_CAPITAL); assert bt._holdings == {}
        assert "REJECT BUY" in caplog.text; assert "Insufficient funds" in caplog.text
        equity_curve = bt.equity_curve; assert equity_curve is not None; assert not equity_curve.isnull().values.any()
        assert np.allclose(equity_curve[TOTAL_EQUITY_COL].values, INITIAL_CAPITAL), \
            f"Equity curve total equity deviation detected. Expected {INITIAL_CAPITAL} everywhere."

    def test_run_insufficient_holdings(self, backtester_ready_to_run: Tuple[Backtester, pd.DataFrame], caplog: pytest.LogCaptureFixture):
        bt, data = backtester_ready_to_run
        cost_buy, _, _, day1 = self._calculate_buy_cost(data, 'TST1', 10, bt.commission_model) # Buy 10
        day3 = data.index[2]; buy_order = Order(ticker='TST1', action=BUY_ACTION, quantity=10)
        sell_order = Order(ticker='TST1', action=SELL_ACTION, quantity=50) # Sell 50
        bt.strategy = MockStrategyStaticOrders(orders_by_date={ day1: [buy_order], day3: [sell_order] })
        caplog.set_level(logging.WARNING); bt.run()
        # Assertions
        assert len(bt.trades) == 1; assert bt.trades[0].action == BUY_ACTION
        assert "REJECT SELL" in caplog.text; assert "Insufficient holdings. Have 10, Need 50" in caplog.text
        assert bt._cash == pytest.approx(INITIAL_CAPITAL - cost_buy); assert bt._holdings == {'TST1': 10}

    def test_run_missing_execution_price(self, backtester_ready_to_run: Tuple[Backtester, pd.DataFrame], sample_market_data: pd.DataFrame, caplog: pytest.LogCaptureFixture):
        bt, _ = backtester_ready_to_run; data = sample_market_data.copy()
        day_to_modify = data.index[1]; data.loc[day_to_modify, (OPEN, 'TST1')] = np.nan
        mock_fetch_modified = MagicMock(return_value=data); bt.data_fetch_func = mock_fetch_modified
        buy_order = Order(ticker='TST1', action=BUY_ACTION, quantity=10)
        bt.strategy = MockStrategyStaticOrders(orders_by_date={day_to_modify: [buy_order]})
        caplog.set_level(logging.WARNING); bt.run()
        # Assertions
        assert len(bt.trades) == 0; assert bt._cash == pytest.approx(INITIAL_CAPITAL); assert bt._holdings == {}
        assert "Order execution skipped" in caplog.text; assert f"Invalid or missing '{OPEN}' price (nan) for TST1" in caplog.text

    def test_run_missing_valuation_price_fatal(self, backtester_ready_to_run: Tuple[Backtester, pd.DataFrame], sample_market_data: pd.DataFrame):
        bt, _ = backtester_ready_to_run; data = sample_market_data.copy()
        day1 = data.index[0]; day_to_modify = data.index[2]
        if len(data) <= 2: pytest.skip("Sample data too short for this test scenario.")
        buy_order = Order(ticker='TST1', action=BUY_ACTION, quantity=10)
        bt.strategy = MockStrategyStaticOrders(orders_by_date={day1: [buy_order]})
        data.loc[day_to_modify, (CLOSE, 'TST1')] = np.nan # Set close to NaN after buy
        mock_fetch_modified = MagicMock(return_value=data); bt.data_fetch_func = mock_fetch_modified
        expected_msg = f"Missing valuation price for held asset TST1 on {day_to_modify.date()}"
        with pytest.raises(DataError, match=re.escape(expected_msg)): bt.run()
        assert bt.equity_curve is None; assert bt.trades == []

    def test_run_strategy_error_halts(self, backtester_ready_to_run: Tuple[Backtester, pd.DataFrame]):
        bt, data = backtester_ready_to_run; bt.strategy = MockStrategyError()
        fail_date_str = '2023-01-05'
        with pytest.raises(StrategyError, match=f"Deliberate strategy failure on or after {fail_date_str}"): bt.run()
        assert bt.equity_curve is None; assert bt.trades == []

    def test_run_order_for_untracked_ticker(self, backtester_ready_to_run: Tuple[Backtester, pd.DataFrame], caplog: pytest.LogCaptureFixture):
         bt, data = backtester_ready_to_run; day1 = data.index[0]
         invalid_order = Order(ticker='UNKNOWN', action=BUY_ACTION, quantity=10)
         bt.strategy = MockStrategyStaticOrders(orders_by_date={day1: [invalid_order]})
         caplog.set_level(logging.WARNING); bt.run()
         assert len(bt.trades) == 0
         assert "Strategy generated order for untracked ticker 'UNKNOWN'" in caplog.text

class TestBacktesterResults:
     """Tests the get_results method."""

     @pytest.fixture
     def backtester_with_trades(self, valid_backtester_params: Dict[str, Any], sample_market_data: pd.DataFrame):
         """Fixture that sets up and runs a backtester GUARANTEED to have trades."""
         the_data = sample_market_data
         first_day_ts: pd.Timestamp = the_data.index[0] # Get the Timestamp object

         tst1_open_price = the_data.loc[first_day_ts, (OPEN, 'TST1')]
         assert not pd.isna(tst1_open_price) and tst1_open_price > 0, "Fixture data setup error: Invalid TST1 Open price on first day."

         params = valid_backtester_params.copy()

         buy_qty = 10
         commission_model = params["commission_model"]
         