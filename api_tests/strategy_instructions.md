## Methodology for Writing Custom Trading Strategies for the Backtesting Engine

This document outlines how to design and implement custom trading strategies compatible with the provided Python backtesting engine (`backtester.py`). It details the required structure, available data, supported features, limitations, and best practices.

### 1. Core Concept: The `Strategy` Class

The foundation of any custom strategy is a Python class that inherits from the abstract base class `backtester.Strategy`. Your custom class **must** implement the `generate_signals` method. This method contains the core logic where your strategy analyzes market data and portfolio state to decide which trades (if any) to propose.

```python
# --- Necessary Imports ---
import pandas as pd
from typing import List, Dict, Any, Optional

# Import core components from the backtester module
from backtester import Strategy, Order, Timestamp, HoldingsDict, TickerSymbol
from backtester import BUY_ACTION, SELL_ACTION # Use constants for actions

# --- Your Strategy Class Definition ---
class MyCustomStrategy(Strategy):
    """
    A descriptive docstring explaining what your strategy does.
    e.g., A simple moving average crossover strategy.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize your strategy.

        - Call the parent constructor: super().__init__(parameters)
        - Process and store any strategy-specific parameters (e.g., moving average windows).
        - Initialize any internal state needed (e.g., dictionaries to track calculations per ticker).
        - Perform parameter validation using _validate_parameters() if needed.
        """
        super().__init__(parameters) # Essential: Initializes self.parameters

        # Example: Extract parameters with defaults
        self.short_window = self.parameters.get('short_window', 20)
        self.long_window = self.parameters.get('long_window', 50)

        # Example: Initialize internal state (e.g., store calculated indicators)
        self.indicators = {} # Dictionary to hold DataFrames or Series per ticker

        # Optional: Validate parameters specific to this strategy
        self._validate_parameters()

        print(f"MyCustomStrategy initialized with short={self.short_window}, long={self.long_window}")


    def _validate_parameters(self) -> None:
        """
        Optional: Implement custom validation for your strategy's parameters.
        Raise ConfigError (imported from backtester) if validation fails.
        """
        from backtester import ConfigError # Import locally if needed or globally

        if not isinstance(self.short_window, int) or self.short_window <= 0:
            raise ConfigError("'short_window' parameter must be a positive integer.")
        if not isinstance(self.long_window, int) or self.long_window <= 0:
            raise ConfigError("'long_window' parameter must be a positive integer.")
        if self.short_window >= self.long_window:
            raise ConfigError("'short_window' must be less than 'long_window'.")


    def generate_signals(
        self,
        current_dt: Timestamp,
        data_slice: pd.DataFrame,
        current_holdings: HoldingsDict,
        current_cash: float
    ) -> List[Order]:
        """
        The core logic of your strategy. Called by the backtester for each time step.

        Args:
            current_dt (Timestamp): The current date/time of the simulation step.
                                    Your logic should act based on data *up to* this point.
            data_slice (pd.DataFrame): A DataFrame containing historical market data
                                       (MultiIndex columns: ['Open', 'High', ...], ['TICKER1', 'TICKER2', ...])
                                       for all tickers, indexed by Timestamp, up to and
                                       *including* `current_dt`. **Crucially, do not use
                                       data from the future within this slice.**
            current_holdings (HoldingsDict): A dictionary showing the quantity of each
                                             ticker currently held *before* executing any
                                             orders generated in this step. Format: {'TICKER': quantity}.
            current_cash (float): The cash balance available *before* executing any
                                  orders generated in this step.

        Returns:
            List[Order]: A list of `Order` objects (or an empty list) representing the
                         trades your strategy wants to make based on the current state.
                         These orders will be simulated by the backtester using the
                         price at `current_dt` (specifically, the `execution_price_col`, default 'Open').
        """
        orders = [] # Initialize list to hold generated orders for this step

        # --- Strategy Logic Example ---
        # Iterate through each ticker the backtester is tracking
        # Access available tickers via data_slice.columns.get_level_values(1).unique()
        # Or use the backtester's configured tickers if needed (less direct)

        available_tickers = data_slice.columns.get_level_values(1).unique()

        for ticker in available_tickers:
            # 1. Access Price Data for the ticker (handle potential missing columns/data)
            try:
                # Example: Get the 'Close' price series for indicator calculation
                # Adjust column name ('Close') if needed
                price_series = data_slice[('Close', ticker)]

                # Check if enough data exists for calculation (e.g., for MA)
                if len(price_series.dropna()) < self.long_window:
                    # Not enough data yet for this ticker, skip
                    continue

                # 2. Perform Calculations (e.g., Moving Averages)
                # Ensure calculations only use data up to current_dt (data_slice handles this)
                short_mavg = price_series.rolling(window=self.short_window).mean()
                long_mavg = price_series.rolling(window=self.long_window).mean()

                # Get the most recent values (as of current_dt)
                latest_short_mavg = short_mavg.iloc[-1]
                latest_long_mavg = long_mavg.iloc[-1]
                # Optional: Get previous values for crossover detection
                prev_short_mavg = short_mavg.iloc[-2] if len(short_mavg) > 1 else None
                prev_long_mavg = long_mavg.iloc[-2] if len(long_mavg) > 1 else None

            except KeyError:
                 # Handle cases where a column might be missing for a ticker in the slice
                 print(f"Warning: Price data ('Close', {ticker}) not found at {current_dt}. Skipping ticker.")
                 continue
            except Exception as e:
                 # Handle potential calculation errors (e.g., from rolling means)
                 print(f"Error calculating indicators for {ticker} at {current_dt}: {e}")
                 continue # Skip this ticker on error

            # Check for NaN values in indicators (can happen at the start)
            if pd.isna(latest_short_mavg) or pd.isna(latest_long_mavg) or \
               (prev_short_mavg is not None and pd.isna(prev_short_mavg)) or \
               (prev_long_mavg is not None and pd.isna(prev_long_mavg)):
                continue # Skip if indicators aren't ready

            # --- Decision Logic ---
            ticker_holding = current_holdings.get(ticker, 0)

            # Example: Buy Signal (Golden Cross)
            if latest_short_mavg > latest_long_mavg and \
               (prev_short_mavg is not None and prev_long_mavg is not None and prev_short_mavg <= prev_long_mavg): # Check for crossover
                if ticker_holding == 0: # Only buy if not already holding
                    # --- Create a Buy Order ---
                    # Determine quantity (e.g., fixed size, based on cash, risk)
                    quantity_to_buy = 10 # Simple fixed quantity

                    # Optional: Check if affordable (Backtester also checks, but good practice)
                    # Estimate cost using a recent price (e.g., last close) - BE CAREFUL not to look ahead!
                    # Execution price ('Open' by default) at current_dt will be used by backtester.
                    last_close_price = price_series.iloc[-1]
                    estimated_cost = quantity_to_buy * last_close_price # Approximation
                    if current_cash > estimated_cost: # Basic check
                        print(f"{current_dt.date()}: BUY signal for {ticker}")
                        orders.append(Order(ticker=ticker, action=BUY_ACTION, quantity=quantity_to_buy))
                    else:
                        print(f"{current_dt.date()}: BUY signal for {ticker}, but insufficient cash (estimated).")


            # Example: Sell Signal (Death Cross)
            elif latest_short_mavg < latest_long_mavg and \
                 (prev_short_mavg is not None and prev_long_mavg is not None and prev_short_mavg >= prev_long_mavg): # Check for crossover
                if ticker_holding > 0: # Only sell if currently holding
                     # --- Create a Sell Order ---
                     # Sell the entire position
                     quantity_to_sell = ticker_holding
                     print(f"{current_dt.date()}: SELL signal for {ticker}")
                     orders.append(Order(ticker=ticker, action=SELL_ACTION, quantity=quantity_to_sell))

        # --- Return the list of orders for this time step ---
        return orders

```

### 2. Step-by-Step Guide to Writing Your Strategy Code

1.  **Create a Python File:** Start a new `.py` file (e.g., `my_strategies.py`).
2.  **Import Necessary Modules:** Import `Strategy`, `Order`, `Timestamp`, `HoldingsDict`, `TickerSymbol`, `BUY_ACTION`, `SELL_ACTION` from `backtester`. Import `pandas` and any other libraries you need (like `numpy` or technical analysis libraries like `TA-Lib` or `pandas_ta`).
3.  **Define Your Class:** Create a class that inherits from `backtester.Strategy`. Give it a descriptive name (e.g., `SMACrossoverStrategy`, `VolatilityBreakout`).
4.  **Implement `__init__`:**
    *   Always call `super().__init__(parameters)` first.
    *   Define default values for your strategy's parameters (e.g., indicator periods, thresholds).
    *   Use `self.parameters.get('param_name', default_value)` to allow users to override defaults when initializing the `Backtester`.
    *   Initialize any state variables your strategy needs to maintain across time steps (e.g., dictionaries to store calculated values, flags).
    *   Optionally, call `self._validate_parameters()` if you implement custom validation.
5.  **Implement `generate_signals` (The Core):**
    *   **Understand the Inputs:**
        *   `current_dt`: The timestamp you are currently evaluating. Your decisions must be based *only* on information available up to this point.
        *   `data_slice`: Your window into the past. Contains OHLCV data for *all* tracked tickers up to `current_dt`. Use this for calculations. Access specific data using MultiIndex slicing: `data_slice[(COLUMN_NAME, TICKER_SYMBOL)]` (e.g., `data_slice[('Close', 'AAPL')]`).
        *   `current_holdings`: A snapshot of your portfolio *before* any trades generated at `current_dt` are executed. Use this to know what you own.
        *   `current_cash`: Your available cash *before* trades at `current_dt`. Use this for position sizing or affordability checks.
    *   **Perform Calculations:** Calculate indicators, signals, or any metrics needed using the `data_slice`. Be mindful of required data lengths (e.g., a 50-day MA needs at least 50 data points). Handle `NaN` values appropriately (often occur at the start of calculations).
    *   **Implement Decision Logic:** Based on your calculations, current holdings, and cash, decide if you want to buy or sell any assets.
    *   **Create `Order` Objects:** If your logic decides on a trade, create an instance of `backtester.Order`.
        *   `ticker`: The ticker symbol (must be one the backtester is tracking).
        *   `action`: Use `BUY_ACTION` or `SELL_ACTION` constants.
        *   `quantity`: A **positive integer** representing the number of shares.
    *   **Return Orders:** Collect all generated `Order` objects for the current `current_dt` into a list and return it. Return an empty list (`[]`) if no trades are desired for this step.
    *   **Error Handling:** Wrap potentially problematic code (e.g., complex calculations, accessing potentially missing data) in `try...except` blocks. If a non-recoverable strategy error occurs, you can raise `StrategyError` (imported from `backtester`) to halt the backtest clearly indicating the source.
6.  **Implement `_validate_parameters` (Optional):** If your strategy takes parameters, add checks in this method to ensure they are valid (correct type, range, etc.). Raise `ConfigError` if validation fails. This helps catch configuration issues early.

### 3. What the Backtester Provides (Within `generate_signals`)

*   **Time Context:** `current_dt` (The current simulation timestamp).
*   **Historical Market Data:** `data_slice` (OHLCV DataFrame up to `current_dt` for all configured tickers). **Crucially, this prevents lookahead bias.**
*   **Current Portfolio State:** `current_holdings` (Dictionary `{ticker: quantity}`) and `current_cash` (float). This reflects the state *before* orders for `current_dt` are processed.

### 4. What the Backtester Handles (What Your Strategy *Doesn't* Need To Do)

*   **Data Fetching:** The `Backtester` fetches the required historical data using the `data_handler`.
*   **Time Iteration:** The engine loops through the dates/timestamps in the data.
*   **Order Execution Simulation:** The engine takes the `Order` objects you return and simulates their execution based on:
    *   Execution Price: Looks up the price in the `execution_price_col` (default: 'Open') for `current_dt`.
    *   Cash Check: Verifies sufficient funds for buys (using calculated cost + commission).
    *   Holdings Check: Verifies sufficient shares for sells.
    *   **Note:** Insufficient funds/holdings result in rejected orders (logged) but typically don't stop the backtest. Execution failure due to missing price data *can* stop the backtest (`DataError`).
*   **Commission Calculation:** Applies the configured `CommissionModel` to executed trades.
*   **Portfolio State Updates:** Updates internal cash and holdings based on executed trades.
*   **Portfolio Valuation:** Calculates the value of holdings daily (using `valuation_price_col`, default: 'Close') and tracks the equity curve (`Cash`, `Holdings_Value`, `Total_Equity`).
*   **Trade Logging:** Records details of successfully executed trades.

### 5. Supported Strategy Features

Based on the engine's design, you can implement strategies with the following characteristics:

*   **Technical Indicator Based:** Calculate any indicator using the provided `data_slice` (e.g., MAs, RSI, MACD, Bollinger Bands).
*   **Multi-Asset:** The strategy logic can handle and generate orders for any/all of the tickers configured in the `Backtester`.
*   **Portfolio-Aware Logic:** Use `current_holdings` and `current_cash` for:
    *   Position sizing (e.g., allocate a percentage of capital, fixed dollar amount).
    *   Risk management (e.g., avoid adding to excessively large positions).
    *   Deciding whether to enter or exit positions.
*   **Event-Driven:** Reacts to each new time step's data.
*   **Signal Generation:** Generates discrete BUY/SELL orders.
*   **Daily (or Data Frequency) Execution:** Assumes decisions and executions happen once per timestamp available in the data (typically daily).
*   **Parameterization:** Strategies can be configured using the `parameters` dictionary passed during `Backtester` initialization.
*   **Long-Only and Basic Long/Short (Selling owned assets):** Can generate BUY orders and SELL orders *for assets currently held*.

### 6. Limitations / Unsupported Features (and Why)

The current backtesting engine has limitations. The following are **NOT** directly supported:

*   **True Short Selling (Selling Borrowed Shares):** The `SELL_ACTION` requires you to already hold the shares (`InsufficientHoldingsError` otherwise). *Why:* The engine doesn't model borrowing mechanics or margin accounts.
*   **Intraday Trading Strategies:** The engine iterates based on the frequency of the input data (typically daily from `yfinance`). Simulating minute-by-minute or tick-level trading requires different data granularity and a much faster event loop. *Why:* Data fetching and the main loop are designed for lower frequencies (like daily); execution is modeled at a single point per bar (e.g., 'Open').
*   **Complex Order Types (Limit, Stop-Loss, Trailing Stops):** The engine only simulates market-like orders executed at the `execution_price_col` price for `current_dt`. *Why:* The `Order` object is simple; the execution logic (`_execute_order`) doesn't check for price conditions during the bar.
*   **Futures, Options, Forex, Cryptocurrencies:** The data handler uses `yfinance`, primarily for stocks/ETFs. The backtester's state management (simple quantity holdings) and OHLCV focus are equity-centric. *Why:* Requires different data sources, pricing models, contract specifications, and state tracking (e.g., margin, expirations).
*   **Dynamic Universe Selection (Adding/Removing Tickers Mid-Backtest):** The set of tickers is fixed when the `Backtester` is initialized. *Why:* Data is fetched only once at the start for the configured tickers.
*   **Direct Use of Fundamental Data within `generate_signals`:** The `data_slice` only contains OHLCV. While you *could* technically make external calls (e.g., fetch fundamentals) inside `generate_signals`, it's highly discouraged. *Why:* It would be very slow, might hit API limits, and introduces significant risk of lookahead bias if not handled extremely carefully (fetching data "as of" `current_dt` is complex).
*   **News/Sentiment Data Integration:** Similar to fundamental data. *Why:* Not part of the standard data flow provided to the strategy.
*   **Advanced Execution Models (Partial Fills, Slippage):** Execution is assumed to be complete at the target price if sufficient funds/holdings exist. *Why:* `_execute_order` is simplified; doesn't model order book depth, volume impact, or random slippage factors.
*   **Live Trading Execution:** This is purely a backtesting simulation environment. *Why:* No integration with broker APIs for real-time data or order placement.

### 7. Best Practices and Tips

*   **Prevent Lookahead Bias:** This is critical. **NEVER** use information in `data_slice` that comes from a time *after* `current_dt` in your decision logic for `current_dt`. The provided `data_slice` structure helps, but be careful with index manipulation or calculations that might inadvertently peek forward.
*   **Handle `NaN` Data:** Especially at the beginning of the backtest or when calculating indicators with lookback periods, expect `NaN` values. Check for them (`pd.isna()`) before using calculated values in your logic.
*   **Use Parameters:** Define strategy parameters (like window lengths, thresholds) in `__init__` and access them via `self.parameters`. This makes your strategy reusable and configurable.
*   **Manage State Carefully:** If your strategy needs to remember things between time steps (e.g., previous signal state, trailing stop levels), store them as instance attributes (e.g., in `self.indicators` dictionary example).
*   **Clear Logging/Printing:** Use `print` or `logging` within your strategy (especially during development) to understand its decisions at each step.
*   **Focus:** Keep the `generate_signals` method focused on calculation and decision logic. Avoid complex I/O or external API calls here.
*   **Error Handling:** Use `try...except` for robustness, and consider raising `StrategyError` for critical strategy failures.
*   **Code Clarity:** Write clean, well-commented code.

### 8. Integrating Your Strategy

Once your `MyCustomStrategy` class is defined (e.g., in `my_strategies.py`):

1.  **Import it** into the script where you are running the `Backtester`.
    ```python
    from my_strategies import MyCustomStrategy
    from backtester import Backtester, BasicCommission
    # ... other necessary imports
    ```
2.  **Pass the class name** (not an instance) to the `Backtester` constructor:
    ```python
    # Configure parameters for your strategy
    strategy_specific_params = {
        'short_window': 15,
        'long_window': 60
    }

    bt = Backtester(
        initial_capital=100000,
        tickers=['AAPL', 'MSFT', 'GOOG'],
        strategy_class=MyCustomStrategy, # Pass the class itself
        start_date='2020-01-01',
        end_date='2023-12-31',
        strategy_params=strategy_specific_params, # Pass parameters here
        commission_model=BasicCommission(fixed_fee=0.5, percent_fee=0.0005)
        # ... other backtester params if needed
    )

    # Run the backtest
    bt.run()

    # Get results
    equity_curve, trades = bt.get_results()
    ```