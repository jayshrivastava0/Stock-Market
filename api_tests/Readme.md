### 1. `data_handler.py`

* **Core Purpose:** This module's main job is to reliably fetch historical stock market data. It acts as a dedicated interface to get Open, High, Low, Close, and Volume (OHLCV), plus Adjusted Close prices, primarily using the `yfinance` library.
* **Key Functionality:**
    * `Workspace_stock_data`: This is the central function. You give it a list of stock tickers (like `['AAPL', 'GOOG']`), a start date, and an end date, and it retrieves the historical data for all those tickers within that timeframe. It can fetch data for multiple stocks in a single efficient call.
    * **Input Handling:** It cleans up the ticker list you provide (makes them uppercase, removes duplicates, checks for empty strings) and validates the date formats to ensure they are usable.
    * **Data Retrieval:** It uses `yfinance.download()` to get the data. It cleverly adjusts the end date when calling `yfinance` to ensure the data *includes* the requested end date.
    * **Data Cleaning:** After fetching, it makes sure the data has the expected structure (using a Pandas MultiIndex for columns when multiple tickers are involved), checks if data was returned for all requested tickers (and warns if some are missing), ensures the index is a proper DatetimeIndex, and removes timezone information to keep things consistent. It also filters the data strictly to the requested start and end dates.
    * **Error Handling:** It defines specific error types (`InvalidInputError`, `DataFetchError`, `DataProcessingError`) to clearly signal what went wrong if there's an issue with your input, the data fetching process (network errors, ticker not found), or the internal processing of the retrieved data.

### 2. `portfolio_manager.py`

* **Core Purpose:** This module focuses on tracking and analyzing the value and composition of a stock portfolio over a specific period. It starts with an initial set of holdings and lets you record changes (buys/sells) over time.
* **Key Functionality:**
    * **Initialization (`__init__`):** You start by giving it your initial portfolio (a dictionary like `{'AAPL': 10, 'MSFT': 5}`) and a time period string (e.g., '1y', '6M', 'from 2022', 'this year'). It cleans the tickers (uppercase) and stores this initial state.
    * **Recording Changes (`add_change`):** You can tell the manager about transactions using this method. You provide the date of the change, the ticker, and how the quantity changed. This change can be an *absolute* new quantity (e.g., `15` means you now hold 15 shares) or a *relative* change (e.g., `'+5'` to add 5 shares, `'-2'` to remove 2 shares). Adding changes clears any previously calculated results, forcing recalculation.
    * **Date Range Parsing (`_determine_date_range`, `_parse_...`):** It has internal logic to understand the flexible time period strings you provide (like '1y', '6M', 'from 2023-01', 'this year') and determines the exact start and end dates needed for analysis.
    * **Data Fetching (`_fetch_all_data`):** When needed, it uses the `data_handler.fetch_stock_data` function (from the previous module) to get the historical prices for *all* tickers involved in the portfolio (both initial holdings and those added via changes) for the calculated date range.
    * **Portfolio Calculation (`build_portfolio`):** This is the core calculation engine. It takes the initial holdings, applies all the recorded changes chronologically, and combines this daily quantity information with the historical price data (fetched via `_fetch_all_data`). It produces a detailed Pandas DataFrame indexed by date, showing:
        * The quantity held for each stock (`<TICKER>_qnty`) on each day.
        * The market value of each holding (`<TICKER>_value`) on each day.
        * The total value of the entire portfolio (`Total_Value`) on each day.
        This result is cached so it doesn't have to be recalculated unless changes are added.
    * **Accessing Results (`get_portfolio_data`, `get_composition`):** You use `get_portfolio_data` to retrieve the final DataFrame with daily values. `get_composition` allows you to see the percentage breakdown of your portfolio's value across different stocks on a specific date.
    * **Error Handling:** It relies on `data_handler`'s exceptions for fetching issues and defines its own (`PortfolioError`, `InvalidInputError`, `DataFetchError`, `PortfolioBuildError`) for problems related to input, fetching, or the portfolio construction process.

### 3. `backtester.py`

* **Core Purpose:** This is the engine for simulating how a trading strategy would have performed using historical market data. It aims for realistic simulation by managing cash, holdings, handling orders, and calculating commissions.
* **Key Functionality:**
    * **Strategy Interface (`Strategy` class):** It defines a blueprint (an Abstract Base Class) for how trading strategies should be written. Any strategy you create needs to inherit from this and implement the `generate_signals` method. This method receives the current market data (up to the simulation time), the current portfolio holdings, and available cash, and must decide which trades (if any) to make, returning them as a list of `Order` objects. This design enforces that strategies can't "look ahead" at future data.
    * **Order & Trade Representation (`Order`, `Trade` classes):** Defines simple structures to represent a requested trade (`Order`: ticker, action, quantity) and a successfully executed trade (`Trade`: includes timestamp, execution price, commission, etc.).
    * **Commission Handling (`CommissionModel`, `BasicCommission`):** Provides a way to model transaction costs. `BasicCommission` allows for simple fixed and/or percentage fees per trade. You can create more complex models if needed.
    * **Backtester Engine (`Backtester` class):**
        * **Initialization:** You set up the backtester with the starting cash, the list of tickers to consider, your custom strategy *class* (not an instance), the start/end dates for the simulation, any parameters your strategy needs, and the commission model. It validates these inputs.
        * **Data Fetching:** It uses a data fetching function (which defaults to `data_handler.fetch_stock_data`) to get the necessary historical OHLCV data for the specified tickers and date range. It validates the fetched data.
        * **Simulation Loop (`run` method):** This is the heart of the backtester. It iterates day by day through the historical data. On each day:
            1.  It calls your strategy's `generate_signals` method, passing the data available *up to that day*, along with the current cash and holdings.
            2.  It takes the `Order` objects returned by your strategy.
            3.  It simulates executing these orders (`_execute_order`) using a specified price from the *next* day's data (e.g., the 'Open' price, configurable via `execution_price_col`).
            4.  Execution involves checking if there's enough cash to buy or enough shares to sell (`InsufficientFundsError`, `InsufficientHoldingsError`) and calculating the commission using the commission model.
            5.  If an order executes successfully, it updates the internal cash balance and holdings dictionary (`_cash`, `_holdings`) and records the details in the `_trades` list.
        * **Performance Tracking (`_update_equity_curve`):** Throughout the simulation, it keeps track of the portfolio's total value (cash + market value of holdings) daily. The market value is calculated using a specified price (e.g., 'Close', configurable via `valuation_price_col`). This creates an "equity curve" showing performance over time.
    * **Results (`results` property):** After the `run` method completes, you can access the results, which typically include the equity curve DataFrame and the list of executed trades.
    * **Error Handling:** Has a robust set of custom exceptions (`BacktesterError`, `ConfigError`, `DataError`, `ExecutionError`, `StrategyError`) to handle issues during setup, data loading, trade execution, or within the strategy's logic.

### 4. `stock_plotter.py`

* **Core Purpose:** A simpler utility focused solely on creating visualizations (plots) of a single stock's price trend over time using the Plotly library.
* **Key Functionality:**
    * **Initialization (`__init__`):** You create an instance by providing a single stock ticker and a time period string (similar flexible format as `PortfolioManager`: '1y', '6M', 'from 2022', etc.).
    * **Date Range Parsing (`_parse_date_range`, `_parse_relative_period`):** Contains logic to interpret the various ways you can specify the time period and determine the actual start/end dates or the period string needed for data fetching.
    * **Data Fetching (`_determine_fetch_params`, `Workspace_data`):** Figures out the correct parameters (start/end dates or period string) based on the input `time_period` and then uses `yfinance.download()` directly to fetch the historical price data for the *single* specified ticker. It cleans the data slightly, ensuring the date is in the correct format and selects the 'Close' price.
    * **Plotting (`plot_trend`):** Takes the fetched data and uses `plotly.graph_objects` to generate a line chart showing the stock's closing price over the requested time frame. It sets titles and labels for clarity. The plot can be automatically displayed or returned as a Plotly figure object.
    * **Command-Line Usage:** The script is set up with `argparse` so you can run it directly from your terminal (e.g., `python stock_plotter.py AAPL 1y`) to quickly generate and view a plot.

In summary:

* `data_handler.py` gets the raw stock price data.
* `portfolio_manager.py` uses that data to track the value of a portfolio with changing holdings over time.
* `backtester.py` uses the data and a user-defined strategy to simulate trading and evaluate the strategy's performance, including cash management and commissions.
* `stock_plotter.py` is a straightforward tool to quickly visualize a single stock's price history.