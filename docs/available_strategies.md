# Trading Strategies for the Custom Backtester

This directory contains custom trading strategy implementations designed to be compatible with the Python backtesting engine (`backtester.py`).

Each strategy is defined in its own Python file and inherits from the base `backtester.Strategy` class. They all implement the core `generate_signals` method, which analyzes market data and portfolio state to produce a list of desired trades (`Order` objects) for each time step.

## Available Strategies

Below is a list of the currently available strategies:

1.  [Moving Average Crossover Strategy (`mac_strategy.py`)](#moving-average-crossover-strategy-mac_strategypy)
2.  [Momentum Strategy (`momentum_strategy.py`)](#momentum-strategy-momentum_strategypy)
3.  [Rolling Sharpe Ratio Strategy (`rolling_sharpe_strategy.py`)](#rolling-sharpe-ratio-strategy-rolling_sharpe_strategypy)
4.  [Maximum Drawdown Risk Strategy (`max_drawdown_risk_strategy.py`)](#maximum-drawdown-risk-strategy-max_drawdown_risk_strategypy)

---

### Moving Average Crossover Strategy (`mac_strategy.py`)

*   **File:** `mac_strategy.py` (or `my_strategies.py` if combined)
*   **Class:** `MACrossoverStrategy`
*   **Concept:** A classic trend-following strategy based on the crossover of two simple moving averages (SMAs) of the closing price.
*   **Signals Generated:**
    *   **BUY:** When the short-term moving average crosses *above* the long-term moving average ("Golden Cross"). A buy order is generated only if the asset is not currently held.
    *   **SELL:** When the short-term moving average crosses *below* the long-term moving average ("Death Cross"). A sell order is generated for the *entire* currently held position.
*   **Parameters:**
    *   `short_window` (int): The lookback period for the short-term SMA. Default: `5`.
    *   `long_window` (int): The lookback period for the long-term SMA. Default: `20`.
    *   `quantity` (int): The fixed number of shares to purchase on a BUY signal. Default: `10`.
*   **Notes:** This is a simple implementation using fixed share quantity for buys and selling the full position on exit signals. It requires `short_window < long_window`.

---

### Momentum Strategy (`momentum_strategy.py`)

*   **File:** `momentum_strategy.py`
*   **Class:** `MomentumStrategy`
*   **Concept:** A strategy that invests in assets exhibiting the strongest recent performance (momentum). It calculates the rate of change over a defined period for each asset, ranks them, and holds the top performers.
*   **Signals Generated:**
    *   **BUY:** Enters positions in the top-ranked assets (up to `num_holdings`) that are not currently held and meet the optional `require_positive_momentum` criteria. Position size determined by `position_sizing`.
    *   **SELL:** Exits positions in assets that fall out of the top rank (below the top `num_holdings`). Sells the entire position.
*   **Parameters:**
    *   `momentum_window` (int): The lookback period (number of data points) for calculating the rate of change (momentum). Default: `90`.
    *   `num_holdings` (int): The target number of top-performing assets to hold simultaneously. Default: `3`.
    *   `position_sizing` (str): Method for determining buy quantity. Options:
        *   `'equal_weight'` (Default): Attempts to allocate capital equally among the target number of holdings based on estimated total equity.
        *   `'fixed_quantity'`: Buys a fixed number of shares defined by `buy_quantity`.
    *   `buy_quantity` (int): The fixed number of shares to buy per signal if `position_sizing` is `'fixed_quantity'`. Default: `10`.
    *   `require_positive_momentum` (bool): If `True`, only assets with a positive calculated momentum are considered eligible for buying. Default: `True`.
*   **Notes:** This strategy involves ranking and rebalancing based on relative performance within the asset universe.

---

### Rolling Sharpe Ratio Strategy (`rolling_sharpe_strategy.py`)

*   **File:** `rolling_sharpe_strategy.py`
*   **Class:** `RollingSharpeStrategy`
*   **Concept:** Selects assets based on their risk-adjusted returns, measured by the annualized Sharpe Ratio calculated over a rolling window. It aims to hold assets that have historically provided better returns for their level of volatility.
*   **Signals Generated:**
    *   **BUY:** Enters positions in the top-ranked assets (up to `num_holdings`) whose rolling annualized Sharpe Ratio exceeds `min_sharpe_ratio` and are not currently held. Uses equal weight position sizing.
    *   **SELL:** Exits positions in assets that fall out of the top rank or whose Sharpe Ratio drops below the `min_sharpe_ratio` threshold. Sells the entire position.
*   **Parameters:**
    *   `sharpe_window` (int): The lookback period (number of data points) for calculating returns and standard deviation for the Sharpe Ratio. Default: `90`.
    *   `num_holdings` (int): The target number of top-ranked, qualifying assets to hold. Default: `3`.
    *   `min_sharpe_ratio` (float): The minimum acceptable annualized Sharpe Ratio for an asset to be considered for purchase. Default: `0.5`.
    *   `risk_free_rate` (float): The annualized risk-free rate used in the Sharpe calculation (e.g., 0.02 for 2%). Default: `0.01`.
    *   `data_frequency` (str): The frequency of the input data ('daily', 'hourly', etc.), used for annualizing the Sharpe Ratio. Must match a key in `ANNUALIZATION_FACTORS`. Default: `'daily'`.
*   **Notes:** Requires `numpy`. Assumes the `data_frequency` parameter correctly reflects the input data to ensure proper annualization.

---

### Maximum Drawdown Risk Strategy (`max_drawdown_risk_strategy.py`)

*   **File:** `max_drawdown_risk_strategy.py`
*   **Class:** `MaxDrawdownRiskStrategy`
*   **Concept:** Acts purely as a risk management overlay. It monitors the current drawdown of each held asset from its peak price over a rolling window. If the drawdown exceeds a specified percentage, it triggers a sell signal.
*   **Signals Generated:**
    *   **BUY:** **None.** This strategy does not generate buy signals.
    *   **SELL:** Generates a sell order for the *entire* position of an asset if its current price drops below the `max_drawdown_pct` threshold from its rolling high price over the `drawdown_window`.
*   **Parameters:**
    *   `drawdown_window` (int): The lookback period (number of data points) for calculating the rolling high price. Default: `60`.
    *   `max_drawdown_pct` (float): The maximum allowable drawdown percentage (e.g., 0.15 for 15%). Expressed as a decimal between 0 and 1. Default: `0.15`.
    *   `price_source` (str): The price column ('Open', 'High', 'Low', 'Close', etc.) used to calculate the rolling high and current price for drawdown monitoring. Default: `'Close'`.
*   **Notes:** This strategy is intended to limit losses on existing positions. It **must** be used either in conjunction with another strategy that generates buy signals or on a portfolio where initial positions are established manually, as it will never enter new positions on its own.



## Usage

To use these strategies with the backtester:

1.  Ensure the strategy's Python file (e.g., `momentum_strategy.py`) is accessible in your Python environment.
2.  Import the strategy class into your main backtesting script.
3.  Pass the **class name** (not an instance) to the `strategy_class` parameter of the `Backtester`.
4.  Provide any desired strategy-specific parameters via the `strategy_params` dictionary when initializing the `Backtester`.

**Example:**

```python
from backtester import Backtester, BasicCommission
from momentum_strategy import MomentumStrategy # Import the desired strategy class
# Other necessary imports (pandas, data_handler, etc.)

# Define parameters for the chosen strategy
mom_params = {
    'momentum_window': 90,
    'num_holdings': 3,
    'position_sizing': 'equal_weight'
}

# Configure and run the backtester
bt = Backtester(
    initial_capital=100000,
    tickers=['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'],
    strategy_class=MomentumStrategy, # Pass the class itself
    start_date='2020-01-01',
    end_date='2023-12-31',
    strategy_params=mom_params, # Pass the specific parameters here
    commission_model=BasicCommission(fixed_fee=1.0) # Example commission
    # ... other backtester parameters ...
)

bt.run()
equity_curve, trades = bt.get_results()

# Analyze results...
print(equity_curve.tail())
print(trades.head())