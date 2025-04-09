# Portfolio Analytics Engine

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add more relevant badges as the project evolves, e.g., build status, code coverage -->
<!-- [![Build Status](https://travis-ci.org/your_username/your_repo.svg?branch=main)](https://travis-ci.org/your_username/your_repo) -->
<!-- [![Coverage Status](https://coveralls.io/repos/github/your_username/your_repo/badge.svg?branch=main)](https://coveralls.io/github/your_username/your_repo?branch=main) -->

**A robust and extensible Python toolkit for fetching financial data, visualizing market trends, and managing multi-asset portfolio performance over time.**



## Overview

The Portfolio Analytics Engine provides a foundational framework for quantitative financial analysis and portfolio management. It leverages industry-standard libraries to offer reliable data retrieval, insightful visualizations, and accurate tracking of portfolio holdings and value. Designed with modularity and extensibility in mind, it serves as a powerful base for developing and testing sophisticated investment strategies and risk management techniques.

Whether you are an individual investor seeking better tools, a developer building fintech applications, or a quantitative analyst prototyping strategies, this engine offers core functionalities wrapped in a clean, well-documented, and test-covered codebase.



## Key Features (Current)

*   **Flexible Data Retrieval (`stock_plotter`):**
    *   Fetch historical stock price data (Close prices) from Yahoo Finance.
    *   Supports standard Yahoo Finance time periods (e.g., `1y`, `6mo`, `ytd`).
    *   Parses intuitive natural language date ranges (e.g., `"from 2022"`, `"this year"`).
    *   Handles relative time periods (e.g., `3M`, `10D`, `2W`).
    *   Robust error handling for invalid tickers or time periods.
*   **Market Trend Visualization (`stock_plotter`):**
    *   Generate interactive time-series plots of stock prices using Plotly.
    *   Includes features like range selectors and hover-over data inspection.
    *   Customizable and embeddable plot objects.
*   **Portfolio Tracking (`portfolio_manager`):**
    *   Initialize portfolios with starting asset quantities.
    *   Record and apply historical changes to holdings (buys, sells, rebalances) using absolute quantities or relative adjustments (e.g., `+10`, `-5`).
    *   Handles multiple tickers simultaneously.
*   **Performance Calculation (`portfolio_manager`):**
    *   Calculates the daily market value of individual holdings based on historical prices.
    *   Computes the total portfolio value across all assets for each day in the specified period.
    *   Handles missing price data gracefully using forward-fill.
*   **Composition Analysis (`portfolio_manager`):**
    *   Determine the percentage allocation (based on value) of each asset within the portfolio for any given historical date or the latest available date.
*   **Robust Foundation:**
    *   Comprehensive logging for diagnostics and monitoring.
    *   Custom exceptions for clear error identification.
    *   Type hinting for improved code clarity and maintainability.
    *   Unit tested using `pytest` for core functionalities.
    *   Dockerized for consistent development and deployment environments.



## Technology Stack

*   **Core:** Python (3.11+)
*   **Data Handling:** Pandas, NumPy
*   **Financial Data Source:** yfinance
*   **Testing:** pytest
*   **Containerization:** Docker



## Getting Started

### Prerequisites

*   Git
*   Python 3.11 or higher
*   `pip` and `venv` (recommended)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd portfolio-analytics-engine # Or your chosen directory name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```



## Usage Examples

### 1. Plotting Stock Trends (Command Line)

Use the `stock_plotter.py` script directly from your terminal:

```bash
# Plot Apple's stock price for the last year
python stock_plotter.py AAPL 1y

# Plot Microsoft's stock price from the start of 2023
python stock_plotter.py MSFT "from 2023"

# Plot Tesla's stock price for the last 6 months (relative)
python stock_plotter.py TSLA 6M

# Enable debug logging
python stock_plotter.py GOOGL 1mo --debug
```
*(Note: Plotting requires a graphical environment if `show=True`)*

### 2. Using Modules Programmatically (Python)

Integrate the engine's components into your own Python scripts:

```python
import pandas as pd
from stock_plotter import StockTrendPlotter
from portfolio_manager import PortfolioManager, InvalidInputError, DataFetchError

# --- Using StockTrendPlotter ---
try:
    plotter = StockTrendPlotter(ticker='NVDA', time_period='6mo')
    # Fetch data
    price_data = plotter.fetch_data()
    print("Fetched NVDA Data:\n", price_data.tail())
    # Generate plot object (don't show automatically)
    # fig = plotter.plot_trend(show=False)
    # fig.write_html("nvda_plot.html") # Save plot to file
except (ValueError, IOError) as e:
    print(f"Error plotting NVDA: {e}")

# --- Using PortfolioManager ---
initial_holdings = {'AAPL': 10, 'GOOGL': 5}
time_frame = '1y' # Data lookback period

try:
    manager = PortfolioManager(initial_portfolio=initial_holdings, time_period=time_frame)

    # Add portfolio changes (transactions)
    manager.add_change('2023-08-15', 'AAPL', '+5')    # Bought 5 more AAPL
    manager.add_change('2023-10-01', 'GOOGL', 2)     # Holdings adjusted to 2 GOOGL
    manager.add_change('2023-11-20', 'MSFT', 8)      # New holding MSFT
    manager.add_change('2023-12-05', 'AAPL', '-3')     # Sold 3 AAPL

    # Calculate portfolio history (fetches data for AAPL, GOOGL, MSFT)
    portfolio_history_df = manager.get_portfolio_data() # Builds if not already built

    if portfolio_history_df is not None:
        print("\nPortfolio History (Tail):\n", portfolio_history_df.tail())

        # Get portfolio composition for a specific date (or latest if None)
        composition_latest = manager.get_composition()
        composition_specific = manager.get_composition(date='2023-11-25') # Finds nearest if exact not found

        print("\nLatest Portfolio Composition (%):\n", composition_latest)
        print("\nPortfolio Composition on 2023-11-25 (%):\n", composition_specific)

except (InvalidInputError, DataFetchError, PortfolioBuildError) as e:
    print(f"Portfolio management error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

```


## Testing

Unit tests are implemented using `pytest`. To run the tests:

1.  Ensure you have installed the development dependencies (if any are added later, e.g., `pytest-mock`). Currently, `pytest` is included in `requirements.txt`.
2.  Navigate to the project's root directory in your terminal.
3.  Run `pytest`:
    ```bash
    pytest -v # -v for verbose output
    ```
    *(Note: Some tests may require network access to fetch sample data or interact with APIs indirectly via mocks)*



## Docker Support

A `Dockerfile` is provided to build a container image with the necessary environment and dependencies. This ensures consistency across different machines.

1.  **Build the Docker image:**
    ```bash
    docker build -t portfolio-engine .
    ```

2.  **Run a container (example: execute a script):**
    You can run scripts within the container environment. Mount your local project directory to access your files inside the container.
    ```bash
    # Example: Run the stock_plotter script inside the container
    # (Note: Plotting directly might require X11 forwarding setup)
    docker run --rm -v "$(pwd):/app" portfolio-engine python stock_plotter.py AAPL 1mo

    # Example: Run an interactive python session inside the container
    docker run --rm -it -v "$(pwd):/app" portfolio-engine python
    ```



## Project Structure

```
portfolio-analytics-engine/
│
├── api_tests/              # Placeholder or actual API/integration tests
│   ├── __pycache__/
│   └── .pytest_cache/
│
├── portfolio_manager.py    # Core portfolio tracking and calculation logic
├── stock_plotter.py        # Data fetching and plotting logic
│
├── test_portfolio_manager.py # Unit tests for portfolio_manager
├── test_stock_plotter.py     # Unit tests for stock_plotter
│
├── yahoo_finance_setup.ipynb # Jupyter Notebook for experimentation/setup (if used)
│
├── .gitignore              # Specifies intentionally untracked files for Git
├── Dockerfile              # Defines the Docker container environment
├── README.md               # This file
└── requirements.txt        # Project dependencies
```


## Roadmap & Future Directions

This engine provides a strong foundation. Future development aims to incorporate more advanced analytics and strategy evaluation capabilities:

### Planned Enhancements

*   **📈 Enhanced Data Handling:**
    *   Integration with multiple data sources (e.g., Alpha Vantage, IEX Cloud, paid providers).
    *   Support for fetching broader datasets (OHLCV, fundamental data, economic indicators).
    *   Robust data cleaning and persistence mechanisms (local caching/database).
*   **💼 Sophisticated Portfolio Management:**
    *   Explicit cash management and currency handling.
    *   Simulation of transaction costs (brokerage fees, slippage).
    *   Handling of corporate actions (dividends, stock splits).
*   **⚙️ Backtesting Framework:**
    *   A dedicated engine for simulating strategy execution on historical data.
    *   Calculation of standard performance metrics (Sharpe Ratio, Sortino Ratio, Max Drawdown, CAGR).
    *   Comparison against benchmarks (e.g., S&P 500).
    *   Visualization of backtest results, including equity curves and trade logs.
    *   "Historical Event Replay" feature to analyze strategy performance during specific market conditions (e.g., 2008 crisis, COVID crash).
*   **⚖️ Risk Management & Optimization:**
    *   **Monte Carlo Simulation:** Forecasting potential portfolio outcomes and visualizing risk distributions.
    *   **Convex Optimization:** Implementation of Mean-Variance Optimization (MVO) to find efficient frontiers and optimal risk-return trade-offs (using `scipy.optimize`).
*   **🛠️ Infrastructure & Quality:**
    *   Configuration management for API keys, parameters, etc.
    *   Potential implementation of CI/CD pipelines for automated testing and deployment.

### Exploratory Advanced Features

*   **🤖 Reinforcement Learning (RL) for Strategy Optimization:**
    *   Investigating the feasibility of using RL agents (e.g., DQN, PPO) to learn dynamic asset allocation strategies based on market state indicators. Requires careful state/action/reward design and robust training infrastructure.
*   **📊 Explainable AI (XAI) for Model Interpretation:**
    *   If complex models (like RL) are implemented, exploring techniques like SHAP to understand feature importance and model decision drivers.



## Contributing

Contributions are welcome! If you have suggestions for improvements, find a bug, or want to add new features:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  **Make your changes.** Ensure you add or update tests accordingly.
4.  **Run tests** (`pytest`) to confirm everything passes.
5.  **Commit your changes** (`git commit -m 'Add some feature'`).
6.  **Push to the branch** (`git push origin feature/your-feature-name`).
7.  **Open a Pull Request** against the `main` branch of the original repository.

Please open an issue first to discuss any significant changes.