#!/usr/bin/env python3
"""
Stock Trend Plotter

A utility for visualizing stock price trends with flexible date range options.
Supports standard Yahoo Finance periods, natural language date specifications,
and relative time patterns.

Requirements:
    - pandas
    - plotly
    - yfinance
    - python-dateutil
    (See requirements.txt for specific versions)

Usage:
    python stock_plotter.py TICKER TIME_PERIOD
    python stock_plotter.py --test  # Runs basic demonstration tests

Examples:
    python stock_plotter.py AAPL 1y
    python stock_plotter.py MSFT "from 2020"
    python stock_plotter.py TSLA 6M
    python stock_plotter.py GOOGL "this year"
"""

import re
import sys
import argparse
import logging
from datetime import datetime
from typing import Tuple, Optional, Union

# Third-party imports
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dateutil.relativedelta import relativedelta

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Log to stdout
)


class StockTrendPlotter:
    """
    Class to handle stock price plotting with flexible date options.

    Provides methods to fetch historical stock data from Yahoo Finance
    and visualize it using Plotly with customizable date ranges.
    Supports standard Yahoo Finance periods, natural language date specifications,
    and relative time patterns.

    Attributes:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        time_period (str): The time period for data retrieval.
    """

    # Valid period strings accepted by Yahoo Finance (now immutable)
    VALID_PERIODS = frozenset({
        '1d', '5d', '1mo', '3mo', '6mo',
        '1y', '2y', '5y', '10y', 'ytd', 'max'
    })

    def __init__(self, ticker: str, time_period: str):
        """
        Initialize the StockTrendPlotter.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL').
            time_period (str): The time period for fetching data.
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker symbol must be a non-empty string.")
        if not time_period or not isinstance(time_period, str):
            raise ValueError("Time period must be a non-empty string.")

        self.ticker = ticker.upper()
        self.time_period = time_period
        logging.info("Initialized StockTrendPlotter for %s with period '%s'", self.ticker, self.time_period)

    def _parse_date_range(self, date_input: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Parse natural language date ranges like 'this year', 'from 2020'.

        Args:
            date_input (str): Natural language date range string.

        Returns:
            tuple: (start_date, end_date) as datetime objects or (None, None).
        """
        today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) # Normalize today
        date_input_lower = date_input.lower().strip()

        if 'from' in date_input_lower:
            match = re.match(r'.*from\s+(\d{4})(?:-(\d{1,2}))?', date_input_lower)
            if match:
                year = int(match.group(1))
                month_str = match.group(2)
                month = int(month_str) if month_str else 1
                try:
                    start_date = datetime(year, month, 1)
                    logging.debug("Parsed 'from' date: %s", start_date)
                    # Ensure start date is not in the future
                    return (start_date, today) if start_date <= today else (None, None)
                except ValueError: # Invalid date like 2020-13
                    logging.warning("Invalid date components parsed from '%s'", date_input)
                    return None, None

        if 'this month' == date_input_lower:
            start_date = today.replace(day=1)
            logging.debug("Parsed 'this month', start_date: %s", start_date)
            return start_date, today

        if 'this year' == date_input_lower:
            start_date = today.replace(month=1, day=1)
            logging.debug("Parsed 'this year', start_date: %s", start_date)
            return start_date, today

        logging.debug("No natural language date pattern matched for '%s'", date_input)
        return None, None

    def _parse_relative_period(self, period_input: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Parse relative time period patterns like '7M', '2W', '10D', '3Y'.

        Args:
            period_input (str): Relative time period string.

        Returns:
            tuple: (start_date, end_date) as datetime objects or (None, None).
        """
        today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) # Normalize today
        match = re.match(r'^(\d+)([DdWwMmYy])$', period_input.strip())
        if not match:
            logging.debug("No relative period pattern matched for '%s'", period_input)
            return None, None

        qty = int(match.group(1))
        unit = match.group(2).lower()
        delta_args = {}

        if unit == 'd':
            delta_args['days'] = qty
        elif unit == 'w':
            delta_args['weeks'] = qty
        elif unit == 'm':
            delta_args['months'] = qty
        elif unit == 'y':
            delta_args['years'] = qty
        else:
             # Should not happen due to regex, but defensive check
            logging.warning("Unexpected unit '%s' in relative period parsing", unit)
            return None, None

        if qty <= 0:
            logging.warning("Relative period quantity must be positive, got %d", qty)
            return None, None

        try:
            start_date = today - relativedelta(**delta_args)
            logging.debug("Parsed relative period '%s', start_date: %s", period_input, start_date)
            return start_date, today
        except Exception as e:
            logging.error("Error calculating relative delta for '%s': %s", period_input, e)
            return None, None


    def _validate_period(self) -> Tuple[Optional[datetime], Optional[datetime], Optional[str]]:
        """
        Validate and determine the correct date range or period type.

        Returns:
            tuple: (start_date, end_date, yf_period)
        Raises:
            ValueError: If the time period format is invalid.
        """
        # 1. Try natural language
        start, end = self._parse_date_range(self.time_period)
        if start is not None and end is not None:
            logging.info("Using natural language date range: %s to %s", start.date(), end.date())
            return start, end, None

        # 2. Try standard Yahoo Finance period
        period_lower = self.time_period.lower()
        if period_lower in self.VALID_PERIODS:
            logging.info("Using standard Yahoo Finance period: '%s'", period_lower)
            return None, None, period_lower

        # 3. Try relative date pattern
        start, end = self._parse_relative_period(self.time_period)
        if start is not None and end is not None:
            logging.info("Using relative date range: %s to %s", start.date(), end.date())
            return start, end, None

        # If all fail
        error_msg = (
            f"Invalid time period: '{self.time_period}'.\n"
            f"Use a valid Yahoo Finance period (e.g., {', '.join(self.VALID_PERIODS)}), "
            f"a natural language range (e.g., 'from 2020', 'this month'), "
            f"or a relative pattern (e.g., '7M', '10D', '2W', '3Y')."
        )
        logging.error(error_msg)
        raise ValueError(error_msg)

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch stock price data from Yahoo Finance.

        Returns:
            pandas.DataFrame: DataFrame with 'Date' and 'Price' columns.
        Raises:
            ValueError: If no data is found or time period is invalid.
            IOError: If there's an issue fetching data from yfinance.
        """
        try:
            start, end, yf_period = self._validate_period()
        except ValueError as e:
            # Validation error already logged, re-raise
            raise e

        logging.info("Fetching data for ticker %s...", self.ticker)
        yf_ticker = yf.Ticker(self.ticker)
        df = pd.DataFrame() # Initialize empty DataFrame

        try:
            if start is not None and end is not None:
                logging.debug("Fetching history with start=%s, end=%s", start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
                df = yf_ticker.history(start=start, end=end)
            elif yf_period is not None:
                logging.debug("Fetching history with period=%s", yf_period)
                df = yf_ticker.history(period=yf_period)
            else:
                # This case should ideally not be reached if _validate_period works correctly
                 logging.error("Internal validation error: No valid period or date range determined.")
                 raise ValueError("Internal error determining fetch parameters.")

        except Exception as e:
            # Catch potential errors from yfinance (network, ticker not found, etc.)
            logging.error("Failed to download data for %s from yfinance: %s", self.ticker, e, exc_info=True)
            raise IOError(f"Could not fetch data for {self.ticker} from Yahoo Finance.") from e

        if df.empty:
            logging.warning("No data returned for %s in the specified time range.", self.ticker)
            raise ValueError(f"No data found for {self.ticker} in the given time range.")

        logging.info("Successfully fetched %d data points for %s.", len(df), self.ticker)

        # Prepare data for plotting
        df = df.reset_index()
        # Ensure 'Date' column exists and handle potential timezone awareness issues
        if 'Date' not in df.columns and 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'}) # Adapt if column name differs

        if 'Date' not in df.columns:
             logging.error("Expected 'Date' column not found in yfinance output.")
             raise ValueError("Data format error: Missing 'Date' column.")

        # Convert to datetime and remove timezone if present (for consistency)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df[['Date', 'Close']].rename(columns={'Close': 'Price'})

        return df

    def plot_trend(self, show: bool = True) -> go.Figure:
        """
        Plot the stock price trend using Plotly.

        Args:
            show (bool, optional): Whether to display the plot. Defaults to True.

        Returns:
            plotly.graph_objects.Figure: The generated Plotly figure object.

        Raises:
            ValueError: If the time period is invalid or no data is found.
            IOError: If data fetching fails.
            Exception: For other unexpected errors during plotting.
        """
        logging.info("Generating plot for %s...", self.ticker)
        try:
            df = self.fetch_data()

            fig = go.Figure(go.Scatter(
                x=df['Date'],
                y=df['Price'],
                mode='lines',
                name=f'{self.ticker} Price',
                line=dict(width=2)
            ))

            fig.update_layout(
                title=f'{self.ticker} Stock Price Trend ({self.time_period})',
                xaxis_title='Date',
                yaxis_title='Price (USD)', # Assuming USD, might need adjustment
                template='plotly_dark',
                hovermode='x unified',
                width=1000,
                height=500,
                margin=dict(l=50, r=50, t=50, b=50)
            )

            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=[
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]
                )
            )

            logging.info("Plot figure created successfully for %s.", self.ticker)

            if show:
                logging.info("Displaying plot...")
                fig.show()

            return fig

        except (ValueError, IOError) as e:
            # Logged in fetch_data or _validate_period, just re-raise
            logging.error("Plot generation failed due to data error: %s", e)
            raise
        except Exception as e:
            logging.exception("An unexpected error occurred during plot generation for %s.", self.ticker)
            raise


def run_demonstration_tests() -> None:
    """
    Run basic demonstration cases.
    Note: These are not exhaustive unit tests. Use pytest for proper testing.
    """
    print("\n=== Running Basic Demonstrations (requires network) ===")
    test_cases = [
        ('AAPL', '1mo'),
        ('MSFT', 'this year'),
        ('GOOGL', 'from 2023'),
        ('TSLA', '3M'),
        # ('INVALIDTICKER', '1y'), # Example failure case (requires yfinance error handling)
        # ('AAPL', 'invalid period') # Example failure case
    ]

    for ticker, period in test_cases:
        print(f"\n--- Testing: {ticker} for period '{period}' ---")
        try:
            plotter = StockTrendPlotter(ticker, period)
            # Fetch data first to catch errors before plotting attempt
            plotter.fetch_data()
            print(f"Data fetched successfully for {ticker}, {period}.")
            # Optionally plot if needed for visual check, but not required for demo
            # plotter.plot_trend(show=True) # Uncomment to show plots
        except (ValueError, IOError) as e:
            print(f"Caught expected error: {e}")
        except Exception as e:
            print(f"Caught unexpected error: {e}")

    print("\n--- Testing Invalid Input (Should Raise ValueError) ---")
    try:
        StockTrendPlotter('AAPL', 'invalid range').plot_trend(show=False)
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")

    print("\n=== Demonstrations Finished ===")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot stock price trends with flexible date ranges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stock_plotter.py AAPL 1y
  python stock_plotter.py MSFT "from 2020"
  python stock_plotter.py TSLA 6M
  python stock_plotter.py GOOGL "this year"
  python stock_plotter.py --test
        """
    )

    parser.add_argument('ticker', nargs='?', help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('time_period', nargs='?', help='Time period (e.g., 1y, "from 2020", 6M)')
    parser.add_argument('--test', action='store_true', help='Run basic demonstration tests (requires network)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    return parser.parse_args()


def main():
    """Main entry point of the script."""
    args = parse_args()

    # Adjust log level if debug flag is set
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")

    if args.test:
        run_demonstration_tests()
        return # Exit after tests

    if not args.ticker or not args.time_period:
        # Use ArgumentParser's error handling for missing arguments
        # This branch might not be strictly necessary if nargs='?' handles it,
        # but provides a clearer message if needed.
        logging.error("Both ticker and time_period arguments are required.")
        print("Usage: python stock_plotter.py TICKER TIME_PERIOD", file=sys.stderr)
        print("Run with --help for more information.", file=sys.stderr)
        # argparse should exit here if arguments are missing, but being explicit
        sys.exit(2) # Use 2 for command line usage errors

    try:
        plotter = StockTrendPlotter(args.ticker, args.time_period)
        plotter.plot_trend(show=True) # Show plot by default when run from CLI
        logging.info("Script finished successfully.")

    except (ValueError, IOError) as e:
        # Specific, expected errors related to input or data fetching
        logging.error("Execution failed: %s", e)
        sys.exit(1) # General error exit code
    except Exception as e:
        # Unexpected errors
        logging.exception("An unexpected critical error occurred.") # Includes traceback
        sys.exit(1)


if __name__ == "__main__":
    main()