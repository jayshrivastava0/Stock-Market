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

Usage:
    python stock_plotter.py TICKER TIME_PERIOD
    
    Examples:
        python stock_plotter.py AAPL 1y
        python stock_plotter.py MSFT from 2020
        python stock_plotter.py TSLA 6M
"""

import re
import sys
import argparse
from datetime import datetime
from typing import Tuple, Optional, Union

# Third-party imports
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dateutil.relativedelta import relativedelta


class StockTrendPlotter:
    """
    Class to handle stock price plotting with flexible date options.
    
    This class provides methods to fetch historical stock data from Yahoo Finance
    and visualize it with customizable date ranges. It supports standard Yahoo Finance
    periods, natural language date specifications, and relative time patterns.
    
    Attributes:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        time_period (str): The time period for data retrieval, which can be:
            - Standard Yahoo Finance periods ('1d', '5d', '1mo', '3mo', '6mo', 
              '1y', '2y', '5y', '10y', 'ytd', 'max')
            - Natural language ('this month', 'this year', 'from 2020')
            - Relative patterns ('7M', '10D', '2W', '3Y')
    """

    # Valid period strings accepted by Yahoo Finance
    VALID_PERIODS = {
        '1d', '5d', '1mo', '3mo', '6mo', 
        '1y', '2y', '5y', '10y', 'ytd', 'max'
    }

    def __init__(self, ticker: str, time_period: str):
        """
        Initialize the StockTrendPlotter with a ticker and time period.
        
        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL').
            time_period (str): The time period for fetching data.
        """
        self.ticker = ticker.upper()  # Ensure ticker is uppercase
        self.time_period = time_period

    def _parse_date_range(self, date_input: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Parse natural language date ranges.
        
        Supports formats like:
        - 'this year'
        - 'this month'
        - 'from 2020'
        - 'from 2020-05'
        
        Args:
            date_input (str): Natural language date range string.
            
        Returns:
            tuple: (start_date, end_date) as datetime objects or (None, None) if no match.
        """
        today = datetime.today()
        date_input_lower = date_input.lower()

        # Match "from YYYY" or "from YYYY-MM" pattern
        if 'from' in date_input_lower:
            match = re.match(r'.*from (\d{4})(?:-(\d{2}))?', date_input_lower)
            if match:
                year = int(match.group(1))
                month = int(match.group(2)) if match.group(2) else 1
                start_date = datetime(year, month, 1)
                return start_date, today

        # Match "this month" pattern
        if 'this month' in date_input_lower:
            return today.replace(day=1), today

        # Match "this year" pattern
        if 'this year' in date_input_lower:
            return today.replace(month=1, day=1), today

        return None, None

    def _parse_relative_period(self, period_input: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Parse relative time period patterns.
        
        Supports formats like:
        - '7M' (7 months ago until now)
        - '2W' (2 weeks ago until now)
        - '10D' (10 days ago until now)
        - '3Y' (3 years ago until now)
        
        Args:
            period_input (str): Relative time period string.
            
        Returns:
            tuple: (start_date, end_date) as datetime objects or (None, None) if no match.
        """
        # Match digit + unit pattern (e.g., 7M, 10D)
        match = re.match(r'^(\d+)([DdWwMmYy])$', period_input)
        if not match:
            return None, None  # No valid pattern detected

        qty = int(match.group(1))
        unit = match.group(2).lower()  # Normalized unit (d, w, m, y)
        today = datetime.today()

        # Calculate relative date based on unit
        if unit == 'd':
            return today - relativedelta(days=qty), today
        if unit == 'w':
            return today - relativedelta(weeks=qty), today
        if unit == 'm':
            return today - relativedelta(months=qty), today
        if unit == 'y':
            return today - relativedelta(years=qty), today

        return None, None

    def _validate_period(self) -> Tuple[Optional[datetime], Optional[datetime], Optional[str]]:
        """
        Validate and determine the correct date range or period type.
        
        This method tries to interpret the time_period attribute in different ways:
        1. As a natural language date range
        2. As a standard Yahoo Finance period string
        3. As a relative time pattern
        
        Returns:
            tuple: (start_date, end_date, yf_period) where:
                - start_date, end_date: datetime objects if a date range is used
                - yf_period: string if a standard Yahoo period is used
                
        Raises:
            ValueError: If the time period format is invalid or can't be interpreted.
        """
        # Try natural language date range
        start, end = self._parse_date_range(self.time_period)
        if start and end:
            return start, end, None  # Use start-end range

        # Try standard Yahoo Finance period
        if self.time_period.lower() in self.VALID_PERIODS:
            return None, None, self.time_period.lower()  # Use Yahoo period

        # Try relative date pattern
        start, end = self._parse_relative_period(self.time_period)
        if start and end:
            return start, end, None  # Use custom range

        # If all parsing attempts fail, raise error with guidance
        raise ValueError(
            f"Invalid time period: '{self.time_period}'.\n"
            f"Use a valid Yahoo Finance period ({', '.join(self.VALID_PERIODS)}), "
            f"a natural language range like 'from 2020'/'this month', "
            f"or a relative pattern like '7M'/'10D'/'2W'/'3Y'."
        )

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch stock price data from Yahoo Finance.
        
        Returns:
            pandas.DataFrame: DataFrame with 'Date' and 'Price' columns.
            
        Raises:
            ValueError: If no data is found for the specified ticker and time range.
            Exception: For other errors during data retrieval.
        """
        start, end, yf_period = self._validate_period()
        yf_ticker = yf.Ticker(self.ticker)

        # Fetch data using either date range or period
        if start and end:
            df = yf_ticker.history(start=start, end=end)
        else:
            df = yf_ticker.history(period=yf_period)

        # Check if data was retrieved
        if df.empty:
            raise ValueError(f"No data found for {self.ticker} in the given time range.")
        
        # Prepare data for plotting
        df = df.reset_index()[['Date', 'Close']].rename(columns={'Close': 'Price'})
        df['Date'] = pd.to_datetime(df['Date'])

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
            Exception: For other errors during plotting.
        """
        try:
            # Fetch stock price data
            df = self.fetch_data()
            
            # Create plotly figure
            fig = go.Figure(go.Scatter(
                x=df['Date'], 
                y=df['Price'], 
                mode='lines', 
                name=f'{self.ticker} Stock Price',
                line=dict(width=2)
            ))

            # Configure plot layout
            fig.update_layout(
                title=f'{self.ticker} Stock Price Trend ({self.time_period})',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                template='plotly_dark',
                hovermode='x unified',
                width=1000,
                height=500,
                margin=dict(l=50, r=50, t=50, b=50)
            )

            # Add range slider
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            if show:
                fig.show()
                
            return fig

        except ValueError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise


def run_tests() -> None:
    """Run test cases demonstrating the functionality of StockTrendPlotter."""
    print("\n=== Testing StockTrendPlotter ===")
    
    print("\n✔️  Testing Valid Yahoo Finance Periods")
    StockTrendPlotter('AAPL', '6mo').plot_trend()
    StockTrendPlotter('AAPL', '1y').plot_trend()

    print("\n✔️  Testing Natural Language Date Specifications")
    StockTrendPlotter('AAPL', 'this month').plot_trend()
    StockTrendPlotter('GOOGL', 'this year').plot_trend()
    StockTrendPlotter('MSFT', 'from 2019-06').plot_trend()

    print("\n✔️  Testing Relative Time Patterns")
    StockTrendPlotter('TSLA', '7M').plot_trend()  # 7 months ago -> now
    StockTrendPlotter('TSLA', '10D').plot_trend() # 10 days ago -> now
    StockTrendPlotter('TSLA', '2W').plot_trend()  # 2 weeks ago -> now
    StockTrendPlotter('TSLA', '3Y').plot_trend()  # 3 years ago -> now

    print("\n❌  Testing Invalid Patterns (Should Raise Errors)")
    try:
        StockTrendPlotter('AAPL', 'invalid range').plot_trend()
    except ValueError as e:
        print(f"Expected error: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot stock price trends with flexible date ranges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stock_plotter.py AAPL 1y
  python stock_plotter.py MSFT from 2020
  python stock_plotter.py TSLA 6M
  python stock_plotter.py GOOGL this year
  python stock_plotter.py --test
        """
    )
    
    parser.add_argument('ticker', nargs='?', help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('time_period', nargs='?', help='Time period (e.g., 1y, from 2020, 6M)')
    parser.add_argument('--test', action='store_true', help='Run test cases')
    
    return parser.parse_args()


def main():
    """Main entry point of the script."""
    args = parse_args()
    
    if args.test:
        run_tests()
        return
    
    if not args.ticker or not args.time_period:
        print("Error: Both ticker and time_period are required.")
        print("Run with --help for usage information.")
        sys.exit(1)
    
    try:
        plotter = StockTrendPlotter(args.ticker, args.time_period)
        plotter.plot_trend()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()