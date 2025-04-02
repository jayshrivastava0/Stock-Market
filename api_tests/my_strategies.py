# import pandas as pd
# from typing import List, Dict, Any, Optional

# # Import core components from the backtester module.
# from backtester import Strategy, Order, Timestamp, HoldingsDict, TickerSymbol, BUY_ACTION, SELL_ACTION

# class BuyAndHoldStrategy(Strategy):
#     """
#     A simple Buy and Hold strategy.
    
#     This strategy purchases a fixed number of shares for each tracked ticker 
#     on the first available time step and then holds the position indefinitely.
    
#     Strategy Parameters:
#         - quantity (int): Number of shares to buy per ticker (default: 10).
#     """
    
#     def __init__(self, parameters: Optional[Dict[str, Any]] = None):
#         """
#         Initialize the BuyAndHoldStrategy.
        
#         Args:
#             parameters (Optional[Dict[str, Any]]): Dictionary containing strategy parameters.
#                                                    Expected parameter:
#                                                    - 'quantity': (int) number of shares to buy per ticker.
#         """
#         super().__init__(parameters)
#         # Use a fixed quantity for each buy order; default is 10 shares.
#         self.quantity = self.parameters.get('quantity', 10)
#         # Track tickers that have already been purchased to avoid repeat buys.
#         self.bought = set()
#         print(f"BuyAndHoldStrategy initialized with quantity={self.quantity}")

#     def generate_signals(
#         self,
#         current_dt: Timestamp,
#         data_slice: pd.DataFrame,
#         current_holdings: HoldingsDict,
#         current_cash: float
#     ) -> List[Order]:
#         """
#         Generate trading signals at the current time step.
        
#         This method checks each ticker in the data slice. If the ticker has not yet been purchased,
#         it generates a BUY order for the fixed quantity if there is sufficient cash.
        
#         Args:
#             current_dt (Timestamp): The current timestamp.
#             data_slice (pd.DataFrame): Historical market data (up to current_dt) in a MultiIndex format.
#             current_holdings (HoldingsDict): Current portfolio holdings.
#             current_cash (float): Available cash before orders.
            
#         Returns:
#             List[Order]: A list of Order objects (empty if no trades are generated).
#         """
#         orders = []
#         # Retrieve unique tickers from the data slice (MultiIndex: level 1 contains ticker symbols)
#         available_tickers = data_slice.columns.get_level_values(1).unique()
        
#         for ticker in available_tickers:
#             # Only generate a buy order if we haven't bought this ticker yet.
#             if ticker not in self.bought:
#                 try:
#                     # Access the most recent 'Close' price for this ticker.
#                     price_series = data_slice[('Close', ticker)]
#                     last_price = price_series.iloc[-1]
#                 except Exception as e:
#                     print(f"Error accessing data for ticker {ticker} at {current_dt}: {e}")
#                     continue
                
#                 # Estimate the cost for the fixed quantity.
#                 estimated_cost = self.quantity * last_price
#                 if current_cash >= estimated_cost:
#                     # Generate a BUY order and mark the ticker as purchased.
#                     orders.append(Order(ticker=ticker, action=BUY_ACTION, quantity=self.quantity))
#                     self.bought.add(ticker)
#                     print(f"{current_dt.date()}: BUY signal for {ticker} (Quantity: {self.quantity}, Cost: {estimated_cost:.2f})")
#                     # Reduce the current_cash locally to simulate allocation for subsequent tickers.
#                     current_cash -= estimated_cost
#                 else:
#                     print(f"{current_dt.date()}: Insufficient cash to buy {ticker} (Needed: {estimated_cost:.2f}, Available: {current_cash:.2f})")
                    
#         return orders







# File: my_strategies.py
import pandas as pd
from typing import List, Dict, Any, Optional

# Import required components from your backtester module.
from backtester import Strategy, Order, Timestamp, HoldingsDict, TickerSymbol, BUY_ACTION, SELL_ACTION

class MACrossoverStrategy(Strategy):
    """
    A Moving Average Crossover Strategy.
    
    For each ticker, this strategy computes a short and a long moving average on the 'Close' price.
    A BUY signal is generated when the short MA crosses above the long MA (golden cross), and a SELL signal 
    is generated when the short MA crosses below the long MA (death cross).
    
    Strategy Parameters:
        - short_window (int): Period for short moving average (default: 5).
        - long_window (int): Period for long moving average (default: 20).
        - quantity (int): Number of shares to buy on a BUY signal (default: 10).
    """
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.short_window = self.parameters.get('short_window', 5)
        self.long_window = self.parameters.get('long_window', 20)
        self.quantity = self.parameters.get('quantity', 10)
        print(f"MACrossoverStrategy initialized with short_window={self.short_window}, long_window={self.long_window}, quantity={self.quantity}")

    def generate_signals(
        self,
        current_dt: Timestamp,
        data_slice: pd.DataFrame,
        current_holdings: HoldingsDict,
        current_cash: float
    ) -> List[Order]:
        orders = []
        # Retrieve unique tickers from the data slice (assumes a MultiIndex where level 1 is ticker)
        available_tickers = data_slice.columns.get_level_values(1).unique()
        
        for ticker in available_tickers:
            try:
                price_series = data_slice[('Close', ticker)]
            except Exception as e:
                print(f"Error accessing 'Close' price for {ticker} at {current_dt}: {e}")
                continue
            
            # Require enough data for the long moving average calculation
            if len(price_series.dropna()) < self.long_window:
                continue
            
            # Compute moving averages
            short_mavg = price_series.rolling(window=self.short_window).mean()
            long_mavg = price_series.rolling(window=self.long_window).mean()
            
            # Get the latest and previous values for both MAs
            latest_short = short_mavg.iloc[-1]
            latest_long = long_mavg.iloc[-1]
            if len(short_mavg) >= 2:
                prev_short = short_mavg.iloc[-2]
                prev_long = long_mavg.iloc[-2]
            else:
                continue
            
            # Skip if any values are missing
            if pd.isna(latest_short) or pd.isna(latest_long) or pd.isna(prev_short) or pd.isna(prev_long):
                continue
            
            holding_qty = current_holdings.get(ticker, 0)
            
            # Generate BUY signal on a golden cross if not already holding
            if latest_short > latest_long and prev_short <= prev_long:
                if holding_qty == 0:
                    orders.append(Order(ticker=ticker, action=BUY_ACTION, quantity=self.quantity))
                    print(f"{current_dt.date()}: BUY signal for {ticker}")
            
            # Generate SELL signal on a death cross if currently holding shares
            elif latest_short < latest_long and prev_short >= prev_long:
                if holding_qty > 0:
                    orders.append(Order(ticker=ticker, action=SELL_ACTION, quantity=holding_qty))
                    print(f"{current_dt.date()}: SELL signal for {ticker}")
        
        return orders
