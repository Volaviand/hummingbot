import logging
from decimal import Decimal
from typing import List
import math
from math import floor, ceil
import pandas as pd
import pandas_ta as ta  # noqa: F401
import requests

import numpy as np
import random
import time
import datetime
import datetime as dt

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase, Dict
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory

from arch import arch_model


### attempt to add your own code from earlier
import sqlite3
import json
import sys
import os
import csv

sys.path.append('/home/tyler/quant/API_call_tests/')
from Kraken_Calculations import BuyTrades, SellTrades


class KrakenAPI:
    def __init__(self, symbol, start_timestamp=None, end_timestamp=None):
        self.symbol = symbol
        self.base_url = 'https://api.kraken.com/0/public/Trades'
        self.data = []
        self.start_timestamp = start_timestamp
        self.last_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

    def fetch_trades(self, since):
        try:
            url = f'{self.base_url}?pair={self.symbol}&since={since}'
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if "result" in data and self.symbol in data["result"]:
                trades = data["result"][self.symbol]
                last_timestamp = int(data["result"].get("last", self.last_timestamp))
                # print(f"Data Saved. Last Timestamp: {last_timestamp}")
            
                return True, trades, last_timestamp
            else:
                # print(f"No data found or error in response for symbol: {self.symbol}")
                return False, [], self.last_timestamp
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return False, [], self.last_timestamp

    def get_trades_since(self):
        initial_start_timestamp = self.start_timestamp  # Store the initial start timestamp
        while True:
            success, trades, last_timestamp = self.fetch_trades(self.last_timestamp)
            # print(len(trades))
            if not success or not trades:
                # print("No more data to fetch.")
                break

            self.data.extend(trades)
            self.last_timestamp = last_timestamp

            # Stop if last timestamp exceeds end timestamp or if no new data is returned
            if  self.last_timestamp >= self.end_timestamp:
                # print("Reached the end timestamp.")
                break

            if len(trades) == 1:
                # print("No more trades.")
                break

            # # Limit the loop to avoid excessive requests
            # if len(self.data) > 100000:  # Example limit
            #     print("Data limit reached.")
            #     break

            # Rate limit to avoid hitting API too hard
            time.sleep(1)

        return self.data
        
    def fetch_kraken_ohlc_data(self, since, interval):
        try:
            url = "https://api.kraken.com/0/public/OHLC"
            
            # Define parameters
            params = {
                'pair': self.symbol,  # Asset pair (e.g., 'XBTEUR' for Bitcoin/EUR)
                'interval': interval,     # Time frame in minutes (1440 = 1 day)
                'since': since        # Unix timestamp for fetching data since a specific time
            }
            
            # Make the GET request to Kraken API with params
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the response JSON
            data = response.json()
    
            # Check if the response contains the 'result' and data for the given symbol
            if "result" in data and self.symbol in data["result"]:
                trades = data["result"][self.symbol]  # OHLC data
                last_timestamp = int(data["result"].get("last", self.last_timestamp))  # Timestamp of the last data point
                
                # print(f"Data Saved. Last Timestamp: {last_timestamp}")
            
                return True, trades, last_timestamp
            else:
                # print(f"No data found or error in response for symbol: {self.symbol}")
                return False, [], self.last_timestamp
    
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return False, [], self.last_timestamp

    def get_ohlc_since(self, interval):
        initial_start_timestamp = self.start_timestamp  # Store the initial start timestamp
        while True:
            success, trades, last_timestamp = self.fetch_kraken_ohlc_data(self.last_timestamp, interval)
            # print(len(trades))
            if not success or not trades:
                # print("No more data to fetch.")
                break

            self.data.extend(trades)
            self.last_timestamp = last_timestamp

            # Stop if last timestamp exceeds end timestamp or if no new data is returned
            if  self.last_timestamp >= self.end_timestamp:
                # print("Reached the end timestamp.")
                break

            if len(trades) == 1:
                # print("No more trades.")
                break

            # # Limit the loop to avoid excessive requests
            # if len(self.data) > 100000:  # Example limit
            #     print("Data limit reached.")
            #     break

            # Rate limit to avoid hitting API too hard
            time.sleep(1)

        return self.data
    def call_kraken_ohlc_data(self, hist_days = 720, market = 'XXLMZEUR', interval = 60):
        # Calculate the timestamp for hist_days ago
        since_input = datetime.datetime.now() - datetime.timedelta(days=hist_days)
        since_timestamp = int(time.mktime(since_input.timetuple())) * 1000000000  # Convert to nanoseconds
        # print(f'Since Timestamp, {since_timestamp}')
        # Calculate the timestamp for now
        now_timestamp = int(time.time() * 1000000000)  # Current time in nanoseconds
        # print(f'Now Timestamp {now_timestamp}')
        markets = market 
        # Initialize Kraken API object with your symbol and start timestamp
        api = KrakenAPI(market, since_timestamp, end_timestamp=now_timestamp)
        trades = api.get_ohlc_since(interval)
        # Convert to DataFrame
        #[int <time>, string <open>, string <high>, string <low>, string <close>, string <vwap>, string <volume>, int <count>]
        df = pd.DataFrame(trades, columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'VWAP', 'Volume', 'Count'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        # Pop out last data point, which is a duplicate of current data etc * 
        df = df.iloc[:-1]
        # print(df)
        return df





class SimplePMM(ScriptStrategyBase):
    """
    BotCamp Cohort: Sept 2022
    Design Template: https://hummingbot-foundation.notion.site/Simple-PMM-63cc765486dd42228d3da0b32537fc92
    Video: -
    Description:
    The bot will place two orders around the price_source (mid price or last traded price) in a trading_pair on
    exchange, with a distance defined by the ask_spread and bid_spread. Every order_refresh_time in seconds,
    the bot will cancel and replace the orders.
    """
    # bid_spread = 0.05
    # ask_spread = 0.05
    min_profitability = 0.015
    target_profitability = min_profitability
    # _order_refresh_tolerance_pct = 0.0301




    #order_refresh_time = 30
    quote_order_amount = Decimal(3.5)
    order_amount = Decimal(40)
    min_order_size_bid = Decimal(0)
    min_order_size_ask = Decimal(0)


    create_timestamp = 0
    create_garch_timestamp = 0
    trading_pair = "XLM-EUR"
    exchange = "kraken"
    base_asset = "XLM"
    quote_asset = "EUR"
    history_market = 'XXLMZEUR'

    #Maximum amount of orders  Bid + Ask
    maximum_orders = 170

    inv_target_percent = Decimal(0.0)   

    ## how fast/gradual does inventory rebalance? bigger= more rebalance
    order_shape_factor = Decimal(2.0) 
    # Here you can use for example the LastTrade price to use in your strategy
    #MidPrice 
    #BestBid 
    #BestAsk 
    #LastTrade 
    #LastOwnTrade 
    #InventoryCost 
    #Custom 
    # _last_trade_price = None
    _vwap_midprice = None
    #price_source = self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.LastOwnTrade)

    markets = {exchange: {trading_pair}}

    report_interval = 60 * 60 * 6  # 6 hours



    ## Breakeven Initialization
    ## Trading Fee for Round Trip side Limit
    fee_percent = 1 / 4 / 100  # Convert percentage to a fraction method
    total_spent = 0
    total_bought = 0
    total_earned = 0
    total_sold = 0
    break_even_price = None  # Store the break-even price

    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)

        min_refresh_time = 30
        max_refresh_time = 120

        # Generate a random integer between min and max using randint
        self.order_refresh_time = random.randint(min_refresh_time, max_refresh_time)

        self.last_time_reported = 0


        self.garch_refresh_time = 480 ### Same as volatility interval
        self_last_garch_time_reported = 0



        # Trade History Timestamp
        # self.last_trade_timestamp = 1726652280000

        ## Initialize Trading Flag for use 
        self.initialize_flag = True

        self._bid_baseline = None
        self._ask_baseline = None

        self.initialize_startprice_flag = True
        # self.buy_counter = 2
        # self.sell_counter = 1


        # Volume Depth Init
        self.bought_volume_depth = 0.0000000001
        self.sold_volume_depth =   0.0000000001

        # Volatility 
        self.max_vola = 0.0
        self.current_vola = 0.0
        self.volatility_rank = 0.0

        # Order Status Variables
        self.ask_percent = 0
        self.bid_percent = 0
        self.b_r_p = 0
        self.a_r_p = 0
        self.b_d = 0
        self.a_d = 0
        self.q_imbalance = 0
        self.inventory_diff = 0
        
        self.b_be = 0
        self.s_be = 0
        self.pnl = 0
        self.u_pnl = 0
        self.n_v = 0

        self._last_buy_price = 0
        self._last_sell_price = 0

        self.trade_position_text = ""

    def get_ohlc_calculations(self, df, rolling_period=72):
        df['Open'] = pd.to_numeric(df['Open'])
        df['High'] = pd.to_numeric(df['High'])
        df['Low'] =pd.to_numeric(df['Low'])
        source = ( df['High'] + df['Low']) / 2 # pd.to_numeric(df['Close'])# 
        df['Source'] = source
        df['Close'] = pd.to_numeric(df['Close'])
        close = df['Close']
        log_returns = np.log(source/np.roll(source,shift=1))[1:].dropna()
        log_returns_series = pd.Series(log_returns)
        
        # Define the GARCH model with automatic rescaling
        model = arch_model(log_returns_series, vol='Garch', mean='constant', p=3, q=3, power=2.0, rescale=True)

        # Fit the model
        model_fit = model.fit(disp="off")
        # fig1 = model_fit.plot()
        print(model_fit.summary())
        # Extract standardized residuals
        std_residuals = model_fit.std_resid
        
        # Extract conditional volatility
        volatility_rescaled = model_fit.conditional_volatility

        # Convert the percent to decimal percent
        volatility = volatility_rescaled / 100 

        dates = log_returns_series.index

        #######################::::::::::::::::::::::::::::::::::::::::::
        ############## Volatility of volatility, Secondary Volatility::::
        secondary_log_returns = np.log(volatility / np.roll(volatility, shift=1))[1:]
        
        # Convert log returns to a pandas Series
        secondary_log_returns_series = pd.Series(secondary_log_returns) #* scale

        # Define the GARCH model with automatic rescaling
        secondary_model = arch_model(secondary_log_returns, vol='Garch', mean='constant', p=3, q=3, power=2.0, rescale=True, dist="StudentsT")

        # Fit the model
        secondary_model_fit = secondary_model.fit(disp="off")

        # Extract conditional volatility
        secondary_volatility_rescaled = np.exp(secondary_model_fit.conditional_volatility) - 1
        
        # Convert the percent to decimal percent
        secondary_volatility = secondary_volatility_rescaled / 100 

        #Append values to df
        df['Mean'] = source.dropna().mean()
        df['Log Returns'] = log_returns_series.dropna()
        df['Volatility'] = volatility.dropna()
        df['Secondary Volatility'] = secondary_volatility.dropna()
        df['Standard Residuals'] = std_residuals.dropna()

        # Edit Volume for calculations
        df['Volume'] = pd.to_numeric(df['Volume'])
        # rolling_period = 72
        
        # Extent of tail capturing, currently 0.8413, 0.1587 for 1 Standard deviation around median
        IQR3_vola = df['Volatility'].quantile(0.8413) # rolling(window=rolling_period).
        vola_median = df['Volatility'].quantile(0.50) # rolling(window=rolling_period).
        IQR1_vola = df['Volatility'].quantile(0.1587) # rolling(window=rolling_period).
        
        IQR3_Source = df['High'].rolling(window=rolling_period).quantile(0.8413)
        IQR1_Source = df['Low'].rolling(window=rolling_period).quantile(0.1587)   
        
        # Assign these values to the entire DataFrame in new columns
        df['IQR3_vola'] = IQR3_vola
        df['Vola_Median'] = vola_median
        df['IQR1_vola'] = IQR1_vola
        df['IQR3_Source'] = IQR3_Source
        df['IQR1_Source'] = IQR1_Source
        
        dt = 1  # np.sqrt(window)# / len(df['Volatility']))
        


        df['Rolling Volatility'] = df['Volatility'].rolling(window=rolling_period).mean()



        # Rank the Volatility
        #init vol rank

        self.max_vola = df['Volatility'].iloc[-rolling_period:].max()
        min_vola = df['Volatility'].iloc[-rolling_period:].min()
        self.current_vola = df['Volatility'].iloc[-1]


        # Prevent division by zero
        if self.max_vola != min_vola:
            self.volatility_rank = (self.current_vola - min_vola) / (self.max_vola - min_vola)
        else:
            self.volatility_rank = 1  # Handle constant volatility case

        df['volatility_rank'] = self.volatility_rank

        # print(f"Max Volatility :: {max_vola}")
        # print(f"Volatility Rank :: {df['volatility_rank'].tail()}")    



        ## Boolean Values for plotting areas;:
        # Initialize flags and trackers before the loop
        latest_low_tail_value = None  # Store the most recent low tail value
        latest_high_tail_value = None  # Store the most recent high tail value
        # high_tail_flag = False  # Initially set to False
        # low_tail_flag = False  # Initially set to False
        
        # Create conditions for high and low tails
        high_volatility = df['Volatility'] > df['IQR3_vola']
        low_tail = (df['Low'] < df['IQR1_Source']) & high_volatility
        high_tail = (df['High'] > df['IQR3_Source']) & high_volatility
        
        # Initialize Low Line and High Line with NaN values and fill the first values of IQR1_Source and IQR3_Source
        df['Low Line'] = np.nan
        df['High Line'] = np.nan
        df.loc[0, 'Low Line'] = df.loc[0, 'Low']  # Start Low Line with the first value of IQR1_Source
        df.loc[0, 'High Line'] = df.loc[0, 'High']  # Start High Line with the first value of IQR3_Source
        
        # Iterate through each row and update the Low Line and High Line
        for i in range(1, len(df)):
            # Get the previous Low and High Line values
            previous_low_line = df.loc[i-1, 'Low Line']
            previous_high_line = df.loc[i-1, 'High Line']
            
            # Handle Low Line updates (when a high tail happens)
            if low_tail[i-1]:
                latest_low_tail_value = np.minimum(df.loc[i-1, 'Low'], df.loc[i, 'Low'])
        
            # If High Tail and there is a last low value, use it.
            if high_tail[i] and latest_low_tail_value is not None:
                df.loc[i, 'Low Line'] = latest_low_tail_value
        
            # If High Tail and there is no last low tail value, use the lowest low up to that point
            elif high_tail[i] and latest_low_tail_value is None:
                # Get the minimum Low value up to this point (from 0 to i)
                df.loc[i, 'Low Line'] = df.loc[:i, 'Low'].min()
        
            else:
                df.loc[i, 'Low Line'] = previous_low_line
        
            # Handle High Line updates (when a low tail happens)
            if high_tail[i-1]:
                latest_high_tail_value = np.maximum(df.loc[i-1, 'High'], df.loc[i, 'High'])
        
            # If Low Tail and there is a last high value, use it.
            if low_tail[i] and latest_high_tail_value is not None:
                df.loc[i, 'High Line'] = latest_high_tail_value
        
            # If Low Tail and there is no last high tail value, use the highest high up to that point
            elif low_tail[i] and latest_high_tail_value is None:
                # Get the maximum High value up to this point (from 0 to i)
                df.loc[i, 'High Line'] = df.loc[:i, 'High'].max()
        
            else:
                df.loc[i, 'High Line'] = previous_high_line


        df['Mid Line'] = (df['High Line'] + df['Low Line']) / 2

        self._bid_baseline = df['Low Line'].iloc[-1]
        self._ask_baseline = df['High Line'].iloc[-1]
        return df

    def call_trade_history(self, file_name='trades_XLM.csv'):
        '''Call your CSV of trade history in order to determine Breakevens, PnL, and other metrics'''
        
        # Start with default values
        last_net_value = 0
        prev_net_value = 0  # This tracks the previous net value for comparison

        # Specify the path to your CSV file
        csv_file_path = f'/home/tyler/hummingbot/hummingbot/data/{file_name}.csv'
        # Check if the CSV file exists
        if not os.path.isfile(csv_file_path):
            # Return zeros on the class variables
            self.b_be = 0
            self.s_be = 0
            self.pnl = 0
            self.n_v = 0
            # Return zeros if the file doesn't exist
            return 0, 0, 0, 0        # Read the CSV file into a Pandas DataFrame

        df = pd.read_csv(csv_file_path)


        # Variables to store trade cycle start point
        cycle_start_index = 0

        # Filter the DataFrame for BUY and SELL trades
        buy_trades = df[df['trade_type'] == 'BUY']
        sell_trades = df[df['trade_type'] == 'SELL']
        
        # Get the last traded price for BUY and SELL, or set to 0 if no trades exist
        self._last_buy_price = buy_trades['price'].iloc[-1] if not buy_trades.empty else 0
        self._last_sell_price = sell_trades['price'].iloc[-1] if not sell_trades.empty else 0


        # self._last_trade_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.MidPrice)

        # Iterate through the trade history in reverse order
        for index, row in df.iterrows():
            trade_type = row['trade_type']
            trade_price = row['price']
            trade_amount = row['amount']
            trade_fee = row['trade_fee_in_quote'] if 'trade_fee_in_quote' in row else 0

            # Update previous net value before calculating the new one
            prev_net_value = last_net_value

            # Calculate the new net value
            if trade_type == 'BUY':
                # print(f"Buy Trade # {index}")
                intermediate_buy_cost = trade_price * trade_amount + trade_fee
                # print(f"intermediate Buy Cost : {intermediate_buy_cost}")
                last_net_value += intermediate_buy_cost
                # print(f"NET {last_net_value}")
            elif trade_type == 'SELL':
                # print(f"Sell Trade # {index}")
                intermediate_sell_proceeds = trade_price * trade_amount - trade_fee
                # print(f"intermediate Sell Proceeds : -{intermediate_sell_proceeds}")
                last_net_value -= intermediate_sell_proceeds
                # print(f"NET {last_net_value}")

            # Detect crossover in net value (crossing zero)
            if (last_net_value <= 0 and prev_net_value > 0) or (last_net_value >= 0 and prev_net_value < 0):
                # new_trade_cycle = True
                cycle_start_index = index  # Update to the most recent crossover index
                # print(f"{cycle_start_index}=====================CROSS=============================")

        
        # print(f"Cycle Starting Index = {cycle_start_index}")
        # Filter out trades after the identified cycle start point
        if cycle_start_index == 0:
            filtered_df = df.iloc[cycle_start_index:]
        else:
            filtered_df = df.iloc[cycle_start_index + 1 :]

        # Filter out buy and sell trades
        buy_trades = filtered_df[filtered_df['trade_type'] == 'BUY']
        sell_trades = filtered_df[filtered_df['trade_type'] == 'SELL']

        # Calculate weighted sums with fees
        sum_of_buy_prices = (buy_trades['price'] * buy_trades['amount']).sum()
        sum_of_buy_amount = buy_trades['amount'].sum()
        sum_of_buy_fees = (buy_trades['trade_fee_in_quote']).sum() if 'trade_fee_in_quote' in buy_trades else 0

        sum_of_sell_prices = (sell_trades['price'] * sell_trades['amount']).sum()
        sum_of_sell_amount = sell_trades['amount'].sum()
        sum_of_sell_fees = (sell_trades['trade_fee_in_quote']).sum() if 'trade_fee_in_quote' in sell_trades else 0

        # Calculate the total buy cost after  fees
        # This isnt a price movement, but a comparison of sum amount.  
        # If I bought $100 worth and paid 0.50, then I paid a total of $100.50 after fees
        # If I sold $100 worth, but paid 0.50 to do so, then I only sold $99.5 after fees
        total_buy_cost = sum_of_buy_prices + sum_of_buy_fees

        # Calculate the total sell proceeds after fees
        total_sell_proceeds = sum_of_sell_prices - sum_of_sell_fees

        # Calculate net value in quote
        # Needed to change since the net value here used to calculate only based on the history of the current situation, 
        # It is now updated for the net of the entirety of the bot's history 
        net_value =  total_buy_cost - total_sell_proceeds


        # Calculate the breakeven prices
        breakeven_buy_price = total_buy_cost / sum_of_buy_amount if sum_of_buy_amount > 0 else 0
        # print(f"Total Buy Cost : {total_buy_cost} / sum_buys {sum_of_buy_amount}")
        
        breakeven_sell_price = total_sell_proceeds / sum_of_sell_amount if sum_of_sell_amount > 0 else 0
        # print(f"Total Sell Proceeds : {total_sell_proceeds} / sum_sells {sum_of_sell_amount}")

        # Calculate realized P&L: only include the amount of buys and sells that have balanced each other out
        balance_text = None
        if min(sum_of_buy_amount, sum_of_sell_amount) == sum_of_buy_amount:
            balance_text = "Unbalanced Sells (Quote)"
        elif min(sum_of_buy_amount, sum_of_sell_amount) == sum_of_sell_amount:
            balance_text = "Unbalanced Buys (Base)"
        else:
            balance_text = "Balanced"

        realized_pnl = min(sum_of_buy_amount, sum_of_sell_amount) * (breakeven_sell_price - breakeven_buy_price)

        # # Calculate Unrealized PnL (for the remaining open position)
        open_position_size = Decimal(abs(float(sum_of_buy_amount) - float(sum_of_sell_amount)))

        if open_position_size > 0:
            vwap_bid = self.connectors[self.exchange].get_vwap_for_volume(self.trading_pair,
                                                    False,
                                                    open_position_size).result_price

            vwap_ask = self.connectors[self.exchange].get_vwap_for_volume(self.trading_pair,
                                                    True,
                                                    open_position_size).result_price

            # Unrealized PnL is based on the current midprice and the breakeven of the open position
            if sum_of_buy_amount > sum_of_sell_amount:
                unrealized_pnl = open_position_size * (Decimal(vwap_bid) - Decimal(breakeven_buy_price))
            elif sum_of_buy_amount < sum_of_sell_amount:
                unrealized_pnl = open_position_size * (Decimal(breakeven_sell_price) - Decimal(vwap_ask))
            else:
                unrealized_pnl = 0

            self.u_pnl = unrealized_pnl
        else:
            self.u_pnl = 0

        ## Update Global Values
        self.b_be = breakeven_buy_price
        self.s_be = breakeven_sell_price
        self.pnl = realized_pnl
        self.n_v = net_value

        return breakeven_buy_price, breakeven_sell_price, realized_pnl, net_value



    def on_tick(self):
        #Calculate garch every so many seconds
        if self.create_garch_timestamp <= self.current_timestamp:
                ### Call Historical Calculations
                kraken_api = KrakenAPI(self.history_market)
                df = kraken_api.call_kraken_ohlc_data(720, 'XXLMZEUR',  60)    
                ohlc_calc_df = self.get_ohlc_calculations(df)

                #msg_gv = (f"GARCH Volatility {garch_volatility:.8f}")
                #self.log_with_clock(logging.INFO, msg_gv)
                self.target_profitability = max(self.min_profitability, self.current_vola)
                self.create_garch_timestamp = self.garch_refresh_time + self.current_timestamp

        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()

            proposal: List[OrderCandidate] = self.create_proposal()
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp

            

        
        # Update the timestamp model 
        if self.current_timestamp - self.last_time_reported > self.report_interval:
            self.last_time_reported = self.current_timestamp

        




    def create_proposal(self) -> List[OrderCandidate]:
        time.sleep(10)
        bp, sp = self.determine_log_multipliers()
        # Fetch balances and optimal bid/ask prices
        _, _, _, _, _, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        optimal_bid_price, optimal_ask_price, order_size_bid, order_size_ask, bid_reservation_price, ask_reservation_price, optimal_bid_percent, optimal_ask_percent = self.optimal_bid_ask_spread()

        # Save Values for Status use without recalculating them over and over again
        self.bid_percent = optimal_bid_percent
        self.ask_percent = optimal_ask_percent
        self.b_r_p = bid_reservation_price
        self.a_r_p = ask_reservation_price

        # Initial prices
        buy_price = optimal_bid_price 
        sell_price = optimal_ask_price

        # Number of levels to create (customizable)
        num_levels = 1  # e.g., 3 buy and 3 sell levels
        
        # Multiplier values for buy and sell price adjustments
        buy_multiplier = bp  # Reduce buy price by bp% ## Decimal(0.96) #
        sell_multiplier =  sp  # Increase sell price by sp% ## Decimal(1.04)
        
        # Store orders
        order_counter = []

        # Loop through each level
        for level in range(num_levels):
            # Adjust buy price and create buy order
            if buy_price <= bid_reservation_price:
                # Calculate adjusted order size to keep the same dollar value
                adjusted_order_size_bid = order_size_bid * (optimal_bid_price / buy_price)
                buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                        order_side=TradeType.BUY, amount=Decimal(adjusted_order_size_bid), price=buy_price)
                if adjusted_order_size_bid >= self.min_order_size_bid:
                    order_counter.append(buy_order)
                else:
                    msg = (f" order_size_bid |{adjusted_order_size_bid}| below minimum_size for bid order |{self.min_order_size_bid}| ")
                    self.log_with_clock(logging.INFO, msg)
            
            # Adjust sell price and create sell order
            if sell_price >= ask_reservation_price:
                # Calculate adjusted order size to keep the same dollar value
                adjusted_order_size_ask = order_size_ask * (optimal_ask_price / sell_price)
                sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                            order_side=TradeType.SELL, amount=Decimal(adjusted_order_size_ask), price=sell_price)
                if adjusted_order_size_ask >= self.min_order_size_ask:
                    order_counter.append(sell_order)
                else:
                    msg = (f" order_size_ask |{adjusted_order_size_ask}| below minimum_size for ask order |{self.min_order_size_ask}| ")
                    self.log_with_clock(logging.INFO, msg)
            
            # Update prices for the next level
            buy_price *= buy_multiplier
            sell_price *= sell_multiplier

        return order_counter

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)
        return proposal_adjusted

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                      order_type=order.order_type, price=order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                     order_type=order.order_type, price=order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)


    def did_fill_order(self, event: OrderFilledEvent):
        t, y_bid, y_ask, bid_volatility_in_base, ask_volatility_in_base, bid_reservation_price, ask_reservation_price = self.reservation_price()


        self.initialize_flag = False

        # Update Trade CSV after a trade completes
        breakeven_buy_price, breakeven_sell_price, realized_pnl, net_value = self.call_trade_history('trades_XLM')



        self.fee_percent = Decimal(self.fee_percent)
        
        # Print log
        msg = (f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} {self.exchange} at {round(event.price, 2)}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

        time.sleep(10)


    ##########################
    ###====== Status Screen
    ###########################

    def format_status(self) -> str:
        """Returns status of the current strategy on user balances and current active orders. This function is called
        when status command is issued. Override this function to create custom status display output.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []
        warning_lines = []
        warning_lines.extend(self.network_warning(self.get_market_trading_pair_tuples()))

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        lines.extend(["", f"Direction :: {self.trade_position_text} "])

        lines.extend(["", "| Inventory Imbalance | Trade History |"])
        lines.extend([f"q(d%) :: {self.q_imbalance:.8f} | Inventory Difference :: {self.inventory_diff:.8f}"])
        lines.extend([f"R_PnL (Quote) :: {self.pnl:.8f} | U_PnL (Quote) :: {self.u_pnl:.8f} | Net Quote Value :: {self.n_v:.8f}"])


        lines.extend(["", "| Reservation Prices | Baselines | Breakevens | Profit Targets |"])
        lines.extend([f"RP /: Ask :: {self.a_r_p:.8f} | | Bid :: {self.b_r_p:.8f}"])
        lines.extend([f"LT /: Ask :: {self._last_sell_price:.8f} || Bid :: {self._last_buy_price:.8f}"])
        lines.extend([f"Bl /: Ask :: {self._ask_baseline} | Bid :: {self._bid_baseline}"])
        lines.extend([f"BE /: Ask :: {self.s_be} | Bid :: {self.b_be}"])
        lines.extend([f"PT /: Ask(%) :: {self.ask_percent:.4f} | Bid(%) :: {self.bid_percent:.4f}"])


        lines.extend(["", "| Market Depth |"])
        lines.extend([f"Ask :: {self.a_d:.8f} | Bid :: {self.b_d:.8f}"])


        lines.extend(["", "| Volatility Measurements |"])
        lines.extend([f"Current Volatility(d%) :: {self.current_vola:.8f} | Volatility Rank :: {self.volatility_rank:.8f}"])

        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])



        warning_lines.extend(self.balance_warning(self.get_market_trading_pair_tuples()))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)

        return "\n".join(lines)

    def determine_log_multipliers(self):
        """Determine the best placement of percentages based on the percentage/log values 
        (log(d)) / (log(p)) = n, breakding this down with a fixed n to solve for p value turns into  p = d**(1/n).  Or closer p = e^(ln(d) / n)
        Match against minimum profit wanted.  Allows us to determine an approximate exhaustion drop percent.
        
        Multiply against breakevens to determine profit target levels (With fees included) that account for multiplicative/log aspects of charts."""

        ## If doing a 50/50 it would be /2 since each side is trading equally
        ## If I am doing a single side (QFL), then the maximum orders should account for only the buy side entry. 
        # n = math.floor(self.maximum_orders/2)

        ### Trades into the more volatile markets should be deeper to account for this
        ## for example, buying illiquid or low volume coins(more volatile than FIAT) should be harder to do than selling/ (profiting) from the trade. 

        n = self.maximum_orders
        # n = math.floor(self.maximum_orders/2)

        ## Buys
        #Minimum Distance in percent. 0.01 = a drop of 99% from original value
        bd = 1 / 30
        bp = math.exp(math.log(bd)/n)
        
        bp = np.minimum(1 - self.min_profitability, bp)

        ## Include Fees
        bp = Decimal(bp)  * (Decimal(1.0) - Decimal(self.fee_percent))
        
        ## Sells
        ## 3 distance move,(distance starts at 1 or 100%) 200% above 100 %
        sd = 30
        sp = math.exp(math.log(sd)/n)

        sp = np.maximum(1 + self.min_profitability, sp)

        ## Include Fees
        sp = Decimal(sp) * (Decimal(1.0) + Decimal(self.fee_percent))


        #Decimalize for later use

        # msg = (f"sp :: {sp:.8f} , bp :: {bp:.8f}")
        # self.log_with_clock(logging.INFO, msg)
        return bp, sp




            ### OLD METHOD
            ### Now using CSV for more precise trade information
            ######################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # def determine_log_breakeven_levels(self):
        # bp,sp = self.determine_log_multipliers()
        # buy_counter_adjusted = self.buy_counter - 1
        # sell_counter_adjusted = self.sell_counter - 1

        # additive_buy = 0
        # additive_sell = 0
        
        # avg_buy_mult = 1
        # avg_sell_mult = 1

        # buy_breakeven_mult = 1
        # sell_breakeven_mult = 1

        # #Average the trade distance percentages(this assumes an even volume on every trade, can implement volume in the future)
        # if buy_counter_adjusted > 0:
        #     for i in range(1, buy_counter_adjusted + 1):
        #         if i == 1 : # First trade has no initial price drop
        #             additive_buy = 1 + self.fee_percent
        #         elif i > 1 :   # Next trades decay log wise
        #             additive_buy += bp**(i-1) + self.fee_percent
        #     # Find the avg percent of all trades        
        #     avg_buy_mult = (additive_buy) / (buy_counter_adjusted)
        #     # Divide the average price by the lowest price to get your multiplier for breakeven
        #     buy_breakeven_mult = avg_buy_mult / (bp**buy_counter_adjusted)
        # else:
        #     additive_buy = 0
        #     avg_buy_mult = 1
        #     buy_breakeven_mult = 1

        # if sell_counter_adjusted > 0:
        #     for i in range(1, sell_counter_adjusted + 1):
        #         if i == 1: # First trade has no initial price drop
        #             additive_sell = 1 - self.fee_percent
        #         elif i > 1:  # Next trades decay log wise
        #             additive_sell += sp**(i-1) - self.fee_percent
        #     # Find the avg percent of all trades        
        #     avg_sell_mult = additive_sell / sell_counter_adjusted
        #     # Divide the average price by the highest price to get your multiplier for breakeven
        #     sell_breakeven_mult = avg_sell_mult / (sp**sell_counter_adjusted)  

        # else:
        #     additive_sell = 0
        #     avg_sell_mult = 1
        #     sell_breakeven_mult = 1


        # return buy_breakeven_mult, sell_breakeven_mult
        

    def get_current_top_bid_ask(self):
        ''' Find the current spread bid and ask prices'''
        top_bid_price = self.connectors[self.exchange].get_price(self.trading_pair, False)
        top_ask_price = self.connectors[self.exchange].get_price(self.trading_pair, True) 
        return top_bid_price, top_ask_price
    
    def get_vwap_bid_ask(self):
        '''Find the bid/ask VWAP of a set price for market depth positioning.'''
        q, _, _, _,_, _, _ = self.get_current_positions()


        # Call the method (Market Buy into ask, Sell into bid)
        bid_volume_cdf_value = Decimal(self.sold_volume_depth) #Decimal(sell_trades_instance.get_volume_cdf(target_percentile, window_size))
        ask_volume_cdf_value = Decimal(self.bought_volume_depth) #Decimal(buy_trades_instance.get_volume_cdf(target_percentile, window_size))


        bid_depth_difference = abs(bid_volume_cdf_value )
        ask_depth_difference = abs(ask_volume_cdf_value )
        
        # Determine the strength ( size ) of volume by how much you want to balance
        if q > 0:
            bid_depth = bid_volume_cdf_value
            ask_depth = max(self.min_order_size_bid, ask_volume_cdf_value) 
        elif q < 0:
            bid_depth = max(self.min_order_size_ask, bid_volume_cdf_value ) 
            ask_depth = ask_volume_cdf_value
        else:
            bid_depth = bid_volume_cdf_value
            ask_depth = ask_volume_cdf_value

        self.b_d = bid_depth
        self.a_d = ask_depth
        # msg_q = (f"bid_depth :: {bid_depth:8f}% :: ask_depth :: {ask_depth:8f}")
        # self.log_with_clock(logging.INFO, msg_q)

        vwap_bid = self.connectors[self.exchange].get_vwap_for_volume(self.trading_pair,
                                                False,
                                                bid_depth).result_price

        vwap_ask = self.connectors[self.exchange].get_vwap_for_volume(self.trading_pair,
                                                True,
                                                ask_depth).result_price
        return  vwap_bid, vwap_ask

    def get_current_positions(self):

        top_bid_price, top_ask_price = self.get_current_top_bid_ask()

        # adjust to hold 0.5% of balance in base. Over time with profitable trades, this will hold a portion of profits in coin: 
        amount_base_to_hold = Decimal(0.005)
        amount_base_rate = Decimal(1.0) - amount_base_to_hold
        
        amount_quote_to_hold = Decimal(0.005)
        amount_quote_rate = Decimal(1.0) - amount_quote_to_hold
        

        # Get currently held balances in each asset base
        maker_base_balance = (self.connectors[self.exchange].get_balance(self.base_asset) * amount_base_rate)
        maker_quote_balance = (self.connectors[self.exchange].get_balance(self.quote_asset) * amount_quote_rate)
        
        #Convert to Quote asset at best sell into bid price
        quote_balance_in_base = maker_quote_balance / top_ask_price

        # Get the total balance in base
        total_balance_in_base = quote_balance_in_base + maker_base_balance


        maximum_number_of_orders = self.maximum_orders    

        if total_balance_in_base == 0:
            # Handle division by zero
            return 0, 0, 0, 0
        ### For Entry Size to have /10 (/2 for) orders on each side of the bid/ask
        ### In terms of Maker Base asset
        entry_size_by_percentage = (total_balance_in_base * self.inv_target_percent) / maximum_number_of_orders 
        # minimum_size = max(self.connectors[self.exchange].quantize_order_amount(self.trading_pair, self.order_amount), entry_size_by_percentage)



        ## Q relation in percent relative terms, later it is in base(abolute)terms
        target_inventory = total_balance_in_base * self.inv_target_percent
        # Inventory Deviation, base inventory - target inventory. 
        inventory_difference = maker_base_balance  - target_inventory
        q = (inventory_difference) / total_balance_in_base
        q = Decimal(q)

        self.q_imbalance = q
        self.inventory_diff = inventory_difference


        # Total Abs(imbalance)
        total_imbalance = abs(inventory_difference)
        
        # Adjust base and quote balancing volumes based on shape factor and entry size by percentage
        # This method reduces the size of the orders which are overbalanced
        #if I have too much base, more base purchases are made small
        #if I have too much quote, more quote purchases are made small
        #When there is too much of one side, it makes the smaller side easier to trade in bid/ask, so 
        #having more orders of the unbalanced side while allowing price go to lower decreases it's loss
        #to market overcorrection
        if q > 0 :
            #If base is overbought, I want to sell more Quote to balance it
            base_balancing_volume =  total_imbalance ##abs(minimum_size) *  Decimal.exp(self.order_shape_factor * q)
            quote_balancing_volume =  max ( self.min_order_size_bid, abs(self.min_order_size_bid) * Decimal.exp(-self.order_shape_factor * q) )


        elif q < 0 :
            base_balancing_volume = max( self.min_order_size_ask, abs(self.min_order_size_ask) *  Decimal.exp(-self.order_shape_factor * q))
            quote_balancing_volume = total_imbalance ##abs(minimum_size) * Decimal.exp(self.order_shape_factor * q) 


         
        else :
            ## Adjust this logic just for one sided entries :: if you are completely sold out, then you should not have the capability to sell in the first place. 
            base_balancing_volume = self.min_order_size_bid
            quote_balancing_volume = self.min_order_size_ask



        
        base_balancing_volume = Decimal(base_balancing_volume)
        quote_balancing_volume = Decimal(quote_balancing_volume)
        #Return values
        return q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,  entry_size_by_percentage, maker_base_balance, quote_balance_in_base
    
    def percentage_order_size(self, bid_op, ask_op):
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        
        self.min_order_size_bid = max(self.quote_order_amount /  bid_op , self.order_amount)
        self.min_order_size_ask = max(self.quote_order_amount / ask_op , self.order_amount)

        self.min_order_size_bid = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, self.min_order_size_bid)
        self.min_order_size_ask = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, self.min_order_size_ask)

        order_size_bid = max(quote_balancing_volume, self.min_order_size_bid)
        order_size_ask = max(base_balancing_volume, self.min_order_size_ask)
        
        # msg_debug = (f"self.min_order_size_bid{self.min_order_size_bid} | self.min_order_size_ask{self.min_order_size_ask} ")
        # self.log_with_clock(logging.INFO, msg_debug) 

        # msg_debug = (f"quote_balancing_volume{quote_balancing_volume} |base_balancing_volume {base_balancing_volume} ")
        # self.log_with_clock(logging.INFO, msg_debug) 

        # msg_debug = (f"order_size_bid {order_size_bid} |order_size_ask{order_size_ask} ")
        # self.log_with_clock(logging.INFO, msg_debug) 

        # if quote_balancing_volume < self.min_order_size_bid  :
        #     order_size_bid = quote_balancing_volume
        #     msg_b = (f"Order Size Bid is too small for trade {order_size_bid:8f}")
        #     self.log_with_clock(logging.INFO, msg_b) 
        if quote_balance_in_base < self.min_order_size_bid:
            order_size_bid = quote_balancing_volume

            msg_b = (f"Not Enough Balance for bid trade {quote_balance_in_base:8f}")
            self.log_with_clock(logging.INFO, msg_b) 
        else:
            order_size_bid = max(quote_balancing_volume , self.min_order_size_bid )



        # if base_balancing_volume < self.min_order_size_ask  :
        #     order_size_ask = base_balancing_volume
        #     msg_a = (f"Order Size Ask is too small for trade {order_size_ask:8f}")
        #     self.log_with_clock(logging.INFO, msg_a)  
        if  maker_base_balance < self.min_order_size_ask:
            order_size_ask = base_balancing_volume

            msg_a = (f"Not Enough Balance for ask trade {maker_base_balance:8f}")
            self.log_with_clock(logging.INFO, msg_a)  
        else:
            order_size_ask = max(base_balancing_volume , self.min_order_size_ask )

        order_size_bid = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, order_size_bid)
        order_size_ask = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, order_size_ask)


        return order_size_bid, order_size_ask
    
    # def get_midprice(self):
    #     sold_baseline, bought_baseline, log_returns_list, self.bought_volume_depth, self.sold_volume_depth = call_kraken_data()

    #     if self._last_trade_price == None :
    #         if self.initialize_flag == True:
    #             # Fetch midprice only during initialization
    #             if self._last_trade_price is None:

    #                 ## If I have to manually restart the bot mid trade, this is the last traded price. 
    #                 manual_price = 0.087009
                    
    #                 #self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.MidPrice)

    #                 # Ensure midprice is not None before converting and assigning
    #                 if manual_price is not None:
    #                     self._last_trade_price = (manual_price)

    #                 self.initialize_flag = False  # Set flag to prevent further updates with midprice

    #     else:
    #         self._last_trade_price = self._last_trade_price

    #     self._bid_baseline = (sold_baseline)
    #     self._ask_baseline = (bought_baseline)
    #     # msg_gv = (f"self._bid_baseline  { self._bid_baseline}, self._ask_baseline  { self._ask_baseline}")
    #     # self.log_with_clock(logging.INFO, msg_gv)
    #     return self._last_trade_price

    # def call_garch_model(self):
    #     sold_baseline, bought_baseline, log_returns_list, self.bought_volume_depth, self.sold_volume_depth = call_kraken_data()
    #     self._bid_baseline = (sold_baseline)
    #     self._ask_baseline = (bought_baseline)

    #     # Retrieve the log returns from the DataFrame
    #     log_returns = log_returns_list##self.log_returns

    #     # Ensure log_returns is a one-dimensional pd.Series
    #     if isinstance(log_returns, list):
    #         log_returns = pd.Series(log_returns)

    #     # Convert to numeric, forcing any errors to NaN
    #     log_returns_numeric = pd.to_numeric(log_returns, errors='coerce')

    #     # Remove any NaN or infinite values from log_returns
    #     log_returns_clean = log_returns_numeric.replace([np.inf, -np.inf], np.nan).dropna()


    #     # Fit GARCH model to log returns
    #     current_variance = []
    #     current_volatility = []
    #     length = 0
    #     # Define the GARCH model with automatic rescaling
    #     model = arch_model(log_returns_clean, vol='Garch', mean='constant', p=1, q=1, power=2.0, rescale=True)

    #     # Fit the model
    #     model_fit = model.fit(disp="off")
    #     msg_gv = (f"GARCH  { model_fit.summary()}")
    #     self.log_with_clock(logging.INFO, msg_gv)
        
    #     # Adjust Volatility to Decimal Percent values 
    #     volatility = model_fit.conditional_volatility / 100 

    #     length = len(model_fit.conditional_volatility)

    
    #     # Convert current_volatility to a pandas Series to apply rolling
    #     current_volatility_series = pd.Series(volatility)

    #     # Define the rolling window size (square root of length)
    #     window = int(np.round(np.sqrt(len(current_volatility_series))))

    #     # Apply the rolling window to smooth volatility
    #     rolling_volatility = current_volatility_series#.rolling(window=window).mean()

    #     # Rank the Volatility
    #     self.max_vola = rolling_volatility.max()
    #     min_vola = rolling_volatility.min()
    #     self.current_vola = current_volatility_series.iloc[-1]

    #     # Prevent division by zero
    #     if self.max_vola != min_vola:
    #         self.volatility_rank = (self.current_vola - min_vola) / (self.max_vola - min_vola)
    #     else:
    #         self.volatility_rank = 1  # Handle constant volatility case

    #     # msg = (f"Volatility :: Rank:{self.volatility_rank}, Max:{self.max_vola}, Min:{min_vola}, Current:{self.current_vola}")
    #     # self.log_with_clock(logging.INFO, msg)            



    def reservation_price(self):
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        
        #self._last_trade_price = self.get_midprice()

        breakeven_buy_price, breakeven_sell_price, realized_pnl, net_value = self.call_trade_history('trades_XLM')


        # msg_4 = (f"breakeven_buy_price @ {breakeven_buy_price:.8f} ::: breakeven_sell_price @ {breakeven_sell_price:.8f}, realized_pnl :: {realized_pnl:.8f}, net_value :: {net_value:.8f}")
        # self.log_with_clock(logging.INFO, msg_4)

        TWO = Decimal(2.0)
        HALF = Decimal(0.5)

        s_bid = self._bid_baseline
        s_ask = self._ask_baseline

        is_buy_data = breakeven_buy_price > 0
        is_sell_data = breakeven_sell_price > 0

        is_buy_net = net_value > 0
        is_sell_net = net_value < 0
        is_neutral_net = net_value == 0 


        # Adjust Breakeven for 2nd half of fees (Move BE bid up, Move BE ask down the opposite side fee amount)
        breakeven_buy_price =  Decimal(breakeven_buy_price) * (Decimal(1.0) + Decimal(self.fee_percent))
        breakeven_sell_price = Decimal(breakeven_sell_price) * (Decimal(1.0) - Decimal(self.fee_percent))
        
        # Quantize Price
        breakeven_buy_price = self.connectors[self.exchange].quantize_order_price(self.trading_pair, breakeven_buy_price)
        breakeven_sell_price = self.connectors[self.exchange].quantize_order_price(self.trading_pair, breakeven_sell_price)

         # There is no data, Use baselines
        if not is_buy_data and not is_sell_data:
            self.trade_position_text = "No Trades, Use Baseline"
            s_bid = self._bid_baseline
            s_ask = self._ask_baseline
        
        # You have started a Buy Cycle, use Bid BE
        elif is_buy_data and not is_sell_data:
            self.trade_position_text = "Buy Cycle"
            s_bid = self._last_buy_price # breakeven_buy_price
            s_ask = breakeven_buy_price
        
        # You have started a Sell Cycle, use Ask BE
        elif not is_buy_data and is_sell_data:
            self.trade_position_text = "Sell Cycle"
            s_bid = breakeven_sell_price
            s_ask = self._last_sell_price # breakeven_sell_price

        # You are mid trade, use net values to determine locations
        elif is_buy_data and is_sell_data:
            if is_buy_net: # Mid Buy Trade, Buy Below BE, Sell for profit
                self.trade_position_text = "Unfinished Buy Cycle"
                s_bid = self._last_buy_price
                s_ask = breakeven_buy_price
            elif is_sell_net: # Mid Sell Trade, Sell Above BE, Buy for profit
                self.trade_position_text = "Unfinished Sell Cycle"
                s_bid = breakeven_sell_price
                s_ask = self._last_sell_price
            elif is_neutral_net: # Price is perfectly neutral, use prospective levels
                self.trade_position_text = "Neutral Cycle"
                s_bid = self._last_buy_price # breakeven_buy_price
                s_ask = self._last_sell_price # breakeven_sell_price


        ## Convert to Decimal
        s_bid = Decimal(s_bid)
        s_ask = Decimal(s_ask)
        # Quantize price
        s_bid = self.connectors[self.exchange].quantize_order_price(self.trading_pair, s_bid)
        s_ask = self.connectors[self.exchange].quantize_order_price(self.trading_pair, s_ask)

            # It doesn't make sense to use mid_price_variance because its units would be absolute price units ^2, yet that side of the equation is subtracted
            # from the actual mid price of the asset in absolute price units
            # gamma / risk_factor gains a meaning of a fraction (percentage) of the volatility (standard deviation between ticks) to be subtraced from the
            # current mid price
            # This leads to normalization of the risk_factor and will guaranetee consistent behavior on all price ranges of the asset, and across assets

        ### Convert Volatility Percents into Absolute Prices

        max_bid_volatility= Decimal(self.current_vola) 
        bid_volatility_in_base = (max_bid_volatility) * s_bid

        max_ask_volatility = Decimal(self.current_vola) 
        ask_volatility_in_base = (max_ask_volatility) * s_ask


        # msg_4 = (f"max_bid_volatility @ {bid_volatility_in_base:.8f} ::: max_ask_volatility @ {ask_volatility_in_base:.8f}")
        # self.log_with_clock(logging.INFO, msg_4)

        #INVENTORY RISK parameter, 0 to 1, higher = more risk averse, as y <-- 0, it behaves more like usual
        # Adjust the width of the y parameter based on volatility, the more volatile , the wider the spread becomes, y goes higher
        y = Decimal(1.0)
        y_min = Decimal(0.000000001)
        y_max = Decimal(1.0)
        y_difference = Decimal(y_max - y_min)
        # konstant = Decimal(5)
        y_bid = y_min + (y_difference * Decimal(self.volatility_rank))  
        y_ask = y_min + (y_difference * Decimal(self.volatility_rank))  

        y_bid = min(y_bid,y_max)
        y_bid = max(y_bid,y_min)

        y_ask = min(y_ask,y_max)
        y_ask = max(y_ask,y_min)

        t = Decimal(1.0)

        #1 is replacement for time (T= 1 - t=0)
        bid_risk_rate = q * y_bid
        ask_risk_rate = q * y_ask
        bid_reservation_adjustment = bid_risk_rate * bid_volatility_in_base * t
        ask_reservation_adjustment = ask_risk_rate * ask_volatility_in_base * t

        bid_reservation_price = (s_bid) - (bid_reservation_adjustment) 
        bid_reservation_price = self.connectors[self.exchange].quantize_order_price(self.trading_pair, bid_reservation_price)

        ask_reservation_price = (s_ask) - (ask_reservation_adjustment)
        ask_reservation_price = self.connectors[self.exchange].quantize_order_price(self.trading_pair, ask_reservation_price)


        
        return  t, y_bid, y_ask, bid_volatility_in_base, ask_volatility_in_base, bid_reservation_price, ask_reservation_price




    def optimal_bid_ask_spread(self):
        t, y_bid, y_ask, bid_volatility_in_base, ask_volatility_in_base, bid_reservation_price, ask_reservation_price = self.reservation_price()
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        bp,sp = self.determine_log_multipliers()

        TWO = Decimal(2.0)
        HALF = Decimal(0.5)

        maximum_volatility = Decimal(np.maximum(self.max_vola , self.target_profitability))

        ###################################
        ## Calculate kappa k (similar to a market depth comparison with k_bid_size,k_ask_size)
        ######################################
        bid_maximum_spread_in_price = (TWO * Decimal(maximum_volatility) * bid_reservation_price)
        bid_maximum_spread_in_price = self.connectors[self.exchange].quantize_order_price(self.trading_pair, bid_maximum_spread_in_price)


        ask_maximum_spread_in_price = (TWO * Decimal(maximum_volatility) * ask_reservation_price)
        ask_maximum_spread_in_price = self.connectors[self.exchange].quantize_order_price(self.trading_pair, ask_maximum_spread_in_price)


        bid_inside_exp = ((bid_maximum_spread_in_price * y_bid) - (bid_volatility_in_base**TWO * y_bid**TWO)) / TWO
        bid_inside_exp = Decimal(bid_inside_exp).exp()
        if bid_inside_exp == 1 :
            bid_inside_exp = Decimal(0.99999999)

        k_bid_size = y_bid / (bid_inside_exp - Decimal(1))
        k_bid_size = Decimal(k_bid_size)

        bid_k_division = y_bid / k_bid_size
        bid_k_division = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, bid_k_division)

        bid_spread_rate = 1 +  (bid_k_division)
        bid_spread_rate = Decimal(bid_spread_rate)
        bid_log_term = Decimal.ln(bid_spread_rate)



        ask_inside_exp = ((ask_maximum_spread_in_price * y_ask) - (ask_volatility_in_base**TWO * y_ask**TWO)) / TWO
        ask_inside_exp = Decimal(ask_inside_exp).exp()
        if ask_inside_exp == 1 :
            ask_inside_exp = Decimal(0.99999999)

        k_ask_size = y_ask / (ask_inside_exp - Decimal(1))
        k_ask_size = Decimal(k_ask_size)

        ask_k_division = y_ask / k_ask_size
        ask_k_division = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, ask_k_division)

        ask_spread_rate = 1 +  (ask_k_division)
        ask_spread_rate = Decimal(ask_spread_rate)
        ask_log_term = Decimal.ln(ask_spread_rate)  

        # msg_1 = (f"k_bid Depth Volume @ {k_bid_size:.8f} ::: k_ask Depth Volume @ {k_ask_size:.8f}")
        # self.log_with_clock(logging.INFO, msg_1)



        optimal_bid_spread = (y_bid * (Decimal(1) * bid_volatility_in_base) * t) + ((TWO  * bid_log_term) / y_bid)
        optimal_ask_spread = (y_ask * (Decimal(1) * ask_volatility_in_base) * t) + ((TWO  * ask_log_term) / y_ask)

    
        ## Optimal Spread in comparison to the min profit wanted
        min_profit_bid =  bid_reservation_price * bp
        min_profit_ask = ask_reservation_price * sp

        # Spread calculation price vs the minimum profit price for entries
        optimal_bid_price = min_profit_bid # np.minimum(bid_reservation_price - (optimal_bid_spread  / TWO), min_profit_bid)
        optimal_ask_price = min_profit_ask # np.maximum(ask_reservation_price + (optimal_ask_spread / TWO), min_profit_ask)

        ## Market Depth Check to allow for hiding further in the orderbook by the volume vwap
        top_bid_price, top_ask_price = self.get_current_top_bid_ask()
        vwap_bid, vwap_ask = self.get_vwap_bid_ask()

        deepest_bid = min(vwap_bid, top_bid_price)
        deepest_ask = max(vwap_ask, top_ask_price)


        # Calculate the quantum for both bid and ask prices (Convert to chart price decimals)
        bid_price_quantum = self.connectors[self.exchange].get_order_price_quantum(
            self.trading_pair,
            top_bid_price
        )
        ask_price_quantum = self.connectors[self.exchange].get_order_price_quantum(
            self.trading_pair,
            top_ask_price
        )

        # Calculate the price just above the top bid and just below the top ask (Allow bot to place at widest possible spread)
        price_above_bid = (ceil(top_bid_price / bid_price_quantum) + 1) * bid_price_quantum
        price_below_ask = (floor(top_ask_price / ask_price_quantum) - 1) * ask_price_quantum

        if q > 0:
            optimal_bid_price = min( optimal_bid_price, price_above_bid)
            optimal_ask_price = max( optimal_ask_price, price_below_ask)
        if q < 0:
            optimal_bid_price = min( optimal_bid_price, price_above_bid)
            optimal_ask_price = max( optimal_ask_price, price_below_ask)
        if q == 0:
            optimal_bid_price = min( optimal_bid_price, price_above_bid)
            optimal_ask_price = max( optimal_ask_price, price_below_ask)


        if optimal_bid_price <= 0 :
            msg_2 = (f"Error ::: Optimal Bid Price @ {optimal_bid_price} below 0.")
            self.log_with_clock(logging.INFO, msg_2)



        # Apply quantum adjustments for final prices
        optimal_bid_price = (floor(optimal_bid_price / bid_price_quantum)) * bid_price_quantum
        optimal_ask_price = (ceil(optimal_ask_price / ask_price_quantum)) * ask_price_quantum

        optimal_bid_percent = ((bid_reservation_price - optimal_bid_price) / bid_reservation_price) * 100
        optimal_ask_percent = ((optimal_ask_price - ask_reservation_price) / ask_reservation_price) * 100

        order_size_bid, order_size_ask = self.percentage_order_size(optimal_bid_price, optimal_ask_price)

        
        return optimal_bid_price, optimal_ask_price, order_size_bid, order_size_ask, bid_reservation_price, ask_reservation_price, optimal_bid_percent, optimal_ask_percent