import logging
import math 
from math import floor, ceil

import time
import requests
import json
from decimal import Decimal
from typing import List
from random import gauss, seed

import datetime
import datetime as dt

import pandas as pd
import numpy as np
from scipy.stats import norm, poisson, stats
from scipy.optimize import minimize_scalar
from scipy.signal import argrelextrema

from py_vollib_vectorized import vectorized_implied_volatility as implied_vol
from arch import arch_model



from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.connector.budget_checker import BudgetChecker


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
# from Kraken_Calculations import BuyTrades, SellTrades

########## Profiling example to find time/speed of code

# import cProfile
# import pstats
# import io

class KrakenAPI:
    def __init__(self, symbol, start_timestamp, end_timestamp=None):
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
                print(f"No data found or error in response for symbol: {self.symbol}")
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
                print("No more data to fetch.")
                break

            self.data.extend(trades)
            self.last_timestamp = last_timestamp

            # Stop if last timestamp exceeds end timestamp or if no new data is returned
            if  self.last_timestamp >= self.end_timestamp:
                print("Reached the end timestamp.")
                break

            if len(trades) == 1:
                print("No more trades.")
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
                'pair': self.symbol,  # Asset pair (e.g., 'XBTUSD' for Bitcoin/USD)
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
                
                print(f"Data Saved. Last Timestamp: {last_timestamp}")
            
                return True, trades, last_timestamp
            else:
                print(f"No data found or error in response for symbol: {self.symbol}")
                return False, [], self.last_timestamp
    
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return False, [], self.last_timestamp

    def get_ohlc_since(self, interval):
        initial_start_timestamp = self.start_timestamp  # Store the initial start timestamp
        while True:
            success, trades, last_timestamp = self.fetch_kraken_ohlc_data(self.last_timestamp, interval)
            if not success or not trades:
                print("No more data to fetch.")
                break

            self.data.extend(trades)
            self.last_timestamp = last_timestamp

            # Stop if last timestamp exceeds end timestamp or if no new data is returned
            if  self.last_timestamp >= self.end_timestamp:
                print("Reached the end timestamp.")
                break

            if len(trades) == 1:
                print("No more trades.")
                break

            # # Limit the loop to avoid excessive requests
            # if len(self.data) > 100000:  # Example limit
            #     print("Data limit reached.")
            #     break

            # Rate limit to avoid hitting API too hard
            time.sleep(1)

        return self.data

# csv_file_path = f'/home/tyler/hummingbot/hummingbot/data/KrakenData/{file_name}.csv'
class KRAKENQFL():
    def __init__(self, filepath, symbol, interval, volatility_periods, rolling_periods):
        self.filepath = f'/home/tyler/hummingbot/hummingbot/data/KrakenData/{filepath}'
        self.symbol = symbol
        self.base_url = 'https://api.kraken.com/0/public/Trades'
        self.data = []
        self.volatility_periods = volatility_periods
        self.rolling_periods = rolling_periods
        self.interval = interval

    def call_csv_history(self):
        """
        Opens a CSV file in OHLCVT format and saves the data to a pandas DataFrame.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            pandas.DataFrame: The DataFrame containing the data from the CSV file.
        """

        try:
            # Read the CSV file using pandas.read_csv, specifying the header as the first row
            df = pd.read_csv(self.filepath, header=0)
        
            # If the first row contains column names, use them. Otherwise, create default column names.
            if df.columns[0].isdigit():
                # Create default column names (adjust as needed)
                df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades']
        
            # Save the data to the class attribute (optional)
            self.data = df
        
            return df
    
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
            return None

    def get_ohlc_calculations(self, df, fitted_params=None):
        """ Run calculations for traing bot information"""

        # Set the number of rows to keep (1440 as default or length of dataframe if it's shorter)
        slice_length = np.minimum(8760, len(df))
        
        # Slice the dataframe to keep only the last 'slice_length' rows
        df = df.iloc[-slice_length:].copy()
        
        # Reset index after slicing
        df = df.reset_index(drop=True)

        
        df['Open'] = pd.to_numeric(df['Open'])
        df['High'] = pd.to_numeric(df['High'])
        df['Low'] =pd.to_numeric(df['Low'])
        source = ( df['High'] + df['Low']) / 2 # pd.to_numeric(df['Close'])# 
        df['Source'] = source
        df['Close'] = pd.to_numeric(df['Close'])
        close = df['Close']
        log_returns = np.log(source/np.roll(source,shift=1))[1:].dropna()
    
        log_returns_series = pd.Series(log_returns)
        # df['LR_High'] = log_returns.rolling(720).quantile(0.8413)
        # df['LR_Low'] = log_returns.rolling(720).quantile(0.1587)
    
        if fitted_params is not None:
            # Fit GARCH on higher range data with initial values from shorter-range model
            model = arch_model(log_returns_series, vol='EGARCH', mean='HARX', p=3, q=3, rescale=True, dist="StudentsT")
            model_fit = model.fit(starting_values=fitted_params)
            print(model_fit.summary())
        else:
            # Define the GARCH model with automatic rescaling
            model = arch_model(log_returns_series, vol='EGARCH', mean='HARX', p=3, q=3, rescale=True, dist="StudentsT")
            # Fit the model
            model_fit = model.fit(disp="off")
    
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
        secondary_model = arch_model(secondary_log_returns, vol='EGARCH', mean='HARX', p=1, q=1, rescale=True, dist="StudentsT")
    
        # Fit the model
        secondary_model_fit = secondary_model.fit(disp="off")
    
        # Extract conditional volatility
        secondary_volatility_rescaled = secondary_model_fit.conditional_volatility
        
        # Convert the percent to decimal percent
        secondary_volatility = secondary_volatility_rescaled / 100 
    
        #Append values to df
        df['Mean'] = source.dropna().mean()
        df['Log Returns'] = log_returns_series.dropna()
        df['Secondary Volatility'] = secondary_volatility.dropna()
        df['Standard Residuals'] = std_residuals.dropna()
    

        window_length = max(1, min(len(df), self.volatility_periods))
        # df['Volatility'] = df['Close'].pct_change().rolling(window=2).std().dropna() #  volatility.dropna() # 
        df['Volatility'] = volatility.dropna() # 

        df['Rolling Volatility'] = df['Volatility'].rolling(window=self.rolling_periods).mean()
        # # Filter out the first self.volatility_periods rows from the volatility data
        # filtered_volatility = df.iloc[self.volatility_periods:].dropna(subset=['Volatility'])
        
        # if len(df) >= 2 * self.volatility_periods:
        #     # Use full rolling window quantiles for larger datasets
        #     IQR3_vola = df['Volatility'].rolling(window=window_length).quantile(0.8413)
        #     vola_median = df['Volatility'].rolling(window=window_length).quantile(0.50)
        #     IQR1_vola = df['Volatility'].rolling(window=window_length).quantile(0.1587)
        # else:
        #     # Use filtered volatility values for smaller datasets
        #     IQR3_vola = filtered_volatility['Volatility'].quantile(0.8413)
        #     vola_median = filtered_volatility['Volatility'].quantile(0.50)
        #     IQR1_vola = filtered_volatility['Volatility'].quantile(0.1587)

        # Entilre Dataset Method ** Perhaps more accurate to true distribution
        IQR3_vola = df['Volatility'].rolling(window= self.volatility_periods).quantile(0.8413)
        vola_median = df['Volatility'].rolling(window= self.volatility_periods).quantile(0.50)
        IQR1_vola = df['Volatility'].rolling(window= self.volatility_periods).quantile(0.1587)
            
        IQR3_Source = df['High'].rolling(window=self.rolling_periods).quantile(0.8413)
        IQR1_Source = df['Low'].rolling(window=self.rolling_periods).quantile(0.1587)    
        
        # # Assign these values to the entire DataFrame in new columns
        # df['Local Volatility Min Event'] = df['Volatility'].iloc[argrelextrema(df['Volatility'].values, np.less_equal, order=self.rolling_periods)[0]]
        # # Local maxima (resistance)
        # df['Local Volatility Max Event'] = df['Volatility'].iloc[argrelextrema(df['Volatility'].values, np.greater_equal, order=self.rolling_periods)[0]]
        # # Fill na values for horizontal lines
        # df['Local Volatility Min PreLine'] = df['Local Volatility Min Event'].ffill()
        # df['Local Volatility Max PreLine'] = df['Local Volatility Max Event'].ffill()

        # if df['Local Volatility Max Event'] is not None:
        #     df['Local Volatility Min'] = df['Local Volatility Min PreLine']
        # if df['Local Volatility Min Event'] is not None:
        #     df['Local Volatility Max'] = df['Local Volatility Max PreLine']


        
        df['IQR3_vola'] = IQR3_vola #df['Local Volatility Max'] 
        df['Vola_Median'] = vola_median
        df['IQR1_vola'] =  IQR1_vola #df['Local Volatility Min']
        df['IQR3_Source'] = IQR3_Source
        df['IQR1_Source'] = IQR1_Source
        
    
        # Rank the Volatility
        max_vola = df['Volatility'].iloc[-self.rolling_periods:].min()
        min_vola = df['Volatility'].iloc[-self.rolling_periods:].max()
    
        # Prevent division by zero
        if max_vola != min_vola:
            df['volatility_rank'] = (df['Volatility'] - min_vola) / (max_vola - min_vola)
        else:
            df['volatility_rank'] = 1  # Handle constant volatility case
        df['volatility_rank'] = np.maximum(0.00000001, df['volatility_rank'] )
        # print(f"Max Volatility :: {max_vola}")
        # print(f"Volatility Rank :: {df['volatility_rank'].tail()}")    
    
        ## Inventory Balance d%, 0 = perfectly balanced,  > 0 is too much base, < 0 is too much quote
        q = 0.0
        min_profit = 0.01
        
        df['max_volatility'] = np.maximum(min_profit  , df['Volatility'] )
    
        
        # Max volatility of the moment * Value of unbalance (q)
        df['Risk_Rate'] = np.maximum(0.01 * df['volatility_rank']  , df['Volatility'] * df['volatility_rank'] ) * q
    
        # Call local values for init values if no events ::         
        # Fill any remaining NaN values forward, so the line is continuous

        df['Local Min'] = df['Low'].iloc[argrelextrema(df['Low'].values, np.less_equal, order=self.rolling_periods)[0]]
        df['Local Min'] = df['Local Min'].ffill()
        
        df['Local Max'] = df['High'].iloc[argrelextrema(df['High'].values, np.greater_equal, order=self.rolling_periods)[0]]
        df['Local Max'] = df['Local Max'].ffill()

        
        # Create conditions for high and low tails
        # Average High is greater than upper percentile (1 StDev)
        high_avg_volatility = df['Rolling Volatility'] > df['IQR3_vola']
        high_spike_volatility = df['Volatility'].shift(-1) > 3 * df['IQR3_vola']
        high_volatility = (high_avg_volatility) | (high_spike_volatility)   

        # low_tail = (df['Low'] < df['IQR1_Source'])  & high_volatility
        # high_tail = (df['High'] > df['IQR3_Source'])  & high_volatility
        
        low_tail =  high_volatility  & (df['Low'] <= df['Local Min'].shift(1))   
        high_tail =  high_volatility  & (df['High'] >= df['Local Max'].shift(1))   
        
        # Initialize Low Line and High Line with NaN values and fill the first values of IQR1_Source and IQR3_Source
    
        ## Initialize
        df['Trailing High Line'] = np.nan
        df['Trailing Low Line'] = np.nan
        
        df['Low Line'] = np.nan
        df['High Line'] = np.nan
        # df.loc[0, 'Low Line'] = df.loc[0, 'Low']  # Start Low Line with the first value of IQR1_Source
        # df.loc[0, 'High Line'] = df.loc[0, 'High']  # Start High Line with the first value of IQR3_Source



    
    
        def determine_tail_levels(i, previous_low_index, latest_low_index, previous_high_index, latest_high_index, trailing):

            
            # Initialize values to the last used level
            # If no tails are triggered, the past value is used. 
            if trailing:
                new_high_tail_value = df.loc[i-1, 'Trailing High Line']
                new_low_tail_value = df.loc[i-1, 'Trailing Low Line']
            else:
                new_high_tail_value = df.loc[i-1, 'High Line']
                new_low_tail_value = df.loc[i-1, 'Low Line']
            
            # Handle Low Line Location: (A new high tail triggers a previous base)
            if high_tail[i]:
                if previous_high_index is not None and latest_low_index is not None:
                    # A new Base is formed from a bounce
                    if previous_high_index < latest_low_index:
                        new_low_tail_value = lowest_between 
                    else:
                        new_low_tail_value = np.minimum(lowest_between, lowest_consecutive)
                        
                        if trailing:
                            # Trailing price upwards as it makes new Tops
                            new_high_tail_value = highest_consecutive # np.maximum(new_high_tail_value, highest_consecutive)
                else:
                    new_low_tail_value = df.loc[i, 'Local Min']
                    # new_low_tail_value = df.loc[1: i, 'Low'].min()

            
            # Handle High Line Location (A new low tail triggers a previous top)
            if low_tail[i]:
                if previous_low_index is not None and latest_high_index is not None:
                    # A new Top is formed from a bounce
                    if previous_low_index < latest_high_index:
                        new_high_tail_value = highest_between 
                    else:
                        new_high_tail_value = np.maximum(highest_between, highest_consecutive)
                        
                        if trailing:
                            # Trailing price downwards as it makes new Bases
                            new_low_tail_value = lowest_consecutive # np.minimum(new_low_tail_value, lowest_consecutive)
                else:
                    new_high_tail_value = df.loc[i, 'Local Max']
                    # new_high_tail_value = df.loc[1: i, 'High'].max()
                
            return new_high_tail_value, new_low_tail_value


        
        # Initialize indexers to keep track of when an event happened
        last_low_indices = []
        last_high_indices = []
    
        # Iterate through each row and update the Low Line and High Line
        for i in range(1, len(df)):
        # Keep track of the last two indices for low_tail events
            if low_tail[i]:
                last_low_indices.append(i)
                if len(last_low_indices) > 2:
                    last_low_indices.pop(0)  # Maintain only the last 2 indices
        
            # Keep track of the last two indices for high_tail events
            if high_tail[i]:
                last_high_indices.append(i)
                if len(last_high_indices) > 2:
                    last_high_indices.pop(0)  # Maintain only the last 2 indices
                    
            # Initialize indexes based on # of events in the symbol/market
            if len(last_low_indices) == 2:
                previous_low_index = last_low_indices[0]
                latest_low_index = last_low_indices[1]
            elif len(last_low_indices) ==  1:
                previous_low_index = last_low_indices[0]
                latest_low_index = None
            else:
                previous_low_index = None
                latest_low_index = None
        
            if len(last_high_indices) == 2:
                previous_high_index = last_high_indices[0]
                latest_high_index = last_high_indices[1]
            elif len(last_high_indices) ==  1:
                previous_high_index = last_high_indices[0]
                latest_high_index = None
            else:
                previous_high_index = None
                latest_high_index = None
    

            #Find Values between points        
            if previous_high_index is not None and latest_high_index is not None:
                lowest_between = df.loc[np.minimum(previous_high_index,latest_high_index):\
                np.maximum(previous_high_index,latest_high_index), 'Low'].min() 
            # elif previous_high_index is not None and latest_high_index is None:
            #     lowest_between = df.loc[1: previous_high_index, 'Low'].min() 
            else:
                lowest_between = df.loc[i, 'Local Min'] #df.loc[1: i, 'Low'].min() 
    
            
            if previous_low_index is not None and latest_low_index is not None:
                highest_between = df.loc[np.minimum(previous_low_index, latest_low_index):\
                np.maximum(previous_low_index, latest_low_index), 'High'].max()
            # elif previous_low_index is not None and latest_low_index is None:
            #     highest_between = df.loc[1: previous_low_index, 'High'].max()
            else:
                highest_between = df.loc[i, 'Local Max'] #df.loc[1: i, 'High'].max()
    
            # Find Values Consecutively. 
            if latest_high_index is not None and latest_low_index is not None:
                lowest_consecutive = df.loc[np.minimum(latest_high_index,latest_low_index) :\
                np.maximum(latest_high_index,latest_low_index), 'Low'].min()
    
                highest_consecutive = df.loc[np.minimum(latest_low_index, latest_high_index):\
                np.maximum(latest_low_index, latest_high_index), 'High'].max()
    
            
            # elif latest_high_index is not None and latest_low_index is None:
            #     lowest_consecutive = df.loc[latest_high_index: i, 'Low'].min()
            #     highest_consecutive = df.loc[1: latest_high_index, 'High'].max()
    
            
            # elif latest_high_index is None and latest_low_index is not None:
            #     lowest_consecutive = df.loc[1: latest_low_index, 'Low'].min()
            #     highest_consecutive = df.loc[latest_low_index: i, 'High'].max()
            else:
                lowest_consecutive = df.loc[i, 'Local Min']#df.loc[1: i, 'Low'].min() 
                highest_consecutive = df.loc[i, 'Local Max']#df.loc[1: i, 'High'].max()
    
    
            
            if self.symbol == 'BSXUSD' or self.symbol == 'CPOOLEUR':
                new_high_trailing = df.loc[i, 'Local Max']
                new_low_trailing = df.loc[i, 'Local Min']
                new_high = df.loc[i, 'Local Max']
                new_low = df.loc[i, 'Local Min']
            else:
                new_high_trailing, new_low_trailing = determine_tail_levels(i, previous_low_index, latest_low_index, previous_high_index, latest_high_index, True)
                new_high, new_low = determine_tail_levels(i, previous_low_index, latest_low_index, previous_high_index, latest_high_index, False)
           
            # new_high_trailing, new_low_trailing = determine_tail_levels(i, previous_low_index, latest_low_index, previous_high_index, latest_high_index, True)
            # new_high, new_low = determine_tail_levels(i, previous_low_index, latest_low_index, previous_high_index, latest_high_index, False)


            # # Update the DataFrame with the new values
            df.loc[i, 'Trailing High Line'] = new_high_trailing
            df.loc[i, 'Trailing Low Line'] = new_low_trailing
            
            # Finalize the High Line and Low Line
            df.loc[i, 'High Line'] = new_high
            df.loc[i, 'Low Line'] = new_low

        # Check if the whole 'High Line' series is NaN and fill with local maxima if necessary
        if df['High Line'].isna().all():
            df['High Line'] = df['High'].iloc[argrelextrema(df['High'].values, np.greater_equal, order=self.rolling_periods)[0]]
            # Fill any remaining NaN values forward, so the line is continuous
            df['High Line'] = df['High Line'].ffill()
            df['Trailing High Line'] = df['High Line']

        # Check if the whole 'Low Line' series is NaN and fill with local minima if necessary
        if df['Low Line'].isna().all():
            df['Low Line'] = df['Low'].iloc[argrelextrema(df['Low'].values, np.less_equal, order=self.rolling_periods)[0]]
            df['Low Line'] = df['Low Line'].ffill()
            df['Trailing Low Line'] = df['Low Line']

        # Initialize DataFrames for low and high tails
        df_low_tails = pd.DataFrame()
        df_high_tails = pd.DataFrame()
        
        # Condition for Low, avoiding initialized Low Line
        df['Lowest Base'] = np.minimum(df['Low Line'], df['Trailing Low Line'])
        df['Highest Top'] =  np.maximum(df['High Line'], df['Trailing High Line'])

        low_below_base = (df['Low'] < df['Lowest Base'])  # Avoid using the initialized low line
        # Condition for High
        high_above_top = (df['High'] > df['Highest Top'])

        
        # # Fill df_low_tails for Low conditions while keeping alignment with the full DataFrame
        df['% Diff Low'] = np.where(low_below_base, (df['Lowest Base'] - df['Low']) / df['Lowest Base'], np.nan)
        # # Calculate the percentage difference straightforwardly
        # df['% Diff Low'] = (df['Lowest Base'] - df['Low']) / df['Lowest Base']
        
        # # Filter out values where the percentage difference is less than 0
        # df['% Diff Low'] = df['% Diff Low'].where(df['% Diff Low'] >= 0, np.nan)
        
        # Calculate rolling quantiles while keeping the DataFrame aligned
        df['% Diff IQR3 Low'] = df['% Diff Low'].quantile(0.9332)
        df['% Diff Median Low'] = df['% Diff Low'].quantile(0.50)
        df['% Diff IQR1 Low'] = df['% Diff Low'].quantile(0.0668)
        df['Max % Diff Low'] = df['% Diff Low'].max()
        

        
        # Fill df_high_tails for High conditions while keeping alignment
        df['% Diff High'] = np.where(high_above_top, (df['High'] - df['Highest Top']) / df['High'], np.nan)
        
        # Calculate rolling quantiles while keeping the DataFrame aligned
        df['% Diff IQR3 High'] = df['% Diff High'].quantile(0.9332)
        df['% Diff Median High'] = df['% Diff High'].quantile(0.50)
        df['% Diff IQR1 High'] = df['% Diff High'].quantile(0.0668)
        df['Max % Diff High'] = df['% Diff High'].max()

        # Fill for plotting
        df['% Diff Low'] = df['% Diff Low'].fillna(0)
        df['% Diff High'] = df['% Diff High'].fillna(0)

    
        df['Mid Line'] = (df['Highest Top'] + df['Lowest Base']) / 2

        _bid_trailing_baseline = df['Trailing Low Line'].iloc[-1]
        _ask_trailing_baseline = df['Trailing High Line'].iloc[-1]

        _bid_baseline = df['Mid Line'].iloc[-1] # df['Low Line'].iloc[-1]
        _ask_baseline = df['Mid Line'].iloc[-1] # df['High Line'].iloc[-1]
        # print(df)
        current_vola = df['Volatility'].iloc[-1]
        volatility_rank = df['volatility_rank'].iloc[-1]
    
        return df, df_low_tails, df_high_tails, _bid_baseline, _ask_baseline, \
         _bid_trailing_baseline, _ask_trailing_baseline, volatility_rank, current_vola

    def call_kraken_ohlc_data(self):
        #Convert string interval to numerical
        interval = pd.to_numeric(self.interval)
        # Calculate the timestamp for x Days ago
        since_input = datetime.datetime.now() - datetime.timedelta(days=720)
        since_timestamp = int(time.mktime(since_input.timetuple())) * 1000000000  # Convert to nanoseconds
        # Calculate the timestamp for now
        now_timestamp = int(time.time() * 1000000000)  # Current time in nanoseconds
        # Initialize Kraken API object with your symbol and start timestamp
        api = KrakenAPI(self.symbol, since_timestamp, end_timestamp=now_timestamp)
        trades = api.get_ohlc_since(interval)
        # Convert to DataFrame
        #[int <time>, string <open>, string <high>, string <low>, string <close>, string <vwap>, string <volume>, int <count>]
        df = pd.DataFrame(trades, columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'VWAP', 'Volume', 'Trades'])
        # df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        # Pop out last data point, which is a duplicate of current data etc * 
        df = df.iloc[:-1]
        return df

    def append_non_overlapping_data(self, csv_df, api_df):
        """Append non-overlapping data based on Timestamp and only keep the necessary columns."""
        columns_to_keep = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades']
        
        # Ensure both dataframes contain only the relevant columns
        csv_df_filtered = csv_df[columns_to_keep]
        api_df_filtered = api_df[columns_to_keep]
        
        if csv_df_filtered.empty:
            return api_df_filtered  # No historical data, just return the API data
        
        # Find the max timestamp in the CSV data
        last_timestamp = csv_df_filtered['Timestamp'].max()

        # Filter API data to get only the rows that have timestamps later than the CSV's last timestamp
        new_data = api_df_filtered[api_df_filtered['Timestamp'] > last_timestamp]
        
        if not new_data.empty:
            # Append new rows to the historical data
            combined_df = pd.concat([csv_df_filtered, new_data], ignore_index=True)
            print(f"Appending {len(new_data)} new rows.")
        else:
            combined_df = csv_df_filtered
            print("No new data to append.")
        
        return combined_df
    
    def save_to_csv(self, df):
        """Save the DataFrame back to CSV."""
        df.to_csv(self.filepath, index=False)
        print(f"Data saved to {self.filepath}")





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


    ## Trade Halting Process

    #Flag to avoid trading unless a cycle is complete
    trade_in_progress = False


    #order_refresh_time = 30
    quote_order_amount = Decimal(5.0)
    order_amount = Decimal(130000)
    max_order_amount = Decimal(1000000)
    min_order_size_bid = Decimal(0)
    min_order_size_ask = Decimal(0)


    trading_pair = "BSX-USD"
    exchange = "kraken"
    base_asset = "BSX"
    quote_asset = "USD"
    history_market = 'BSXUSD'


    #Maximum amount of orders  Bid + Ask
    maximum_orders = 170

    inv_target_percent = Decimal(0)   

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

        # Define Market Parameters and Settings
        self.Kraken_QFL = KRAKENQFL('BSXUSD_60.csv', self.history_market, '60', volatility_periods=168, rolling_periods=12)

        # Cooldown for how long an order stays in place. 
        self.create_timestamp = 0

        min_refresh_time = 90
        max_refresh_time = 300
        
        self.order_refresh_time = 59 # random.randint(min_refresh_time, max_refresh_time)

        # Cooldown for Volatility calculations 
        self.create_garch_timestamp = 0
        self.garch_refresh_time = 600 
        self_last_garch_time_reported = 0
        
        # Cooldown after a fill
        self.wait_after_fill_timestamp = 0
        self.fill_cooldown_duration = 27
        # Cooldown after cancelling orders
        self.wait_after_cancel_timestamp = 0
        self.cancel_cooldown_duration = 7



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

    def call_trade_history(self, file_name='trades_BSX_USD.csv'):
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
            return 0, 0, 0, 0, 0        # Read the CSV file into a Pandas DataFrame

        df = pd.read_csv(csv_file_path)

        # Convert to numeric
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['trade_fee_in_quote'] = pd.to_numeric(df['trade_fee_in_quote'], errors='coerce')

        # Variables to store trade cycle start point
        cycle_start_index = 0

        # Filter the DataFrame for BUY and SELL trades
        u_buy_trades = df[df['trade_type'] == 'BUY']
        u_sell_trades = df[df['trade_type'] == 'SELL']


        # Get the last traded price for BUY and SELL, or set to 0 if no trades exist
        self._last_buy_price = u_buy_trades['price'].iloc[-1] if not u_buy_trades.empty else 0
        self._last_sell_price = u_sell_trades['price'].iloc[-1] if not u_sell_trades.empty else 0


        # self._last_trade_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.MidPrice)

        # Flag a new trading cycle to make logic trade on baselines, regardless of net value
        new_trade_cycle = False
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
                new_trade_cycle = True
                cycle_start_index = index  # Update to the most recent crossover index
                # print(f"{cycle_start_index}=====================CROSS=============================")
            
                # Update the filtered DataFrame only when a crossover happens
                filtered_df = df.iloc[cycle_start_index:]
                # Update the 'amount' at the crossover index based on the last_net_value
                adjusted_amount = last_net_value / filtered_df.loc[cycle_start_index, 'price']
                filtered_df.loc[cycle_start_index, 'amount'] = adjusted_amount
                # Adjust trade_fee_in_quote based on the updated amount
                fee_percentage = float(self.fee_percent) # Fee Rate
                filtered_df.loc[cycle_start_index, 'trade_fee_in_quote'] = abs(adjusted_amount) * filtered_df.loc[cycle_start_index, 'price'] * fee_percentage
            
                # print(f'Start of Trade Amount :: {filtered_df.loc[cycle_start_index, 'amount']:.8f}, Quote {last_net_value:.8f}')
            else:
                new_trade_cycle = False
                filtered_df = df.iloc[cycle_start_index:]


        # Filter out buy and sell trades

        buy_trades = filtered_df[filtered_df['trade_type'] == 'BUY'].copy()
        sell_trades = filtered_df[filtered_df['trade_type'] == 'SELL'].copy()

        # Ensure amounts are treated as absolute values after editing
        buy_trades.loc[:, 'amount'] = np.abs(buy_trades['amount'])
        sell_trades.loc[:, 'amount'] = np.abs(sell_trades['amount'])
        
        # Check if there are any buy trades
        if not buy_trades.empty:
            sum_of_buy_prices = (buy_trades['price'] * buy_trades['amount']).sum()
            sum_of_buy_amount = buy_trades['amount'].sum()
            sum_of_buy_fees = (buy_trades['trade_fee_in_quote']).sum() if 'trade_fee_in_quote' in buy_trades else 0
        else:
            sum_of_buy_prices = 0
            sum_of_buy_amount = 0
            sum_of_buy_fees = 0
        
        # Check if there are any sell trades
        if not sell_trades.empty:
            sum_of_sell_prices = (sell_trades['price'] * sell_trades['amount']).sum()
            sum_of_sell_amount = sell_trades['amount'].sum()
            sum_of_sell_fees = (sell_trades['trade_fee_in_quote']).sum() if 'trade_fee_in_quote' in sell_trades else 0
        else:
            sum_of_sell_prices = 0
            sum_of_sell_amount = 0
            sum_of_sell_fees = 0

        # Calculate the total buy cost after  fees
        # This isnt a price movement, but a comparison of sum amount.  
        # If I bought $100 worth and paid 0.50, then I paid a total of $100.50 after fees
        # If I sold $100 worth, but paid 0.50 to do so, then I only sold $99.5 after fees
        total_buy_cost = sum_of_buy_prices + sum_of_buy_fees

        # Calculate the total sell proceeds after fees
        total_sell_proceeds = sum_of_sell_prices - sum_of_sell_fees

        # Calculate net value in quote
        # print(last_net_value)
        # print(prev_net_value)
        # Needed to change since the net value here used to calculate only based on the history of the current situation, not updated
        net_value = total_buy_cost - total_sell_proceeds
        # print(f'Net Value :: {net_value}')

        # Calculate the breakeven prices
        breakeven_buy_price = total_buy_cost / sum_of_buy_amount if sum_of_buy_amount > 0 else 0
        # print(f"Total Buy Cost : {total_buy_cost} / sum_buys {sum_of_buy_amount}")
        # print(f'Breakeven Buy Price : {breakeven_buy_price}')

        breakeven_sell_price = total_sell_proceeds / sum_of_sell_amount if sum_of_sell_amount > 0 else 0
        # print(f"Total Sell Proceeds : {total_sell_proceeds} / sum_sells {sum_of_sell_amount}")
        # print(f'Breakeven Sell Price : {breakeven_sell_price}')
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

        return breakeven_buy_price, breakeven_sell_price, realized_pnl, net_value, new_trade_cycle



    def on_tick(self):
        ########## Profiling example to find time/speed of code
        # Start profiling
        # profiler = cProfile.Profile()
        # profiler.enable()


        #Calculate garch every so many seconds
        if self.create_garch_timestamp <= self.current_timestamp:
            ### Call Historical Calculations
            csv_df = self.Kraken_QFL.call_csv_history()
            api_df = self.Kraken_QFL.call_kraken_ohlc_data()
            if csv_df is not None and not csv_df.empty:
                # print(f"CSV DF :: \n{csv_df}")
                if api_df is not None and not api_df.empty:
                    # print(f"API DF :: \n{api_df}")
                    
                    # update CSV with new Data!
                    ###########################################################
                    # Merge data safely (without duplicates)
                    combined_df = self.Kraken_QFL.append_non_overlapping_data(csv_df, api_df)
            
                    # Save the updated data to CSV (only combined original data, no calculations)
                    self.Kraken_QFL.save_to_csv(combined_df)

                    # Perform any additional calculations separately (not saved to CSV)
                    calculated_df, df_low_tails, df_high_tails, self._bid_baseline, \
                    self._ask_baseline, self._bid_trailing_baseline, self._ask_trailing_baseline, \
                    self.volatility_rank, self.current_vola = self.Kraken_QFL.get_ohlc_calculations(combined_df)                
                else:
                    print(f'API DF Empty, Using only historical Data')
                    # Perform any additional calculations separately (not saved to CSV)
                    calculated_df, df_low_tails, df_high_tails, self._bid_baseline, \
                    self._ask_baseline, self._bid_trailing_baseline, self._ask_trailing_baseline, \
                    self.volatility_rank, self.current_vola = self.Kraken_QFL.get_ohlc_calculations(csv_df)

            else:
                print(f"No CSV data found. Initializing new dataset.")
            self.target_profitability = max(self.min_profitability, self.current_vola)
            self.create_garch_timestamp = self.garch_refresh_time + self.current_timestamp

        # Ensure enough time has passed since the last order fill before placing new orders
        if self.create_timestamp <= self.current_timestamp:
            # self.cancel_all_orders()
            self.cancel_bid_orders()
            self.cancel_ask_orders()

            # # Call the balance dataframe
            # self.get_balance_df()

            # If there was a fill or cancel, this timer will halt new orders until timers are met   
            if self.wait_after_fill_timestamp <= self.current_timestamp and \
            self.wait_after_cancel_timestamp <= self.current_timestamp:


                # Update Timestamps
                self.wait_after_cancel_timestamp = self.current_timestamp + self.cancel_cooldown_duration + self.order_refresh_time   # e.g., 10 seconds

                # Reset the Trade Cycle Execution After Timers End
                self.trade_in_progress = False




                # Open Orders if the halt timer is changed to False
                if not self.trade_in_progress:
                    # Call the balance dataframe
                    self.get_balance_df()
                    # Flag the start of a trade Execution
                    self.trade_in_progress = True
                    proposal: List[OrderCandidate] = self.create_proposal()
                    proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
                    self.place_orders(proposal_adjusted)
                    
                    # Update Length of order open Timestamp
                    self.create_timestamp = self.order_refresh_time + self.current_timestamp
        
        ########## Profiling example to find time/speed of code
        # # Stop profiling
        # profiler.disable()
        # # Save the profiling results to a string buffer
        # s = io.StringIO()
        # sortby = pstats.SortKey.CUMULATIVE  # Sort by cumulative time
        # ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        # ps.print_stats()

        # # Print the profiling results to the console
        # print(s.getvalue())

        # # Optionally save to a file
        # with open('profiling_results.txt', 'a') as f:
        #     f.write(s.getvalue())


    def create_proposal(self) -> List[OrderCandidate]:
        bp, sp = self.determine_log_multipliers()
        # Fetch balances and optimal bid/ask prices
        _, _, _, _, _, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        optimal_bid_price, optimal_ask_price, bid_reservation_price,\
        ask_reservation_price, optimal_bid_percent, optimal_ask_percent = self.optimal_bid_ask_spread()

        # Save Values for Status use without recalculating them over and over again
        self.bid_percent = optimal_bid_percent
        self.ask_percent = optimal_ask_percent
        self.b_r_p = bid_reservation_price
        self.a_r_p = ask_reservation_price

        # Generate order sizes lists for both buy and sell sides
        bid_order_levels, ask_order_levels = self.determine_entry_placement(max_levels=7)  # Limit to 5 levels as an example

        # Initial prices
        buy_price = optimal_bid_price 
        sell_price = optimal_ask_price

        # Multiplier values for buy and sell price adjustments
        buy_multiplier = bp  # Reduce buy price by bp%
        sell_multiplier = sp  # Increase sell price by sp%

        # Store orders
        order_counter = []

        # Initialize cumulative order sizes
        cumulative_order_size_bid = 0
        cumulative_order_size_ask = 0

        # Loop through bid levels
        for i in range(len(bid_order_levels)):
            order_size_bid = bid_order_levels.at[i, 'size']
            order_price_bid = bid_order_levels.at[i, 'price']  # Use the price directly from the DataFrame

            # Check if there's enough balance to place the order
            if cumulative_order_size_bid + order_size_bid <= quote_balance_in_base:
                buy_order = OrderCandidate(
                    trading_pair=self.trading_pair,
                    is_maker=True,
                    order_type=OrderType.LIMIT,
                    order_side=TradeType.BUY,
                    amount=Decimal(order_size_bid),
                    price=order_price_bid,
                    from_total_balances=False
                )
                order_counter.append(buy_order)
                cumulative_order_size_bid += order_size_bid  # Update cumulative order size

        # Loop through ask levels
        for i in range(len(ask_order_levels)):
            order_size_ask = ask_order_levels.at[i, 'size']
            order_price_ask = ask_order_levels.at[i, 'price']  # Use the price directly from the DataFrame

            # Check if there's enough balance to place the order
            if cumulative_order_size_ask + order_size_ask <= maker_base_balance:
                sell_order = OrderCandidate(
                    trading_pair=self.trading_pair,
                    is_maker=True,
                    order_type=OrderType.LIMIT,
                    order_side=TradeType.SELL,
                    amount=Decimal(order_size_ask),
                    price=order_price_ask,
                    from_total_balances=True
                )
                order_counter.append(sell_order)
                cumulative_order_size_ask += order_size_ask  # Update cumulative order size

        return order_counter




    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)
        return proposal_adjusted

    def manual_reset_locked_collateral(self):
        self.connectors[self.exchange].budget_checker.reset_locked_collateral()

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
            self.manual_reset_locked_collateral()

            ## Print object attributes. 
            # print(dir(order))
            # print('/n')



    def cancel_bid_orders(self):
        for order in self.get_active_orders(connector_name=self.exchange):
            if order.is_buy :
                self.cancel(self.exchange, order.trading_pair, order.client_order_id)
                self.manual_reset_locked_collateral()

    def cancel_ask_orders(self):
        for order in self.get_active_orders(connector_name=self.exchange):            
            if not order.is_buy:
                self.cancel(self.exchange, order.trading_pair, order.client_order_id)
                self.manual_reset_locked_collateral()





    def did_fill_order(self, event: OrderFilledEvent):
        t, y_bid, y_ask, bid_volatility_in_base, ask_volatility_in_base, bid_reservation_price, ask_reservation_price = self.reservation_price()


        # Update Trade CSV after a trade completes
        breakeven_buy_price, breakeven_sell_price, realized_pnl, net_value, new_trade_cycle = self.call_trade_history('trades_BSX_USD')



        self.fee_percent = Decimal(self.fee_percent)
        

        # Print log
        msg = (f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} {self.exchange} at {round(event.price, 2)}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

         # Set a delay before placing new orders after a fill
        self.wait_after_fill_timestamp = self.current_timestamp + self.fill_cooldown_duration  + self.order_refresh_time  # e.g., 10 seconds




    ##########################
    ###====== Status Screen
    ###########################

    def format_status(self) -> str:
        """Returns status of the current strategy on user balances and current active orders. This function is called
        when status command is issued. Override this function to create custom status display output.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."

        _, _, _, _,_, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        bp, sp = self.determine_log_multipliers()

        lines = []
        warning_lines = []
        warning_lines.extend(self.network_warning(self.get_market_trading_pair_tuples()))

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        lines.extend(["", f"Direction :: {self.trade_position_text} "])

        lines.extend(["", "| Inventory Imbalance | Trade History |"])
        lines.extend([f"Bal ::maker_base_balance {maker_base_balance:.8f} | quote_balance_in_base :: {quote_balance_in_base:.8f}"])

        lines.extend([f"q(d%) :: {self.q_imbalance:.8f} | Inventory Difference :: {self.inventory_diff:.8f}| bp/sp :: {bp:.4f}:{sp:.4f}"] )
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

        # n = self.maximum_orders
        # # n = math.floor(self.maximum_orders/2)

        # ## Buys
        # #Minimum Distance in percent. 0.01 = a drop of 99% from original value
        # bd = 1 / 30
        # bp = math.exp(math.log(bd)/n)
        
        # bp = np.minimum(1 - self.min_profitability, bp)

        # ## Include Fees
        # bp = Decimal(bp)  * (Decimal(1.0) - Decimal(self.fee_percent))
        
        # ## Sells
        # ## 3 distance move,(distance starts at 1 or 100%) 200% above 100 %
        # sd = 30
        # sp = math.exp(math.log(sd)/n)

        # sp = np.maximum(1 + self.min_profitability, sp)

        # ## Include Fees
        # sp = Decimal(sp) * (Decimal(1.0) + Decimal(self.fee_percent))


        #Decimalize for later use

        # msg = (f"sp :: {sp:.8f} , bp :: {bp:.8f}")
        # self.log_with_clock(logging.INFO, msg)
        # New Method
        # Function to transform the metric
        
        def log_transform_reverse(q, m_0, m_min, k):
            abs_q = Decimal(abs(q))
            m_0 = Decimal(m_0)
            m_min = Decimal(m_min)
            k = Decimal(k)
            ONE = Decimal(1.0)
            if m_0 < ONE:  # Drop situation (values < 1)
                adj_m_min = ONE - m_min
                transformed_value = adj_m_min + ((m_0 - adj_m_min) * (ONE - Decimal.ln(k * abs_q + ONE)))
                return min(transformed_value, adj_m_min)
            #
            elif m_0 > ONE:  # Rise situation (values > 1)
                adj_m_min = ONE + m_min
                # Here m_0 > 1 and the transformed value should decrease towards m_min=1
                transformed_value = adj_m_min - ((adj_m_min  - m_0) * (ONE - Decimal.ln(k * abs_q + ONE)))
                return max(transformed_value, adj_m_min)  # Prevent exceeding m_0
            else:
                print('Error, trade depth set at 0% (m_0 = 1)')
                return m_0

        q, _, _, _,_, _, _ = self.get_current_positions()

        # Ratio of how strong the reverse transform is K, modified by 
        # the strength of volatility.  1 - vr = as volatility ^, % distance decreases
        k = Decimal(1.0) * (Decimal(1.0) - Decimal(self.volatility_rank))
        # Deepest entry % to start off the trade
        maximum_bp = 0.970
        maximum_sp = 1.03
        # Log Transform the values based on q balance and k rate
        bp = log_transform_reverse(q, maximum_bp, self.min_profitability, k)
        sp = log_transform_reverse(q, maximum_sp, self.min_profitability, k)

        # Decimal values for use
        bp = Decimal(bp)
        sp = Decimal(sp)

        # msg = (f"sp :: {sp:.8f} , bp :: {bp:.8f}, q :: {q}")
        # self.log_with_clock(logging.INFO, msg)

        # # Bypass with manual numbers for now
        # bp = Decimal(0.970)
        # sp = Decimal(1.03)
        return bp, sp


        

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
        # if q > 0:
        #     bid_depth = bid_volume_cdf_value
        #     ask_depth = max(self.min_order_size_bid, ask_volume_cdf_value) 
        # elif q < 0:
        #     bid_depth = max(self.min_order_size_ask, bid_volume_cdf_value ) 
        #     ask_depth = ask_volume_cdf_value
        # else:
        #     bid_depth = bid_volume_cdf_value
        #     ask_depth = ask_volume_cdf_value

        bid_depth = self.max_order_amount # self.order_amount
        ask_depth = self.max_order_amount # self.order_amount
        self.b_d = bid_depth # bid_depth
        self.a_d = ask_depth # ask_depth
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
        percent_base_to_hold = Decimal(0.005)
        # percent_base_rate = Decimal(1.0) - percent_base_to_hold
        
        percent_quote_to_hold = Decimal(0.005)
        # percent_quote_rate = Decimal(1.0) - percent_quote_to_hold
        

        # Get currently held balances in each asset base
        original_maker_base_balance = self.connectors[self.exchange].get_balance(self.base_asset) 
        original_maker_quote_balance = self.connectors[self.exchange].get_balance(self.quote_asset) 
        
        #Convert to Quote asset at best buy into ask price
        original_quote_balance_in_base = original_maker_quote_balance / top_ask_price

        # Get the total balance in base
        total_balance_in_base = original_quote_balance_in_base + original_maker_base_balance

        # Calculate the target amounts to hold in base and quote assets
        target_base_balance = total_balance_in_base * percent_base_to_hold
        target_quote_balance = total_balance_in_base * percent_quote_to_hold

        # Update Balances to reflect wanted held values
        maker_base_balance = Decimal(original_maker_base_balance) - target_base_balance # max(Decimal(original_maker_base_balance) - target_base_balance, Decimal(0))
        quote_balance_in_base = Decimal(original_quote_balance_in_base) - target_quote_balance #max(Decimal(original_quote_balance_in_base) - target_quote_balance, Decimal(0))

        # Recalculate the total base balance after adjusting for held amounts
        adjusted_total_balance_in_base = maker_base_balance + quote_balance_in_base

        maximum_number_of_orders = self.maximum_orders    

        if total_balance_in_base == 0:
            # Handle division by zero
            return 0, 0, 0, 0
        ### For Entry Size to have /10 (/2 for) orders on each side of the bid/ask
        ### In terms of Maker Base asset
        entry_size_by_percentage = (adjusted_total_balance_in_base * self.inv_target_percent) / maximum_number_of_orders 
        # minimum_size = max(self.connectors[self.exchange].quantize_order_amount(self.trading_pair, self.order_amount), entry_size_by_percentage)



        ## Q relation in percent relative terms, later it is in base(abolute)terms
        target_inventory = adjusted_total_balance_in_base * self.inv_target_percent
        # Inventory Deviation, base inventory - target inventory. 
        inventory_difference = maker_base_balance  - target_inventory
        q = (inventory_difference) / adjusted_total_balance_in_base
        q = Decimal(q)

        self.q_imbalance = q
        self.inventory_diff = inventory_difference


        # Total Abs(imbalance)
        total_imbalance = abs(inventory_difference)

        # Log entry sizes
        def log_transform(q, M_0, k):
            abs_q = abs(q)  # Make sure we are using |q|
            return M_0 * (1 + np.log(1 + k * abs_q))
        # Adjust base and quote balancing volumes based on shape factor and entry size by percentage
        # This method reduces the size of the orders which are overbalanced
        #if I have too much base, more base purchases are made small
        #if I have too much quote, more quote purchases are made small
        #When there is too much of one side, it makes the smaller side easier to trade in bid/ask, so 
        #having more orders of the unbalanced side while allowing price go to lower decreases it's loss
        #to market overcorrection


        if q > 0 :
            # In order to balance the base, I want to sell more of ask to balance it
            # Using total imbalance for quick rebalancing to reduce risk, vs more gradual rebalancing:
            base_balancing_volume =   total_imbalance # abs(self.min_order_size_ask) *  Decimal.exp(self.order_shape_factor * q) #
            quote_balancing_volume =  max ( self.order_amount, abs(self.order_amount) * Decimal.exp(-self.order_shape_factor * q) )

        elif q < 0 :
            base_balancing_volume = max( self.order_amount, abs(self.order_amount) *  Decimal.exp(self.order_shape_factor * q))

            # In order to balance the Quote, I want to buy more of bid to balance it
            # Using total imbalance for quick rebalancing to reduce risk, vs more gradual rebalancing:
            quote_balancing_volume = total_imbalance # abs(self.min_order_size_bid) * Decimal.exp(self.order_shape_factor * q) 


         
        else :
            ## Adjust this logic just for one sided entries :: if you are completely sold out, then you should not have the capability to sell in the first place. 
            base_balancing_volume = self.min_order_size_bid
            quote_balancing_volume = self.min_order_size_ask



        
        base_balancing_volume = Decimal(base_balancing_volume)
        quote_balancing_volume = Decimal(quote_balancing_volume)
        #Return values
        return q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,  entry_size_by_percentage, maker_base_balance, quote_balance_in_base
    

    def determine_entry_placement(self, max_levels=5):
        # Retrieve current positions and calculate minimum order sizes
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base, entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        
        bp, sp = self.determine_log_multipliers()

        optimal_bid_price, optimal_ask_price, bid_reservation_price,\
        ask_reservation_price, optimal_bid_percent, optimal_ask_percent = self.optimal_bid_ask_spread()

        # Call Trade History
        breakeven_buy_price, breakeven_sell_price, realized_pnl, net_value, new_trade_cycle = self.call_trade_history('trades_BSX_USD')

        s_bid = self._bid_baseline
        s_ask = self._ask_baseline

        is_buy_data = breakeven_buy_price > 0
        is_sell_data = breakeven_sell_price > 0

        is_buy_net = net_value > 0
        is_sell_net = net_value < 0
        is_neutral_net = net_value == 0 

        # Set initial minimum order sizes based on configuration or calculations
        self.min_order_size_bid = self.order_amount
        self.min_order_size_ask = self.order_amount

        # Quantize order sizes according to the exchange's rules
        self.min_order_size_bid = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, self.min_order_size_bid)
        self.min_order_size_ask = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, self.min_order_size_ask)

        # Define maximum order size (25th percentile, for instance)
        max_order_size = self.max_order_amount

        # Calculate order sizes for both sides with dynamic levels
        order_levels = pd.DataFrame(columns=['price', 'size'])

        def calculate_max_orders(min_order_size, max_order_size ):
            if min_order_size <= 0:
                raise ValueError("Order size must be greater than zero.")
            
            # Calculate the maximum number of full orders that can fit
            max_full_orders = max_order_size // min_order_size
            return max_full_orders

        # Function to calculate order sizes
        def calculate_dynamic_order_sizes(balance, min_order_size, max_order_size, max_levels):
            order_levels = pd.DataFrame(columns=['price', 'size'])  # Create an empty DataFrame for order levels
            total_size = 0

            max_full_orders = calculate_max_orders(min_order_size, max_order_size)
            max_full_distance = max_levels * max_full_orders
            max_full_distance = int(max_full_distance)

            # Handle the case where min and max sizes are the same
            if min_order_size == max_order_size:
                for level in range(max_full_distance):
                    remaining_balance = balance - total_size

                    if remaining_balance < min_order_size:
                        # Add the remainder to the last order if it's too small to be an independent order
                        if level > 0:  # Ensure there's at least one previous order
                            order_levels.at[level - 1, 'size'] += remaining_balance
                        break  # Exit the loop after adjusting the last order

                    order_levels.loc[level] = {'price': None, 'size': min_order_size}
                    total_size += min_order_size
            else:
                # Handle the case where min and max sizes differ
                for level in range(max_full_distance):
                    remaining_balance = balance - total_size

                    if remaining_balance < min_order_size:
                        # Add the remainder to the last order if it's too small to be an independent order
                        if level > 0:
                            order_levels.at[level - 1, 'size'] += remaining_balance
                        break  # Exit loop after adjusting the last order

                    order_size = min_order_size  # You can add custom sizing logic here if desired
                    order_size = min(order_size, remaining_balance)  # Prevent overshooting the balance
                    order_size = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, order_size)

                    if order_size >= min_order_size:
                        order_levels.loc[level] = {'price': None, 'size': order_size}
                        total_size += order_size

            return order_levels, max_full_orders



        # Function to calculate prices based on the order levels
        def calculate_prices(order_levels, starting_price, price_multiplier, max_orders):
            # Assuming max_orders is your reset interval
            for i in range(len(order_levels)):
                # Determine the current group of orders based on the max orders
                current_group = i // max_orders
                if i>0:
                    if price_multiplier > 1:
                        base_increment = (price_multiplier - 1) / max_orders
                        increment_multiplier = base_increment / (1 + Decimal.ln(i % max_orders + 1))  # Logarithmic adjustment

                        if i % max_orders == 0 and i != 0:  # At the start of a new group
                            # Use full multiplier
                            order_levels.at[i, 'price'] = \
                                self.connectors[self.exchange].quantize_order_price(self.trading_pair,
                                starting_price * (price_multiplier**current_group))  
                        else:
                            # Incrementally adjust from the last order price
                            order_levels.at[i, 'price'] = \
                                self.connectors[self.exchange].quantize_order_price(self.trading_pair,
                                order_levels.at[i - 1, 'price'] * (1 + increment_multiplier))

                    elif price_multiplier < 1:
                        base_increment = (1 - price_multiplier) / max_orders
                        increment_multiplier = base_increment / (1 + Decimal.ln(i % max_orders + 1))

                        if i % max_orders == 0 and i != 0:
                            # Use full multiplier
                            order_levels.at[i, 'price'] = \
                                self.connectors[self.exchange].quantize_order_price(self.trading_pair,
                                starting_price * (price_multiplier**current_group))  
                        else:
                            # Incrementally adjust from the last order price
                            order_levels.at[i, 'price'] = \
                                self.connectors[self.exchange].quantize_order_price(self.trading_pair,
                                order_levels.at[i - 1, 'price'] * (1 - increment_multiplier))
                else:
                    order_levels.at[i, 'price'] = starting_price

            return order_levels

        # Main logic for determining order sizes and prices
        def create_order_levels(is_buy_data, is_sell_data, new_trade_cycle, max_levels):
            # Depending on the cycle, calculate order sizes
            if (not is_buy_data and not is_sell_data) or (new_trade_cycle):
                bid_order_levels, bid_max_full_orders = calculate_dynamic_order_sizes(quote_balance_in_base, self.min_order_size_bid, self.min_order_size_bid, max_levels)
                ask_order_levels, ask_max_full_orders = calculate_dynamic_order_sizes(maker_base_balance, self.min_order_size_ask, self.min_order_size_ask, max_levels)

            elif (is_buy_data and not is_sell_data) and (not new_trade_cycle):
                bid_order_levels, bid_max_full_orders = calculate_dynamic_order_sizes(quote_balance_in_base, self.min_order_size_bid, self.min_order_size_bid, max_levels)
                ask_order_levels, ask_max_full_orders = calculate_dynamic_order_sizes(maker_base_balance, self.min_order_size_ask, max_order_size, max_levels)

            elif (not is_buy_data and is_sell_data) and (not new_trade_cycle):
                bid_order_levels, bid_max_full_orders = calculate_dynamic_order_sizes(quote_balance_in_base, self.min_order_size_bid, max_order_size, max_levels)
                ask_order_levels, ask_max_full_orders = calculate_dynamic_order_sizes(maker_base_balance, self.min_order_size_ask, self.min_order_size_ask, max_levels)

            # Mid trade logic
            elif (is_buy_data and is_sell_data) and (not new_trade_cycle):
                if is_buy_net: 
                    bid_order_levels, bid_max_full_orders = calculate_dynamic_order_sizes(quote_balance_in_base, self.min_order_size_bid, self.min_order_size_bid, max_levels)
                    ask_order_levels, ask_max_full_orders = calculate_dynamic_order_sizes(maker_base_balance, self.min_order_size_ask, max_order_size, max_levels)
                elif is_sell_net: 
                    bid_order_levels, bid_max_full_orders = calculate_dynamic_order_sizes(quote_balance_in_base, self.min_order_size_bid, max_order_size, max_levels)
                    ask_order_levels, ask_max_full_orders = calculate_dynamic_order_sizes(maker_base_balance, self.min_order_size_ask, self.min_order_size_ask, max_levels)
                elif is_neutral_net: 
                    bid_order_levels, bid_max_full_orders = calculate_dynamic_order_sizes(quote_balance_in_base, self.min_order_size_bid, self.min_order_size_bid, max_levels)
                    ask_order_levels, ask_max_full_orders = calculate_dynamic_order_sizes(maker_base_balance, self.min_order_size_ask, self.min_order_size_ask, max_levels)

            # Calculate prices for both bid and ask order levels
            bid_order_levels = calculate_prices(bid_order_levels, optimal_bid_price, bp, bid_max_full_orders)
            ask_order_levels = calculate_prices(ask_order_levels, optimal_ask_price, sp, ask_max_full_orders)

            # Log insufficient balance for clarity
            if bid_order_levels['size'].sum() < self.min_order_size_bid:
                msg_b = f"Not Enough Balance for bid trade: {quote_balance_in_base:.8f}"
                self.log_with_clock(logging.INFO, msg_b)
            if ask_order_levels['size'].sum() < self.min_order_size_ask:
                msg_a = f"Not Enough Balance for ask trade: {maker_base_balance:.8f}"
                self.log_with_clock(logging.INFO, msg_a)

            # print(bid_order_levels)
            # print(ask_order_levels)
            return bid_order_levels, ask_order_levels

        # Example usage in your main function or workflow
        bid_order_levels, ask_order_levels = create_order_levels(is_buy_data, is_sell_data, new_trade_cycle, max_levels)
        return bid_order_levels, ask_order_levels



    

    def reservation_price(self):
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        
        #self._last_trade_price = self.get_midprice()

        breakeven_buy_price, breakeven_sell_price, realized_pnl, net_value, new_trade_cycle = self.call_trade_history('trades_BSX_USD')


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
        if (not is_buy_data and not is_sell_data) or (new_trade_cycle):
            self.trade_position_text = "No Trades, Use Baseline"
            s_bid = self._bid_baseline
            s_ask = self._ask_baseline
        
        # You have started a Buy Cycle, use Bid BE
        elif (is_buy_data and not is_sell_data) and (not new_trade_cycle):
            self.trade_position_text = "Buy Cycle"
            s_bid = self._last_buy_price # breakeven_buy_price
            s_ask = breakeven_buy_price
        
        # You have started a Sell Cycle, use Ask BE
        elif (not is_buy_data and is_sell_data) and (not new_trade_cycle):
            self.trade_position_text = "Sell Cycle"
            s_bid = breakeven_sell_price
            s_ask = self._last_sell_price # breakeven_sell_price

        # You are mid trade, use net values to determine locations
        elif (is_buy_data and is_sell_data) and (not new_trade_cycle):
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


        breakeven_buy_price, breakeven_sell_price, realized_pnl, net_value, new_trade_cycle = self.call_trade_history('trades_BSX_USD')

        is_buy_data = breakeven_buy_price > 0
        is_sell_data = breakeven_sell_price > 0

        is_buy_net = net_value > 0
        is_sell_net = net_value < 0
        is_neutral_net = net_value == 0 
    
        ## Optimal Spread in comparison to the min profit wanted
        # if not is_buy_data and not is_sell_data:
        #     min_profit_bid = bid_reservation_price
        #     min_profit_ask = ask_reservation_price
        # else:
        min_profit_bid =  bid_reservation_price * bp
        min_profit_ask = ask_reservation_price * sp





        # Spread calculation price vs the minimum profit price for entries
        optimal_bid_price = min_profit_bid # np.minimum(bid_reservation_price - (optimal_bid_spread  / TWO), min_profit_bid)
        optimal_ask_price = min_profit_ask # np.maximum(ask_reservation_price + (optimal_ask_spread / TWO), min_profit_ask)

        # Market Depth Check to allow for hiding further in the orderbook by the volume vwap
        top_bid_price, top_ask_price = self.get_current_top_bid_ask()

        # # # Specified Volume Depth VWAP in the order book
        # depth_vwap_bid, depth_vwap_ask = self.get_vwap_bid_ask()
        # top_bid_price = depth_vwap_bid
        # top_ask_price = depth_vwap_ask

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
            optimal_bid_price = min( optimal_bid_price, price_above_bid)#, depth_vwap_bid)
            optimal_ask_price = max( optimal_ask_price, price_below_ask)#, depth_vwap_ask)
        if q < 0:
            optimal_bid_price = min( optimal_bid_price, price_above_bid)#, depth_vwap_bid)
            optimal_ask_price = max( optimal_ask_price, price_below_ask)#, depth_vwap_ask)
        if q == 0:
            optimal_bid_price = min( optimal_bid_price, price_above_bid)#, depth_vwap_bid)
            optimal_ask_price = max( optimal_ask_price, price_below_ask)#, depth_vwap_ask)


        if optimal_bid_price <= 0 :
            msg_2 = (f"Error ::: Optimal Bid Price @ {optimal_bid_price} below 0.")
            self.log_with_clock(logging.INFO, msg_2)



        # Apply quantum adjustments for final prices
        optimal_bid_price = (floor(optimal_bid_price / bid_price_quantum)) * bid_price_quantum
        optimal_ask_price = (ceil(optimal_ask_price / ask_price_quantum)) * ask_price_quantum

        optimal_bid_percent = ((bid_reservation_price - optimal_bid_price) / bid_reservation_price) * 100
        optimal_ask_percent = ((optimal_ask_price - ask_reservation_price) / ask_reservation_price) * 100


        
        return optimal_bid_price, optimal_ask_price, bid_reservation_price, ask_reservation_price, optimal_bid_percent, optimal_ask_percent