import logging
from decimal import Decimal
from typing import List
import math
import pandas as pd
import pandas_ta as ta  # noqa: F401

import numpy as np
import random
import time

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase, Dict
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory

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
    bid_spread = 0.05
    ask_spread = 0.05
    min_profitability = 0.01
    target_profitability = min_profitability
    _order_refresh_tolerance_pct = 0.0301

    # Define the order size increase factor for geometric progression
    _order_size_increase_factor = Decimal(1.01)


    #order_refresh_time = 30
    order_amount = Decimal(0.00301)
    create_timestamp = 0
    trading_pair = "PAXG-BTC"
    exchange = "kraken"
    base_asset = "PAXG"
    quote_asset = "BTC"

    #Initialize Max orders
    maximum_orders = 3
    max_buy_orders = 1
    max_sell_orders = 1

    inv_target_percent = Decimal(0.50)   

    ## how fast/gradual does inventory rebalance? bigger= more rebalance
    order_shape_factor = Decimal(1.1) 
    # Here you can use for example the LastTrade price to use in your strategy
    #MidPrice 
    #BestBid 
    #BestAsk 
    #LastTrade 
    #LastOwnTrade 
    #InventoryCost 
    #Custom 
    _last_trade_price = None
    _vwap_midprice = None




    price_source = PriceType.LastOwnTrade

    markets = {exchange: {trading_pair}}


################ Volatility Initializtions  
    trading_pairs = ["PAXG-BTC"] #"BTC-USD", "ETH-USD", "PAXG-USD", "PAXG-BTC", "BSX-EUR",, "EUR-USD"]
                    # "LPT-USDT", "SOL-USDT", "LTC-USDT", "DOT-USDT", "LINK-USDT", "UNI-USDT", "AAVE-USDT"]

    intervals = ["1m"]
    max_records = 720

    volatility_interval = 480
    columns_to_show = ["trading_pair", "interval", "volatility", "volatility_bid", "volatility_ask"]
    sort_values_by = ["interval", "volatility"]
    top_n = 20
    report_interval = 60 * 60 * 6  # 6 hours





    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        #Base and Top flags
        self.first_base_flag = True
        self.first_top_flag = True

        min_refresh_time = 60
        max_refresh_time = 120

        # Generate a random integer between min and max using randint
        self.order_refresh_time = random.randint(min_refresh_time, max_refresh_time)

        self.last_time_reported = 0
        combinations = [(trading_pair, interval) for trading_pair in self.trading_pairs for interval in
                        self.intervals]

        self.candles = {f"{combinations[0]}_{combinations[1]}": None for combinations in combinations}
        # we need to initialize the candles for each trading pair
        for combination in combinations:
            candle = CandlesFactory.get_candle(
                CandlesConfig(connector=self.exchange, trading_pair=combination[0], interval=combination[1],
                              max_records=self.max_records))
            candle.start()
            self.candles[f"{combination[0]}_{combination[1]}"] = candle

        ## Initialize Trading Flag for use 
        self._last_trade_price = None
        self._vwap_midprice = None
        # add base and top initializaion
        self.top_bid_calc = None
        self.top_ask_calc = None
        
        # Create Database for top bid/ask calcs

        self.base = None
        self.top = None

        self.highest_high = None
        self.lowest_low = None

        ## Order Tracker
        self.buy_order_sizes = []
        self.sell_order_sizes = []

        self.buy_fills = []
        self.break_even_price = None
        self.buy_levels_list = []
        self.sell_levels_list = []

        self.top_bid_ask_df = pd.DataFrame(columns=['timestamp', 'top_bid', 'top_ask'])

        self.first_order_creation = True

    def on_tick(self):


        time_passed = self.create_timestamp <= self.current_timestamp


        # Calculate new geometric levels for buy and sell orders
        buy_entry_percents, sell_entry_percents = self.geometric_entry_levels()
        # Calculate a move that would trigger a new base or top
        self.buy_levels_list, base_volatility_threshold, base_change_significant = self.get_purchase_levels()
        self.sell_levels_list, top_volatility_threshold, top_change_significant = self.get_sell_levels()

        if time_passed and (self.first_order_creation is True or base_change_significant or top_change_significant):
            # Check if the base has changed or if the minimum timeframe has passed since the last update
            # Adjust maximum orders based on current balances
            self.max_buy_orders, self.max_sell_orders = self.adjust_maximum_orders()
            self.first_order_creation = False
            self.cancel_all_orders()

            if len(self.get_active_orders(connector_name=self.exchange)) == 0:
                proposal: List[OrderCandidate] = self.create_proposal()
                proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
                self.place_orders(proposal_adjusted)
                self.create_timestamp = self.order_refresh_time + self.current_timestamp

            

        for trading_pair, candles in self.candles.items():
            if not candles.is_ready:
                self.logger().info(
                    f"Candles not ready yet for {trading_pair}! Missing {candles._candles.maxlen - len(candles._candles)}")
        if all(candle.is_ready for candle in self.candles.values()):
            if self.current_timestamp - self.last_time_reported > self.report_interval:
                self.last_time_reported = self.current_timestamp
                self.notify_hb_app(self.get_formatted_market_analysis())
                self.update_top_bid_ask_df(self.top_bid_calc, self.top_ask_calc)
                self.top_bid_calc = self.connectors[self.exchange].get_price(self.trading_pair, False)
                self.top_ask_calc = self.connectors[self.exchange].get_price(self.trading_pair, True)    




    def create_proposal(self):
        # Retrieve dynamic order sizes and maximum orders
        self.max_buy_orders, self.max_sell_orders = self.adjust_maximum_orders()
        latest_highest_high, latest_lowest_low = self.calculate_highest_high_lowest_low()



        order_candidates = []


        time_passed = self.create_timestamp <= self.current_timestamp
        if time_passed:

            # For logging or debugging purposes
            self.log_with_clock(logging.INFO, f"Adjusted max buy orders: {self.max_buy_orders}, max sell orders: {self.max_sell_orders}")

            self.log_with_clock(logging.INFO, f"Highest High :: {latest_highest_high}")
            msg = f"Top Level ::: {self.top}"
            self.log_with_clock(logging.INFO, msg)

            self.log_with_clock(logging.INFO, f"Lowest Low :: {latest_lowest_low}")
            msg2 = (f"Base Level ::: {self.base}")
            self.log_with_clock(logging.INFO, msg2)

        # Iterate over buy levels and sizes together
        for level, size in zip(self.buy_levels_list, self.buy_order_sizes):
            buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                    order_side=TradeType.BUY, amount=Decimal(size), price=Decimal(level))
            order_candidates.append(buy_order)

        # Iterate over sell levels and sizes together
        for level, size in zip(self.sell_levels_list, self.sell_order_sizes):
            sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                        order_side=TradeType.SELL, amount=Decimal(size), price=Decimal(level))
            order_candidates.append(sell_order)

        return order_candidates




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
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()


        # Print log
        msg = (f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} {self.exchange} at {round(event.price, 2)}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)
        
        if event.trade_type.name == "BUY":
            self.buy_fills.append((event.price, event.amount))
        elif event.trade_type.name == "SELL":
            self.adjust_for_sell(event.amount)

        self.update_break_even_point()

    def adjust_for_sell(self, sell_amount):
        # Adjust the buy fills list based on the sell amount
        remaining_sell_amount = sell_amount
        while remaining_sell_amount > 0 and self.buy_fills:
            buy_price, buy_amount = self.buy_fills[0]
            if buy_amount > remaining_sell_amount:
                # Update the first tuple with the remaining amount
                self.buy_fills[0] = (buy_price, buy_amount - remaining_sell_amount)
                remaining_sell_amount = 0  # Fully handled the sell
            else:
                # Remove the first tuple and decrease the remaining sell amount
                remaining_sell_amount -= buy_amount
                self.buy_fills.pop(0)  # Remove this buy fill as it's fully sold

    def update_break_even_point(self):
        total_spent = sum(price * amount for price, amount in self.buy_fills)
        total_units = sum(amount for _, amount in self.buy_fills)
        self.break_even_price = total_spent / total_units if total_units else None

        if self.break_even_price < self.base:
            self.break_even_price = self.base * Decimal(1.01)
        print(f"Updated Break-Even Price: {self.break_even_price:.2f}")


    def adjust_maximum_orders(self):
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base, entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()

        # Initialize variables to track total sizes for buy and sell orders
        total_buy_size = Decimal(0)
        total_sell_size = Decimal(0)
        self.buy_order_sizes = []
        self.sell_order_sizes = []

        # Assuming self.order_amount is the starting size for both buy and sell orders
        current_buy_size = self.order_amount
        current_sell_size = self.order_amount

        # Calculate the maximum number of buy and sell orders based on dynamic sizing
        while total_buy_size + current_buy_size <= quote_balance_in_base :
            self.buy_order_sizes.append(current_buy_size)
            total_buy_size += current_buy_size
            current_buy_size *= self._order_size_increase_factor
        
        while total_sell_size + current_sell_size <= maker_base_balance :
            self.sell_order_sizes.append(current_sell_size)
            total_sell_size += current_sell_size
            current_sell_size *= self._order_size_increase_factor

        max_buy_orders = len(self.buy_order_sizes)
        max_sell_orders = len(self.sell_order_sizes)

        self.max_buy_orders = max_buy_orders
        self.max_sell_orders = max_sell_orders


        return self.max_buy_orders, self.max_sell_orders
    
    def geometric_buy_levels(self, max_buy_orders):
        q, _, _, _, _, _, _ = self.get_current_positions()

        # Define the range for adjustment
        max_percent = 0.99
        min_percent = 0.75
        percent_difference = max_percent - min_percent
        percent_difference = Decimal(percent_difference)

        # Adjust max_percent for buy levels based on q
        if q < 0:  # Implies need to be more aggressive in buying
            q_adjusted_max_percent = max_percent - (abs(q) * percent_difference)
        else:
            q_adjusted_max_percent = max_percent

        q_adjusted_max_percent = float(q_adjusted_max_percent)

        # Logarithmically spaced percentages for buy levels
        buy_geom_percents = np.geomspace(self.target_profitability, q_adjusted_max_percent, max_buy_orders).astype(float)

        # Create a dictionary for buy entry percents
        buy_entry_percents = {i: buy_geom_percents[i - 1] for i in range(1, max_buy_orders + 1)}

        return buy_entry_percents


    def geometric_sell_levels(self, max_sell_orders):
        q, _, _, _,_, _, _ = self.get_current_positions()

        # Define the range for adjustment
        max_percent = 3
        min_percent = 0.5
        percent_difference = max_percent - min_percent
        percent_difference = Decimal(percent_difference)

        # Adjust max_percent for buy levels based on q
        if q > 0:  # Implies need to be more aggressive in selling
            q_adjusted_max_percent = max_percent - (abs(q) * percent_difference)
        else:
            q_adjusted_max_percent = max_percent

        q_adjusted_max_percent = float(q_adjusted_max_percent)

        # Logarithmically spaced percentages for sell levels
        sell_geom_percents = np.geomspace(self.target_profitability, q_adjusted_max_percent, max_sell_orders).astype(float)
        
        # Create a dictionary for sell entry percents
        sell_entry_percents = {i: sell_geom_percents[i - 1] for i in range(1, max_sell_orders + 1)}
        
        return sell_entry_percents

    def geometric_entry_levels(self):
        self.max_buy_orders, self.max_sell_orders = self.adjust_maximum_orders()

        # Get separate geometric levels for buy and sell orders
        buy_entry_percents = self.geometric_buy_levels(self.max_buy_orders)
        sell_entry_percents = self.geometric_sell_levels(self.max_sell_orders)
        
        return buy_entry_percents, sell_entry_percents

    def on_stop(self):
        for candle in self.candles.values():
            candle.stop()

    def get_formatted_market_analysis(self):
        latest_highest_high, latest_lowest_low = self.calculate_highest_high_lowest_low()
        volatility_metrics_df = self.get_market_analysis()
        volatility_metrics_pct_str = format_df_for_printout(
            volatility_metrics_df[self.columns_to_show].sort_values(by=self.sort_values_by, ascending=False).head(self.top_n),
            table_format="psql")
        return volatility_metrics_pct_str

    def format_status(self) -> str:
        
        if all(candle.is_ready for candle in self.candles.values()):
            lines = []
            lines.extend(["Configuration:", f"Volatility Interval: {self.volatility_interval}"])
            lines.extend(["", "Volatility Metrics", ""])
            lines.extend([self.get_formatted_market_analysis()])
            return "\n".join(lines)
        else:
            return "Candles not ready yet!"

    def get_market_analysis(self):
        market_metrics = {}
        for trading_pair_interval, candle in self.candles.items():
            df = candle.candles_df
            df["trading_pair"] = trading_pair_interval.split("_")[0]
            df["interval"] = trading_pair_interval.split("_")[1]
            # adding volatility metrics
            df["volatility"] = df["close"].pct_change().rolling(self.volatility_interval).std()
            df["volatility_bid"] = df["low"].pct_change().rolling(self.volatility_interval).std()
            df["volatility_bid_max"] = df["low"].pct_change().rolling(self.volatility_interval).std().max()
            df["volatility_bid_min"] = df["low"].pct_change().rolling(self.volatility_interval).std().min()
            
            df["volatility_ask"] = df["high"].pct_change().rolling(self.volatility_interval).std()
            df["volatility_ask_max"] = df["high"].pct_change().rolling(self.volatility_interval).std().max()
            df["volatility_ask_min"] = df["high"].pct_change().rolling(self.volatility_interval).std().min()

            #df["volatility_pct"] = df["volatility"] / df["close"]
            #df["volatility_pct_mean"] = df["volatility_pct"].rolling(self.volatility_interval).mean()

            #Bar Metrics 

            # adding bbands metrics
            #df.ta.bbands(length=self.volatility_interval, append=True)
            #df["bbands_width_pct"] = df[f"BBB_{self.volatility_interval}_2.0"]
            #df["bbands_width_pct_mean"] = df["bbands_width_pct"].rolling(self.volatility_interval).mean()
            #df["bbands_percentage"] = df[f"BBP_{self.volatility_interval}_2.0"]
            #df["natr"] = ta.natr(df["high"], df["low"], df["close"], length=self.volatility_interval)
            market_metrics[trading_pair_interval] = df.iloc[-1]
        volatility_metrics_df = pd.DataFrame(market_metrics).T
        self.target_profitability = max(self.min_profitability, 2 * volatility_metrics_df["volatility"].iloc[-1])


        
        ## Call other Database 
        return volatility_metrics_df

    def update_top_bid_ask_df(self, top_bid, top_ask):
        # Current timestamp can be obtained in various ways depending on your requirements
        current_timestamp = pd.Timestamp.now()
        
        # Create a DataFrame from the new data
        new_data_df = pd.DataFrame([{'timestamp': current_timestamp, 'top_bid': top_bid, 'top_ask': top_ask}])
        
        # If self.top_bid_ask_df does not exist, initialize it
        if not hasattr(self, 'top_bid_ask_df') or self.top_bid_ask_df is None:
            self.top_bid_ask_df = new_data_df
        else:
            # Use concat instead of append
            self.top_bid_ask_df = pd.concat([self.top_bid_ask_df, new_data_df], ignore_index=True)
        
        # Optionally, limit the size of the DataFrame to keep only recent data
        self.top_bid_ask_df = self.top_bid_ask_df.tail(1000)  # Keep the last 1000 entries, for example

    def calculate_highest_high_lowest_low(self):
        self.update_top_bid_ask_df(self.top_bid_calc, self.top_ask_calc)
        volatility_metrics_df= self.get_market_analysis()


        df = volatility_metrics_df

        #volatility_ask = df["volatility_ask"].iloc[-1]
        #volatility_bid = df["volatility_bid"].iloc[-1]

        # Ensure there's enough data
        initial_top = self.top_bid_ask_df["top_ask"].iloc[-1] #+ (self.top_bid_ask_df["top_ask"].iloc[-1] * volatility_ask )
        initial_base = self.top_bid_ask_df["top_bid"].iloc[-1] #- (self.top_bid_ask_df["top_bid"].iloc[-1] * volatility_bid )


        if len(self.top_bid_ask_df) > self.volatility_interval:
            self.top_bid_ask_df["highest_high"] = self.top_bid_ask_df["top_bid"].rolling(window=self.volatility_interval).max()
            self.top_bid_ask_df["lowest_low"] = self.top_bid_ask_df["top_ask"].rolling(window=self.volatility_interval).min()
            
            # Access the latest calculated values
            latest_highest_high = self.top_bid_ask_df["highest_high"].iloc[-1] #+ (self.top_bid_ask_df["highest_high"].iloc[-1] * volatility_ask)
            latest_lowest_low = self.top_bid_ask_df["lowest_low"].iloc[-1] #- (self.top_bid_ask_df["lowest_low"].iloc[-1] * volatility_bid)
            
            return latest_highest_high, latest_lowest_low
        else:
            return initial_top, initial_base  # Not enough data to calculate    

##########
    ### Added calculations
    #################
    def get_top_bid_ask(self):
        self.top_bid_calc = self.connectors[self.exchange].get_price(self.trading_pair, False)
        self.top_ask_calc = self.connectors[self.exchange].get_price(self.trading_pair, True)  

        vwap_bid = self.connectors[self.exchange].get_vwap_for_volume(self.trading_pair,
                                                False,
                                                self.order_amount).result_price

        vwap_ask = self.connectors[self.exchange].get_vwap_for_volume(self.trading_pair,
                                                True,
                                                self.order_amount).result_price
        return self.top_bid_calc, self.top_ask_calc, vwap_bid, vwap_ask

    def get_current_positions(self):
        top_bid_price, top_ask_price, vwap_bid, vwap_ask = self.get_top_bid_ask()


        amount_base_to_hold = Decimal(0.20)
        amount_base_rate = Decimal(1.0) - amount_base_to_hold
        
        amount_quote_to_hold = Decimal(0.20)
        amount_quote_rate = Decimal(1.0) - amount_base_to_hold
        

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


        ## Q relation in percent relative terms, later it is in base(abolute)terms
        target_inventory = total_balance_in_base * self.inv_target_percent
        # Inventory Deviation, base inventory - target inventory. 
        inventory_difference = maker_base_balance  - target_inventory
        q = (inventory_difference) / total_balance_in_base
        q = Decimal(q)


        # Adjust base and quote balancing volumes based on shape factor and entry size by percentage
        # This method reduces the size of the orders which are overbalanced
        #if I have too much base, more base purchases are made small
        #if I have too much quote, more quote purchases are made small
        #When there is too much of one side, it makes the smaller side easier to trade in bid/ask, so 
        #having more orders of the unbalanced side while allowing price go to lower decreases it's loss
        #to market overcorrection
        if q > 0 :
            base_balancing_volume =  abs(entry_size_by_percentage) *  Decimal.exp(-self.order_shape_factor * q)
            quote_balancing_volume = entry_size_by_percentage
        elif q < 0 :
            base_balancing_volume = entry_size_by_percentage
            quote_balancing_volume = abs(entry_size_by_percentage) * Decimal.exp(self.order_shape_factor * q)     
        else :
            base_balancing_volume = entry_size_by_percentage
            quote_balancing_volume = entry_size_by_percentage

            

        base_balancing_volume = Decimal(base_balancing_volume)
        quote_balancing_volume = Decimal(quote_balancing_volume)
        #Return values
        return q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,  entry_size_by_percentage, maker_base_balance, quote_balance_in_base

    

    def percentage_order_size(self):
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        


        minimum_size = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, self.order_amount)

        order_size_bid = base_balancing_volume #max(minimum_size, quote_balancing_volume)
        order_size_ask = quote_balancing_volume  #max(minimum_size, base_balancing_volume)

        order_size_bid = max(minimum_size, self.connectors[self.exchange].quantize_order_amount(self.trading_pair, order_size_bid))
        order_size_ask = max(minimum_size, self.connectors[self.exchange].quantize_order_amount(self.trading_pair, order_size_ask))


        return order_size_bid, order_size_ask
    
    def get_purchase_levels(self):
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()

        buy_entry_percents, sell_entry_percents = self.geometric_entry_levels()

        top_bid_price, top_ask_price, vwap_bid, vwap_ask = self.get_top_bid_ask()

        latest_highest_high, latest_lowest_low = self.calculate_highest_high_lowest_low()

        # Get Metrics
        volatility_metrics_df = self.get_market_analysis()
        df = volatility_metrics_df
        volatility_bid = Decimal(df["volatility_bid"].iloc[-1])
        volatility_ask = Decimal(df["volatility_ask"].iloc[-1])

        if self.first_base_flag:
            self.base = latest_lowest_low
            self.first_base_flag = False


        if self.base is None:
            self.log_with_clock(logging.INFO, "Init Base not calculating Correctly")

        self.base = Decimal(self.base)

        if not self.first_base_flag:
            if latest_lowest_low < self.base - (self.base * volatility_bid):
                self.base = latest_lowest_low



        latest_lowest_low = Decimal(latest_lowest_low)
        self.base = Decimal(self.base)
        volatility_bid = Decimal(volatility_bid)

        base_distance = volatility_bid * self.base * Decimal(2)
        self.buy_levels_list = [self.base - base_distance]

        # Adjust the loop to start geometric levels from 1 and to make dynamic adjustments
        for i in range(1, self.max_buy_orders + 1):
            if i in buy_entry_percents:
                next_bid_level = self.buy_levels_list[-1] * (1 - Decimal(buy_entry_percents[i]))
                self.buy_levels_list.append(next_bid_level)


        # Calculate the current volatility-based threshold for the base change
        base_volatility_threshold = self.base * volatility_bid

        # Determine if the base price has changed significantly in either direction
        base_change_significant = (latest_lowest_low < self.base - base_volatility_threshold) or \
                                (latest_lowest_low > self.base + base_volatility_threshold)

        if base_change_significant:
            self.base = latest_lowest_low  # Adjust this assignment as needed based on your logic



        return self.buy_levels_list, base_volatility_threshold, base_change_significant

    def get_sell_levels(self):
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        buy_entry_percents, sell_entry_percents = self.geometric_entry_levels()

        top_bid_price, top_ask_price, vwap_bid, vwap_ask = self.get_top_bid_ask()

        latest_highest_high, latest_lowest_low = self.calculate_highest_high_lowest_low()



        # Get Metrics
        volatility_metrics_df = self.get_market_analysis()
        df = volatility_metrics_df
        volatility_bid = Decimal(df["volatility_bid"].iloc[-1])
        volatility_ask = Decimal(df["volatility_ask"].iloc[-1])

        if self.first_top_flag:
            self.top = latest_highest_high
            self.first_top_flag = False


        if self.top is None:
            self.log_with_clock(logging.INFO, "Top not calculating Correctly")

        self.top = Decimal(self.top)

        if not self.first_top_flag:
            if latest_highest_high > self.top + (self.top * volatility_ask):
                self.top = latest_highest_high

        latest_highest_hight = Decimal(latest_highest_high)
        self.top = Decimal(self.top)
        volatility_ask = Decimal(volatility_ask)
        top_distance = volatility_ask * self.top * Decimal(2)

        # Initial sell level determination with placeholder for future adjustments
        if self.break_even_price is not None:
            if self.break_even_price > self.top:
                # Scenario: Break-even price is above the top level
                # Decision needed: Fast balance recovery vs. waiting for a profit above breakeven
                self.sell_levels_list = [self.break_even_price + top_distance]  # Placeholder logic
            elif self.break_even_price <= self.top:
                # Scenario: Break-even price is at or below the top level
                # Potential for more nuanced strategy here, such as waiting or immediate action
                self.sell_levels_list = [self.break_even_price + top_distance]  # Placeholder logic
        else:
            # If there's no break-even price set, default to using self.top + top_distance
            self.sell_levels_list = [self.top + top_distance]

        # Adjust the loop to start geometric levels from 1 and to make dynamic adjustments for sell levels
        for i in range(1, self.max_sell_orders + 1):
            if i in sell_entry_percents:
                next_sell_level = self.sell_levels_list[-1] * (1 + Decimal(sell_entry_percents[i]))
                self.sell_levels_list.append(next_sell_level)

        # Calculate the current volatility-based threshold for the top change
        top_volatility_threshold = self.top * volatility_ask

        # Determine if the top price has changed significantly in either direction
        top_change_significant = (latest_highest_high > self.top + top_volatility_threshold) or \
                                (latest_highest_high < self.top - top_volatility_threshold)

        if top_change_significant:
            self.top = latest_highest_high  # Adjust this assignment as needed based on your logic



        
        return self.sell_levels_list, top_volatility_threshold, top_change_significant
