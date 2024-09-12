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
import sys
sys.path.append('/home/tyler/quant/API_call_tests/')
from Kraken_Calculations import BuyTrades, SellTrades


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
            print(len(trades))
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

            # Limit the loop to avoid excessive requests
            if len(self.data) > 100000:  # Example limit
                print("Data limit reached.")
                break

            # Rate limit to avoid hitting API too hard
            time.sleep(1)

        return self.data

def call_kraken_data(hist_days = 3, market = 'XXLMZEUR'):
    # Calculate the timestamp for 1 day ago
    since_input = datetime.datetime.now() - datetime.timedelta(days=hist_days)
    since_timestamp = int(time.mktime(since_input.timetuple())) * 1000000000  # Convert to nanoseconds

    # Calculate the timestamp for now
    now_timestamp = int(time.time() * 1000000000)  # Current time in nanoseconds
    #print(now_timestamp)
    market = market

    # Initialize Kraken API object with your symbol and start timestamp
    api = KrakenAPI(market, since_timestamp, end_timestamp=now_timestamp)
    trades = api.get_trades_since()

    # Convert to DataFrame
    kdf = pd.DataFrame(trades, columns=["Price", "Volume", "Timestamp", "Buy/Sell", "Blank", "Market/Limit", "TradeNumber"])

    #Convert values to numerics
    kdf['Price'] = pd.to_numeric(kdf['Price'], errors='coerce')
    kdf['Volume'] = pd.to_numeric(kdf['Volume'], errors='coerce')


    # Calculate log returns
    kdf['Log_Returns'] = np.log(kdf['Price'] / kdf['Price'].shift(1))

    # Drop any NaN values created due to shifting
    kdf.dropna(subset=['Log_Returns'], inplace=True)

    # Save log returns to a list
    log_returns_list = kdf['Log_Returns'].tolist()


    # Create separate lists for buys and sells
    buy_trades = []
    sell_trades = []


    # Iterate over each trade and separate buys and sells
    for index, row in kdf.iterrows():
        if row['Buy/Sell'] == 'b':  # Assuming 'b' indicates buy
            buy_trades.append(row)  # Add buy trade to the buy list
        elif row['Buy/Sell'] == 's':  # Assuming 's' indicates sell
            sell_trades.append(row)  # Add sell trade to the sell list
            
    # Convert buy_trades and sell_trades to DataFrames for further analysis
    buy_trades_df = pd.DataFrame(buy_trades)
    sell_trades_df = pd.DataFrame(sell_trades)



    total_buy_volume = buy_trades_df['Volume'].sum()
    total_sell_volume = sell_trades_df['Volume'].sum()

    # Print the totals


    # Find the 25th percentile of buy and sell volumes
    percentile = 25
    bought_volume_percentile =np.percentile(buy_trades_df['Volume'], percentile)
    sold_volume_percentile = np.percentile(sell_trades_df['Volume'], percentile)

    # Sell into Bid (Lower Base) IQR
        # Bypass sold baseline for now for median price :: 
    sold_baseline = np.percentile(sell_trades_df['Price'], 50)
    # # Buy into Ask( Upper Base) IQR
        # Bypass sold baseline for now for median price :: 
    bought_baseline = np.percentile(buy_trades_df['Price'], 50)


    # Drop the 'Blank' column
    # kdf.drop('Blank', axis=1, inplace=True)

    # Convert Timestamp to readable format (from seconds)
    kdf['Timestamp'] = pd.to_datetime(kdf['Timestamp'], unit='s')
    #print(f"Start of Data :: {kdf['Timestamp'][0]}")
    # Check for any missing values and fill or drop them if necessary
    kdf.dropna(inplace=True)

    # Apply a rolling average (optional for smoothing)
    kdf['Price_Smoothed'] = kdf['Price']#.rolling(window=2).mean().dropna()  # Adjust window size as needed
    kdf['Volume_Smoothed'] = kdf['Volume']#.rolling(window=2).mean().dropna()

    return sold_baseline, bought_baseline, log_returns_list

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




    #order_refresh_time = 30
    order_amount = Decimal(40)
    create_timestamp = 0
    create_garch_timestamp = 0
    trading_pair = "XLM-EUR"
    exchange = "kraken"
    base_asset = "XLM"
    quote_asset = "EUR"

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
    _last_trade_price = None
    _vwap_midprice = None
    #price_source = self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.LastOwnTrade)

    markets = {exchange: {trading_pair}}


################ Volatility Initializtions  
    trading_pairs = ["XLM-EUR"] #"BTC-USD", "ETH-USD", "PAXG-USD", "PAXG-BTC", "XLM-EUR",, "EUR-USD"]
                    # "LPT-USDT", "SOL-USDT", "LTC-USDT", "DOT-USDT", "LINK-USDT", "UNI-USDT", "AAVE-USDT"]

    intervals = ["1m"]
    max_records = 720

    volatility_interval = 480
    columns_to_show = ["trading_pair", "interval"]
    sort_values_by = ["interval"]
    top_n = 20
    report_interval = 60 * 60 * 6  # 6 hours



    ## Breakeven Initialization
    ## Trading Fee for Round Trip side Limit
    fee_percent = 1 / 2 / 100  # Convert percentage to a decimal
    total_spent = 0
    total_bought = 0
    total_earned = 0
    total_sold = 0
    break_even_price = None  # Store the break-even price

    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)

        min_refresh_time = 30
        max_refresh_time = 60

        # Generate a random integer between min and max using randint
        self.order_refresh_time = random.randint(min_refresh_time, max_refresh_time)

        self.last_time_reported = 0


        self.garch_refresh_time = 480 ### Same as volatility interval
        self_last_garch_time_reported = 0

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
        self.initialize_flag = True
        self._vwap_midprice = None
        self.ask_entry_percents, self.bid_entry_percents = self.geometric_entry_levels()


        self.initialize_startprice_flag = True
        self.buy_counter = 1
        self.sell_counter = 1


        #history_values
        self.close_history = []
        self.log_returns = []
        # self.rolling_mean = 0.08

        # Volatility 
        self.max_vola = 0.0
        self.current_vola = 0.0
        self.volatility_rank = 0.0

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            #time.sleep(10)

            proposal: List[OrderCandidate] = self.create_proposal()
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp

            

        for trading_pair, candles in self.candles.items():
            if not candles.ready:
                self.logger().info(
                    f"Candles not ready yet for {trading_pair}! Missing {candles._candles.maxlen - len(candles._candles)}")

        #All candles are ready for calculation
        if all(candle.ready for candle in self.candles.values()):
            #Calculate garch every so many seconds
            if self.create_garch_timestamp<= self.current_timestamp:
                    ### Call Garch Test
                    self.call_garch_model()
                    #msg_gv = (f"GARCH Volatility {garch_volatility:.8f}")
                    #self.log_with_clock(logging.INFO, msg_gv)
                    self.target_profitability = max(self.min_profitability, self.current_vola)
                    self.create_garch_timestamp = self.garch_refresh_time + self.current_timestamp
            
            #Update the timestamp model 
            if self.current_timestamp - self.last_time_reported > self.report_interval:
                self.last_time_reported = self.current_timestamp
                self.notify_hb_app(self.get_formatted_market_analysis())

        


    def refresh_tolerance_met(self, proposal: List[OrderCandidate]) -> List[OrderCandidate] :
            vwap_bid, vwap_ask = self.get_vwap_bid_ask()
            # if spread diff is more than the tolerance or order quantities are different, return false.
            current = self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.MidPrice)
            if self._order_refresh_tolerance_pct > 0:
                if abs(proposal - current)/current > self._order_refresh_tolerance_pct:
                    return False
                else:
                    return True


    def create_proposal(self) -> List[OrderCandidate]:
        # self._last_trade_price, self._vwap_midprice = self.get_midprice()
        optimal_bid_price, optimal_ask_price, optimal_bid_price2, optimal_ask_price2, optimal_bid_price3, optimal_ask_price3, order_size_bid, order_size_ask, bid_reservation_price, ask_reservation_price, k_bid_size, k_ask_size, optimal_bid_percent, optimal_ask_percent= self.optimal_bid_ask_spread()
        bid_starting_price, ask_starting_price = self.get_starting_prices()

        #ref_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        buy_price = optimal_bid_price ##ref_price * Decimal(1 - self.bid_spread)
        sell_price = optimal_ask_price ##ref_price * Decimal(1 + self.ask_spread)

        buy_price2 = optimal_bid_price2 ##ref_price * Decimal(1 - self.bid_spread)
        sell_price2 = optimal_ask_price2 ##ref_price * Decimal(1 + self.ask_spread)

        buy_price3 = optimal_bid_price3 ##ref_price * Decimal(1 - self.bid_spread)
        sell_price3 = optimal_ask_price3 ##ref_price * Decimal(1 + self.ask_spread)

        if buy_price < self._last_trade_price:
            buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                    order_side=TradeType.BUY, amount=Decimal(order_size_bid), price=buy_price)
           
            #buy_order2 = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
            #                        order_side=TradeType.BUY, amount=Decimal(order_size_bid), price=buy_price2)
            
            #buy_order3 = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
            #                        order_side=TradeType.BUY, amount=Decimal(order_size_bid), price=buy_price3)
        if sell_price > self._last_trade_price:
            sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                        order_side=TradeType.SELL, amount=Decimal(order_size_ask), price=sell_price)
            
            #sell_order2 = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
            #                            order_side=TradeType.SELL, amount=Decimal(order_size_ask), price=sell_price2)

            #sell_order3 = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
            #                            order_side=TradeType.SELL, amount=Decimal(order_size_ask), price=sell_price3)
        
        #msg1 = (f" Trades Placed ::  Bid Price : {buy_price:.8f} , Ask Price : {sell_price:.8f}")
        #self.log_with_clock(logging.INFO, msg1)
        minimum_size = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, self.order_amount)

        order_counter = []
        if order_size_bid >= minimum_size:
            order_counter.append(buy_order)

        if order_size_ask >= minimum_size:
            order_counter.append(sell_order)

        msg2 = (f"Bid % : {optimal_bid_percent:.4f} , Ask % : {optimal_ask_percent:.4f}, Buy Counter {self.buy_counter}, Sell Counter{self.sell_counter}")
        self.log_with_clock(logging.INFO, msg2)           
        # msg3 = (f"Min Order  : {minimum_size:.8f} , Ask Order : {order_size_ask:.8f}, Bid Order {order_size_bid:.8f}")
        # self.log_with_clock(logging.INFO, msg3)

        #msgbe = (f"BreakEven : {self.break_even_price} , Total Spent : {self.total_spent}, Total Bought : {self.total_bought}, Total Earned : {self.total_earned},  Total Sold : {self.total_sold}")
        #self.log_with_clock(logging.INFO, msgbe)
        #self.notify_hb_app_with_timestamp(msg)

        msgce = (f"Bid Starting Price : {bid_starting_price:.8f}, Ask Starting Price : {ask_starting_price:.8f}")
        self.log_with_clock(logging.INFO, msgce)

        return order_counter #[buy_order , sell_order]

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
        s, t, y_bid, y_ask, bid_volatility_in_base, ask_volatility_in_base, bid_reservation_price, ask_reservation_price, bid_stdev_price, ask_stdev_price = self.reservation_price()

        ### Counter Method for Constant Trading without using breakeven levels
        # if event.price < self._last_trade_price or event.price <= bid_reservation_price:
        #     self.sell_counter -= 1
        #     self.buy_counter += 1
            
        # if event.price > self._last_trade_price or event.price >= ask_reservation_price:
        #     self.sell_counter += 1
        #     self.buy_counter -= 1

        # if self.sell_counter <= 0:
        #     self.sell_counter = 1

        # if self.buy_counter<= 0:
        #     self.buy_counter = 1

        ### Counter method that resets the buy or sells if a breakeven trade is made. 
        if event.price < self._last_trade_price or event.price <= bid_reservation_price:
            self.sell_counter = 1
            self.buy_counter += 1
            
        if event.price > self._last_trade_price or event.price >= ask_reservation_price:
            self.sell_counter += 1
            self.buy_counter = 1

        if self.sell_counter <= 0:
            self.sell_counter = 1

        if self.buy_counter<= 0:
            self.buy_counter = 1

        self.initialize_flag = False





        #reset S midprice to last traded value
        self._last_trade_price = event.price

        # Update totals and calculate break-even price based on trade type
        fee = event.price * event.amount * self.fee_percent
        if event.price < self._last_trade_price:
            self.total_spent += (event.price * event.amount) + fee
            self.total_bought += event.amount
            if self.total_bought > 0:
                self.break_even_price = self.total_spent / self.total_bought
        if event.price > self._last_trade_price:
            self.total_earned += (event.price * event.amount) - fee
            self.total_sold += event.amount
            if self.total_bought > 0:
                net_spent = self.total_spent - self.total_earned
                self.break_even_price = net_spent / self.total_bought if self.total_bought > 0 else None

        
        # Print log
        msg = (f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} {self.exchange} at {round(event.price, 2)}, Buy Counter {self.buy_counter}, Sell Counter{self.sell_counter}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

        time.sleep(10)


    #def trade_completion_counter(self, event: OrderFilledEvent):
    def determine_log_multipliers(self):
        """Determine the best placement of percentages based on the percentage/log values 
        (log(d)) / (log(p)) = n, breakding this down with a fixed n to solve for p value turns into  p = d**(1/n).  Or closer p = e^(ln(d) / n)"""

        ## If doing a 50/50 it would be /2 since each side is trading equally
        ## If I am doing a single side (QFL), then the maximum orders should account for only the buy side entry. 
        # n = math.floor(self.maximum_orders/2)

        ### Trades into the more volatile markets should be deeper to account for this
        ## for example, buying XLM(more volatile than FIAT) should be harder to do than selling/ (profiting) from the trade. 

        n = self.maximum_orders
        
        ## Buys
        #Minimum Distance in percent. 0.01 = a drop of 99% from original value
        bd = 1 / 30
        ## Percent multiplier, <1 = buy(goes down), >1 = sell(goes up) 
        #p = (1 - 0.05)
        #bp = min( 1 - self.min_profitability, bd**(1/n) )
        bp = math.exp(math.log(bd)/n)
        ## Sells
        ## 3 distance move,(distance starts at 1 or 100%) 200% above 100 %
        sd = 30
        #sp = max(1 + self.min_profitability, (sd**(1/n)) )
        sp = math.exp(math.log(sd)/n)

        msg = (f"sp :: {sp:.8f} , bp :: {bp:.8f}")
        self.log_with_clock(logging.INFO, msg)
        return bp, sp

    def determine_log_breakeven_levels(self):
        bp,sp = self.determine_log_multipliers()
        buy_counter_adjusted = self.buy_counter - 1
        sell_counter_adjusted = self.sell_counter - 1

        additive_buy = 0
        additive_sell = 0
        
        avg_buy_mult = 1
        avg_sell_mult = 1

        buy_breakeven_mult = 1
        sell_breakeven_mult = 1

        #Average the trade distance percentages(this assumes an even volume on every trade, can implement volume in the future)
        if buy_counter_adjusted > 0:
            for i in range(1, buy_counter_adjusted + 1):
                if i == 1 : # First trade has no initial price drop
                    additive_buy = 1 + self.fee_percent
                elif i > 1 :   # Next trades decay log wise
                    additive_buy += bp**(i-1) + self.fee_percent
            # Find the avg percent of all trades        
            avg_buy_mult = (additive_buy) / (buy_counter_adjusted)
            # Divide the average price by the lowest price to get your multiplier for breakeven
            buy_breakeven_mult = avg_buy_mult / (bp**buy_counter_adjusted)
        else:
            additive_buy = 0
            avg_buy_mult = 1
            buy_breakeven_mult = 1

        if sell_counter_adjusted > 0:
            for i in range(1, sell_counter_adjusted + 1):
                if i == 1: # First trade has no initial price drop
                    additive_sell = 1 - self.fee_percent
                elif i > 1:  # Next trades decay log wise
                    additive_sell += sp**(i-1) - self.fee_percent
            # Find the avg percent of all trades        
            avg_sell_mult = additive_sell / sell_counter_adjusted
            # Divide the average price by the highest price to get your multiplier for breakeven
            sell_breakeven_mult = avg_sell_mult / (sp**sell_counter_adjusted)  

        else:
            additive_sell = 0
            avg_sell_mult = 1
            sell_breakeven_mult = 1

        msg2 = (f"additive_sell :: {additive_sell:.8f} , additive_buy :: {additive_buy:.8f}")
        self.log_with_clock(logging.INFO, msg2)
        msg = (f"avg_sell_mult :: {avg_sell_mult:.8f} , avg_buy_mult :: {avg_buy_mult:.8f}")
        self.log_with_clock(logging.INFO, msg)
        msg3 = (f"Sell Breakeven Mult(s * this = where buy level should be at) :: {sell_breakeven_mult:.8f} , Buy Breakeven Mult(s * this = where sell level should be above) :: {buy_breakeven_mult:.8f}")
        self.log_with_clock(logging.INFO, msg3)


        return buy_breakeven_mult, sell_breakeven_mult
        
    def geometric_entry_levels(self):
        num_trades = math.floor(self.maximum_orders/2)
        max_ask_percent = 2  # Maximum Rise planned for, Numbers are addative so 2 = 200% rise, example: (1 + max_ask_percent)*current ask price  = ask order price
        max_bid_percent = 1 # Numbers are subtractive so 1 = 100% drop,  example:  (1 - max_bid_percent)*current bid price = bid order price 
        # Calculate logarithmically spaced entry percents
        ask_geom_entry_percents = np.geomspace(self.target_profitability, max_ask_percent, num_trades).astype(float)
        bid_geom_entry_percents = np.geomspace(self.target_profitability, max_bid_percent, num_trades).astype(float)

        # Reverse the order and transform values
        ask_transformed_percents = abs(max_ask_percent - ask_geom_entry_percents[::-1])
        bid_transformed_percents = abs(max_bid_percent - bid_geom_entry_percents[::-1])
        # Create an empty dictionary to store the adjusted entry percentages
        ask_entry_percents = {}
        bid_entry_percents = {}

        # Initialize and adjust the values of the dictionary with transformed geometric progression
        for i in range(1, num_trades + 1):
            # Ensure each entry percent is at least i * min_profitability
            min_threshold = self.target_profitability # * i
            ask_entry_percents[i] =max(ask_transformed_percents[i - 1], min_threshold)
            bid_entry_percents[i] =max(bid_transformed_percents[i - 1], min_threshold)

        #msg_lastrade = (f"ask_entry_percents {ask_entry_percents}, bid_entry_percents{bid_entry_percents}")
        #self.log_with_clock(logging.INFO, msg_lastrade)
        return ask_entry_percents, bid_entry_percents

    def get_geometric_entry_levels(self, bid_num, ask_num):
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()



        #if q > 0:
        geom_bid_percent = self.bid_entry_percents.get(bid_num , None)
        geom_ask_percent = self.ask_entry_percents.get(ask_num , None)##self.min_profitability #


        
        geom_bid_percent = Decimal(geom_bid_percent)
        geom_ask_percent = Decimal(geom_ask_percent)

        geom_bid_percent2 = Decimal(geom_bid_percent)
        geom_ask_percent2 = Decimal(geom_ask_percent)

        geom_bid_percent3 = Decimal(geom_bid_percent)
        geom_ask_percent3 = Decimal(geom_ask_percent)
        return geom_bid_percent, geom_ask_percent, geom_bid_percent2, geom_ask_percent2, geom_bid_percent3, geom_ask_percent3

    def on_stop(self):
        for candle in self.candles.values():
            candle.stop()

    def get_formatted_market_analysis(self):
        volatility_metrics_df, log_returns= self.get_market_analysis()
        volatility_metrics_pct_str = format_df_for_printout(
            volatility_metrics_df[self.columns_to_show].sort_values(by=self.sort_values_by, ascending=False).head(self.top_n),
            table_format="psql")
        return volatility_metrics_pct_str

    def format_status(self) -> str:
        if all(candle.ready for candle in self.candles.values()):
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
            #df["volatility"] = df["close"].pct_change().rolling(self.volatility_interval).std()
            #df["volatility_bid"] = df["low"].pct_change().rolling(self.volatility_interval).std()
            #df["volatility_bid_max"] = df["low"].pct_change().rolling(self.volatility_interval).std().max()
            #df["volatility_bid_min"] = df["low"].pct_change().rolling(self.volatility_interval).std().min()
            
            #df["volatility_ask"] = df["high"].pct_change().rolling(self.volatility_interval).std()
            #df["volatility_ask_max"] = df["high"].pct_change().rolling(self.volatility_interval).std().max()
            #df["volatility_ask_min"] = df["high"].pct_change().rolling(self.volatility_interval).std().min()

            #df["volatility_pct"] = df["volatility"] / df["close"]
            #df["volatility_pct_mean"] = df["volatility_pct"].rolling(self.volatility_interval).mean()

            

            # adding bbands metrics
            #df.ta.bbands(length=self.volatility_interval, append=True)
            #df["bbands_width_pct"] = df[f"BBB_{self.volatility_interval}_2.0"]
            #df["bbands_width_pct_mean"] = df["bbands_width_pct"].rolling(self.volatility_interval).mean()
            #df["bbands_percentage"] = df[f"BBP_{self.volatility_interval}_2.0"]
            #df["natr"] = ta.natr(df["high"], df["low"], df["close"], length=self.volatility_interval)
            market_metrics[trading_pair_interval] = df.iloc[-1]

            # Compute rolling window of close prices
            # self.rolling_mean = df["close"].rolling(self.volatility_interval).mean()
            # Calculate log returns using rolling windows
            log_returns = []
            
            # Iterate through the DataFrame starting from the end of the rolling window
            for i in range(self.volatility_interval, len(df)):
                # Extract the window values
                window_values = df["close"].iloc[i - self.volatility_interval:i]
                
                # Calculate log returns for each value in the rolling window
                for j in range(1, len(window_values)):
                    log_return = np.log(window_values.iloc[j] / window_values.iloc[j - 1])
                    log_returns.append(log_return)
            
            # Convert log_returns to a DataFrame or Series
            log_returns_df = pd.Series(log_returns)
            
            # Store log returns
            self.log_returns = log_returns_df.tolist()

            ##self.log_returns.append(df["close"].pct_change().rolling(self.volatility_interval).dropna() )

        volatility_metrics_df = pd.DataFrame(market_metrics).T
        

        return volatility_metrics_df, self.log_returns


##########
    ### Added calculations
    #################
    def get_current_top_bid_ask(self):
        top_bid_price = self.connectors[self.exchange].get_price(self.trading_pair, False)
        top_ask_price = self.connectors[self.exchange].get_price(self.trading_pair, True) 
        return top_bid_price, top_ask_price
    
    def get_vwap_bid_ask(self):
        q, _, _, _,_, _, _ = self.get_current_positions()

        # Create an instance of Trades (Market Trades, don't confuse with Limit)
        buy_trades_instance = BuyTrades('XXLMZEUR')
        sell_trades_instance = SellTrades('XXLMZEUR')
        # Assuming you want to calculate the 97.5th percentile CDF of buy volumes within the last {window_size} data points
        # Data points are in trades collected
        target_percentile = 25
        window_size = 6000

        # Call the method (Market Buy into ask, Sell into bid)
        bid_volume_cdf_value = Decimal(sell_trades_instance.get_volume_cdf(target_percentile, window_size))
        ask_volume_cdf_value = Decimal(buy_trades_instance.get_volume_cdf(target_percentile, window_size))


        bid_depth_difference = abs(bid_volume_cdf_value - self.order_amount)
        ask_depth_difference = abs(ask_volume_cdf_value - self.order_amount)
        
        # Determine the strength ( size ) of volume by how much you want to balance
        if q > 0:
            bid_depth = bid_volume_cdf_value
            ask_depth = max(self.order_amount, ask_volume_cdf_value - (Decimal(ask_depth_difference) * q) )
        elif q < 0:
            bid_depth = max(self.order_amount, bid_volume_cdf_value - abs(Decimal(bid_depth_difference) * q) )
            ask_depth = ask_volume_cdf_value
        else:
            bid_depth = bid_volume_cdf_value
            ask_depth = ask_volume_cdf_value


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
        
        amount_quote_to_hold = Decimal(0)
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
        minimum_size = max(self.connectors[self.exchange].quantize_order_amount(self.trading_pair, self.order_amount), entry_size_by_percentage)



        ## Q relation in percent relative terms, later it is in base(abolute)terms
        target_inventory = total_balance_in_base * self.inv_target_percent
        # Inventory Deviation, base inventory - target inventory. 
        inventory_difference = maker_base_balance  - target_inventory
        q = (inventory_difference) / total_balance_in_base
        q = Decimal(q)
        msg_q = (f"Inventory Balance :: {q:4f}% :: Token :: {inventory_difference:8f}.  + = too much base - = too much quote")
        self.log_with_clock(logging.INFO, msg_q)
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
            quote_balancing_volume =  abs(minimum_size) * Decimal.exp(-self.order_shape_factor * q) 
            # Ensure base balancing volume does not exceed the amount needed to balance
            if quote_balancing_volume > total_imbalance:
                quote_balancing_volume = total_imbalance

        elif q < 0 :
            base_balancing_volume = abs(minimum_size) *  Decimal.exp(-self.order_shape_factor * q)
            quote_balancing_volume = total_imbalance ##abs(minimum_size) * Decimal.exp(self.order_shape_factor * q) 

            # Ensure base balancing volume does not exceed the amount needed to balance
            if base_balancing_volume > total_imbalance:
                base_balancing_volume = total_imbalance
         
        else :
            ## Adjust this logic just for one sided entries :: if you are completely sold out, then you should not have the capability to sell in the first place. 
            base_balancing_volume = 0.0 #minimum_size
            quote_balancing_volume = minimum_size



        
        base_balancing_volume = Decimal(base_balancing_volume)
        quote_balancing_volume = Decimal(quote_balancing_volume)
        #Return values
        return q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,  entry_size_by_percentage, maker_base_balance, quote_balance_in_base
    
    def percentage_order_size(self):
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        


        minimum_size = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, self.order_amount)

        order_size_bid = quote_balancing_volume 
        order_size_ask = base_balancing_volume 

        order_size_bid = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, order_size_bid)
        order_size_ask = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, order_size_ask)


        return order_size_bid, order_size_ask
    
    def get_midprice(self):
        sold_baseline, bought_baseline, log_returns_list = call_kraken_data()
        if self._last_trade_price == None:
            if self.initialize_flag == True:
                # Fetch midprice only during initialization
                if self._last_trade_price is None:
                    midprice = sold_baseline ##0.08506 #self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.MidPrice)
                    # Ensure midprice is not None before converting and assigning
                    if midprice is not None:
                        self._last_trade_price = Decimal(midprice)
                    self.initialize_flag = False  # Set flag to prevent further updates with midprice
        elif self.buy_counter == 1 and self.sell_counter == 1:
            self._last_trade_price =  sold_baseline
        else:
            self._last_trade_price = Decimal(self._last_trade_price)

        msg_lastrade = (f"_last_trade_price @ {self._last_trade_price}")
        self.log_with_clock(logging.INFO, msg_lastrade)

        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()



        price_vwap_bid = self.connectors[self.exchange].get_vwap_for_volume(self.trading_pair,
                                                False,
                                                total_balance_in_base).result_price

        price_vwap_ask = self.connectors[self.exchange].get_vwap_for_volume(self.trading_pair,
                                                True,
                                                total_balance_in_base).result_price

        price_vwap_bid = Decimal(price_vwap_bid)
        price_vwap_ask = Decimal(price_vwap_ask)
        self._vwap_midprice = (price_vwap_bid + price_vwap_ask) / 2

        
  
          
        self._vwap_midprice = self._last_trade_price

        return self._last_trade_price, self._vwap_midprice

    def call_garch_model(self):
        sold_baseline, bought_baseline, log_returns_list = call_kraken_data()

        # Retrieve the log returns from the DataFrame
        log_returns = log_returns_list##self.log_returns

        # Ensure log_returns is a one-dimensional pd.Series
        if isinstance(log_returns, list):
            log_returns = pd.Series(log_returns)

        # Convert to numeric, forcing any errors to NaN
        log_returns_numeric = pd.to_numeric(log_returns, errors='coerce')

        # Remove any NaN or infinite values from log_returns
        log_returns_clean = log_returns_numeric.replace([np.inf, -np.inf], np.nan).dropna()

        # Scale small factors for easy use
        scale_factor = 1000
        log_returns_clean *= scale_factor

        # Fit GARCH model to log returns
        try:
            model = arch_model(log_returns_clean, vol='Garch',mean="AR", p=1, q=1, power=2.0)  # Default model is GARCH(1,1)
            model_fit = model.fit(disp="off")  # Fit the model without display // update_freq = 20
            msg_gv = (f"GARCH  { model_fit.summary()}")
            self.log_with_clock(logging.INFO, msg_gv)
           
            current_variance = []
            current_volatility = []
            length = 0
            # Check if `conditional_volatility` is available
            if hasattr(model_fit, 'conditional_volatility'):
                # Retrieve the latest (current) GARCH volatility
                for i in range(len(model_fit.conditional_volatility)):
                    # Calculate variance
                    variance = model_fit.conditional_volatility.iloc[i]**2
                    # Calculate volatility from variance
                    volatility = np.sqrt(variance)
                    
                    # De-scale the variance and volatility (if needed)
                    variance /= scale_factor
                    volatility /= np.sqrt(scale_factor)
                    
                    # Append to the lists
                    current_variance.append(variance)
                    current_volatility.append(volatility)
                length = len(model_fit.conditional_volatility)
            else:
                print(" USING FORECAST VOLATILTY")
                # Alternative way to get volatility if `conditional_volatility` is not available
                forecast = model_fit.forecast(start=None)
                for i in range(len(forecast.variance)):
                    variance = forecast.variance[i]
                    volatility = np.sqrt(variance) / np.sqrt(scale_factor) # de scale

                    variance /= scale_factor # de scale

                    # Append to the lists
                    current_variance.append(variance)
                    current_volatility.append(volatility)
                length = len(forecast.variance)
        
            # Convert current_volatility to a pandas Series to apply rolling
            current_volatility_series = pd.Series(current_volatility)

            # Define the rolling window size (square root of length)
            window = int(np.round(np.sqrt(len(current_volatility_series))))

            # Apply the rolling window to smooth volatility
            rolling_volatility = current_volatility_series.rolling(window=window).mean()

            # Rank the Volatility
            self.max_vola = rolling_volatility.max()
            min_vola = rolling_volatility.min()
            self.current_vola = current_volatility_series.iloc[-1]

            # Prevent division by zero
            if self.max_vola != min_vola:
                self.volatility_rank = (self.current_vola - min_vola) / (self.max_vola - min_vola)
            else:
                self.volatility_rank = 1  # Handle constant volatility case

            msg = (f"Volatility :: Rank:{self.volatility_rank}, Max:{self.max_vola}, Min:{min_vola}, Current:{self.current_vola}")
            self.log_with_clock(logging.INFO, msg)            
            # return current_vola, max_vola, min_vola

        except Exception as e:
            # Handle any exceptions that occur during model fitting or volatility retrieval
            print(f"An error occurred while fitting the GARCH model: {e}")
            #return None

    def reservation_price(self):
        volatility_metrics_df, log_returns = self.get_market_analysis()
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        
        self._last_trade_price, self._vwap_midprice = self.get_midprice()
       
        buy_breakeven_mult, sell_breakeven_mult = self.determine_log_breakeven_levels()

        TWO = Decimal(2.0)
        HALF = Decimal(0.5)


        s = self._last_trade_price
        s = Decimal(s)

            # It doesn't make sense to use mid_price_variance because its units would be absolute price units ^2, yet that side of the equation is subtracted
            # from the actual mid price of the asset in absolute price units
            # gamma / risk_factor gains a meaning of a fraction (percentage) of the volatility (standard deviation between ticks) to be subtraced from the
            # current mid price
            # This leads to normalization of the risk_factor and will guaranetee consistent behavior on all price ranges of the asset, and across assets




        ### Convert Volatility Percents into Absolute Prices




        max_bid_volatility= Decimal(self.target_profitability) 
        bid_volatility_in_base = (max_bid_volatility) * s 

        max_ask_volatility = Decimal(self.target_profitability) 
        ask_volatility_in_base = (max_ask_volatility) * s 


        msg_4 = (f"max_bid_volatility @ {bid_volatility_in_base:.8f} ::: max_ask_volatility @ {ask_volatility_in_base:.8f}")
        self.log_with_clock(logging.INFO, msg_4)

        #INVENTORY RISK parameter, 0 to 1, higher = more risk averse, as y <-- 0, it behaves more like usual
        # Adjust the width of the y parameter based on volatility, the more volatile , the wider the spread becomes, y goes higher
        y = Decimal(1.0)
        y_min = Decimal(0.000000001)
        y_max = Decimal(1.0)
        y_difference = Decimal(y_max - y_min)
        # konstant = Decimal(5)
        y_bid = y_min + (y_difference * Decimal(self.volatility_rank))  #y_difference * Decimal(math.exp(konstant * max_bid_volatility)) ##y - (volatility_bid_rank * y_difference)
        y_ask = y_min + (y_difference * Decimal(self.volatility_rank))  #y_difference * Decimal(math.exp(konstant * max_ask_volatility)) ##y - (volatility_ask_rank * y_difference)

        y_bid = min(y_bid,y_max)
        y_bid = max(y_bid,y_min)

        y_ask = min(y_ask,y_max)
        y_ask = max(y_ask,y_min)

        msg_1 = (f"y_bid @ {y_bid:.8f} ::: y_ask @ {y_ask:.8f}")
        self.log_with_clock(logging.INFO, msg_1)
        t = Decimal(1.0)
        #1 is replacement for time (T= 1 - t=0)
        bid_risk_rate = q * y_bid
        ask_risk_rate = q * y_ask
        bid_reservation_adjustment = bid_risk_rate * bid_volatility_in_base * t
        ask_reservation_adjustment = ask_risk_rate * ask_volatility_in_base * t

        bid_reservation_price = (s*Decimal(sell_breakeven_mult)) - (bid_reservation_adjustment) 
        ask_reservation_price = (s*Decimal(buy_breakeven_mult)) - (ask_reservation_adjustment)

        #msg_6 = (f" q {q} , bid_risk_rate {bid_risk_rate},ask_risk_rate {ask_risk_rate}  bid_reservation_adjustment {bid_reservation_adjustment}, ask_reservation_adjustment {ask_reservation_adjustment}")
        #self.log_with_clock(logging.INFO, msg_6)

        # Get 3 stdevs from price to use in volatility measurements upcoming. 
        msg = (f"Ask_RP :: {ask_reservation_price:.8f}  , Last TP :: {self._last_trade_price:.8f} , Bid_RP :: {bid_reservation_price:.8f}")
        self.log_with_clock(logging.INFO, msg)

        bid_stdev_price = bid_reservation_price - (Decimal(3) * (max_bid_volatility * s))
        ask_stdev_price = ask_reservation_price + (Decimal(3) * (max_ask_volatility * s))

        
        return s, t, y_bid, y_ask, bid_volatility_in_base, ask_volatility_in_base, bid_reservation_price, ask_reservation_price, bid_stdev_price, ask_stdev_price

    def get_starting_prices(self):
        s, t, y_bid, y_ask, bid_volatility_in_base, ask_volatility_in_base, bid_reservation_price, ask_reservation_price, bid_stdev_price, ask_stdev_price = self.reservation_price()

        if self.initialize_startprice_flag == True:
            bid_starting_price = self._last_trade_price#Decimal(0.0000985)
            ask_starting_price = self._last_trade_price#Decimal(0.0000631)
            self.initialize_startprice_flag == False
        else:
            bid_starting_price = self._last_trade_price#bid_starting_price
            ask_starting_price = self._last_trade_price#ask_starting_price
            
            ####  Use Highest and Lowest trade to determine where you should start your percentage entries. 
            ## When a new trend is placed, the trader will always start at the top of the trend until it is completely broken. 
            if self.buy_counter == 1:
                bid_starting_price = self._last_trade_price#bid_reservation_price
            
            if self.sell_counter == 1:
                ask_starting_price = self._last_trade_price#ask_reservation_price

        return bid_starting_price, ask_starting_price

    def optimal_bid_ask_spread(self):
        s, t, y_bid, y_ask, bid_volatility_in_base, ask_volatility_in_base, bid_reservation_price, ask_reservation_price, bid_stdev_price, ask_stdev_price = self.reservation_price()
        order_size_bid, order_size_ask = self.percentage_order_size()
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()

        geom_bid_percent, geom_ask_percent, geom_bid_percent2, geom_ask_percent2, geom_bid_percent3, geom_ask_percent3 = self.get_geometric_entry_levels(self.buy_counter, self.sell_counter)
        
        bid_starting_price, ask_starting_price = self.get_starting_prices()
        bp, sp = self.determine_log_multipliers()
        bp = Decimal(bp)
        sp = Decimal(sp)
        bp_inprice = Decimal(1) - bp
        sp_inprice = sp - Decimal(1)

        msg_7 = (f"bp {bp:.8f} ::: sp {sp:.8f}")
        self.log_with_clock(logging.INFO, msg_7)
        TWO = Decimal(2.0)
        HALF = Decimal(0.5)



        ## Calculate kappa k (similar to market depth for a percent, can perhaps modify it to represent 50th percentile etc, look into it)
        #e_value = math.e
        bid_maximum_spread_in_price = (TWO * Decimal(bp_inprice) * bid_reservation_price)
        bid_maximum_spread_in_price = self.connectors[self.exchange].quantize_order_price(self.trading_pair, bid_maximum_spread_in_price)


        ask_maximum_spread_in_price = (TWO * Decimal(sp_inprice) * ask_reservation_price)
        ask_maximum_spread_in_price = self.connectors[self.exchange].quantize_order_price(self.trading_pair, ask_maximum_spread_in_price)


        bid_inside_exp = ((bid_maximum_spread_in_price * y_bid) - (bid_volatility_in_base**TWO * y_bid**TWO)) / TWO
        #bid_inside_exp = e_value ** float(bid_inside_exp) 
        bid_inside_exp = Decimal(bid_inside_exp).exp()


        k_bid_size = y_bid / (bid_inside_exp - Decimal(1))
        k_bid_size = Decimal(k_bid_size)

        bid_k_division = y_bid / k_bid_size
        bid_k_division = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, bid_k_division)

        bid_spread_rate = 1 +  (bid_k_division)
        bid_spread_rate = Decimal(bid_spread_rate)
        bid_log_term = Decimal.ln(bid_spread_rate)



        ask_inside_exp = ((ask_maximum_spread_in_price * y_ask) - (ask_volatility_in_base**TWO * y_ask**TWO)) / TWO
        #ask_inside_exp = e_value ** float(ask_inside_exp) 
        ask_inside_exp = Decimal(ask_inside_exp).exp()

        k_ask_size = y_ask / (ask_inside_exp - Decimal(1))
        k_ask_size = Decimal(k_ask_size)

        ask_k_division = y_ask / k_ask_size
        ask_k_division = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, ask_k_division)

        ask_spread_rate = 1 +  (ask_k_division)
        ask_spread_rate = Decimal(ask_spread_rate)
        ask_log_term = Decimal.ln(ask_spread_rate)  

        msg_1 = (f"k_bid Depth Volume @ {k_bid_size:.8f} ::: k_ask Depth Volume @ {k_ask_size:.8f}")
        self.log_with_clock(logging.INFO, msg_1)


        if q > 0:
            optimal_bid_spread = (y_bid * (Decimal(1) * bid_volatility_in_base) * t) + ((TWO  * bid_log_term) / y_bid)
            optimal_ask_spread = (y_ask * (Decimal(1) * ask_volatility_in_base) * t) + ((TWO  * ask_log_term) / y_ask)
        if q < 0:
            optimal_bid_spread = (y_bid * (Decimal(1) * bid_volatility_in_base) * t) + ((TWO  * bid_log_term) / y_bid)
            optimal_ask_spread = (y_ask * (Decimal(1) * ask_volatility_in_base) * t) + ((TWO  * ask_log_term) / y_ask)
        else:
            optimal_bid_spread = (y_bid * (Decimal(1) * bid_volatility_in_base) * t) + ((TWO  * bid_log_term) / y_bid)
            optimal_ask_spread = (y_ask * (Decimal(1) * ask_volatility_in_base) * t) + ((TWO  * ask_log_term) / y_ask)
     

        #1
        geom_spread_bid = 1 - Decimal(geom_bid_percent)
        geom_spread_ask = 1 + Decimal(geom_ask_percent)




        geom_limit_bid = Decimal(bid_starting_price) * bp ##geom_spread_bid 
        geom_limit_ask = Decimal(ask_starting_price) * sp ##geom_spread_ask 
        #2
        geom_spread_bid2 = 1 - Decimal(geom_bid_percent2)
        geom_spread_ask2 = 1 + Decimal(geom_ask_percent2)

        geom_limit_bid2 = bid_reservation_price * geom_spread_bid2 
        geom_limit_ask2 = ask_reservation_price * geom_spread_ask2 
        #3
        geom_spread_bid3 = 1 - Decimal(geom_bid_percent3)
        geom_spread_ask3 = 1 + Decimal(geom_ask_percent3)

        geom_limit_bid3 = bid_reservation_price * geom_spread_bid3
        geom_limit_ask3 = ask_reservation_price * geom_spread_ask3


        geom_limit_bid = max(geom_limit_bid, 0)
        
        

        optimal_bid_price = bid_reservation_price -  (optimal_bid_spread  / TWO)
        optimal_ask_price = ask_reservation_price +  (optimal_ask_spread / TWO)

        top_bid_price, top_ask_price = self.get_current_top_bid_ask()
        vwap_bid, vwap_ask = self.get_vwap_bid_ask()

        deepest_bid = min(vwap_bid, top_bid_price)
        deepest_ask = max(vwap_ask, top_ask_price)


        # Calculate the quantum for both bid and ask prices
        bid_price_quantum = self.connectors[self.exchange].get_order_price_quantum(
            self.trading_pair,
            top_bid_price
        )
        ask_price_quantum = self.connectors[self.exchange].get_order_price_quantum(
            self.trading_pair,
            top_ask_price
        )

        # Calculate the price just above the top bid and just below the top ask
        price_above_bid = (ceil(top_bid_price / bid_price_quantum) + 1) * bid_price_quantum
        price_below_ask = (floor(top_ask_price / ask_price_quantum) - 1) * ask_price_quantum

        if q > 0:
            optimal_bid_price = min(deepest_bid, optimal_bid_price, price_above_bid)
            optimal_ask_price = max(top_ask_price, optimal_ask_price, price_below_ask)
        if q < 0:
            optimal_bid_price = min(top_bid_price, optimal_bid_price, price_above_bid)
            optimal_ask_price = max(deepest_ask, optimal_ask_price, price_below_ask)
        if q == 0:
            optimal_bid_price = min(deepest_bid, optimal_bid_price, price_above_bid)
            optimal_ask_price = max(deepest_ask, optimal_ask_price, price_below_ask)


        if optimal_bid_price <= 0 :
            msg_2 = (f"Optimal Bid Price @ {optimal_bid_price} below 0. Setting at {geom_limit_bid}. Reservation Price = {bid_reservation_price}")
            self.log_with_clock(logging.INFO, msg_2)
            optimal_bid_price = geom_limit_bid
            

        # Apply quantum adjustments for final prices
        optimal_bid_price = (floor(optimal_bid_price / bid_price_quantum)) * bid_price_quantum
        optimal_ask_price = (ceil(optimal_ask_price / ask_price_quantum)) * ask_price_quantum

        optimal_bid_price = min(optimal_bid_price, geom_limit_bid)
        optimal_ask_price = max(optimal_ask_price , geom_limit_ask)

        optimal_bid_percent = ((bid_reservation_price - optimal_bid_price) / bid_reservation_price) * 100
        optimal_ask_percent = ((optimal_ask_price - ask_reservation_price) / ask_reservation_price) * 100
        #2
        optimal_bid_price2 = min( vwap_bid * geom_spread_bid2, geom_limit_bid2)
        optimal_ask_price2 = max( vwap_ask * geom_spread_ask2, geom_limit_ask2)
        #3
        optimal_bid_price3 = min( vwap_bid * geom_spread_bid3, geom_limit_bid3)
        optimal_ask_price3 = max( vwap_ask * geom_spread_ask3 , geom_limit_ask3)
        
        return optimal_bid_price, optimal_ask_price, optimal_bid_price2, optimal_ask_price2, optimal_bid_price3, optimal_ask_price3, order_size_bid, order_size_ask, bid_reservation_price, ask_reservation_price, k_bid_size, k_ask_size, optimal_bid_percent, optimal_ask_percent


    