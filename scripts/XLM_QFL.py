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
            #print(len(trades))
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
    # print(now_timestamp)

    # Initialize Kraken API object with your symbol and start timestamp
    api = KrakenAPI(market, since_timestamp, end_timestamp=now_timestamp)
    trades = api.get_trades_since()

    # Convert to DataFrame
    kdf = pd.DataFrame(trades, columns=["Price", "Volume", "Timestamp", "Buy/Sell", "Blank", "Market/Limit", "TradeNumber"])

    #Convert values to numerics
    kdf['Price'] = pd.to_numeric(kdf['Price'], errors='coerce')
    kdf['Volume'] = pd.to_numeric(kdf['Volume'], errors='coerce').fillna(0)
    
    # Calculate log returns
    kdf['Log_Returns'] = np.log(kdf['Price'] / kdf['Price'].shift(1))

    # Drop any NaN values created due to shifting
    kdf.dropna(subset=['Log_Returns'], inplace=True)

    # Save log returns to a list
    log_returns_list = kdf['Log_Returns'].tolist()


    # Create separate lists for buys and sells
    buy_trades = []
    sell_trades = []


    # Initialize the total volumes
    total_buy_volume = 0
    total_sell_volume = 0

    # Separate buys and sells and track volumes
    for index, row in kdf.iterrows():
        if row['Buy/Sell'] == 'b':  # Assuming 'b' indicates buy
            # Add buy trade with actual volume
            buy_trades.append(row)
            total_buy_volume += row['Volume']
            # Add corresponding sell trade with zero volume if needed
            sell_trade = row.copy()
            sell_trade['Volume'] = 0
            sell_trades.append(sell_trade)
        elif row['Buy/Sell'] == 's':  # Assuming 's' indicates sell
            # Add sell trade with actual volume
            sell_trades.append(row)
            total_sell_volume += row['Volume']
            # Add corresponding buy trade with zero volume if needed
            buy_trade = row.copy()
            buy_trade['Volume'] = 0
            buy_trades.append(buy_trade)
            
    # Convert buy_trades and sell_trades to DataFrames for further analysis
    buy_trades_df = pd.DataFrame(buy_trades)
    sell_trades_df = pd.DataFrame(sell_trades)



    kdf['Timestamp'] = pd.to_datetime(kdf['Timestamp'], unit='s')

    # Ensure the 'Timestamp' column is retained and converted to datetime if needed
    buy_trades_df['Timestamp'] = pd.to_datetime(buy_trades_df['Timestamp'], unit='s')
    sell_trades_df['Timestamp'] = pd.to_datetime(sell_trades_df['Timestamp'], unit='s')

    # Sort DataFrames by Timestamp if you want them ordered
    buy_trades_df = buy_trades_df.sort_values(by='Timestamp')
    sell_trades_df = sell_trades_df.sort_values(by='Timestamp')


    # Find the 75th percentile of buy and sell volumes
    percentile = 25

    # Calculate percentiles, excluding zeros
    nonzero_buy_volumes = buy_trades_df[buy_trades_df['Volume'] > 0]['Volume']
    nonzero_sell_volumes = sell_trades_df[sell_trades_df['Volume'] > 0]['Volume']

    # Find the 75th percentile of buy and sell volumes
    percentile = 25
    bought_volume_depth = np.percentile(nonzero_buy_volumes, percentile) if not nonzero_buy_volumes.empty else 0
    sold_volume_depth = np.percentile(nonzero_sell_volumes, percentile) if not nonzero_sell_volumes.empty else 0

    # Calculate the percentile window
    percentile_window = int(np.round(np.sqrt(len(kdf['Price']))))

    # Function to calculate the baseline percentile for sell trades
    def calculate_sold_baseline(row, percentile):
        # Get the relevant prices from sell_trades_df within the window
        relevant_prices = sell_trades_df['Price'][(sell_trades_df['Timestamp'] <= row['Timestamp'])]#.tail(percentile_window)
        return np.percentile(relevant_prices, percentile) if not relevant_prices.empty else np.nan  # Return NaN if no relevant prices

    # Function to calculate the baseline percentile for buy trades
    def calculate_bought_baseline(row, percentile):
        # Get the relevant prices from buy_trades_df within the window
        relevant_prices = buy_trades_df['Price'][(buy_trades_df['Timestamp'] <= row['Timestamp'])]#.tail(percentile_window)
        return np.percentile(relevant_prices, percentile) if not relevant_prices.empty else np.nan  # Return NaN if no relevant prices


    # Define the desired percentile values
    sell_percentile = 25  # You can change this to any value
    buy_percentile =  75  # You can change this to any value

    # Apply the functions to create new columns in kdf with variable percentiles
    kdf['sold_baseline'] = kdf.apply(lambda row: calculate_sold_baseline(row, sell_percentile), axis=1)
    kdf['bought_baseline'] = kdf.apply(lambda row: calculate_bought_baseline(row, buy_percentile), axis=1)

    sold_baseline = kdf['sold_baseline'].iloc[-1]
    bought_baseline = kdf['bought_baseline'].iloc[-1]

    # Drop the 'Blank' column
    kdf.drop('Blank', axis=1, inplace=True)


    # Check for any missing values and fill or drop them if necessary
    kdf.dropna(inplace=True)

    return sold_baseline, bought_baseline, log_returns_list, bought_volume_depth, sold_volume_depth

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

    report_interval = 60 * 60 * 6  # 6 hours



    ## Breakeven Initialization
    ## Trading Fee for Round Trip side Limit
    fee_percent = 1 / 2 / 100  # Convert percentage to a fraction method
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


        ## Initialize Trading Flag for use 
        self.initialize_flag = True

        self._bid_baseline = None
        self._ask_baseline = None

        self.initialize_startprice_flag = True
        self.buy_counter = 1
        self.sell_counter = 1


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
        

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            #time.sleep(10)

            proposal: List[OrderCandidate] = self.create_proposal()
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp

            
        #Calculate garch every so many seconds
        if self.create_garch_timestamp<= self.current_timestamp:
                ### Call Garch Test
                self.call_garch_model()
                #msg_gv = (f"GARCH Volatility {garch_volatility:.8f}")
                #self.log_with_clock(logging.INFO, msg_gv)
                self.target_profitability = max(self.min_profitability, self.current_vola)
                self.create_garch_timestamp = self.garch_refresh_time + self.current_timestamp
        
        # Update the timestamp model 
        if self.current_timestamp - self.last_time_reported > self.report_interval:
            self.last_time_reported = self.current_timestamp

        




    def refresh_tolerance_met(self, proposal: List[OrderCandidate]) -> List[OrderCandidate] :
            #vwap_bid, vwap_ask = self.get_vwap_bid_ask()
            # if spread diff is more than the tolerance or order quantities are different, return false.
            current = self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.MidPrice)
            if self._order_refresh_tolerance_pct > 0:
                if abs(proposal - current)/current > self._order_refresh_tolerance_pct:
                    return False
                else:
                    return True


    def create_proposal(self) -> List[OrderCandidate]:
        self._last_trade_price, self._ask_baseline, self._bid_baseline = self.get_midprice()
        optimal_bid_price, optimal_ask_price, order_size_bid, order_size_ask, bid_reservation_price, ask_reservation_price, optimal_bid_percent, optimal_ask_percent= self.optimal_bid_ask_spread()

        # Save Values for Status use without recalculating them over and over again
        self.bid_percent = optimal_bid_percent
        self.ask_percent = optimal_ask_percent
        self.b_r_p = bid_reservation_price
        self.a_r_p = ask_reservation_price

        buy_price = optimal_bid_price 
        sell_price = optimal_ask_price 


        if buy_price <= bid_reservation_price:
            buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                    order_side=TradeType.BUY, amount=Decimal(order_size_bid), price=buy_price)
           

        if sell_price >= ask_reservation_price:
            sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                        order_side=TradeType.SELL, amount=Decimal(order_size_ask), price=sell_price)

        minimum_size = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, self.order_amount)

        order_counter = []
        if order_size_bid >= minimum_size:
            order_counter.append(buy_order)

        if order_size_ask >= minimum_size:
            order_counter.append(sell_order)
    

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
        t, y_bid, y_ask, bid_volatility_in_base, ask_volatility_in_base, bid_reservation_price, ask_reservation_price, bid_stdev_price, ask_stdev_price = self.reservation_price()

        ### Counter Method for Constant Trading without using breakeven levels
        # if event.price < self._bid_baseline or event.price <= bid_reservation_price:
        #     self.sell_counter -= 1
        #     self.buy_counter += 1
            
        # if event.price > self._ask_baseline or event.price >= ask_reservation_price:
        #     self.sell_counter += 1
        #     self.buy_counter -= 1

        # if self.sell_counter <= 0:
        #     self.sell_counter = 1

        # if self.buy_counter<= 0:
        #     self.buy_counter = 1

        ### Counter method that resets the buy or sells if a breakeven trade is made. 
        if event.price < self._bid_baseline or event.price <= bid_reservation_price:
            self.sell_counter = 1
            self.buy_counter += 1
            
        if event.price > self._ask_baseline or event.price >= ask_reservation_price:
            self.sell_counter += 1
            self.buy_counter = 1

        if self.sell_counter <= 0:
            self.sell_counter = 1

        if self.buy_counter<= 0:
            self.buy_counter = 1

        self.initialize_flag = False





        #reset S midprice to last traded value
        self._last_trade_price = event.price
        self._last_trade_price = self.connectors[self.exchange].quantize_order_price(self.trading_pair, self._last_trade_price)

        # Update totals and calculate break-even price based on trade type
        fee = event.price * event.amount * self.fee_percent
        if event.price < self._bid_baseline:
            self.total_spent += (event.price * event.amount) + fee
            self.total_bought += event.amount
            if self.total_bought > 0:
                self.break_even_price = self.total_spent / self.total_bought
        if event.price > self._ask_baseline:
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

        lines.extend(["", "| Inventory Imbalance |"])
        lines.extend([f"q(d%) :: {self.q_imbalance:.8f} | Inventory Difference :: {self.inventory_diff:.8f}"])

        lines.extend(["", "| Reservation Prices | Profit Target |"])
        lines.extend([f"Ask :: {self.a_r_p:.8f} | Last Trade Price :: {self._last_trade_price} | Bid :: {self.b_r_p:.8f}"])
        lines.extend([f"Ask(d%) :: {self.ask_percent:.4f} | Bid(d%) :: {self.bid_percent:.4f}"])

        lines.extend(["", "| Market Depth |"])
        lines.extend([f"Ask :: {self.a_d:.8f} | Bid :: {self.b_d:.8f}"])

        lines.extend(["", "| Order History |"])
        lines.extend([f"Buys :: {self.buy_counter - 1} | Sells :: {self.sell_counter - 1}"])

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
        bp = 0.985 # math.exp(math.log(bd)/n)
        ## Sells
        ## 3 distance move,(distance starts at 1 or 100%) 200% above 100 %
        sd = 30
        #sp = max(1 + self.min_profitability, (sd**(1/n)) )
        sp = 1.015 # math.exp(math.log(sd)/n)

        # msg = (f"sp :: {sp:.8f} , bp :: {bp:.8f}")
        # self.log_with_clock(logging.INFO, msg)
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

        # msg2 = (f"additive_sell :: {additive_sell:.8f} , additive_buy :: {additive_buy:.8f}")
        # self.log_with_clock(logging.INFO, msg2)
        # msg = (f"avg_sell_mult :: {avg_sell_mult:.8f} , avg_buy_mult :: {avg_buy_mult:.8f}")
        # self.log_with_clock(logging.INFO, msg)
        # msg3 = (f"Sell Breakeven Mult(s * this = where buy level should be at) :: {sell_breakeven_mult:.8f} , Buy Breakeven Mult(s * this = where sell level should be above) :: {buy_breakeven_mult:.8f}")
        # self.log_with_clock(logging.INFO, msg3)


        return buy_breakeven_mult, sell_breakeven_mult
        

    def get_current_top_bid_ask(self):
        ''' Find the current spread bid and ask prices'''
        top_bid_price = self.connectors[self.exchange].get_price(self.trading_pair, False)
        top_ask_price = self.connectors[self.exchange].get_price(self.trading_pair, True) 
        return top_bid_price, top_ask_price
    
    def get_vwap_bid_ask(self):
        '''Find the bid/ask VWAP of a set price for market depth positioning.'''
        q, _, _, _,_, _, _ = self.get_current_positions()

        # Create an instance of Trades (Market Trades, don't confuse with Limit)
        # buy_trades_instance = BuyTrades('XXLMZEUR')
        # sell_trades_instance = SellTrades('XXLMZEUR')
        # Assuming you want to calculate the 97.5th percentile CDF of buy volumes within the last {window_size} data points
        # Data points are in trades collected
        # target_percentile = 25
        # window_size = 6000

        # Call the method (Market Buy into ask, Sell into bid)
        bid_volume_cdf_value = Decimal(self.sold_volume_depth) #Decimal(sell_trades_instance.get_volume_cdf(target_percentile, window_size))
        ask_volume_cdf_value = Decimal(self.bought_volume_depth) #Decimal(buy_trades_instance.get_volume_cdf(target_percentile, window_size))


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
        sold_baseline, bought_baseline, log_returns_list, self.bought_volume_depth, self.sold_volume_depth = call_kraken_data()

        if self._last_trade_price == None and ( self.buy_counter != 1 or self.sell_counter != 1):
            if self.initialize_flag == True:
                # Fetch midprice only during initialization
                if self._last_trade_price is None:

                    ## If I have to manually restart the bot mid trade, this is the last traded price. 
                    manual_price = 0.084139
                    
                    #self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.MidPrice)

                    # Ensure midprice is not None before converting and assigning
                    if manual_price is not None:
                        self._last_trade_price = (manual_price)
                        self._last_trade_price = self.connectors[self.exchange].quantize_order_price(self.trading_pair, self._last_trade_price)
                        self._bid_baseline = (self._last_trade_price)
                        self._ask_baseline = (self._last_trade_price)
                    self.initialize_flag = False  # Set flag to prevent further updates with midprice

        elif self.buy_counter == 1 and self.sell_counter == 1 and self._last_trade_price == None:

            self._bid_baseline = (sold_baseline)
            self._bid_baseline = self.connectors[self.exchange].quantize_order_price(self.trading_pair, self._bid_baseline)

            self._ask_baseline = (bought_baseline)
            self._ask_baseline = self.connectors[self.exchange].quantize_order_price(self.trading_pair, self._ask_baseline)

    
        else:
            self._last_trade_price = self._last_trade_price
            self._last_trade_price = self.connectors[self.exchange].quantize_order_price(self.trading_pair, self._last_trade_price)

            self._bid_baseline = self._last_trade_price
            self._bid_baseline = self.connectors[self.exchange].quantize_order_price(self.trading_pair, self._bid_baseline)

            self._ask_baseline = self._last_trade_price
            self._ask_baseline = self.connectors[self.exchange].quantize_order_price(self.trading_pair, self._ask_baseline)


        return self._last_trade_price, self._ask_baseline, self._bid_baseline

    def call_garch_model(self):
        sold_baseline, bought_baseline, log_returns_list, self.bought_volume_depth, self.sold_volume_depth = call_kraken_data()

        # Retrieve the log returns from the DataFrame
        log_returns = log_returns_list##self.log_returns

        # Ensure log_returns is a one-dimensional pd.Series
        if isinstance(log_returns, list):
            log_returns = pd.Series(log_returns)

        # Convert to numeric, forcing any errors to NaN
        log_returns_numeric = pd.to_numeric(log_returns, errors='coerce')

        # Remove any NaN or infinite values from log_returns
        log_returns_clean = log_returns_numeric.replace([np.inf, -np.inf], np.nan).dropna()


        # Fit GARCH model to log returns
        current_variance = []
        current_volatility = []
        length = 0
        # Define the GARCH model with automatic rescaling
        model = arch_model(log_returns_clean, vol='Garch', mean='constant', p=1, q=1, power=2.0, rescale=True)

        # Fit the model
        model_fit = model.fit(disp="off")
        msg_gv = (f"GARCH  { model_fit.summary()}")
        self.log_with_clock(logging.INFO, msg_gv)
        # Extract the scale factor used by the model
        scale_factor = model_fit.scale

        # Extract conditional volatility
        volatility_rescaled = np.exp(model_fit.conditional_volatility) - 1
        
        # Apply the scale factor to descaled volatility
        volatility = volatility_rescaled / scale_factor 

        length = len(model_fit.conditional_volatility)

    
        # Convert current_volatility to a pandas Series to apply rolling
        current_volatility_series = pd.Series(volatility)

        # Define the rolling window size (square root of length)
        window = int(np.round(np.sqrt(len(current_volatility_series))))

        # Apply the rolling window to smooth volatility
        rolling_volatility = current_volatility_series#.rolling(window=window).mean()

        # Rank the Volatility
        self.max_vola = rolling_volatility.max()
        min_vola = rolling_volatility.min()
        self.current_vola = current_volatility_series.iloc[-1]

        # Prevent division by zero
        if self.max_vola != min_vola:
            self.volatility_rank = (self.current_vola - min_vola) / (self.max_vola - min_vola)
        else:
            self.volatility_rank = 1  # Handle constant volatility case

        # msg = (f"Volatility :: Rank:{self.volatility_rank}, Max:{self.max_vola}, Min:{min_vola}, Current:{self.current_vola}")
        # self.log_with_clock(logging.INFO, msg)            



    def reservation_price(self):
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        
        self._last_trade_price, self._ask_baseline, self._bid_baseline = self.get_midprice()
       
        buy_breakeven_mult, sell_breakeven_mult = self.determine_log_breakeven_levels()

        TWO = Decimal(2.0)
        HALF = Decimal(0.5)


        s_bid = self._bid_baseline
        s_bid = Decimal(s_bid)

        s_ask = self._ask_baseline
        s_ask = Decimal(s_ask)

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

        bid_reservation_price = (s_bid*Decimal(sell_breakeven_mult)) - (bid_reservation_adjustment) 
        ask_reservation_price = (s_ask*Decimal(buy_breakeven_mult)) - (ask_reservation_adjustment)


        bid_stdev_price = bid_reservation_price - (Decimal(3) * (max_bid_volatility * s_bid))
        ask_stdev_price = ask_reservation_price + (Decimal(3) * (max_ask_volatility * s_ask))

        
        return  t, y_bid, y_ask, bid_volatility_in_base, ask_volatility_in_base, bid_reservation_price, ask_reservation_price, bid_stdev_price, ask_stdev_price




    def optimal_bid_ask_spread(self):
        t, y_bid, y_ask, bid_volatility_in_base, ask_volatility_in_base, bid_reservation_price, ask_reservation_price, bid_stdev_price, ask_stdev_price = self.reservation_price()
        order_size_bid, order_size_ask = self.percentage_order_size()
        q, base_balancing_volume, quote_balancing_volume, total_balance_in_base,entry_size_by_percentage, maker_base_balance, quote_balance_in_base = self.get_current_positions()
        
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


        k_bid_size = y_bid / (bid_inside_exp - Decimal(1))
        k_bid_size = Decimal(k_bid_size)

        bid_k_division = y_bid / k_bid_size
        bid_k_division = self.connectors[self.exchange].quantize_order_amount(self.trading_pair, bid_k_division)

        bid_spread_rate = 1 +  (bid_k_division)
        bid_spread_rate = Decimal(bid_spread_rate)
        bid_log_term = Decimal.ln(bid_spread_rate)



        ask_inside_exp = ((ask_maximum_spread_in_price * y_ask) - (ask_volatility_in_base**TWO * y_ask**TWO)) / TWO
        ask_inside_exp = Decimal(ask_inside_exp).exp()

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
        min_profit_bid =  bid_reservation_price * (Decimal(1) / (Decimal(1) + Decimal(self.min_profitability)))
        min_profit_ask = ask_reservation_price * (Decimal(1) / (Decimal(1) - Decimal(self.min_profitability)))

        # Spread calculation price vs the minimum profit price for entries
        optimal_bid_price = np.minimum(bid_reservation_price -  (optimal_bid_spread  / TWO), min_profit_bid)
        optimal_ask_price = np.maximum(ask_reservation_price +  (optimal_ask_spread / TWO), min_profit_ask)

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
            optimal_bid_price = min(deepest_bid, optimal_bid_price, price_above_bid)
            optimal_ask_price = max(top_ask_price, optimal_ask_price, price_below_ask)
        if q < 0:
            optimal_bid_price = min(top_bid_price, optimal_bid_price, price_above_bid)
            optimal_ask_price = max(deepest_ask, optimal_ask_price, price_below_ask)
        if q == 0:
            optimal_bid_price = min(deepest_bid, optimal_bid_price, price_above_bid)
            optimal_ask_price = max(deepest_ask, optimal_ask_price, price_below_ask)


        if optimal_bid_price <= 0 :
            msg_2 = (f"Error ::: Optimal Bid Price @ {optimal_bid_price} below 0.")
            self.log_with_clock(logging.INFO, msg_2)

            

        # Apply quantum adjustments for final prices
        optimal_bid_price = (floor(optimal_bid_price / bid_price_quantum)) * bid_price_quantum
        optimal_ask_price = (ceil(optimal_ask_price / ask_price_quantum)) * ask_price_quantum

        optimal_bid_percent = ((bid_reservation_price - optimal_bid_price) / bid_reservation_price) * 100
        optimal_ask_percent = ((optimal_ask_price - ask_reservation_price) / ask_reservation_price) * 100

        
        return optimal_bid_price, optimal_ask_price, order_size_bid, order_size_ask, bid_reservation_price, ask_reservation_price, optimal_bid_percent, optimal_ask_percent