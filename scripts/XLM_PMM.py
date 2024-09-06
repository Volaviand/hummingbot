import logging
from decimal import Decimal
from typing import List
import math
from math import floor, ceil
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


### attempt to add your own code from earlier
import sqlite3
import sys
sys.path.append('/home/tyler/quant/API_call_tests/')
from Kraken_Calculations import BuyTrades, SellTrades



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
    trading_pair = "XLM-EUR"
    exchange = "kraken"
    base_asset = "XLM"
    quote_asset = "EUR"

    #Maximum amount of orders  Bid + Ask
    maximum_orders = 170

    inv_target_percent = Decimal(0.50)   

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
    columns_to_show = ["trading_pair", "interval", "volatility", "volatility_bid", "volatility_ask"]
    sort_values_by = ["interval", "volatility"]
    top_n = 20
    report_interval = 60 * 60 * 6  # 6 hours



    ## Breakeven Initialization
    ## Trading Fee for one side Limit
    fee_percent = 0.25 / 100  # Convert percentage to a decimal
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
        self.buy_counter = 3
        self.sell_counter = 1



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
        if all(candle.ready for candle in self.candles.values()):
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
        self._last_trade_price, self._vwap_midprice = self.get_midprice()
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
        
        msg2 = (f"Bid % : {optimal_bid_percent:.4f} , Ask % : {optimal_ask_percent:.4f}, Buy Counter {self.buy_counter}, Sell Counter{self.sell_counter}")
        self.log_with_clock(logging.INFO, msg2)

        #msgbe = (f"BreakEven : {self.break_even_price} , Total Spent : {self.total_spent}, Total Bought : {self.total_bought}, Total Earned : {self.total_earned},  Total Sold : {self.total_sold}")
        #self.log_with_clock(logging.INFO, msgbe)
        #self.notify_hb_app_with_timestamp(msg)

        msgce = (f"Bid Starting Price : {bid_starting_price:.8f}, Ask Starting Price : {ask_starting_price:.8f}")
        self.log_with_clock(logging.INFO, msgce)

        return [buy_order , sell_order]

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

        if event.price < self._last_trade_price or event.price <= bid_reservation_price:
            self.sell_counter -= 1
            self.buy_counter += 1
            
        if event.price > self._last_trade_price or event.price >= ask_reservation_price:
            self.sell_counter += 1
            self.buy_counter -= 1

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

        n = math.floor(self.maximum_orders/2)
        ## Buys
        #Minimum Distance in percent. 0.01 = a drop of 99% from original value
        bd = 0.25
        ## Percent multiplier, <1 = buy(goes down), >1 = sell(goes up) 
        #p = (1 - 0.05)
        #bp = min( 1 - self.min_profitability, bd**(1/n) )
        bp = math.exp(math.log(bd)/n)
        ## Sells
        ## 3 distance move,(distance starts at 1 or 100%) 200% above 100 %
        sd = 3
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
            for i in range(1, buy_counter_adjusted+1):
                additive_buy += bp**i
                avg_buy_mult = additive_buy / buy_counter_adjusted
                buy_breakeven_mult = avg_buy_mult / (bp**buy_counter_adjusted)
        else:
            additive_buy = 0
            avg_buy_mult = 1
            buy_breakeven_mult = 1

        if sell_counter_adjusted > 0:
            for i in range(1, sell_counter_adjusted+1):
                additive_sell += sp**i
                avg_sell_mult = additive_sell/sell_counter_adjusted
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

        msg_lastrade = (f"ask_entry_percents {ask_entry_percents}, bid_entry_percents{bid_entry_percents}")
        self.log_with_clock(logging.INFO, msg_lastrade)
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
        volatility_metrics_df, self.target_profitability= self.get_market_analysis()
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
            df["volatility"] = df["close"].pct_change().rolling(self.volatility_interval).std()
            df["volatility_bid"] = df["low"].pct_change().rolling(self.volatility_interval).std()
            df["volatility_bid_max"] = df["low"].pct_change().rolling(self.volatility_interval).std().max()
            df["volatility_bid_min"] = df["low"].pct_change().rolling(self.volatility_interval).std().min()
            
            df["volatility_ask"] = df["high"].pct_change().rolling(self.volatility_interval).std()
            df["volatility_ask_max"] = df["high"].pct_change().rolling(self.volatility_interval).std().max()
            df["volatility_ask_min"] = df["high"].pct_change().rolling(self.volatility_interval).std().min()

            df["volatility_pct"] = df["volatility"] / df["close"]
            df["volatility_pct_mean"] = df["volatility_pct"].rolling(self.volatility_interval).mean()

            # adding bbands metrics
            df.ta.bbands(length=self.volatility_interval, append=True)
            df["bbands_width_pct"] = df[f"BBB_{self.volatility_interval}_2.0"]
            df["bbands_width_pct_mean"] = df["bbands_width_pct"].rolling(self.volatility_interval).mean()
            df["bbands_percentage"] = df[f"BBP_{self.volatility_interval}_2.0"]
            df["natr"] = ta.natr(df["high"], df["low"], df["close"], length=self.volatility_interval)
            market_metrics[trading_pair_interval] = df.iloc[-1]
        volatility_metrics_df = pd.DataFrame(market_metrics).T
        self.target_profitability = max(self.min_profitability, volatility_metrics_df["volatility"].iloc[-1])
        return volatility_metrics_df, self.target_profitability


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


        amount_base_to_hold = Decimal(0.10)
        amount_base_rate = Decimal(1.0) - amount_base_to_hold
        
        amount_quote_to_hold = Decimal(0.10)
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
            base_balancing_volume =  abs(minimum_size) *  Decimal.exp(-self.order_shape_factor * q)
            quote_balancing_volume = abs(minimum_size) * ( 1 + ( 1 - Decimal.exp(-self.order_shape_factor * q))) 
            # Ensure base balancing volume does not exceed the amount needed to balance
            if quote_balancing_volume > total_imbalance:
                quote_balancing_volume = total_imbalance

        elif q < 0 :
            base_balancing_volume = abs(minimum_size) *  ( 1 + ( 1 - Decimal.exp(self.order_shape_factor * q)))
            quote_balancing_volume = abs(minimum_size) * Decimal.exp(self.order_shape_factor * q) 

            # Ensure base balancing volume does not exceed the amount needed to balance
            if base_balancing_volume > total_imbalance:
                base_balancing_volume = total_imbalance
         
        else :
            base_balancing_volume = minimum_size
            quote_balancing_volume = minimum_size



        
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
    
    def get_midprice(self):

        if self._last_trade_price == None:
            if self.initialize_flag == True:
                # Fetch midprice only during initialization
                if self._last_trade_price is None:
                    midprice = 0.083782 #self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.MidPrice)
                    # Ensure midprice is not None before converting and assigning
                    if midprice is not None:
                        self._last_trade_price = Decimal(midprice)
                    self.initialize_flag = False  # Set flag to prevent further updates with midprice
    
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

    def reservation_price(self):
        volatility_metrics_df, self.target_profitability = self.get_market_analysis()
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
        df = volatility_metrics_df
        volatility = df["volatility"].iloc[-1]
        volatility_bid = df["volatility_bid"].iloc[-1]
        volatility_ask = df["volatility_ask"].iloc[-1]

        volatility_bid_denominator = df["volatility_bid_max"].iloc[-1] - df["volatility_bid_min"].iloc[-1]
        volatility_ask_denominator = df["volatility_ask_max"].iloc[-1] - df["volatility_ask_min"].iloc[-1]

        if volatility_bid_denominator == 0:
            volatility_bid_denominator = 0.0000000001

        if volatility_ask_denominator == 0:
            volatility_ask_denominator = 0.0000000001            

        volatility_bid_rank = (df["volatility_bid"].iloc[-1] - df["volatility_bid_min"].iloc[-1]) / (volatility_bid_denominator)
        volatility_ask_rank = (df["volatility_ask"].iloc[-1] - df["volatility_ask_min"].iloc[-1]) / (volatility_ask_denominator)




        volatility_bid_rank = Decimal(volatility_bid_rank)
        volatility_ask_rank = Decimal(volatility_ask_rank)

        if volatility_bid_rank <= 0:
            volatility_bid_rank = Decimal(0.000000001)
        elif volatility_bid_rank >= 1:
            volatility_bid_rank = Decimal(1.0)

        if volatility_ask_rank <= 0:
            volatility_ask_rank = Decimal(0.000000001)
        elif volatility_ask_rank >= 1:
            volatility_ask_rank = Decimal(1.0)


        if volatility <=0:
            volatility = 0 
        else:
            volatility = volatility_metrics_df["volatility"].iloc[-1]

        if volatility_bid <=0:
            volatility_bid = 0 
        else:
            volatility_bid = volatility_metrics_df["volatility_bid"].iloc[-1]

        if volatility_ask <=0:
            volatility_ask = 0 
        else:
            volatility_ask = volatility_metrics_df["volatility_ask"].iloc[-1]     

        ### Convert Volatility Percents into Absolute Prices




        max_bid_volatility= Decimal(volatility_bid) 
        bid_volatility_in_base = (max_bid_volatility) * s 

        max_ask_volatility = Decimal(volatility_ask) 
        ask_volatility_in_base = (max_ask_volatility) * s 


        msg_4 = (f"max_bid_volatility @ {bid_volatility_in_base:.8f} ::: max_ask_volatility @ {ask_volatility_in_base:.8f}")
        self.log_with_clock(logging.INFO, msg_4)

        #INVENTORY RISK parameter, 0 to 1, higher = more risk averse, as y <-- 0, it behaves more like usual
        # Adjust the width of the y parameter based on volatility, the more volatile , the wider the spread becomes, y goes higher
        y = Decimal(1.0)
        y_min = Decimal(0.5)
        y_max = Decimal(1.0)
        y_difference = y_max - y_min

        y_bid = y - (volatility_bid_rank * y_difference)
        y_ask = y - (volatility_ask_rank * y_difference)
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




        geom_limit_bid = bid_starting_price * bp ##geom_spread_bid 
        geom_limit_ask = ask_starting_price * sp ##geom_spread_ask 
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


    