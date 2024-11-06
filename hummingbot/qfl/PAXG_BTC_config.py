# strategy_config.py
from decimal import Decimal

# Configuration parameters to be used for instantiating the bot
STRATEGY_CONFIG = {
    'trading_pair': 'PAXG-BTC',
    'exchange': 'kraken',
    'base_asset': 'PAXG',
    'quote_asset': 'BTC',
    'history_market': 'PAXGXBT',
    'min_profitability': Decimal(0.015),
    'buy_p': Decimal(0.975),
    'sell_p': Decimal(1.025),
    'quote_order_amount': Decimal(0.0001),
    'order_amount': Decimal(0.002),
    'max_order_amount': Decimal(0.01),
    'maximum_orders': 50,
    'inv_target_percent': Decimal(0.50),
    'order_shape_factor': Decimal(2.0),
    'history_name': 'PAXGXBT_60.csv',
    'trade_history_name': 'trades.PAXG_BTC',
    'chart_period': '60',
    'volatility_periods': 168,
    'rolling_periods': 12,
    'trading_style': 'QFL', # Account Building or QFL
}