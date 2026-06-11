import pandas as pd
import numpy as np

from lumibot.strategies import Strategy as _LumibotStrategy
from lumibot.entities import Asset

from strategies import params as P
from strategies.features import compute_features, compute_cross_asset_features
from strategies.model import TradingModel
from strategies.risk import DecayTracker, compute_position_sizes

base = Asset(symbol="BTC", asset_type="crypto")
quote = Asset(symbol="USD", asset_type="forex")
last_price = lumibot.strategies.strategy.Strategy.get_last_price(base, quote=quote)