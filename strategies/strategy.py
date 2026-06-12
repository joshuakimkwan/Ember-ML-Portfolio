"""
SoAI 2026 AI Algorithmic Trading Competition - ML Strategy.

XGBoost signal generation + exponential decay adaptive position sizing.
Scores 15 equity/crypto candidates hourly, holds max 5 positions.

The class name MUST remain ``Strategy`` and this file MUST stay at
``strategies/strategy.py`` for the competition execution environment.
"""

import pandas as pd
import numpy as np

from lumibot.strategies import Strategy as _LumibotStrategy
from lumibot.entities import Asset

from strategies import params as P
from strategies.features import compute_features, compute_cross_asset_features
from strategies.model import TradingModel
from strategies.risk import DecayTracker, compute_position_sizes


class Strategy(_LumibotStrategy):
    """ML trading strategy with adaptive position sizing."""

    # ------------------------------------------------------------------
    # Lifecycle: setup
    # ------------------------------------------------------------------
    def initialize(self):
        self.sleeptime = P.SLEEPTIME

        # For trading crypto
<<<<<<< HEAD
        # self.set_market('24/7')
=======
        self.set_market('24/7')
>>>>>>> e1646a59097e74270aee44d37c54b2a25f477887
        
        # Build asset objects
        self.equity_assets = {
            s: Asset(s, asset_type=Asset.AssetType.STOCK)
            for s in P.STOCK_SLEEVE_SYMBOLS
        }
        self.crypto_assets = {
            s: Asset(s, asset_type=Asset.AssetType.CRYPTO)
            for s in P.CRYPTO_SLEEVE_SYMBOLS
        }

        self.all_assets = {**self.equity_assets, **self.crypto_assets}
        self.all_symbols = list(self.all_assets.keys())

        # ML model and adaptive tracker
        self.model = TradingModel()
        self.tracker = DecayTracker(self.all_symbols)

        # State tracking
        self.iteration_count = 0
        self.peak_value = None
        self.prev_prices = {}
        self.last_predictions = {}  # {symbol: direction} from prior iteration
        self.paused = False
        self.asset_returns = {s: pd.Series(dtype=float) for s in self.all_symbols}
        self._prev_features = {}  # features from prior iteration for labeling
<<<<<<< HEAD
        self.target_streak = {} # CHANGED (12062026) {symbol: consecutive iterations with positive target weight}
        self.exit_streak = {} # CHANGED (12062026) {symbol: consecutive iterations a held position is out of targets}
=======
>>>>>>> e1646a59097e74270aee44d37c54b2a25f477887

        self.log_message(
            f"ML Strategy initialized | universe={len(self.all_symbols)} assets | "
            f"max_positions={P.MAX_POSITIONS} | sleeptime={P.SLEEPTIME}"
        )

    # ------------------------------------------------------------------
    # Lifecycle: per-step decision making
    # ------------------------------------------------------------------
    def on_trading_iteration(self):
        
        self.iteration_count += 1
        portfolio_value = self.get_portfolio_value()
        cash = self.get_cash()

        # -- Drawdown tracking --
        if self.peak_value is None or portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        current_drawdown = (
            (self.peak_value - portfolio_value) / self.peak_value
            if self.peak_value > 0 else 0.0
        )

        # -- Pause/resume on max drawdown --
        if self.paused:
            if current_drawdown <= P.DRAWDOWN_RECOVERY:
                self.paused = False
                self.log_message("Drawdown recovered, resuming trading")
            else:
                self.log_message(
                    f"[PAUSED] drawdown={current_drawdown:.2%} | "
                    f"portfolio=${portfolio_value:,.2f}"
                )
                return

        if current_drawdown >= P.MAX_DRAWDOWN:
            self.paused = True
            self._sell_all()
            self.log_message(
                f"MAX DRAWDOWN HIT ({current_drawdown:.2%}), liquidating all positions"
            )
            return

        # -- Collect current prices and historical data --
        current_prices = {}
        historical = {}
        crypto_quote = Asset(symbol="USD", asset_type=Asset.AssetType.CRYPTO)
        for symbol in self.all_symbols:
            try:
                is_crypto = symbol in P.CRYPTO_SYMBOLS
                is_stock = symbol in P.STOCK_SLEEVE_SYMBOLS
                if is_stock:
                    price = self.get_last_price(symbol)
                    if price is None or price <= 0:
                        continue
                    current_prices[symbol] = price
                    
                    bars = self.get_historical_prices(
                        symbol, length=100, timestep="hour"
                    )

                    if bars is not None and hasattr(bars, "df") and len(bars.df) >= P.MIN_HISTORY_BARS:
                        historical[symbol] = bars.df
                if is_crypto:
                    ast = Asset(symbol=symbol, asset_type=Asset.AssetType.CRYPTO)
                    price = self.get_last_price(ast, quote=crypto_quote)
                    
                    if price is None or price <= 0:
                        continue
                    current_prices[symbol] = price
                    
                    bars = self.get_historical_prices(
                        ast, length=100, timestep="hour"
                    )

                    if bars is not None and hasattr(bars, "df") and len(bars.df) >= P.MIN_HISTORY_BARS:
                        historical[symbol] = bars.df
            except Exception as e:
                self.log_message(f"Data error for {symbol}: {e}")
                continue

        if not current_prices:
            self.log_message("No price data available, skipping iteration")
            return

        # -- Update rolling returns for cross-asset features --
        for symbol, price in current_prices.items():
            if symbol in self.prev_prices and self.prev_prices[symbol] > 0:
                ret = (price - self.prev_prices[symbol]) / self.prev_prices[symbol]
                new_entry = pd.Series([ret])
                self.asset_returns[symbol] = pd.concat(
                    [self.asset_returns[symbol], new_entry], ignore_index=True
                ).tail(100)

        # -- Update decay tracker from last iteration's predictions --
        for symbol, predicted_dir in self.last_predictions.items():
            if symbol in current_prices and symbol in self.prev_prices:
                actual_return = (
                    (current_prices[symbol] - self.prev_prices[symbol])
                    / self.prev_prices[symbol]
                )
                if actual_return > P.RETURN_DEAD_ZONE:
                    actual_dir = 1
                elif actual_return < -P.RETURN_DEAD_ZONE:
                    actual_dir = -1
                else:
                    actual_dir = 0
                self.tracker.update(symbol, predicted_dir == actual_dir)

        # -- Add training data from previous iteration --
        if self._prev_features and self.prev_prices:
            for symbol, feat in self._prev_features.items():
                if symbol in current_prices and symbol in self.prev_prices:
<<<<<<< HEAD
                    if self.prev_prices[symbol] == current_prices[symbol]:
                        continue  # CHANGED (12062026) stale bar (market closed, price unchanged) — skip
=======
>>>>>>> e1646a59097e74270aee44d37c54b2a25f477887
                    target = TradingModel.compute_target(
                        self.prev_prices[symbol], current_prices[symbol]
                    )
                    self.model.add_sample(feat, target)

        # -- Feature engineering + prediction --
        signals = {}
        current_features = {}
        for symbol, df in historical.items():
            try:
                features = compute_features(df)
                cross_feats = compute_cross_asset_features(
                    self.asset_returns, symbol, self.all_symbols
                )
                for k, v in cross_feats.items():
                    features[k] = v

                current_features[symbol] = features
                direction, confidence = self.model.predict(features)
                signals[symbol] = (direction, confidence)
            except Exception as e:
                self.log_message(f"Feature/predict error for {symbol}: {e}")
                continue

        # -- Store state for next iteration --
        self._prev_features = current_features
        self.last_predictions = {s: sig[0] for s, sig in signals.items()}
        self.prev_prices = current_prices.copy()

        # -- Train / retrain model --
        if self.iteration_count == 1 or self.iteration_count % P.RETRAIN_INTERVAL == 0:
            if self.model.has_enough_data():
                try:
                    success = self.model.train()
                    self.log_message(
                        f"Model retrain: {'updated' if success else 'kept previous'} | "
                        f"samples={len(self.model.feature_history)} | "
                        f"val_acc={self.model.last_val_accuracy:.3f}"
                    )
                except Exception as e:
                    self.log_message(f"Retrain failed: {e}")

        # -- If model not ready yet, stay in cash --
        if self.model.model is None:
            self.log_message(
                f"Collecting data: {len(self.model.feature_history)}/{P.MIN_TRAINING_SAMPLES} samples"
            )
            return

        # -- Position sizing --
        target_weights = compute_position_sizes(
            signals, self.tracker, portfolio_value, current_drawdown
        )

<<<<<<< HEAD
        # -- Entry persistence: update consecutive-target streaks -- CHANGED (12062026)
        for symbol in self.all_symbols:
            if target_weights.get(symbol, 0) > 0:
                self.target_streak[symbol] = self.target_streak.get(symbol, 0) + 1
            else:
                self.target_streak[symbol] = 0

        # -- Block first-time buys until streak >= ENTRY_PERSISTENCE -- # CHANGED (12062026), now included this
        held = {p.asset.symbol for p in self.get_positions() if p.quantity > 0}
        target_weights = {
            s: w for s, w in target_weights.items()
            if s in held or self.target_streak.get(s, 0) >= P.ENTRY_PERSISTENCE
        }

        # -- Exit persistence: count consecutive iterations held symbols are out of targets -- # CHANGED (12062026), now included this
        for symbol in held:
            if target_weights.get(symbol, 0) <= 0:
                self.exit_streak[symbol] = self.exit_streak.get(symbol, 0) + 1
            else:
                self.exit_streak[symbol] = 0

=======
>>>>>>> e1646a59097e74270aee44d37c54b2a25f477887
        # -- Execute orders --
        self._rebalance(target_weights, current_prices, portfolio_value)

        # -- Logging --
        self.log_message(
            f"iter={self.iteration_count} | portfolio=${portfolio_value:,.2f} | "
            f"cash=${cash:,.2f} | drawdown={current_drawdown:.2%} | "
            f"positions={len(target_weights)} | signals={len(signals)}"
        )

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    def _rebalance(self, target_weights, prices, portfolio_value):
        """Rebalance portfolio to match target weights. Sells first, then buys."""
<<<<<<< HEAD
        available_cash = float(self.get_cash()) # CHANGED (12062025) to check cash before buying to avoid negative cash

=======
>>>>>>> e1646a59097e74270aee44d37c54b2a25f477887
        current_positions = {}
        for p in self.get_positions():
            sym = p.asset.symbol
            current_positions[sym] = p

<<<<<<< HEAD
        # -- Sell: exit positions not in targets or with negative weight -- 
        for symbol, position in current_positions.items():
            target_w = target_weights.get(symbol, 0)
            if target_w <= 0 and position.quantity > 0:
                if self.exit_streak.get(symbol, 0) < P.EXIT_PERSISTENCE:
                    continue  # CHANGED (12062026) now included, signal must stay dead for N consecutive hours before exiting
                order = self.create_order(symbol, position.quantity, "sell")
                self.submit_order(order)
                self.exit_streak.pop(symbol, None) # CHANGED (12062026)
=======
        # -- Sell: exit positions not in targets or with negative weight --
        for symbol, position in current_positions.items():
            target_w = target_weights.get(symbol, 0)
            if target_w <= 0 and position.quantity > 0:
                order = self.create_order(symbol, position.quantity, "sell")
                self.submit_order(order)
>>>>>>> e1646a59097e74270aee44d37c54b2a25f477887
                self.log_message(f"SELL ALL {symbol}: qty={position.quantity}")

        # -- Buy / adjust remaining positions --
        for symbol, weight in target_weights.items():
            if weight <= 0:
                continue

            price = prices.get(symbol)
            if not price or price <= 0:
                continue

<<<<<<< HEAD
            target_value = abs(weight) * portfolio_value * 0.65 # CHANGED (12062026) from *1 to *0.6 to reduce amount transacted each trade
=======
            target_value = abs(weight) * portfolio_value
>>>>>>> e1646a59097e74270aee44d37c54b2a25f477887
            current_value = 0
            current_qty = 0
            if symbol in current_positions:
                current_qty = current_positions[symbol].quantity
                current_value = current_qty * price

<<<<<<< HEAD
            tolerance = 0.4 * current_value # CHANGED (12062026) included now to ignore rebalance for small price movements
            diff_value = target_value - current_value
            cash_reserve = available_cash * (P.CASH_BUFFER) # CHANGED (06122026)
            max_order_notional = 100000 # CHANGED (06122026)
            diff_value = min(diff_value, max_order_notional) # CHANGED (06122026)
            if abs(diff_value) < tolerance or abs(diff_value) > cash_reserve:  # skip tiny adjustments, also CHANGED from 10 to tolerance to ignore rebalance for small price movements
                continue
            # CHANGED (12062026) abs(diff_value) > cash_reserve to prevent overspending.
=======
            diff_value = target_value - current_value
            if abs(diff_value) < 10:  # skip tiny adjustments
                continue
>>>>>>> e1646a59097e74270aee44d37c54b2a25f477887

            # Crypto allows fractional, stocks need whole shares
            is_crypto = symbol in P.CRYPTO_SYMBOLS
            if is_crypto:
                quantity = round(abs(diff_value) / price, 6)
            else:
                quantity = int(abs(diff_value) / price)

            if quantity <= 0:
                continue

            if diff_value > 0:
                order = self.create_order(symbol, quantity, "buy")
                self.log_message(f"BUY {symbol}: qty={quantity} @ ${price:,.2f}")
            else:
                quantity = min(quantity, current_qty)
                if quantity > 0:
                    order = self.create_order(symbol, quantity, "sell")
                    self.log_message(f"SELL {symbol}: qty={quantity} @ ${price:,.2f}")
                else:
                    continue
            self.submit_order(order)

    def _sell_all(self):
        """Liquidate all positions (used when max drawdown is hit)."""
        for position in self.get_positions():
            if position.quantity > 0:
                order = self.create_order(
                    position.asset.symbol, position.quantity, "sell"
                )
                self.submit_order(order)
                self.log_message(
                    f"LIQUIDATE {position.asset.symbol}: qty={position.quantity}"
                )
