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

        # self.set_market('24/7')

        
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
        # self.model = TradingModel()
        self.models = {s: TradingModel(symbol=s) for s in self.all_symbols} # CHANGED (18062026)
        self.tracker = DecayTracker(self.all_symbols)

        # State tracking
        self.iteration_count = 0
        self.peak_value = None
        self.prev_prices = {}
        self.last_predictions = {}  # {symbol: direction} from prior iteration
        self.paused = False
        self.asset_returns = {s: pd.Series(dtype=float) for s in self.all_symbols}
        self._prev_features = {}  # features from prior iteration for labeling

        self.target_streak = {} # CHANGED (12062026) {symbol: consecutive iterations with positive target weight}
        self.exit_streak = {} # CHANGED (12062026) {symbol: consecutive iterations a held position is out of targets}

        self.entry_prices = {}      # CHANGED (21062026) {symbol: price at which we last bought}
        # self.last_sold_prices = {}  # CHANGED (21062026) {symbol: price at which we last sold}
        self.accumulated_fees = {}

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

                    if self.prev_prices[symbol] == current_prices[symbol]:
                        continue  # CHANGED (12062026) stale bar (market closed, price unchanged) — skip

                    target = TradingModel.compute_target(
                        self.prev_prices[symbol], current_prices[symbol]
                    )
                    # self.model.add_sample(feat, target)
                    self.models[symbol].add_sample(feat, target)

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
                # direction, confidence = self.model.predict(features)
                # signals[symbol] = (direction, confidence)
                if self.models[symbol].model is not None: # CHANGED (18062026)
                    direction, confidence = self.models[symbol].predict(features)
                    signals[symbol] = (direction, confidence)
            except Exception as e:
                self.log_message(f"Feature/predict error for {symbol}: {e}")
                continue

        # -- Store state for next iteration --
        self._prev_features = current_features
        self.last_predictions = {s: sig[0] for s, sig in signals.items()}
        self.prev_prices = current_prices.copy()

        # -- Train / retrain model --
        # if self.iteration_count == 1 or self.iteration_count % P.RETRAIN_INTERVAL == 0:
        #     if self.model.has_enough_data():
        #         try:
        #             success = self.model.train()
        #             self.log_message(
        #                 f"Model retrain: {'updated' if success else 'kept previous'} | "
        #                 f"samples={len(self.model.feature_history)} | "
        #                 f"val_acc={self.model.last_val_accuracy:.3f}"
        #             )
        #         except Exception as e:
        #             self.log_message(f"Retrain failed: {e}")
        if self.iteration_count == 1 or self.iteration_count % P.RETRAIN_INTERVAL == 0: # CHANGED (18062026) from above
            # self.last_sold_prices.clear()  # clear stale buyback memory each retrain cycle
            for symbol, mdl in self.models.items():
                if mdl.has_enough_data():
                    try:
                        success = mdl.train()
                        self.log_message(
                            f"Model retrain [{symbol}]: {'updated' if success else 'kept previous'} | "
                            f"samples={len(mdl.feature_history)} | "
                            f"val_acc={mdl.last_val_accuracy:.3f}"
                        )
                    except Exception as e:
                        self.log_message(f"Retrain failed [{symbol}]: {e}")

        # -- If model not ready yet, stay in cash --
        # if self.model.model is None:
        #     self.log_message(
        #         f"Collecting data: {len(self.model.feature_history)}/{P.MIN_TRAINING_SAMPLES} samples"
        #     )
        #     return
        ready_count = sum(1 for m in self.models.values() if m.model is not None) # CHANGED (18062026) from above
        if ready_count == 0:
            total_samples = sum(len(m.feature_history) for m in self.models.values())
            self.log_message(
                f"Collecting data: {total_samples} total samples, no models ready yet"
            )
            return

        # -- Position sizing --
        target_weights = compute_position_sizes(
            signals, self.tracker, portfolio_value, current_drawdown
        )


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

        # -- Execute orders --
        self._rebalance(target_weights, current_prices, portfolio_value)

        # -- Logging --
        self.log_message(
            f"iter={self.iteration_count} | portfolio=${portfolio_value:,.2f} | "
            f"cash=${cash:,.2f} | drawdown={current_drawdown:.2%} | "
            f"positions={len(target_weights)} | signals={len(signals)}"
        )

        # -- Out of sample testing --
        if self.iteration_count == 1:
            self._oos_logged = False
        if not hasattr(self, '_oos_logged'):
            self._oos_logged = False
        if not self._oos_logged and str(self.get_datetime().date()) >= "2025-07-01":
            self.log_message(f"=== OOS START === portfolio=${portfolio_value:,.2f}")
            self._oos_logged = True

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    def _rebalance(self, target_weights, prices, portfolio_value):
        """Rebalance portfolio to match target weights. Sells first, then buys."""

        available_cash = float(self.get_cash()) # CHANGED (12062025) to check cash before buying to avoid negative cash

        current_positions = {}
        for p in self.get_positions():
            sym = p.asset.symbol
            current_positions[sym] = p

        # -- Sell: exit positions not in targets or with negative weight -- 
        for symbol, position in current_positions.items():
            target_w = target_weights.get(symbol, 0)
            if target_w <= 0 and position.quantity > 0:
                if self.exit_streak.get(symbol, 0) < P.EXIT_PERSISTENCE:
                    continue  # CHANGED (12062026) now included, signal must stay dead for N consecutive hours before exiting

                price = prices.get(symbol, 0) # CHANGED (21062026)
                entry_price = self.entry_prices.get(symbol, 0)

                # Stop-loss override: always allow sell if position is down too much
                if entry_price > 0 and price > 0: # CHANGED (21062026)
                    position_return = (price - entry_price) / entry_price
                    is_stop_loss = position_return <= -P.MAX_POSITION_LOSS
                    total_entry_value = position.quantity * entry_price
                    total_buy_fees = self.accumulated_fees.get(symbol, 0)
                    sell_fee = position.quantity * price * P.PERCENT_FEE_PER_SIDE
                    min_profit_needed = (total_buy_fees + sell_fee) * P.MIN_SELL_PROFIT_MULTIPLIER / total_entry_value
                    meets_profit_target = position_return >= min_profit_needed
                    if not meets_profit_target and not is_stop_loss:
                        continue  # price hasn't moved enough to justify selling

                order = self.create_order(symbol, position.quantity, "sell")
                self.submit_order(order)
                self.exit_streak.pop(symbol, None) # CHANGED (12062026)
                # self.last_sold_prices[symbol] = price  # CHANGED (21062026) track sold price
                self.entry_prices.pop(symbol, None)    # CHANGED (21062026) clear entry price
                self.accumulated_fees.pop(symbol, None)
                
                self.log_message(f"SELL ALL {symbol}: qty={position.quantity}")

        # -- Buy / adjust remaining positions --
        for symbol, weight in target_weights.items():
            if weight <= 0:
                continue

            price = prices.get(symbol)
            if not price or price <= 0:
                continue

            # CHANGED (21062026) Buy-back threshold: only buy if price has dropped enough from last sold price
            # last_sold = self.last_sold_prices.get(symbol)
            # if last_sold is not None: # and price > last_sold * (1 - P.MIN_BUYBACK_DROP):
            #     self.log_message(f"[BUYBACK BLOCKED] {symbol}: price=${price:.2f} > sold=${last_sold:.2f} * {1 - P.MIN_BUYBACK_DROP:.4f} = ${last_sold * (1 - P.MIN_BUYBACK_DROP):.2f}")
            #     continue

            target_value = abs(weight) * portfolio_value * 0.65 # CHANGED (12062026) from *1 to *0.6 to reduce amount transacted each trade

            current_value = 0
            current_qty = 0
            if symbol in current_positions:
                current_qty = current_positions[symbol].quantity
                current_value = current_qty * price

            tolerance = 0.05 * current_value # CHANGED (12062026) included now to ignore rebalance for small price movements

            diff_value = target_value - current_value
            cash_reserve = available_cash * (P.CASH_BUFFER) # CHANGED (06122026)
            max_order_notional = 100000 # CHANGED (06122026)
            spendable = available_cash - cash_reserve
            diff_value = min(diff_value, max_order_notional, spendable) # CHANGED (06122026)
            if abs(diff_value) < tolerance or (available_cash - abs(diff_value)) < cash_reserve or diff_value <= 0:  # skip tiny adjustments, also CHANGED from 10 to tolerance to ignore rebalance for small price movements
                continue
            # CHANGED (12062026) abs(diff_value) > cash_reserve to prevent overspending.

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
                self.submit_order(order)
                available_cash -= quantity * price * (1 + P.PERCENT_FEE_PER_SIDE) # To avoid negative cash
                # self.entry_prices[symbol] = price  # track entry price
                old_price = self.entry_prices.get(symbol, 0)
                old_qty = current_qty
                # self.last_sold_prices.pop(symbol, None)  # clear sold price
                if old_price > 0 and old_qty > 0:
                    self.entry_prices[symbol] = (old_qty * old_price + quantity * price) / (old_qty + quantity)
                else:
                    self.entry_prices[symbol] = price
                self.accumulated_fees[symbol] = self.accumulated_fees.get(symbol, 0) + quantity * price * P.PERCENT_FEE_PER_SIDE
                self.log_message(f"BUY {symbol}: qty={quantity} @ ${price:,.2f}")
            else:
                quantity = min(quantity, current_qty)
                if quantity > 0:
                    order = self.create_order(symbol, quantity, "sell")
                    self.submit_order(order)
                    self.log_message(f"SELL {symbol}: qty={quantity} @ ${price:,.2f}")
                else:
                    continue

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
