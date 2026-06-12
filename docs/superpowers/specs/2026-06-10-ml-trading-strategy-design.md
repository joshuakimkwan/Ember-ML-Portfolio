# ML Algorithmic Trading Strategy — Design Spec

**Date:** 2026-06-10
**Competition:** SoAI 2026 AI Algorithmic Trading Competition
**Trading window:** 1 Aug 2026 – 31 Aug 2026
**Return target:** 10% | **Max drawdown:** 35%

---

## 1. Architecture Overview

Single XGBoost model scoring 15 candidate assets hourly, with an exponential decay tracker adjusting position sizes based on recent prediction accuracy. Max 5 concurrent positions.

```
on_trading_iteration() — every 60 minutes:
  1. Feature Engineering → 30 features per asset from historical bars
  2. Signal Generation   → XGBoost predicts direction + confidence
  3. Adaptive Sizing     → Decay tracker weights signals by recent accuracy
  4. Risk Management     → Drawdown checks, position caps, cash buffer
  5. Order Execution     → Sell stale positions, buy new targets
  6. Model Update        → Retrain XGBoost every 24 iterations
```

## 2. Asset Universe (15 candidates, max 5 held)

- **Equities (8):** TSLA, NVDA, AAPL, AMZN, META, MSFT, GOOGL, AMD
- **Crypto (7):** BTC, ETH, SOL, DOGE, XRP, AVAX, LINK

The model scores all 15 each hour and allocates capital to the top 5 by risk-adjusted signal strength.

## 3. Feature Engineering (~30 features per asset)

Computed from 60–100 bars of historical data via `get_historical_prices()`.

### Price-based (12)
- Returns: 1h, 4h, 12h, 24h percentage returns
- Moving averages: SMA(10), SMA(20), EMA(12), EMA(26) as distance-from-price ratios
- Bollinger Bands: upper/lower distance, bandwidth, %B position

### Momentum (6)
- RSI(14), RSI(7)
- MACD line, signal, histogram
- Rate of change (ROC) over 12 periods

### Volatility (4)
- ATR(14) normalized by price
- Std dev of returns (12h, 24h)
- Intraday range ratio (high-low / close)

### Volume (4)
- Volume ratio vs 20-period average
- VWAP deviation
- On-balance volume slope
- Volume spike detector (binary: >2x average)

### Cross-asset (4)
- Correlation with BTC over trailing 24h
- Correlation with SPY-proxy (AAPL) over trailing 24h
- Rank of momentum vs all 15 candidates
- Rank of volatility vs all 15 candidates

### Target Variable
Sign of next-hour return: +1 (buy), 0 (hold), -1 (sell). Dead zone: returns within +/-0.1% labeled as 0.

## 4. Adaptive Decay Tracker & Position Sizing

### Decay formula
```
accuracy_score[asset] = sum(decay^t * was_correct[t]) / sum(decay^t)
```
- `decay = 0.97` (~48-hour half-life)
- `was_correct[t]` = 1 if predicted direction matched actual movement

### Position sizing pipeline
1. **Raw signal:** XGBoost direction (+1/0/-1) and confidence (probability 0.5–1.0)
2. **Adjusted signal:** `raw_confidence * accuracy_score[asset]`
3. **Rank & select:** Top 5 by absolute adjusted signal (non-zero only)
4. **Weight allocation:** Proportional to adjusted signal strength, capped at 40% per position
5. **Drawdown scaling:** Linear reduction from 20% to 35% drawdown; at 35% go fully to cash

```
if drawdown > 20%:
    scale = (0.35 - drawdown) / (0.35 - 0.20)  # 1.0 at 20%, 0.0 at 35%
    all weights *= scale
```

## 5. Model Training & Retraining

### Initial training (in `initialize()`)
- Train on all available historical data
- Train/val split: last 20% held out
- Hyperparameters: `n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`
- Objective: `multi:softprob`, 3 classes
- Class weights balanced for imbalanced labels

### Daily retraining (every 24 iterations)
- Expanding window (all data from start to current time)
- Full retrain, not incremental
- Safety gate: keep old model if new model's validation accuracy is worse
- Minimum 100 training samples before producing live signals; stay in cash until then

### Safeguards
- If retraining fails, log warning and keep previous model
- Features + targets stored in-memory as growing DataFrames on `self`
- If fewer than 60 historical bars available (early iterations), skip feature computation and stay in cash
- Decay tracker starts with neutral score (0.5) for all assets until enough predictions accumulate

## 6. Risk Management & Order Execution

### Risk rules

| Rule | Threshold | Action |
|------|-----------|--------|
| Max drawdown | 35% | Go to cash, pause until recovery to 25% |
| Drawdown scaling | 20–35% | Linear position size reduction |
| Per-position cap | 40% of portfolio | Redistribute excess to next-ranked |
| Max positions | 5 | Only top 5 signals get capital |
| Min confidence | adjusted signal > 0.1 | Below = hold (avoid noise) |
| Cash buffer | 5% minimum | Always reserve for slippage/fees |

### Order execution (each hour)
1. Compute target weights for top 5 assets
2. Diff target vs current positions
3. Sell orders first (free cash), then buy orders
4. Quantity: `target_weight * available_portfolio / price`
5. Market orders (no limit orders at hourly cadence)

### Asset type handling
- Crypto: `Asset(symbol, asset_type=Asset.AssetType.CRYPTO)`
- Equities: `Asset(symbol, asset_type=Asset.AssetType.STOCK)`

## 7. File Structure

```
Algorithmic-Trading/
├── README.md
├── requirements.txt
├── .gitignore
├── backtest.py
├── strategies/
│   ├── __init__.py
│   ├── strategy.py      # Thin orchestrator (~150 lines)
│   ├── params.py         # All tunable constants
│   ├── features.py       # Feature engineering (pure function)
│   ├── model.py          # XGBoost train/retrain/predict
│   └── risk.py           # Decay tracker, sizing, drawdown
└── data/
    └── EXAMPLE_1m_spot.csv
```

### Module responsibilities
- **strategy.py** — Orchestrates initialize + on_trading_iteration. Calls features → model → risk → orders.
- **params.py** — Universe lists, sleeptime, risk thresholds, model hyperparams, decay rate.
- **features.py** — Pure function: historical DataFrame in, feature row out.
- **model.py** — XGBoost wrapper: train(), retrain(), predict(). Safety gate logic.
- **risk.py** — DecayTracker class, position sizing, drawdown scaling.

### Dependencies
- lumibot, pandas, numpy, scikit-learn, xgboost, ta, python-dotenv
