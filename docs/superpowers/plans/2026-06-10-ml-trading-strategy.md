# ML Trading Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an XGBoost + adaptive decay tracker trading strategy for the SoAI 2026 competition targeting 10% monthly returns across 15 equity/crypto assets.

**Architecture:** Single XGBoost model scores 15 candidates hourly, exponential decay tracker adjusts position sizes by recent accuracy, risk layer enforces 35% max drawdown with linear scaling. Max 5 concurrent positions.

**Tech Stack:** Python, Lumibot, XGBoost, pandas, numpy, scikit-learn, ta

---

## File Structure

```
/Users/jclaw/projects/fyp/
├── README.md                          # Approach description
├── requirements.txt                   # All dependencies (UPDATE)
├── .gitignore                         # Python/data artifacts (CREATE)
├── backtest.py                        # Local backtest entrypoint (UPDATE)
├── strategies/
│   ├── __init__.py                    # Package init (CREATE)
│   ├── strategy.py                    # Main Strategy class (CREATE - replaces root strategy.py)
│   ├── params.py                      # All tunable constants (CREATE)
│   ├── features.py                    # Feature engineering - 30 features per asset (CREATE)
│   ├── model.py                       # XGBoost train/retrain/predict (CREATE)
│   └── risk.py                        # Decay tracker, position sizing, drawdown (CREATE)
└── data/
    └── TSLA_1m_spot.csv               # Existing data (MOVE from root)
```

---

### Task 1: Create strategies/params.py

**Files:**
- Create: `strategies/params.py`

- [ ] **Step 1:** Create params.py with all tunable constants — asset universe (8 equities, 7 crypto), sleeptime, risk thresholds (35% DD, 40% position cap, 5% cash buffer), XGBoost hyperparameters, decay rate (0.97), feature params.

---

### Task 2: Create strategies/features.py

**Files:**
- Create: `strategies/features.py`

- [ ] **Step 1:** Implement `compute_features(df)` — takes OHLCV DataFrame, returns ~26 features: price-based (returns, MA distances, Bollinger), momentum (RSI, MACD, ROC), volatility (ATR, stddev, range), volume (ratio, VWAP, OBV, spike).
- [ ] **Step 2:** Implement `compute_cross_asset_features(asset_returns, symbol, all_symbols)` — 4 features: BTC correlation, SPY-proxy correlation, momentum rank, volatility rank.

---

### Task 3: Create strategies/model.py

**Files:**
- Create: `strategies/model.py`

- [ ] **Step 1:** Implement `TradingModel` class — XGBoost wrapper with `add_sample()`, `has_enough_data()`, `train()` (with 80/20 split, balanced weights, safety gate), `predict()` (returns direction + confidence), `compute_target()` (3-class with dead zone).

---

### Task 4: Create strategies/risk.py

**Files:**
- Create: `strategies/risk.py`

- [ ] **Step 1:** Implement `DecayTracker` class — per-asset exponential decay accuracy tracking (decay=0.97, ~48h half-life).
- [ ] **Step 2:** Implement `compute_position_sizes()` — adjusted signals, rank top 5, proportional weights, position cap, drawdown scaling (linear 20%→35%).

---

### Task 5: Create strategies/strategy.py

**Files:**
- Create: `strategies/strategy.py`

**Depends on:** Tasks 1-4

- [ ] **Step 1:** Implement `initialize()` — build asset objects, init model + tracker, set state variables.
- [ ] **Step 2:** Implement `on_trading_iteration()` — drawdown checks, collect prices/history, update tracker, feature engineering, prediction, position sizing, rebalance.
- [ ] **Step 3:** Implement `_rebalance()` — diff target vs current, sell first then buy, handle fractional crypto quantities.
- [ ] **Step 4:** Implement `_sell_all()` — liquidate all positions.

---

### Task 6: Update backtest.py and supporting files

**Files:**
- Update: `backtest.py`
- Update: `requirements.txt`
- Create: `strategies/__init__.py`
- Create: `.gitignore`
- Create: `README.md`
- Move: `TSLA_1m_spot.csv` → `data/TSLA_1m_spot.csv`

- [ ] **Step 1:** Update backtest.py for new structure.
- [ ] **Step 2:** Update requirements.txt with xgboost and ta.
- [ ] **Step 3:** Create __init__.py, .gitignore, README.md.
- [ ] **Step 4:** Move CSV data to data/ directory.
- [ ] **Step 5:** Clean up old root-level strategy.py.
