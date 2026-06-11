# ML Algorithmic Trading Strategy

SoAI 2026 AI Algorithmic Trading Competition entry.

## Approach

XGBoost gradient-boosted classifier generating hourly buy/hold/sell signals across 15 equity and crypto assets, combined with an exponential decay tracker that adapts position sizing based on recent prediction accuracy.

### Key components

- **Signal generation:** XGBoost trained on ~30 technical features (momentum, volatility, volume, cross-asset correlations) predicts next-hour return direction
- **Adaptive sizing:** Exponential decay tracker (half-life ~48h) weights each asset's allocation by the model's recent accuracy on that asset
- **Risk management:** 35% max drawdown with linear position scaling from 20%, per-position cap at 40%, 5% cash buffer
- **Retraining:** Model retrains every 24 hours on expanding window with safety gate (keeps old model if new one underperforms)

### Universe

- **Equities (8):** TSLA, NVDA, AAPL, AMZN, META, MSFT, GOOGL, AMD
- **Crypto (7):** BTC, ETH, SOL, DOGE, XRP, AVAX, LINK
- **Max concurrent positions:** 5 (selected by strongest risk-adjusted signal)

## Setup

```bash
pip install -r requirements.txt
```

## Backtest

Place CSV files named `{SYMBOL}_1m_spot.csv` in the `data/` directory (columns: `open, high, low, close, volume, timestamp`), then run:

```bash
python backtest.py
```

## Structure

```
strategies/
  strategy.py   - Main Strategy class (Lumibot entrypoint)
  params.py     - All tunable constants
  features.py   - Feature engineering (30 features per asset)
  model.py      - XGBoost train/retrain/predict
  risk.py       - Decay tracker, position sizing, drawdown scaling
```
