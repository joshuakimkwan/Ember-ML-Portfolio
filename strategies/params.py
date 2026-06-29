"""Shared parameters for the ML trading strategy."""

# ── Asset universe ──────────────────────────────────────
STOCK_SLEEVE_SYMBOLS = ["TSLA", "NVDA", "AAPL", "GOOGL", "AMD", "LLY", "ABBV", "VRTX"] # Removed AMZN, META, MSFT, included Pharma, for diversification
CRYPTO_SLEEVE_SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX", "LINK"]
CRYPTO_SYMBOLS = set(CRYPTO_SLEEVE_SYMBOLS)
ALL_SYMBOLS = STOCK_SLEEVE_SYMBOLS + CRYPTO_SLEEVE_SYMBOLS

STOCK_BENCH = "SPY"
CRYPTO_BENCH = "BTC"

# ── Strategy ────────────────────────────────────────────
SLEEPTIME = "60M"
MAX_POSITIONS = 5
MAX_WEIGHT_PER_POSITION = 0.40

CASH_BUFFER = 0.05 # CHANGED from 0.05 to prevent spending $100000 on each buy/sell. The dollar amount transacted initially is too high.
MIN_CONFIDENCE = 0.4
ENTRY_PERSISTENCE = 2 # CHANGED (12062026) added line to only BUY if it has appeared in the target list for N consecutive iterations.
EXIT_PERSISTENCE = 2 # CHANGED (12062026) added line to only SELL if it has been out of the target list for N consecutive iterations.

# ── Transaction costs (used for threshold calculations) ──
PERCENT_FEE_PER_SIDE = 0.0007   # 7 bps per side, must match backtest.py

# ── Trade thresholds (as multiples of round-trip cost) ──
ROUND_TRIP_COST = 2 * PERCENT_FEE_PER_SIDE  # 0.0014 = 14 bps TOASK to import backtest??
MIN_SELL_PROFIT_MULTIPLIER = 1.5   # sell only if profit >= 2.5× round-trip cost
# MIN_BUYBACK_MULTIPLIER = 0.5      # buy back only if price dropped >= 0.6× round-trip cost

MIN_SELL_PROFIT = ROUND_TRIP_COST * MIN_SELL_PROFIT_MULTIPLIER      # CHANGED (21062026) only sell if price >= entry * (1 + this)
# MIN_BUYBACK_DROP = ROUND_TRIP_COST * MIN_BUYBACK_MULTIPLIER     # CHANGED (21062026) only buy back if price <= last_sold * (1 - this) 0.00084
MAX_POSITION_LOSS = 0.06      # CHANGED (21062026) stop-loss override: sell if down more than this (recommended). Original is 0.05 drop (5% drop of price bought)

# ── Risk ────────────────────────────────────────────────
MAX_DRAWDOWN = 0.35
DRAWDOWN_SCALING_START = 0.20
DRAWDOWN_RECOVERY = 0.25
STOP_LOSS_COOLDOWN_HOURS = 36

# ── XGBoost model ──────────────────────────────────────
N_ESTIMATORS = 900
MAX_DEPTH = 10
LEARNING_RATE = 0.029241
SUBSAMPLE =0.9653
COLSAMPLE_BYTREE = 0.5291
MIN_TRAINING_SAMPLES = 800
RETRAIN_INTERVAL = 96  # iterations (hours)

# ── Decay tracker ──────────────────────────────────────
DECAY_RATE = 0.97
INITIAL_ACCURACY = 0.5

TRACKER_PRIOR_WEIGHT = 5.0  # CHANGED (06122026) pseudo-observations anchoring score to INITIAL_ACCURACY

# ── Feature engineering ────────────────────────────────
MIN_HISTORY_BARS = 70
RETURN_DEAD_ZONE = 0.003  # +/-0.1% CHANGED (12062026) from 0.001 to 0.003
NUM_SELECTED_FEATURES = 20 # CHANGED (18062026)