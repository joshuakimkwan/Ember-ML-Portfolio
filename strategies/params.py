"""Shared parameters for the ML trading strategy."""

# ── Asset universe ──────────────────────────────────────
STOCK_SLEEVE_SYMBOLS = ["TSLA", "NVDA", "AAPL", "AMZN", "META", "MSFT", "GOOGL", "AMD"]
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
MIN_CONFIDENCE = 0.2
ENTRY_PERSISTENCE = 2 # CHANGED (12062026) added line to only BUY if it has appeared in the target list for N consecutive iterations.
EXIT_PERSISTENCE = 2 # CHANGED (12062026) added line to only SELL if it has been out of the target list for N consecutive iterations.

# ── Risk ────────────────────────────────────────────────
MAX_DRAWDOWN = 0.35
DRAWDOWN_SCALING_START = 0.20
DRAWDOWN_RECOVERY = 0.25

# ── XGBoost model ──────────────────────────────────────
N_ESTIMATORS = 450
MAX_DEPTH = 8
LEARNING_RATE = 0.02
SUBSAMPLE = 0.888
COLSAMPLE_BYTREE = 0.504
MIN_CHILD_WEIGHT = 3
GAMMA = 0.487
REG_ALPHA = 0.0002
REG_LAMBDA = 0.033
MIN_TRAINING_SAMPLES = 100
RETRAIN_INTERVAL = 24  # iterations (hours)

# ── Decay tracker ──────────────────────────────────────
DECAY_RATE = 0.97
INITIAL_ACCURACY = 0.5

TRACKER_PRIOR_WEIGHT = 10.0  # CHANGED (06122026) pseudo-observations anchoring score to INITIAL_ACCURACY

# ── Feature engineering ────────────────────────────────
MIN_HISTORY_BARS = 60
RETURN_DEAD_ZONE = 0.003  # +/-0.1% CHANGED (12062026) from 0.001 to 0.003
