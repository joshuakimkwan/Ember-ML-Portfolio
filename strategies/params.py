"""Shared parameters for the ML trading strategy."""

# ── Asset universe ──────────────────────────────────────
STOCK_SLEEVE_SYMBOLS = ["TSLA", "NVDA", "AAPL", "AMZN", "META", "MSFT", "GOOGL", "AMD"]
CRYPTO_SLEEVE_SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX", "LINK"]
CRYPTO_SYMBOLS = set(CRYPTO_SLEEVE_SYMBOLS)
ALL_SYMBOLS = STOCK_SLEEVE_SYMBOLS + CRYPTO_SLEEVE_SYMBOLS

STOCK_BENCH = "AAPL"
CRYPTO_BENCH = "BTC"

# ── Strategy ────────────────────────────────────────────
SLEEPTIME = "60M"
MAX_POSITIONS = 5
MAX_WEIGHT_PER_POSITION = 0.40
CASH_BUFFER = 0.05
MIN_CONFIDENCE = 0.1

# ── Risk ────────────────────────────────────────────────
MAX_DRAWDOWN = 0.35
DRAWDOWN_SCALING_START = 0.20
DRAWDOWN_RECOVERY = 0.25

# ── XGBoost model ──────────────────────────────────────
N_ESTIMATORS = 300
MAX_DEPTH = 6
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
MIN_TRAINING_SAMPLES = 100
RETRAIN_INTERVAL = 24  # iterations (hours)

# ── Decay tracker ──────────────────────────────────────
DECAY_RATE = 0.97
INITIAL_ACCURACY = 0.5

# ── Feature engineering ────────────────────────────────
MIN_HISTORY_BARS = 60
RETURN_DEAD_ZONE = 0.001  # +/-0.1%
