import numpy as np
import pandas as pd 
import optuna
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from strategies.features import compute_features, compute_cross_asset_features
from strategies import params as P

# ── 1. Build a feature/label dataset from your CSVs ──────────────────────────
def build_dataset(csv_path: str, n_rows: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).tail(n_rows)
    df = df.sort_values("timestamp").reset_index(drop=True)

    X_rows, y_rows = [], []
    for i in range(P.MIN_HISTORY_BARS, len(df) - 1):
        window = df.iloc[i - P.MIN_HISTORY_BARS : i + 1]
        feat = compute_features(window)
        curr_price = df["close"].iloc[i]
        next_price = df["close"].iloc[i + 1]
        if curr_price <= 0:
            continue
        ret = (next_price - curr_price) / curr_price
        label = 1 if ret > P.RETURN_DEAD_ZONE else (-1 if ret < -P.RETURN_DEAD_ZONE else 0)
        X_rows.append(feat.values)
        y_rows.append(label + 1)   # map -1/0/1 → 0/1/2 for XGBoost

    return np.array(X_rows), np.array(y_rows)

X, y = build_dataset("data/SPY_1h_spot.csv")   # swap to any symbol

# ── 2. Objective function ─────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 600, step=50),
        max_depth         = trial.suggest_int("max_depth", 3, 8),
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
        gamma             = trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        objective         = "multi:softprob",
        num_class         = 3,
        eval_metric       = "mlogloss",
        verbosity         = 0,
        random_state      = 42,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        w = compute_sample_weight("balanced", y_tr)
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=w)
        scores.append(accuracy_score(y_val, model.predict(X_val)))

    return np.mean(scores) if scores else 0.0

# ── 3. Run study ──────────────────────────────────────────────────────────────
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\n=== Best trial ===")
print(f"  Accuracy : {study.best_value:.4f}")
for k, v in study.best_params.items():
    print(f"  {k:25s} = {v}")