"""Optuna hyperparameter tuner for XGBoost trading model.

Searches for optimal XGBoost params (n_estimators, max_depth, learning_rate,
subsample, colsample_bytree, num_selected_features) using the same two-pass
training logic as model.py.

Usage:
    python tune.py

After tuning, update strategies/params.py with the best values printed,
then run python backtest.py to verify.
"""

import optuna
import numpy as np
import pandas as pd
import json
import xgboost as xgb
from pathlib import Path
from sklearn.utils.class_weight import compute_sample_weight

from strategies.features import compute_features, compute_cross_asset_features
from strategies import params as P

DATA_DIR = Path(__file__).resolve().parent / "data"
N_TRIALS = 200
SYMBOLS = P.STOCK_SLEEVE_SYMBOLS


def load_bars(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_1h_spot.csv"
    df = pd.read_csv(path, usecols=["open", "high", "low", "close", "volume", "timestamp"])
    df["datetime"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).drop(columns=["timestamp"])
    df = df.set_index("datetime").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def build_dataset(symbol: str, all_bars: dict) -> tuple:
    df = all_bars[symbol]
    features_list = []
    targets = []
    feature_names = None

    asset_returns = {s: pd.Series(dtype=float) for s in all_bars}

    for i in range(P.MIN_HISTORY_BARS, len(df) - 1):
        window = df.iloc[max(0, i - P.MIN_HISTORY_BARS + 1):i + 1]
        if len(window) < P.MIN_HISTORY_BARS:
            continue

        if i > 0 and df["close"].iloc[i] == df["close"].iloc[i - 1]:
            continue

        try:
            feat = compute_features(window)
            cross = compute_cross_asset_features(asset_returns, symbol, list(all_bars.keys()))
            for k, v in cross.items():
                feat[k] = v

            if feature_names is None:
                feature_names = list(feat.index)

            current_price = df["close"].iloc[i]
            next_price = df["close"].iloc[i + 1]
            if current_price <= 0:
                continue
            ret = (next_price - current_price) / current_price
            if ret > P.RETURN_DEAD_ZONE:
                target = 1
            elif ret < -P.RETURN_DEAD_ZONE:
                target = -1
            else:
                target = 0

            features_list.append(feat.values.astype(float))
            targets.append(target)
        except Exception:
            print(f"  ERROR [{symbol}] bar {i}: {e}")
            continue

        if i > 0:
            prev_price = df["close"].iloc[i - 1]
            if prev_price > 0:
                r = (df["close"].iloc[i] - prev_price) / prev_price
                asset_returns[symbol] = pd.concat(
                    [asset_returns[symbol], pd.Series([r])], ignore_index=True
                ).tail(100)

    if not features_list:
        return np.array([]), np.array([]), []
    return np.array(features_list), np.array(targets), feature_names


def objective(trial, datasets: dict):
    n_estimators = trial.suggest_int("n_estimators", 200, 2000, step=100)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.005, 0.15, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0)
    num_selected = trial.suggest_int("num_selected_features", 6, 20)

    xgb_params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        verbosity=0,
        random_state=42,
    )

    accuracies = []
    for symbol, (X, y, _names) in datasets.items():
        if len(X) < 200:
            continue

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_mapped = y + 1

        split = int(len(X_clean) * 0.8)
        X_train, X_val = X_clean[:split], X_clean[split:]
        y_train, y_val = y_mapped[:split], y_mapped[split:]

        if len(np.unique(y_train)) < 2:
            continue

        sample_weights = compute_sample_weight("balanced", y_train)

        model_full = xgb.XGBClassifier(**xgb_params)
        model_full.fit(X_train, y_train, sample_weight=sample_weights)

        importances = model_full.feature_importances_
        top_k = min(num_selected, len(importances))
        top_indices = np.argsort(importances)[-top_k:]
        top_indices.sort()

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train[:, top_indices], y_train, sample_weight=sample_weights)

        val_preds = model.predict(X_val[:, top_indices])
        acc = (val_preds == y_val).mean()
        accuracies.append(acc)

    if not accuracies:
        return 0.0
    return float(np.mean(accuracies))


def main():
    print("Loading data...")
    all_bars = {}
    for symbol in SYMBOLS:
        try:
            all_bars[symbol] = load_bars(symbol)
            print(f"  {symbol}: {len(all_bars[symbol])} bars")
        except FileNotFoundError:
            print(f"  {symbol}: CSV not found, skipping")

    print("\nBuilding datasets...")
    datasets = {}
    for symbol in all_bars:
        X, y, names = build_dataset(symbol, all_bars)
        if len(X) > 0:
            datasets[symbol] = (X, y, names)
            class_counts = {-1: (y == -1).sum(), 0: (y == 0).sum(), 1: (y == 1).sum()}
            print(f"  {symbol}: {len(X)} samples | classes: {class_counts}")

    if not datasets:
        print("No datasets built. Check your CSV files in data/")
        return

    print(f"\nRunning Optuna study ({N_TRIALS} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, datasets), n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("BEST PARAMETERS")
    print("=" * 60)
    print(f"\n  Best mean val accuracy: {study.best_value:.4f}\n")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key.upper()} = {value:.6f}")
        else:
            print(f"  {key.upper()} = {value}")

    print("\n" + "-" * 60)
    print("Copy these into strategies/params.py:")
    print("-" * 60)
    bp = study.best_params
    print(f"N_ESTIMATORS = {bp['n_estimators']}")
    print(f"MAX_DEPTH = {bp['max_depth']}")
    print(f"LEARNING_RATE = {bp['learning_rate']:.6f}")
    print(f"SUBSAMPLE = {bp['subsample']:.4f}")
    print(f"COLSAMPLE_BYTREE = {bp['colsample_bytree']:.4f}")
    print(f"NUM_SELECTED_FEATURES = {bp['num_selected_features']}")

    with open("best_params.json", "w") as f:
        json.dump({"best_params": study.best_params, "best_value": study.best_value}, f, indent=2)
    print(f"\nSaved to best_params.json")

    print("\n" + "-" * 60)
    print("Top 5 trials:")
    print("-" * 60)
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 0, reverse=True)[:5]
    for i, t in enumerate(top_trials, 1):
        print(f"  #{i} acc={t.value:.4f} | {t.params}")


if __name__ == "__main__":
    main()
