"""XGBoost model for trading signal prediction."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from strategies import params as P


class TradingModel:
    """Wraps XGBoost for 3-class direction prediction: sell (-1), hold (0), buy (+1)."""

    def __init__(self):
        self.model = None
        self.feature_history = []  # list of numpy arrays
        self.target_history = []   # list of ints (-1, 0, 1)
        self.last_val_accuracy = 0.0

    def _build_model(self):
        """Create a fresh XGBoost classifier with params from config."""
        return xgb.XGBClassifier(
            n_estimators=P.N_ESTIMATORS,
            max_depth=P.MAX_DEPTH,
            learning_rate=P.LEARNING_RATE,
            subsample=P.SUBSAMPLE,
            colsample_bytree=P.COLSAMPLE_BYTREE,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
        )

    def add_sample(self, features: pd.Series, target: int):
        """Append one training sample (feature vector + target label)."""
        self.feature_history.append(features.values.astype(float))
        self.target_history.append(target)

    def has_enough_data(self) -> bool:
        """Check if we have enough samples to train."""
        return len(self.feature_history) >= P.MIN_TRAINING_SAMPLES

    def train(self) -> bool:
        """
        Train (or retrain) the model on all collected data.

        Uses 80/20 train/val split. Applies balanced class weights.
        Safety gate: only replaces current model if validation accuracy improves.

        Returns True if model was updated, False if previous model was kept.
        """
        if not self.has_enough_data():
            return False

        X = np.array(self.feature_history)
        y = np.array(self.target_history)

        # Replace any NaN/inf in features with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Map labels: -1 -> 0, 0 -> 1, 1 -> 2 (XGBoost needs 0-indexed classes)
        y_mapped = y + 1

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y_mapped[:split], y_mapped[split:]

        # Need at least 2 classes in training set
        if len(np.unique(y_train)) < 2:
            return False

        sample_weights = compute_sample_weight("balanced", y_train)

        model = self._build_model()
        model.fit(X_train, y_train, sample_weight=sample_weights)

        val_preds = model.predict(X_val)
        val_accuracy = (val_preds == y_val).mean()

        # Safety gate: only update if improvement (or first model)
        if self.model is None or val_accuracy >= self.last_val_accuracy:
            self.model = model
            self.last_val_accuracy = val_accuracy
            return True

        return False

    def predict(self, features: pd.Series) -> tuple:
        """
        Predict direction and confidence for given features.

        Returns:
            (direction, confidence) where:
            - direction: -1 (sell), 0 (hold), or 1 (buy)
            - confidence: probability of predicted class (0.33 to 1.0)
        """
        if self.model is None:
            return 0, 0.0

        X = np.nan_to_num(features.values.astype(float).reshape(1, -1),
                          nan=0.0, posinf=0.0, neginf=0.0)
        probs = self.model.predict_proba(X)[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        direction = predicted_class - 1  # Map back: 0->-1, 1->0, 2->1

        return direction, confidence

    @staticmethod
    def compute_target(current_price: float, next_price: float) -> int:
        """
        Compute training target from price movement.
        Uses dead zone to filter noise: returns within +/-0.1% are labeled hold (0).
        """
        if current_price <= 0:
            return 0
        ret = (next_price - current_price) / current_price
        if ret > P.RETURN_DEAD_ZONE:
            return 1
        elif ret < -P.RETURN_DEAD_ZONE:
            return -1
        return 0
