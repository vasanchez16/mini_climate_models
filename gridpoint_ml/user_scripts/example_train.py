"""
example_train.py — Reference implementation using scikit-learn.

This script satisfies the gridpoint_ml training interface:
  - train(X, y) -> model
  - save(model, path)
  - load(path) -> model

Copy and adapt this file for your own model framework.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def train(X: np.ndarray, y: np.ndarray):
    """Train a RandomForestRegressor and return the fitted model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    model.fit(X, y)
    return model


def save(model, path: str) -> str:
    """Persist a scikit-learn model to disk using joblib. Returns the actual file path written."""
    full_path = path + ".joblib"
    joblib.dump(model, full_path)
    return full_path


def load(path: str):
    """Load a persisted scikit-learn model from disk."""
    return joblib.load(path + ".joblib")


def predict(model, X: np.ndarray) -> dict:
    """Run predictions and return a dict of named output arrays, each shape (n_variants,)."""
    mean, std = model.predict(X, return_std=True)
    return {
        "prediction_mean": mean,
        "prediction_std": std,
    }
