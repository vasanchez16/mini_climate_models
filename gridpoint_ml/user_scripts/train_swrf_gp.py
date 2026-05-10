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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def train(X: np.ndarray, y: np.ndarray):
    """Train a RandomForestRegressor and return the fitted model."""

    # create kernel object
    lengthScale = [1]*12
    nu = 0.5
    coefficient = 1.0
    kernel = coefficient * Matern(length_scale = lengthScale, nu = nu)

    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    model.fit(X, y)

    return model


def save(model, path: str) -> str:
    """Persist a scikit-learn model to disk using joblib. Returns the actual file path written."""
    full_path = path + ".joblib"
    joblib.dump(model, full_path)
    return full_path


def load(path: str):
    """Load a persisted scikit-learn model from disk."""
    return joblib.load(path)


def predict(model, X: np.ndarray) -> dict:
    """Run predictions and return a dict of named output arrays, each shape (n_variants,)."""
    mean, std = model.predict(X, return_std=True)
    return {
        "meanResponse": mean,
        "sdResponse": std,
    }
