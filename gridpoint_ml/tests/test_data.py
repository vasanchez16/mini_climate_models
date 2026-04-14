import numpy as np
import pytest
import tempfile
import os
import csv

from gridpoint_ml.data import load_features


def _write_csv(path: str, n_rows: int, n_cols: int) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"feat_{i}" for i in range(n_cols)])
        for r in range(n_rows):
            writer.writerow([float(r * n_cols + c) for c in range(n_cols)])


def test_load_features_shape():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "features.csv")
        _write_csv(csv_path, n_rows=117, n_cols=10)
        X = load_features(csv_path)
        assert X.shape == (117, 10)


def test_load_features_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "features.csv")
        _write_csv(csv_path, n_rows=5, n_cols=3)
        X = load_features(csv_path)
        assert X.dtype == float


def test_load_features_values():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "features.csv")
        _write_csv(csv_path, n_rows=3, n_cols=2)
        X = load_features(csv_path)
        # Row 0: [0.0, 1.0], Row 1: [2.0, 3.0], Row 2: [4.0, 5.0]
        np.testing.assert_array_equal(X[0], [0.0, 1.0])
        np.testing.assert_array_equal(X[1], [2.0, 3.0])
        np.testing.assert_array_equal(X[2], [4.0, 5.0])
