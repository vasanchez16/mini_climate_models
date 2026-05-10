"""
predict_worker.py — Per-gridpoint prediction worker executed in a subprocess.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import numpy as np

from .io import load_training_module

logger = logging.getLogger(__name__)


@dataclass
class PredictWorkerResult:
    npz_path: str
    success: bool
    error: str | None = None


def predict_gridpoint(
    metadata_path: str,
    X_test: np.ndarray,
    output_dir: str,
) -> PredictWorkerResult:
    """
    Load a trained model and run predictions for one gridpoint.

    Intended to run inside a subprocess. Reads its own metadata JSON,
    loads the model, runs predict(), and writes a .npz temp file.
    """
    tmp_dir = os.path.join(output_dir, "tmp")

    try:
        with open(metadata_path) as f:
            meta = json.load(f)

        model_path: str = meta["model_path"]
        training_script: str = meta["training_script"]

        # Reconstruct the gridpoint label from metadata fields
        time: str = meta["time"]
        lat: float = float(meta["lat"])
        lon: float = float(meta["lon"])
        time_safe = time.replace(" ", "T").replace(":", "_")
        label = f"t{time_safe}_lat{lat:.4f}_lon{lon:.4f}"

        module = load_training_module(training_script)
        model = module.load(model_path)
        predictions: dict = module.predict(model, X_test)

        os.makedirs(tmp_dir, exist_ok=True)
        npz_path = os.path.join(tmp_dir, f"pred_{label}.npz")
        np.savez_compressed(npz_path, **predictions)

        return PredictWorkerResult(npz_path=npz_path, success=True)

    except Exception as exc:
        logger.warning("FAILED prediction for %s: %s", metadata_path, exc)
        return PredictWorkerResult(npz_path="", success=False, error=str(exc))
