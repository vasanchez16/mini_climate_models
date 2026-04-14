"""
worker.py — Per-gridpoint training function executed in a subprocess.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .grid import Gridpoint
from .data import load_targets
from .io import load_training_module
from .metadata import save_metadata


@dataclass
class WorkerResult:
    gridpoint: Gridpoint
    model_path: str
    metadata_path: str
    success: bool
    error: str | None = None


def train_gridpoint(
    gridpoint: Gridpoint,
    X,  # np.ndarray passed in; avoid re-importing numpy just for annotation
    config: dict,
) -> WorkerResult:
    """
    Complete training pipeline for a single gridpoint.

    Intended to run inside a subprocess (ProcessPoolExecutor worker).
    All file handles are opened and closed here — none are inherited from parent.
    """
    pipeline_cfg = config["pipeline"]
    output_dir: str = pipeline_cfg["output_dir"]
    training_script: str = pipeline_cfg["training_script"]
    save_meta: bool = config.get("metadata", {}).get("save_metadata", True)

    try:
        # Load user training module fresh in this worker process
        module = load_training_module(training_script)

        # Build target vector for this gridpoint (opens/closes NC files internally)
        y = load_targets(gridpoint, config)

        # Train model
        model = module.train(X, y)

        # Determine output path and save
        os.makedirs(output_dir, exist_ok=True)
        model_filename = f"model_{gridpoint.label()}"
        model_path = os.path.join(output_dir, model_filename)
        module.save(model, model_path)

        # Write sidecar metadata
        meta_path = ""
        if save_meta:
            meta_path = save_metadata(
                gridpoint=gridpoint,
                model_path=model_path,
                training_script=training_script,
                output_dir=output_dir,
            )

        return WorkerResult(
            gridpoint=gridpoint,
            model_path=model_path,
            metadata_path=meta_path,
            success=True,
        )

    except Exception as exc:
        return WorkerResult(
            gridpoint=gridpoint,
            model_path="",
            metadata_path="",
            success=False,
            error=str(exc),
        )
