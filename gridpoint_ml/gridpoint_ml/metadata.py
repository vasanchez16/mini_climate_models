"""
metadata.py — Write per-gridpoint JSON metadata files.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .grid import Gridpoint


def save_metadata(
    gridpoint: Gridpoint,
    model_path: str,
    training_script: str,
    output_dir: str,
) -> str:
    """
    Write a JSON sidecar file next to the saved model.

    Returns the path to the metadata file.
    """
    meta = {
        "time": gridpoint.time,
        "lat": gridpoint.lat,
        "lon": gridpoint.lon,
        "training_script": os.path.abspath(training_script),
        "model_path": os.path.abspath(model_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = os.path.splitext(model_path)[0] + "_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta_path
