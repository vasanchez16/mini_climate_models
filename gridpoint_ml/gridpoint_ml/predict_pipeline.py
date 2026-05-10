"""
predict_pipeline.py — Orchestrate parallel per-gridpoint prediction (Phase 1)
and NetCDF4 assembly (Phase 2).
"""
from __future__ import annotations

import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
from tqdm import tqdm

from .assembler import assemble
from .data import load_features
from .io import load_config
from .predict_worker import predict_gridpoint, PredictWorkerResult

logger = logging.getLogger(__name__)


class PredictPipeline:
    """Load config, dispatch prediction workers, then assemble NetCDF4 outputs."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = load_config(config_path)

    def run(self) -> list[PredictWorkerResult]:
        """
        Phase 1: run predict() for every gridpoint in parallel.
        Phase 2: assemble .npz temp files into per-time NetCDF4 files.

        Returns the list of PredictWorkerResult objects from Phase 1.
        """
        pred_cfg = self.config["prediction"]
        metadata_dir: str = pred_cfg["metadata_dir"]
        test_features_csv: str = pred_cfg["test_features_csv"]
        output_dir: str = pred_cfg["output_dir"]
        max_workers: int = int(pred_cfg["max_workers"])

        X_test: np.ndarray = load_features(test_features_csv)

        metadata_files = sorted(
            glob.glob(os.path.join(metadata_dir, "*_metadata.json"))
        )
        if not metadata_files:
            logger.warning("No metadata JSON files found in %s", metadata_dir)
            return []

        n_total = len(metadata_files)
        logger.info(
            "Phase 1: dispatching %d prediction jobs across %d workers.",
            n_total,
            max_workers,
        )

        results: list[PredictWorkerResult] = []
        worker_fn = partial(predict_gridpoint, X_test=X_test, output_dir=output_dir)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(worker_fn, mpath): mpath for mpath in metadata_files
            }
            with tqdm(total=n_total, desc="Predictions", unit="gp") as pbar:
                for future in as_completed(futures):
                    result: PredictWorkerResult = future.result()
                    results.append(result)
                    if not result.success:
                        logger.warning(
                            "FAILED %s: %s", futures[future], result.error
                        )
                    pbar.update(1)

        n_ok = sum(r.success for r in results)
        logger.info("Phase 1 complete: %d/%d succeeded.", n_ok, n_total)

        # Phase 2: assemble temp .npz files into NetCDF4 outputs
        tmp_dir = os.path.join(output_dir, "tmp")
        if os.path.isdir(tmp_dir):
            logger.info("Phase 2: assembling NetCDF4 output files...")
            assemble(tmp_dir, output_dir)
            logger.info("Phase 2 complete.")
        else:
            logger.warning(
                "No tmp directory at %s — Phase 2 skipped.", tmp_dir
            )

        return results
