"""
pipeline.py — Orchestrate parallel per-gridpoint model training.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
from tqdm import tqdm

from .data import load_features
from .grid import enumerate_gridpoints, Gridpoint
from .io import load_config
from .worker import train_gridpoint, WorkerResult

logger = logging.getLogger(__name__)


class Pipeline:
    """High-level entry point: load config, load features, dispatch workers."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = load_config(config_path)

    def run(self) -> list[WorkerResult]:
        """
        Train one model per gridpoint in parallel.

        Returns a list of WorkerResult objects (one per gridpoint).
        """
        config = self.config
        pipeline_cfg = config["pipeline"]
        max_workers: int = int(pipeline_cfg["max_workers"])

        # Load features once in the parent process; pass as a numpy array
        # (numpy arrays are safely picklable across processes)
        features_csv: str = config["data"]["features_csv"]
        X: np.ndarray = load_features(features_csv)

        gridpoints = enumerate_gridpoints(config)
        n_total = len(gridpoints)
        logger.info("Dispatching %d gridpoint jobs across %d workers.", n_total, max_workers)

        results: list[WorkerResult] = []
        failed: list[WorkerResult] = []

        # Use a top-level function reference so ProcessPoolExecutor can pickle it
        worker_fn = partial(train_gridpoint, X=X, config=config)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker_fn, gp): gp for gp in gridpoints}

            with tqdm(total=n_total, desc="Gridpoints", unit="gp") as pbar:
                for future in as_completed(futures):
                    result: WorkerResult = future.result()
                    results.append(result)
                    if not result.success:
                        failed.append(result)
                        logger.warning(
                            "FAILED %s: %s",
                            result.gridpoint.label(),
                            result.error,
                        )
                    pbar.update(1)

        n_ok = sum(r.success for r in results)
        logger.info("Completed: %d/%d succeeded, %d failed.", n_ok, n_total, len(failed))

        if failed:
            logger.warning("Failed gridpoints:")
            for r in failed:
                logger.warning("  %s — %s", r.gridpoint.label(), r.error)

        return results
