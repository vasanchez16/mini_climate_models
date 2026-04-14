"""
data.py — Load features from CSV and extract target values from NetCDF4 files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .grid import Gridpoint


def load_features(features_csv: str) -> np.ndarray:
    """
    Load the features matrix from a CSV file.

    Rows are simulations (ordered by row index = sim index).
    Returns X of shape (n_simulations, n_features).
    """
    df = pd.read_csv(features_csv)
    return df.to_numpy(dtype=float)


def load_targets(gridpoint: Gridpoint, config: dict) -> np.ndarray:
    """
    Build y of shape (n_simulations,) for a given gridpoint.

    Opens each simulation's NetCDF4 file independently (no shared handles).
    """
    import netCDF4 as nc  # imported here to keep it local to workers

    data_cfg = config["data"]
    pattern: str = data_cfg["target_netcdf_pattern"]
    sim_id_format: str = data_cfg["sim_id_format"]
    n_simulations: int = int(data_cfg["n_simulations"])
    variable: str = data_cfg["target_variable"]

    y = np.empty(n_simulations, dtype=float)

    for sim_idx in range(n_simulations):
        sim_id = format(sim_idx, sim_id_format)
        path = pattern.format(sim_id=sim_id)

        with nc.Dataset(path, "r") as ds:
            y[sim_idx] = _extract_scalar(ds, variable, gridpoint)

    return y


def _extract_scalar(ds, variable: str, gridpoint: Gridpoint) -> float:
    """
    Slice a scalar value from a NetCDF4 dataset at the given (time, lat, lon).

    Looks up the nearest index along each dimension by value.
    """
    var = ds.variables[variable]
    dims = var.dimensions

    idx: dict[str, int] = {}

    for dim in dims:
        if dim not in ds.variables:
            raise KeyError(f"No coordinate variable found for dimension '{dim}'")
        coord = ds.variables[dim][:]

        if dim in ("time", "t"):
            # Match by string representation if stored as numeric
            target = gridpoint.time
            # Try to match via cftime or numeric index
            try:
                import cftime
                times = nc.num2date(coord, units=ds.variables[dim].units)
                time_strs = [t.strftime("%Y-%m-%d %H:%M") for t in times]
                idx[dim] = time_strs.index(target)
            except Exception:
                idx[dim] = 0  # fallback: first time step

        elif dim in ("lat", "latitude"):
            idx[dim] = int(np.argmin(np.abs(coord - gridpoint.lat)))

        elif dim in ("lon", "longitude"):
            idx[dim] = int(np.argmin(np.abs(coord - gridpoint.lon)))

        else:
            idx[dim] = 0  # unknown dimension — take first index

    slices = tuple(idx[d] for d in dims)
    return float(var[slices])
