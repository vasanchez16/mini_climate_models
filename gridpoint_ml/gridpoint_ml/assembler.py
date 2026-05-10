"""
assembler.py — Phase 2: assemble per-gridpoint .npz files into per-time NetCDF4 files.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime

import netCDF4 as nc
import numpy as np

logger = logging.getLogger(__name__)

_NPZ_RE = re.compile(r"^pred_t(.+)_lat(-?[\d.]+)_lon(-?[\d.]+)\.npz$")


def _parse_npz_filename(filename: str) -> tuple[str, float, float] | None:
    """
    Parse time_str, lat, lon from a temp .npz filename.

    Filename format: pred_t{time_safe}_lat{lat:.4f}_lon{lon:.4f}.npz
    where time_safe = original_time.replace(' ', 'T').replace(':', '_')
    """
    m = _NPZ_RE.match(filename)
    if not m:
        return None
    time_safe, lat_str, lon_str = m.group(1), m.group(2), m.group(3)
    # Reverse the label transformation: "2020-01-01T00_00" → "2020-01-01 00:00"
    date_part, time_part = time_safe.split("T", 1)
    time_str = f"{date_part} {time_part.replace('_', ':')}"
    return time_str, float(lat_str), float(lon_str)


def _output_filename(time_str: str) -> str:
    """Convert '2020-01-01 00:00' → 'predictions_2020_01_01_00_00_00.nc'"""
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    return f"predictions_{dt.strftime('%Y_%m_%d_%H_%M_%S')}.nc"


def assemble(tmp_dir: str, output_dir: str) -> list[str]:
    """
    Scan tmp_dir for .npz files, group by time, write one NetCDF4 per time point.

    Output arrays have dimensions (lat, lon, variant) with NaN for any
    gridpoint whose .npz is missing (failed workers).

    Deletes tmp_dir after all NetCDF4 files are successfully written.
    Returns list of written output file paths.
    """
    npz_files = sorted(f for f in os.listdir(tmp_dir) if f.endswith(".npz"))
    if not npz_files:
        logger.warning("No .npz files found in %s — nothing to assemble.", tmp_dir)
        return []

    # Parse filenames into records
    records: list[tuple[str, float, float, str]] = []  # (time_str, lat, lon, full_path)
    for fname in npz_files:
        parsed = _parse_npz_filename(fname)
        if parsed is None:
            logger.warning("Skipping unrecognised file: %s", fname)
            continue
        time_str, lat, lon = parsed
        records.append((time_str, lat, lon, os.path.join(tmp_dir, fname)))

    if not records:
        logger.warning("No parseable .npz files in %s.", tmp_dir)
        return []

    # Infer variable names and n_variants from the first .npz
    first_data = np.load(records[0][3])
    variables: list[str] = list(first_data.files)
    n_variants: int = first_data[variables[0]].shape[0]
    first_data.close()

    # Group records by time
    by_time: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for time_str, lat, lon, path in records:
        by_time[time_str].append((lat, lon, path))

    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []

    for time_str, gridpoints in sorted(by_time.items()):
        lats = sorted({r[0] for r in gridpoints})
        lons = sorted({r[1] for r in gridpoints})
        lat_idx = {v: i for i, v in enumerate(lats)}
        lon_idx = {v: i for i, v in enumerate(lons)}
        n_lats, n_lons = len(lats), len(lons)

        # Pre-initialise with NaN; missing gridpoints stay NaN
        arrays = {
            var: np.full((n_lats, n_lons, n_variants), np.nan, dtype=np.float32)
            for var in variables
        }

        for lat, lon, npz_path in gridpoints:
            data = np.load(npz_path)
            for var in variables:
                if var in data.files:
                    arrays[var][lat_idx[lat], lon_idx[lon], :] = data[var].astype(np.float32)
                else:
                    logger.warning(
                        "Variable '%s' missing from %s — leaving NaN.", var, npz_path
                    )
            data.close()

        out_path = os.path.join(output_dir, _output_filename(time_str))
        with nc.Dataset(out_path, "w") as ds:
            ds.createDimension("lat", n_lats)
            ds.createDimension("lon", n_lons)
            ds.createDimension("variant", n_variants)

            lat_var = ds.createVariable("lat", "f4", ("lat",))
            lat_var[:] = np.array(lats, dtype=np.float32)

            lon_var = ds.createVariable("lon", "f4", ("lon",))
            lon_var[:] = np.array(lons, dtype=np.float32)

            variant_var = ds.createVariable("variant", "i4", ("variant",))
            variant_var[:] = np.arange(n_variants, dtype=np.int32)

            for var_name, arr in arrays.items():
                v = ds.createVariable(
                    var_name, "f4", ("lat", "lon", "variant"), fill_value=np.nan
                )
                v[:] = arr

        logger.info("Wrote %s", out_path)
        written.append(out_path)

    shutil.rmtree(tmp_dir)
    logger.info("Removed temporary directory %s", tmp_dir)

    return written
