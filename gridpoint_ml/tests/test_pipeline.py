import os
import csv
import textwrap

import numpy as np
import pytest

from gridpoint_ml.pipeline import Pipeline

N_SIMS = 5


def _write_features_csv(path: str, n_rows: int = N_SIMS, n_cols: int = 3) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"f{i}" for i in range(n_cols)])
        for r in range(n_rows):
            writer.writerow([float(r * n_cols + c) for c in range(n_cols)])


def _write_training_script(path: str) -> None:
    code = textwrap.dedent("""
        import numpy as np

        def train(X, y):
            return {"mean_y": float(np.mean(y))}

        def save(model, path):
            import json
            with open(path + ".json", "w") as f:
                json.dump(model, f)

        def load(path):
            import json
            with open(path + ".json") as f:
                return json.load(f)
    """)
    with open(path, "w") as f:
        f.write(code)


def _write_nc_files(base_dir: str, pattern: str, n_sims: int,
                    sim_id_format: str, lat: float, lon: float,
                    variable: str) -> None:
    """Create minimal NetCDF4 files that workers can open."""
    import netCDF4 as nc

    for sim_idx in range(n_sims):
        sim_id = format(sim_idx, sim_id_format)
        path = pattern.format(sim_id=sim_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with nc.Dataset(path, "w") as ds:
            ds.createDimension("time", 1)
            ds.createDimension("lat", 1)
            ds.createDimension("lon", 1)

            t_var = ds.createVariable("time", "f8", ("time",))
            t_var.units = "hours since 2020-01-01 00:00:00"
            t_var[:] = [0.0]

            lat_var = ds.createVariable("lat", "f4", ("lat",))
            lat_var[:] = [lat]

            lon_var = ds.createVariable("lon", "f4", ("lon",))
            lon_var[:] = [lon]

            data_var = ds.createVariable(variable, "f4", ("time", "lat", "lon"))
            data_var[:] = [[[float(sim_idx)]]]


def _write_config(
    config_path: str,
    features_csv: str,
    nc_pattern: str,
    output_dir: str,
    training_script: str,
) -> None:
    # Escape backslashes for Windows compatibility, use forward slashes in TOML
    nc_pattern_toml = nc_pattern.replace("\\", "/")
    content = f"""
[grid]
times = ["2020-01-01 00:00"]
lats = [30.0]
lons = [-90.0]

[data]
features_csv = "{features_csv}"
target_netcdf_pattern = "{nc_pattern_toml}"
sim_id_format = "03d"
n_simulations = {N_SIMS}
target_variable = "temperature"

[pipeline]
max_workers = 1
output_dir = "{output_dir}"
training_script = "{training_script}"

[metadata]
save_metadata = true
"""
    with open(config_path, "w") as f:
        f.write(content)


def test_pipeline_end_to_end(tmp_path):
    """Full pipeline smoke test with real (minimal) NetCDF files."""
    features_csv = str(tmp_path / "features.csv")
    training_script = str(tmp_path / "my_train.py")
    output_dir = str(tmp_path / "models")
    config_path = str(tmp_path / "config.toml")
    nc_pattern = str(tmp_path / "sims" / "sim_{sim_id}" / "sim_{sim_id}_var.nc")

    _write_features_csv(features_csv)
    _write_training_script(training_script)
    _write_nc_files(
        base_dir=str(tmp_path / "sims"),
        pattern=nc_pattern,
        n_sims=N_SIMS,
        sim_id_format="03d",
        lat=30.0,
        lon=-90.0,
        variable="temperature",
    )
    _write_config(config_path, features_csv, nc_pattern, output_dir, training_script)

    pipeline = Pipeline(config_path)
    results = pipeline.run()

    assert len(results) == 1  # 1 time * 1 lat * 1 lon
    assert results[0].success, f"Pipeline failed: {results[0].error}"
    assert os.path.exists(results[0].model_path + ".json")
