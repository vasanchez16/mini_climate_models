import csv
import json
import os
import textwrap

import netCDF4 as nc
import numpy as np
import pytest

from gridpoint_ml.predict_pipeline import PredictPipeline


N_VARIANTS = 8
N_FEATURES = 3


def _write_features_csv(path: str, n_rows: int = N_VARIANTS) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"f{i}" for i in range(N_FEATURES)])
        for r in range(n_rows):
            writer.writerow([float(r)] * N_FEATURES)


def _write_training_script(path: str) -> None:
    """Minimal training script implementing all four interface functions."""
    code = textwrap.dedent("""
        import json
        import numpy as np

        def train(X, y):
            return {"intercept": float(np.mean(y))}

        def save(model, path):
            full_path = path + ".json"
            with open(full_path, "w") as f:
                import json; json.dump(model, f)
            return full_path

        def load(path):
            with open(path) as f:
                return json.load(f)

        def predict(model, X):
            n = X.shape[0]
            return {
                "pred_mean": np.full(n, model["intercept"], dtype=np.float32),
                "pred_std": np.zeros(n, dtype=np.float32),
            }
    """)
    with open(path, "w") as f:
        f.write(code)


def _write_metadata_json(
    path: str,
    time: str,
    lat: float,
    lon: float,
    model_path: str,
    training_script: str,
) -> None:
    meta = {
        "time": time,
        "lat": lat,
        "lon": lon,
        "model_path": model_path,
        "training_script": training_script,
        "timestamp_utc": "2024-01-01T00:00:00+00:00",
    }
    with open(path, "w") as f:
        json.dump(meta, f)


def _write_model_file(path: str, intercept: float) -> str:
    """Write a fake model JSON and return the path that load() expects."""
    full_path = path + ".json"
    with open(full_path, "w") as f:
        json.dump({"intercept": intercept}, f)
    return full_path


def test_predict_pipeline_end_to_end(tmp_path):
    """Full predict pipeline: 2 gridpoints → 1 time step → 1 NetCDF4 file."""
    features_csv = str(tmp_path / "test_features.csv")
    training_script = str(tmp_path / "my_train.py")
    metadata_dir = str(tmp_path / "metadata")
    output_dir = str(tmp_path / "output")
    config_path = str(tmp_path / "predict_config.toml")

    os.makedirs(metadata_dir)
    _write_features_csv(features_csv)
    _write_training_script(training_script)

    gridpoints = [
        ("2020-01-01 00:00", 30.0, -90.0, 1.0),
        ("2020-01-01 00:00", 31.0, -89.0, 2.0),
    ]
    for time, lat, lon, intercept in gridpoints:
        model_stem = str(tmp_path / f"model_lat{lat}_lon{lon}")
        _write_model_file(model_stem, intercept)

        time_safe = time.replace(" ", "T").replace(":", "_")
        label = f"t{time_safe}_lat{lat:.4f}_lon{lon:.4f}"
        meta_path = os.path.join(metadata_dir, f"model_{label}_metadata.json")
        _write_metadata_json(
            path=meta_path,
            time=time,
            lat=lat,
            lon=lon,
            model_path=model_stem + ".json",
            training_script=training_script,
        )

    config_content = f"""
[prediction]
metadata_dir = "{metadata_dir}"
test_features_csv = "{features_csv}"
output_dir = "{output_dir}"
max_workers = 1
"""
    with open(config_path, "w") as f:
        f.write(config_content)

    pipeline = PredictPipeline(config_path)
    results = pipeline.run()

    assert len(results) == 2
    assert all(r.success for r in results), [r.error for r in results if not r.success]

    # tmp dir should be cleaned up
    assert not os.path.exists(os.path.join(output_dir, "tmp"))

    # One NetCDF4 file for the single time step
    nc_files = [f for f in os.listdir(output_dir) if f.endswith(".nc")]
    assert len(nc_files) == 1
    assert nc_files[0] == "predictions_2020_01_01_00_00_00.nc"

    with nc.Dataset(os.path.join(output_dir, nc_files[0]), "r") as ds:
        assert "pred_mean" in ds.variables
        assert "pred_std" in ds.variables
        assert ds.variables["pred_mean"].shape == (2, 2, N_VARIANTS)


def test_predict_pipeline_no_metadata(tmp_path):
    """Pipeline returns empty list and logs warning when metadata_dir is empty."""
    features_csv = str(tmp_path / "features.csv")
    metadata_dir = str(tmp_path / "metadata")
    output_dir = str(tmp_path / "output")
    config_path = str(tmp_path / "config.toml")

    os.makedirs(metadata_dir)
    _write_features_csv(features_csv)

    config_content = f"""
[prediction]
metadata_dir = "{metadata_dir}"
test_features_csv = "{features_csv}"
output_dir = "{output_dir}"
max_workers = 1
"""
    with open(config_path, "w") as f:
        f.write(config_content)

    pipeline = PredictPipeline(config_path)
    results = pipeline.run()
    assert results == []
