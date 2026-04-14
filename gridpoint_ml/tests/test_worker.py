import os
import tempfile
import textwrap

import numpy as np
import pytest

from gridpoint_ml.grid import Gridpoint
from gridpoint_ml.worker import train_gridpoint, WorkerResult


def _write_training_script(path: str) -> None:
    """Write a minimal training script that satisfies the interface."""
    code = textwrap.dedent("""
        import numpy as np

        def train(X, y):
            return {"coef": float(np.mean(y))}

        def save(model, path):
            import json
            with open(path + ".json", "w") as f:
                import json; json.dump(model, f)

        def load(path):
            import json
            with open(path + ".json") as f:
                return json.load(f)
    """)
    with open(path, "w") as f:
        f.write(code)


def _make_config(output_dir: str, training_script: str) -> dict:
    return {
        "data": {
            "target_netcdf_pattern": "",  # not used in this test (load_targets is mocked)
            "sim_id_format": "03d",
            "n_simulations": 5,
            "target_variable": "temperature",
        },
        "pipeline": {
            "max_workers": 2,
            "output_dir": output_dir,
            "training_script": training_script,
        },
        "metadata": {"save_metadata": True},
    }


def test_worker_success(monkeypatch):
    """Worker should return success=True and create a model file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "my_train.py")
        _write_training_script(script_path)
        config = _make_config(output_dir=tmpdir, training_script=script_path)

        # Monkeypatch load_targets so we don't need real NetCDF files
        import gridpoint_ml.worker as worker_mod
        monkeypatch.setattr(
            worker_mod,
            "load_targets",
            lambda gp, cfg: np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )

        gp = Gridpoint(time="2020-01-01 00:00", lat=30.0, lon=-90.0)
        X = np.random.rand(5, 4)
        result = train_gridpoint(gp, X, config)

        assert result.success, f"Expected success, got error: {result.error}"
        assert os.path.exists(result.model_path + ".json")
        assert result.metadata_path != ""
        assert os.path.exists(result.metadata_path)


def test_worker_failure_on_bad_script():
    """Worker should catch errors from a bad script and return success=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a script missing the required 'train' function
        bad_script = os.path.join(tmpdir, "bad_train.py")
        with open(bad_script, "w") as f:
            f.write("# intentionally empty\n")

        config = _make_config(output_dir=tmpdir, training_script=bad_script)
        gp = Gridpoint(time="2020-01-01 00:00", lat=30.0, lon=-90.0)
        X = np.random.rand(5, 4)
        result = train_gridpoint(gp, X, config)

        assert not result.success
        assert result.error is not None
