import os
import tempfile

import netCDF4 as nc
import numpy as np
import pytest

from gridpoint_ml.assembler import assemble, _parse_npz_filename, _output_filename


# --- unit tests for filename helpers ---

def test_parse_npz_filename_basic():
    result = _parse_npz_filename("pred_t2020-01-01T00_00_lat30.0000_lon-90.0000.npz")
    assert result is not None
    time_str, lat, lon = result
    assert time_str == "2020-01-01 00:00"
    assert lat == pytest.approx(30.0)
    assert lon == pytest.approx(-90.0)


def test_parse_npz_filename_negative_lat():
    result = _parse_npz_filename("pred_t2020-06-15T12_30_lat-10.5000_lon45.2500.npz")
    assert result is not None
    time_str, lat, lon = result
    assert time_str == "2020-06-15 12:30"
    assert lat == pytest.approx(-10.5)
    assert lon == pytest.approx(45.25)


def test_parse_npz_filename_bad_name():
    assert _parse_npz_filename("not_a_pred_file.npz") is None
    assert _parse_npz_filename("pred_t2020_lat30.npz") is None


def test_output_filename():
    assert _output_filename("2020-01-01 00:00") == "predictions_2020_01_01_00_00_00.nc"
    assert _output_filename("2017-08-01 10:30") == "predictions_2017_08_01_10_30_00.nc"


# --- integration test for assemble() ---

def _write_npz(path: str, **arrays) -> None:
    np.savez_compressed(path, **arrays)


def test_assemble_basic():
    """Assemble 4 gridpoints (2 lats × 2 lons) at 1 time step into one NetCDF4."""
    with tempfile.TemporaryDirectory() as base:
        tmp_dir = os.path.join(base, "tmp")
        output_dir = os.path.join(base, "output")
        os.makedirs(tmp_dir)

        n_variants = 10
        lats = [30.0, 31.0]
        lons = [-90.0, -89.0]
        time_str = "2020-01-01 00:00"
        time_safe = time_str.replace(" ", "T").replace(":", "_")

        for lat in lats:
            for lon in lons:
                label = f"t{time_safe}_lat{lat:.4f}_lon{lon:.4f}"
                npz_path = os.path.join(tmp_dir, f"pred_{label}.npz")
                _write_npz(
                    npz_path,
                    temperature_mean=np.full(n_variants, lat + lon, dtype=np.float32),
                    temperature_std=np.zeros(n_variants, dtype=np.float32),
                )

        written = assemble(tmp_dir, output_dir)

        assert len(written) == 1
        out_path = written[0]
        assert os.path.basename(out_path) == "predictions_2020_01_01_00_00_00.nc"
        assert os.path.exists(out_path)

        with nc.Dataset(out_path, "r") as ds:
            assert "lat" in ds.variables
            assert "lon" in ds.variables
            assert "variant" in ds.variables
            assert "temperature_mean" in ds.variables
            assert "temperature_std" in ds.variables
            assert ds.variables["temperature_mean"].shape == (2, 2, n_variants)
            assert ds.variables["lat"][:].tolist() == pytest.approx(lats)
            assert ds.variables["lon"][:].tolist() == pytest.approx(lons)

        # tmp_dir should be deleted after assembly
        assert not os.path.exists(tmp_dir)


def test_assemble_two_time_steps():
    """Two time steps produce two NetCDF4 files."""
    with tempfile.TemporaryDirectory() as base:
        tmp_dir = os.path.join(base, "tmp")
        output_dir = os.path.join(base, "output")
        os.makedirs(tmp_dir)

        n_variants = 5
        for time_str in ["2020-01-01 00:00", "2020-01-01 12:00"]:
            time_safe = time_str.replace(" ", "T").replace(":", "_")
            label = f"t{time_safe}_lat30.0000_lon-90.0000"
            _write_npz(
                os.path.join(tmp_dir, f"pred_{label}.npz"),
                mean=np.ones(n_variants, dtype=np.float32),
            )

        written = assemble(tmp_dir, output_dir)
        assert len(written) == 2


def test_assemble_nan_for_missing_gridpoint():
    """A lat/lon that appears only in some time steps stays NaN for missing ones."""
    with tempfile.TemporaryDirectory() as base:
        tmp_dir = os.path.join(base, "tmp")
        output_dir = os.path.join(base, "output")
        os.makedirs(tmp_dir)

        n_variants = 4
        # Two gridpoints at time 0; only one at time 1 → (31.0, -89.0) missing at time 1
        entries = [
            ("2020-01-01 00:00", 30.0, -90.0),
            ("2020-01-01 00:00", 31.0, -89.0),
            ("2020-01-01 12:00", 30.0, -90.0),
        ]
        for time_str, lat, lon in entries:
            time_safe = time_str.replace(" ", "T").replace(":", "_")
            label = f"t{time_safe}_lat{lat:.4f}_lon{lon:.4f}"
            _write_npz(
                os.path.join(tmp_dir, f"pred_{label}.npz"),
                val=np.full(n_variants, lat, dtype=np.float32),
            )

        written = assemble(tmp_dir, output_dir)
        assert len(written) == 2

        # The second time step only has one gridpoint → 1×1 grid, no NaN needed
        t1_file = next(f for f in written if "12_00" in f)
        with nc.Dataset(t1_file, "r") as ds:
            assert ds.variables["val"].shape == (1, 1, n_variants)
