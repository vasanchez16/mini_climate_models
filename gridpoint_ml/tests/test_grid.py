import pytest
from gridpoint_ml.grid import Gridpoint, enumerate_gridpoints


def _config():
    return {
        "grid": {
            "times": ["2020-01-01 00:00", "2020-01-01 12:00"],
            "lats": [30.0, 31.0],
            "lons": [-90.0, -89.0],
        }
    }


def test_enumerate_count():
    gps = enumerate_gridpoints(_config())
    # 2 times * 2 lats * 2 lons = 8
    assert len(gps) == 8


def test_enumerate_types():
    gps = enumerate_gridpoints(_config())
    for gp in gps:
        assert isinstance(gp, Gridpoint)
        assert isinstance(gp.time, str)
        assert isinstance(gp.lat, float)
        assert isinstance(gp.lon, float)


def test_gridpoint_label_no_spaces():
    gp = Gridpoint(time="2020-01-01 00:00", lat=30.0, lon=-90.0)
    label = gp.label()
    assert " " not in label
    assert ":" not in label


def test_gridpoint_label_unique():
    gps = enumerate_gridpoints(_config())
    labels = [gp.label() for gp in gps]
    assert len(labels) == len(set(labels)), "Labels must be unique"


def test_enumerate_order():
    """Times vary slowest, then lats, then lons (itertools.product order)."""
    gps = enumerate_gridpoints(_config())
    assert gps[0].time == "2020-01-01 00:00"
    assert gps[0].lat == 30.0
    assert gps[0].lon == -90.0
    assert gps[-1].time == "2020-01-01 12:00"
    assert gps[-1].lat == 31.0
    assert gps[-1].lon == -89.0
