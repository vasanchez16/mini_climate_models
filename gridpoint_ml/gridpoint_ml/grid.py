"""
grid.py — Enumerate all (time, lat, lon) gridpoints from config.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Gridpoint:
    time: str
    lat: float
    lon: float

    def label(self) -> str:
        """Return a filesystem-safe label for this gridpoint."""
        time_safe = self.time.replace(" ", "T").replace(":", "")
        return f"t{time_safe}_lat{self.lat:.4f}_lon{self.lon:.4f}"


def enumerate_gridpoints(config: dict) -> list[Gridpoint]:
    """Return the full list of gridpoints from the [grid] config section."""
    grid_cfg = config["grid"]
    times: list[str] = grid_cfg["times"]
    lats: list[float] = [float(v) for v in grid_cfg["lats"]]
    lons: list[float] = [float(v) for v in grid_cfg["lons"]]

    return [
        Gridpoint(time=t, lat=lat, lon=lon)
        for t, lat, lon in itertools.product(times, lats, lons)
    ]
