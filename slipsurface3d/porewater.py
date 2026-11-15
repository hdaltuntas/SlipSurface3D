"""Pore-pressure grids: scattered (x, y, z, u) samples interpolated to column bases.

Used for pore pressures exported from an external seepage / consolidation analysis or
read from piezometers. Inside the convex hull of the samples the pressure is linearly
interpolated; outside it the nearest sample is used.

Interpolated values are clamped at zero. Where the samples come from a seepage model,
keep the negative pressures above the phreatic surface: with them the zero-pressure line
is located exactly, whereas samples already clamped to zero smear it by up to one sample
spacing. Water standing above the ground surface is not represented by a pressure grid;
model it with a water table or a surcharge load.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError


class PorePressureGrid3D:
    """Interpolator over scattered pore-pressure samples ``(x, y, z, u)`` [m, m, m, kPa]."""

    def __init__(self, samples: Any) -> None:
        pts = np.asarray(samples, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 4 or len(pts) < 1 or not np.all(np.isfinite(pts)):
            raise ValueError("Pore-pressure grid needs finite rows of (x, y, z, u).")
        self.samples = pts
        self._nearest = NearestNDInterpolator(pts[:, :3], pts[:, 3])
        self._linear: LinearNDInterpolator | None = None
        if len(pts) >= 5:
            try:
                self._linear = LinearNDInterpolator(pts[:, :3], pts[:, 3], fill_value=np.nan)
            except QhullError:
                # Coplanar or collinear samples have no 3D triangulation: nearest only.
                self._linear = None

    def __call__(self, x: Any, y: Any, z: Any) -> np.ndarray:
        xa, ya, za = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
        )
        query = np.column_stack([xa.ravel(), ya.ravel(), za.ravel()])
        out = np.asarray(self._nearest(query), dtype=float)
        if self._linear is not None:
            lin = np.asarray(self._linear(query), dtype=float)
            out = np.where(np.isfinite(lin), lin, out)
        return np.maximum(out, 0.0).reshape(xa.shape)

    def coverage(self, x: Any, y: Any, z: Any) -> float:
        """Fraction of the query points that lie inside the sampled volume (0 to 1).

        Points outside it get the nearest sample's pressure, which is an extrapolation.
        """
        xa, ya, za = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
        )
        if xa.size == 0:
            return 1.0
        if self._linear is None:
            return 0.0
        lin = np.asarray(self._linear(np.column_stack([xa.ravel(), ya.ravel(), za.ravel()])))
        return float(np.count_nonzero(np.isfinite(lin))) / xa.size

    def to_list(self) -> list[list[float]]:
        return [[float(v) for v in row] for row in self.samples]


def read_pore_pressure_csv(path: str | Path) -> list[list[float]]:
    """Read ``x, y, z, u`` rows from a CSV / whitespace file, skipping header and comment lines."""
    rows: list[list[float]] = []
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = next(csv.reader([line.replace(";", ",").replace("\t", ",")]))
        if len(parts) < 4:
            parts = line.split()
        try:
            rows.append([float(p) for p in parts[:4]])
        except (ValueError, IndexError):
            continue  # header or malformed line
    if not rows or any(len(r) != 4 for r in rows):
        raise ValueError(f"'{path}' contains no rows of the form x, y, z, u.")
    return rows
