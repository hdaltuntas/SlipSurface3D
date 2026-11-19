"""Pore-pressure grids: scattered (x, y, z, u) samples interpolated to column bases.

Used for pore pressures exported from an external seepage / consolidation analysis or
read from piezometers. Inside the convex hull of the samples the pressure is linearly
interpolated; outside it the nearest sample is used.

Interpolated values are clamped at zero. Samples that were clamped to zero above the
phreatic surface are given their signed pressure back first (see ``restore_suction``), so
the zero-pressure line is located between samples rather than smeared across them. A
positive pressure at the ground surface is treated as standing water of that head.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError


def restore_suction(samples: np.ndarray, gamma_w: float = 9.81) -> tuple[np.ndarray, int]:
    """Give zero-clamped samples above the phreatic surface a signed (negative) pressure.

    Seepage exports often clamp pore pressure at zero above the water. Interpolating such
    data smears the zero-pressure line by up to one sample spacing. Where samples share a
    plan position (a vertical string, as in gridded exports and piezometer strings), every
    zero above the highest positive sample is replaced by the pressure extrapolated upward
    from that sample, using the gradient of the two highest positive samples (hydrostatic
    ``-gamma_w`` when there is only one). Data that already contains negative pressures, and
    plan positions with a single sample, are left untouched.

    Returns the adjusted samples and the number of values changed.
    """
    pts = np.array(samples, dtype=float, copy=True)
    if np.any(pts[:, 3] < 0.0):
        return pts, 0
    _keys, inverse = np.unique(np.round(pts[:, :2], 6), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    order = np.lexsort((pts[:, 2], inverse))
    bounds = np.flatnonzero(np.r_[True, inverse[order][1:] != inverse[order][:-1], True])
    changed = 0
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        idx = order[start:stop]  # one vertical string, ascending z
        if len(idx) < 2:
            continue
        u = pts[idx, 3]
        positive = np.flatnonzero(u > 0.0)
        if len(positive) == 0 or positive[-1] == len(idx) - 1:
            continue
        top = int(positive[-1])
        grad = -gamma_w
        if top >= 1 and u[top - 1] > 0.0:
            dz = pts[idx[top], 2] - pts[idx[top - 1], 2]
            if dz > 1e-9:
                measured = (u[top] - u[top - 1]) / dz
                if measured < 0.0:
                    grad = float(measured)
        for j in range(top + 1, len(idx)):
            value = u[top] + grad * (pts[idx[j], 2] - pts[idx[top], 2])
            if value < 0.0:
                pts[idx[j], 3] = value
                changed += 1
    return pts, changed


class PorePressureGrid3D:
    """Interpolator over scattered pore-pressure samples ``(x, y, z, u)`` [m, m, m, kPa]."""

    def __init__(self, samples: Any, gamma_w: float = 9.81) -> None:
        raw = np.asarray(samples, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 4 or len(raw) < 1 or not np.all(np.isfinite(raw)):
            raise ValueError("Pore-pressure grid needs finite rows of (x, y, z, u).")
        self.samples = raw  # as supplied
        pts, self.n_restored = restore_suction(raw, gamma_w)
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
