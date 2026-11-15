"""Three-dimensional slip surface representations.

Supported 3D slip surfaces:
* ``EllipsoidSurface``              : General tri-axial 3D ellipsoid (covers spheres when
                                      ``rx == ry == rz`` and 2D circular cylinders when ``ry`` is large).
                                      Both sheets of the closed ellipsoid are available via
                                      ``vertical_hits``; ``z(x, y)`` remains the lower (floor) sheet.
* ``CompositeEllipsoidPlaneSurface``: Ellipsoid truncated by a weak geological plane (e.g. Zhang 1988
                                      and Hungr et al. 1989 weak-seam benchmark).
* ``SplinePatchSurface``            : Smooth non-ellipsoidal 3D bowl formed by a tensor-product
                                      elevation perturbation over an ``EllipsoidSurface``.
* ``MeshSlipSurface``               : Triangulated 3D slip surface (optional faces). With faces, a
                                      vertical ray may hit multiple sheets (overhangs / folds);
                                      without faces the vertices are interpolated as a heightfield.
* ``ImplicitSlipSurface``           : General F(x, y, z) = 0 manifold sampled by vertical line search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline

from slipsurface3d.geometry.intersection import (
    MAX_VERTICAL_HITS,
    NZ_FLOOR_MIN,
    BaseSample,
    VerticalMeshIndex,
    empty_hit_buffers,
    graph_hits_from_z,
    orient_triangles,
    pack_hits,
    pick_floor,
    ray_triangle_hits,
)


class SlipSurface3D(ABC):
    """3D basal slip surface: a heightfield, a closed manifold, or a folded triangle mesh.

    ``z(x, y)`` returns the floor sheet used as the Method-of-Columns base when no ground
    elevation is supplied (the highest intersection with positive ``n_z``). Column
    discretisation should call ``sample_base`` so roof-like hits (overhangs) are skipped.
    """

    graph_valued: bool = True

    @abstractmethod
    def z(self, x: Any, y: Any) -> np.ndarray:
        """Return floor-sheet elevation z_b(x, y), or NaN outside the surface footprint."""

    def gradients(self, x: Any, y: Any, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        """Return partial derivatives (dz_dx, dz_dy) of the floor sheet at (x, y)."""
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        dz_dx = (self.z(xa + eps, ya) - self.z(xa - eps, ya)) / (2.0 * eps)
        dz_dy = (self.z(xa, ya + eps) - self.z(xa, ya - eps)) / (2.0 * eps)
        return dz_dx, dz_dy

    def vertical_hits(
        self, x: Any, y: Any, *, max_hits: int = MAX_VERTICAL_HITS
    ) -> tuple[np.ndarray, np.ndarray]:
        """All vertical-line intersections at (x, y).

        Returns ``(z_hits, normals)`` with ``z_hits`` shape ``(..., max_hits)`` sorted
        descending and NaN-padded, and ``normals`` shape ``(..., max_hits, 3)`` pointing
        into the sliding mass (``n_z > 0`` on a floor, ``n_z < 0`` on a roof).
        """
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        zb = self.z(xa, ya)
        dz_dx, dz_dy = self.gradients(xa, ya)
        return graph_hits_from_z(zb, dz_dx, dz_dy, max_hits=max_hits)

    def sample_base(
        self,
        x: Any,
        y: Any,
        z_top: Any,
        z_min: Any,
        *,
        nz_min: float = NZ_FLOOR_MIN,
    ) -> BaseSample:
        """Basal hit of a downward ray from ``z_top``, skipping roof-like intersections."""
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        zt = np.broadcast_to(np.asarray(z_top, dtype=float), xa.shape)
        zm = np.broadcast_to(np.asarray(z_min, dtype=float), xa.shape)
        z_hits, normals = self.vertical_hits(xa, ya)
        return pick_floor(z_hits, normals, zt, zm, nz_min=nz_min)

    def intersect_segment(
        self,
        a: np.ndarray,
        b: np.ndarray,
        n_samples: int = 65,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """First intersection of segment ``a -> b`` with the surface, plus the unit normal."""
        a = np.asarray(a, dtype=float).reshape(3)
        b = np.asarray(b, dtype=float).reshape(3)
        ab = b - a
        total = float(np.linalg.norm(ab))
        if total <= 1e-9:
            return None
        ts = np.linspace(0.0, 1.0, max(8, int(n_samples)))
        pts = a[None, :] + ts[:, None] * ab[None, :]
        z_top = float(np.max(pts[:, 2]) + 1.0)
        z_min = float(np.min(pts[:, 2]) - 1.0)
        samp = self.sample_base(pts[:, 0], pts[:, 1], z_top, z_min)
        diff = pts[:, 2] - samp.z
        valid = np.isfinite(diff)
        if int(np.count_nonzero(valid)) < 2:
            return None
        for k in range(len(ts) - 1):
            if valid[k] and valid[k + 1] and diff[k] >= 0.0 and diff[k + 1] < 0.0:
                denom = diff[k] - diff[k + 1]
                frac = float(diff[k] / denom) if abs(denom) > 1e-12 else 0.5
                t_hit = float(ts[k] + frac * (ts[k + 1] - ts[k]))
                xyz = a + t_hit * ab
                nrm = samp.normal_xyz[k]
                if not np.all(np.isfinite(nrm)):
                    nrm = samp.normal_xyz[k + 1]
                n_len = float(np.linalg.norm(nrm))
                if n_len < 1e-12:
                    return None
                return xyz, nrm / n_len
        return None

    def rotation_center_sz(self, azimuth_deg: float) -> tuple[float, float] | None:
        """Optional moment-equilibrium axis (s_c, z_c) in the sliding frame."""
        return None

    @abstractmethod
    def footprint_bounds(self) -> tuple[float, float, float, float]:
        """Bounding box (x_min, x_max, y_min, y_max) of the slip surface projection."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""

    def to_mesh(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        ground_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        nx: int = 72,
        ny: int = 72,
    ) -> pv.PolyData:
        """Generate a clipped PyVista PolyData surface mesh of the active slip bowl."""
        fx0, fx1, fy0, fy1 = self.footprint_bounds()
        x0 = max(x_bounds[0], fx0)
        x1 = min(x_bounds[1], fx1)
        y0 = max(y_bounds[0], fy0)
        y1 = min(y_bounds[1], fy1)
        if x1 <= x0 or y1 <= y0:
            return pv.PolyData()

        xs = np.linspace(x0, x1, max(8, nx))
        ys = np.linspace(y0, y1, max(8, ny))
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        zz = self.z(xx, yy)

        if ground_fn is not None:
            zg = ground_fn(xx, yy)
            valid = np.isfinite(zz) & (zz <= zg + 1e-6)
            zz_clip = np.where(valid, np.minimum(zz, zg), np.nan)
        else:
            valid = np.isfinite(zz)
            zz_clip = zz

        if not np.any(valid):
            return pv.PolyData()

        zz_filled = np.where(np.isfinite(zz_clip), zz_clip, float(np.nanmean(zz_clip)))
        grid = pv.StructuredGrid(xx, yy, zz_filled)
        grid["valid_mask"] = valid.ravel(order="F").astype(float)
        grid["Elevation"] = zz_filled.ravel(order="F")
        surf = grid.extract_surface()
        clipped = surf.threshold(0.5, scalars="valid_mask", preference="point")
        return clipped.extract_surface()


@dataclass
class EllipsoidSurface(SlipSurface3D):
    """Lower half of a tri-axial 3D ellipsoid centred at ``(xc, yc, zc)``.

    Principal semi-axes:
    * ``rx`` : semi-axis along ``azimuth_deg`` in the horizontal plane [m]
    * ``ry`` : semi-axis transverse to ``azimuth_deg`` in the horizontal plane [m]
    * ``rz`` : vertical semi-axis [m]
    """

    xc: float
    yc: float
    zc: float
    rx: float
    ry: float
    rz: float
    azimuth_deg: float = 180.0
    sheet: str = "lower"  # lower | upper | both

    def __post_init__(self) -> None:
        if self.rx <= 0 or self.ry <= 0 or self.rz <= 0:
            raise ValueError("Ellipsoid semi-axes rx, ry, rz must all be strictly positive.")
        if self.sheet not in ("lower", "upper", "both"):
            raise ValueError("EllipsoidSurface.sheet must be 'lower', 'upper', or 'both'.")

    @property
    def graph_valued(self) -> bool:  # type: ignore[override]
        return self.sheet != "both"

    @property
    def is_spherical(self) -> bool:
        return bool(np.allclose([self.rx, self.ry], self.rz, rtol=1e-6, atol=1e-8))

    def _local_uv(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        psi = float(np.radians(self.azimuth_deg))
        cp, sp = float(np.cos(psi)), float(np.sin(psi))
        dx = x - self.xc
        dy = y - self.yc
        u = dx * cp + dy * sp
        v = -dx * sp + dy * cp
        return u, v, cp, sp

    def _sheet_z(self, xa: np.ndarray, ya: np.ndarray, lower: bool) -> np.ndarray:
        u, v, _, _ = self._local_uv(xa, ya)
        rho2 = (u / self.rx) ** 2 + (v / self.ry) ** 2
        inside = rho2 < 1.0 - 1e-12
        out = np.full(xa.shape, np.nan, dtype=float)
        if np.any(inside):
            half = self.rz * np.sqrt(np.maximum(1.0 - rho2[inside], 0.0))
            out[inside] = self.zc - half if lower else self.zc + half
        return out

    def z(self, x: Any, y: Any) -> np.ndarray:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return self._sheet_z(xa, ya, lower=self.sheet != "upper")

    def _implicit_normal(self, xa: np.ndarray, ya: np.ndarray, za: np.ndarray) -> np.ndarray:
        """Unit normal pointing into the ellipsoid interior (sliding mass of a scoop)."""
        u, v, cp, sp = self._local_uv(xa, ya)
        dF_du = 2.0 * u / (self.rx * self.rx)
        dF_dv = 2.0 * v / (self.ry * self.ry)
        dF_dz = 2.0 * (za - self.zc) / (self.rz * self.rz)
        dF_dx = dF_du * cp - dF_dv * sp
        dF_dy = dF_du * sp + dF_dv * cp
        g = np.stack([dF_dx, dF_dy, dF_dz], axis=-1)
        # F < 0 inside; -∇F points into the interior
        n = -g
        ln = np.linalg.norm(n, axis=-1, keepdims=True)
        return np.where(ln > 1e-14, n / np.maximum(ln, 1e-14), np.nan)

    def vertical_hits(
        self, x: Any, y: Any, *, max_hits: int = MAX_VERTICAL_HITS
    ) -> tuple[np.ndarray, np.ndarray]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        z_lo = self._sheet_z(xa, ya, lower=True)
        z_hi = self._sheet_z(xa, ya, lower=False)
        if self.sheet == "lower":
            n_lo = self._implicit_normal(xa, ya, z_lo)
            return pack_hits([z_lo], [n_lo], xa.shape, max_hits=max_hits)
        if self.sheet == "upper":
            n_hi = self._implicit_normal(xa, ya, z_hi)
            return pack_hits([z_hi], [n_hi], xa.shape, max_hits=max_hits)
        n_lo = self._implicit_normal(xa, ya, z_lo)
        n_hi = self._implicit_normal(xa, ya, z_hi)
        return pack_hits([z_hi, z_lo], [n_hi, n_lo], xa.shape, max_hits=max_hits)

    def gradients(self, x: Any, y: Any, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        u, v, cp, sp = self._local_uv(xa, ya)
        rho2 = (u / self.rx) ** 2 + (v / self.ry) ** 2
        inside = rho2 < 1.0 - 1e-9
        dz_dx = np.full(xa.shape, np.nan, dtype=float)
        dz_dy = np.full(xa.shape, np.nan, dtype=float)
        if np.any(inside):
            rad = np.sqrt(np.maximum(1.0 - rho2[inside], 1e-12))
            dz_du = self.rz * u[inside] / (self.rx * self.rx * rad)
            dz_dv = self.rz * v[inside] / (self.ry * self.ry * rad)
            dz_dx[inside] = dz_du * cp - dz_dv * sp
            dz_dy[inside] = dz_du * sp + dz_dv * cp
        if self.sheet == "upper":
            return -dz_dx, -dz_dy
        return dz_dx, dz_dy

    def rotation_center_sz(self, azimuth_deg: float) -> tuple[float, float]:
        psi = float(np.radians(azimuth_deg))
        sc = self.xc * float(np.cos(psi)) + self.yc * float(np.sin(psi))
        return (sc, float(self.zc))

    def footprint_bounds(self) -> tuple[float, float, float, float]:
        psi = float(np.radians(self.azimuth_deg))
        cp, sp = abs(float(np.cos(psi))), abs(float(np.sin(psi)))
        # Exact axis-aligned bounding half-extents of rotated ellipse
        hx = float(np.hypot(self.rx * cp, self.ry * sp))
        hy = float(np.hypot(self.rx * sp, self.ry * cp))
        return (self.xc - hx, self.xc + hx, self.yc - hy, self.yc + hy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "ellipsoid",
            "xc": float(self.xc),
            "yc": float(self.yc),
            "zc": float(self.zc),
            "rx": float(self.rx),
            "ry": float(self.ry),
            "rz": float(self.rz),
            "azimuth_deg": float(self.azimuth_deg),
            "sheet": self.sheet,
        }


def sphere_surface(xc: float, yc: float, zc: float, radius: float) -> EllipsoidSurface:
    """Create an exact 3D spherical slip surface of radius R centred at (xc, yc, zc)."""
    return EllipsoidSurface(xc=xc, yc=yc, zc=zc, rx=radius, ry=radius, rz=radius, azimuth_deg=180.0)


def cylindrical_surface(
    xc: float, zc: float, radius: float, y_center: float = 0.0, azimuth_deg: float = 180.0
) -> EllipsoidSurface:
    """Create a 2D circular cylinder extruded along the transverse axis (ry -> infinity)."""
    return EllipsoidSurface(
        xc=xc,
        yc=y_center,
        zc=zc,
        rx=radius,
        ry=1.0e9,
        rz=radius,
        azimuth_deg=azimuth_deg,
    )


@dataclass
class CompositeEllipsoidPlaneSurface(SlipSurface3D):
    """Ellipsoid truncated from below by a weak geological plane z_plane(x, y).

    Where the ellipsoid dips below the weak plane, the sliding surface follows the weak plane:
    ``z_b(x, y) = max(z_ellipsoid(x, y), z_plane(x, y))`` inside the ellipsoid footprint.
    Used in classic 3D slope benchmarks (Zhang 1988; Hungr et al. 1989).
    """

    ellipsoid: EllipsoidSurface
    plane_x0: float = 0.0
    plane_y0: float = 0.0
    plane_z0: float = 0.0
    dip_deg: float = 0.0
    dip_dir_deg: float = 180.0

    def _plane_z_and_grad(self, xa: np.ndarray, ya: np.ndarray) -> tuple[np.ndarray, float, float]:
        az = float(np.radians(self.dip_dir_deg))
        slope = float(np.tan(np.radians(self.dip_deg)))
        gx = -slope * float(np.cos(az))
        gy = -slope * float(np.sin(az))
        zp = self.plane_z0 + gx * (xa - self.plane_x0) + gy * (ya - self.plane_y0)
        return zp, gx, gy

    def z(self, x: Any, y: Any) -> np.ndarray:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        ze = self.ellipsoid.z(xa, ya)
        zp, _, _ = self._plane_z_and_grad(xa, ya)
        return np.where(np.isfinite(ze), np.maximum(ze, zp), np.nan)

    def gradients(self, x: Any, y: Any, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        ze = self.ellipsoid.z(xa, ya)
        edz_dx, edz_dy = self.ellipsoid.gradients(xa, ya, eps=eps)
        zp, gx, gy = self._plane_z_and_grad(xa, ya)
        on_plane = np.isfinite(ze) & (zp > ze)
        dz_dx = np.where(on_plane, gx, edz_dx)
        dz_dy = np.where(on_plane, gy, edz_dy)
        return dz_dx, dz_dy

    def rotation_center_sz(self, azimuth_deg: float) -> tuple[float, float]:
        return self.ellipsoid.rotation_center_sz(azimuth_deg)

    def footprint_bounds(self) -> tuple[float, float, float, float]:
        return self.ellipsoid.footprint_bounds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "composite_ellipsoid_plane",
            "ellipsoid": self.ellipsoid.to_dict(),
            "plane_x0": float(self.plane_x0),
            "plane_y0": float(self.plane_y0),
            "plane_z0": float(self.plane_z0),
            "dip_deg": float(self.dip_deg),
            "dip_dir_deg": float(self.dip_dir_deg),
        }


@dataclass
class WeakLayerClippedSurface(SlipSurface3D):
    """A slip surface clipped so it never dips below a set of weak-layer surfaces.

    ``z_b(x, y) = max(z_base(x, y), z_layer_1(x, y), ...)`` inside the base footprint: where
    the base surface would pass under a weak layer, the slip surface runs along the layer
    instead. ``layers`` holds ``(material_name, horizon)`` pairs; ``governing_layer`` tells
    which one, if any, carries each point so the column base can take that layer's strength.
    """

    base: SlipSurface3D
    layers: list[tuple[str, Any]] = field(default_factory=list)
    gradient_step: float = 0.02
    # Seam thickness above each layer surface (same order as ``layers``; missing = 0)
    thicknesses: list[float] = field(default_factory=list)

    def _stack(self, xa: np.ndarray, ya: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (z_base, highest layer z, index of that layer or -1)."""
        zb = self.base.z(xa, ya)
        z_top = np.full(xa.shape, -np.inf, dtype=float)
        idx = np.full(xa.shape, -1, dtype=int)
        for k, (_mat, horizon) in enumerate(self.layers):
            zl = np.asarray(horizon.z(xa, ya), dtype=float)
            higher = np.isfinite(zl) & (zl > z_top)
            z_top = np.where(higher, zl, z_top)
            idx = np.where(higher, k, idx)
        return zb, z_top, idx

    def z(self, x: Any, y: Any) -> np.ndarray:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        zb, z_top, _ = self._stack(xa, ya)
        return np.where(np.isfinite(zb), np.maximum(zb, z_top), np.nan)

    def governing_layer(self, x: Any, y: Any) -> np.ndarray:
        """Index into ``layers`` of the weak layer the surface follows at (x, y), else -1."""
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        zb, z_top, idx = self._stack(xa, ya)
        return np.where(np.isfinite(zb) & (z_top > zb), idx, -1)

    def strength_layer(self, x: Any, y: Any) -> np.ndarray:
        """Index of the weak layer whose strength applies at the slip surface, else -1.

        That is the layer the surface runs along, or failing that a layer whose seam band
        ``[z_layer, z_layer + thickness]`` contains the slip surface.
        """
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        out = self.governing_layer(xa, ya)
        z_slip = self.z(xa, ya)
        for k, (_mat, horizon) in enumerate(self.layers):
            thick = float(self.thicknesses[k]) if k < len(self.thicknesses) else 0.0
            if thick <= 0.0:
                continue
            zl = np.asarray(horizon.z(xa, ya), dtype=float)
            in_band = (out < 0) & np.isfinite(zl) & (z_slip >= zl) & (z_slip <= zl + thick)
            out = np.where(in_band, k, out)
        return out

    def gradients(self, x: Any, y: Any, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        dz_dx, dz_dy = self.base.gradients(xa, ya, eps=eps)
        gov = self.governing_layer(xa, ya)
        h = self.gradient_step
        for k, (_mat, horizon) in enumerate(self.layers):
            on_k = gov == k
            if not np.any(on_k):
                continue
            xk, yk = xa[on_k], ya[on_k]
            gx = (horizon.z(xk + h, yk) - horizon.z(xk - h, yk)) / (2.0 * h)
            gy = (horizon.z(xk, yk + h) - horizon.z(xk, yk - h)) / (2.0 * h)
            dz_dx = np.array(dz_dx, dtype=float, copy=True)
            dz_dy = np.array(dz_dy, dtype=float, copy=True)
            dz_dx[on_k] = gx
            dz_dy[on_k] = gy
        return dz_dx, dz_dy

    def rotation_center_sz(self, azimuth_deg: float) -> tuple[float, float] | None:
        return self.base.rotation_center_sz(azimuth_deg)

    def footprint_bounds(self) -> tuple[float, float, float, float]:
        return self.base.footprint_bounds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "weak_layer_clipped",
            "base": self.base.to_dict(),
            "layers": [{"material": mat, "horizon": hor.to_dict()} for mat, hor in self.layers],
            "thicknesses": [float(t) for t in self.thicknesses],
        }


@dataclass
class SplinePatchSurface(SlipSurface3D):
    """Non-ellipsoidal 3D slip surface parameterized by a tensor-product control grid.

    ``control_dz`` has shape ``(nu, nv)`` over normalized coordinates ``(u/rx, v/ry) in [-1, 1]^2``
    and perturbs ``base`` smoothly via ``z = z_base + (1 - rho^2) * S(u/rx, v/ry)``.
    """

    base: EllipsoidSurface
    control_dz: np.ndarray
    _spline: RectBivariateSpline = field(init=False, repr=False)

    def __post_init__(self) -> None:
        arr = np.asarray(self.control_dz, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2 or not np.all(np.isfinite(arr)):
            raise ValueError("control_dz must be a finite 2D array with shape (nu>=2, nv>=2).")
        self.control_dz = arr
        ug = np.linspace(-1.0, 1.0, arr.shape[0])
        vg = np.linspace(-1.0, 1.0, arr.shape[1])
        kx = min(3, arr.shape[0] - 1)
        ky = min(3, arr.shape[1] - 1)
        self._spline = RectBivariateSpline(ug, vg, arr, kx=kx, ky=ky)

    def z(self, x: Any, y: Any) -> np.ndarray:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        zb = self.base.z(xa, ya)
        inside = np.isfinite(zb)
        if not np.any(inside):
            return zb
        u, v, _, _ = self.base._local_uv(xa[inside], ya[inside])
        un = u / self.base.rx
        vn = v / self.base.ry
        rho2 = un * un + vn * vn
        dz = self._spline.ev(un, vn) * np.maximum(1.0 - rho2, 0.0)
        out = zb.copy()
        out[inside] = zb[inside] + dz
        return out

    def gradients(self, x: Any, y: Any, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        zb = self.base.z(xa, ya)
        bdz_dx, bdz_dy = self.base.gradients(xa, ya, eps=eps)
        inside = np.isfinite(zb) & np.isfinite(bdz_dx) & np.isfinite(bdz_dy)
        if not np.any(inside):
            return bdz_dx, bdz_dy
        u, v, ca, sa = self.base._local_uv(xa[inside], ya[inside])
        un = u / self.base.rx
        vn = v / self.base.ry
        rho2 = un * un + vn * vn
        taper = np.maximum(1.0 - rho2, 0.0)
        s_val = self._spline.ev(un, vn)
        su = self._spline.ev(un, vn, dx=1, dy=0)
        sv = self._spline.ev(un, vn, dx=0, dy=1)
        active_taper = rho2 < 1.0
        d_du = np.where(active_taper, (su * taper - 2.0 * un * s_val) / self.base.rx, 0.0)
        d_dv = np.where(active_taper, (sv * taper - 2.0 * vn * s_val) / self.base.ry, 0.0)
        ddx = d_du * ca - d_dv * sa
        ddy = d_du * sa + d_dv * ca
        out_x = bdz_dx.copy()
        out_y = bdz_dy.copy()
        out_x[inside] = bdz_dx[inside] + ddx
        out_y[inside] = bdz_dy[inside] + ddy
        return out_x, out_y

    def rotation_center_sz(self, azimuth_deg: float) -> tuple[float, float]:
        return self.base.rotation_center_sz(azimuth_deg)

    def footprint_bounds(self) -> tuple[float, float, float, float]:
        return self.base.footprint_bounds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "spline_patch",
            "base": self.base.to_dict(),
            "control_dz": self.control_dz.tolist(),
        }


@dataclass
class MeshSlipSurface(SlipSurface3D):
    """User-defined 3D slip surface from scattered vertices, optionally with triangles.

    Without ``faces`` the surface is a heightfield interpolated over ``(x, y)``.
    With ``faces`` a vertical ray may intersect several sheets (overhangs, folds, closed bowls).

    Triangle winding decides which hits are floors (``n_z > 0``) and which are roofs, and
    imported files have no reliable convention, so ``auto_orient`` rewinds the triangles
    (see ``orient_triangles``): closed meshes inward, open meshes to a net upward normal.
    Set ``auto_orient=False`` to keep the supplied winding verbatim.
    """

    vertices: np.ndarray
    faces: np.ndarray | None = None
    auto_orient: bool = True
    _interp: LinearNDInterpolator = field(init=False, repr=False)
    _z_ray: float = field(init=False, repr=False)
    _index: VerticalMeshIndex | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        pts = np.asarray(self.vertices, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3 or not np.all(np.isfinite(pts)):
            raise ValueError("MeshSlipSurface requires at least 3 finite (x, y, z) vertices.")
        self.vertices = pts
        self._z_ray = float(np.max(pts[:, 2]) + max(1.0, 0.05 * np.ptp(pts[:, 2])))
        if self.faces is not None:
            faces = np.asarray(self.faces, dtype=np.int64)
            if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
                raise ValueError("MeshSlipSurface.faces must be an (M, 3) triangle index array.")
            if int(np.min(faces)) < 0 or int(np.max(faces)) >= len(pts):
                raise ValueError("MeshSlipSurface.faces indexes a missing vertex.")
            self.faces = orient_triangles(pts, faces) if self.auto_orient else faces
            self._index = VerticalMeshIndex(pts, self.faces)
        self._interp = LinearNDInterpolator(pts[:, :2], pts[:, 2], fill_value=np.nan)

    @property
    def graph_valued(self) -> bool:  # type: ignore[override]
        return self.faces is None

    @classmethod
    def from_polydata(cls, mesh: pv.DataSet) -> MeshSlipSurface:
        surf = mesh.extract_surface().triangulate().clean()
        pts = np.asarray(surf.points, dtype=float)
        if surf.n_cells == 0 or len(pts) < 3:
            raise ValueError("PolyData slip surface has no triangles.")
        faces_raw = np.asarray(surf.faces, dtype=np.int64).reshape(-1, 4)
        return cls(vertices=pts, faces=faces_raw[:, 1:4])

    def z(self, x: Any, y: Any) -> np.ndarray:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if self.faces is None:
            vals = self._interp(xa.ravel(), ya.ravel())
            return np.asarray(vals, dtype=float).reshape(xa.shape)
        z_min = float(np.min(self.vertices[:, 2]) - 1.0)
        return self.sample_base(xa, ya, self._z_ray, z_min).z

    def vertical_hits(
        self, x: Any, y: Any, *, max_hits: int = MAX_VERTICAL_HITS
    ) -> tuple[np.ndarray, np.ndarray]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if self.faces is None:
            zb = np.asarray(self._interp(xa.ravel(), ya.ravel()), dtype=float).reshape(xa.shape)
            dz_dx, dz_dy = super().gradients(xa, ya)
            return graph_hits_from_z(zb, dz_dx, dz_dy, max_hits=max_hits)
        assert self._index is not None
        return self._index.hits(xa, ya, max_hits=max_hits)

    def intersect_segment(
        self,
        a: np.ndarray,
        b: np.ndarray,
        n_samples: int = 65,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.faces is None:
            return super().intersect_segment(a, b, n_samples=n_samples)
        a = np.asarray(a, dtype=float).reshape(3)
        b = np.asarray(b, dtype=float).reshape(3)
        ab = b - a
        length = float(np.linalg.norm(ab))
        if length <= 1e-9:
            return None
        direction = ab / length
        xyz, nrm = ray_triangle_hits(
            a.reshape(1, 3), direction, self.vertices, self.faces, t_min=1e-9, t_max=length
        )
        if not np.isfinite(xyz[0, 0, 0]):
            return None
        n = nrm[0, 0]
        n_len = float(np.linalg.norm(n))
        if n_len < 1e-12:
            return None
        if n[2] < 0.0:
            n = -n
            n_len = float(np.linalg.norm(n))
        return xyz[0, 0], n / n_len

    def to_mesh(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        ground_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        nx: int = 72,
        ny: int = 72,
    ) -> pv.PolyData:
        if self.faces is None:
            return super().to_mesh(x_bounds, y_bounds, ground_fn=ground_fn, nx=nx, ny=ny)
        n = len(self.faces)
        faces = np.hstack([np.full((n, 1), 3, dtype=np.int64), self.faces]).ravel()
        poly = pv.PolyData(self.vertices, faces)
        x0, x1 = x_bounds
        y0, y1 = y_bounds
        in_xy = (
            (self.vertices[:, 0] >= x0)
            & (self.vertices[:, 0] <= x1)
            & (self.vertices[:, 1] >= y0)
            & (self.vertices[:, 1] <= y1)
        )
        if not np.any(in_xy):
            return pv.PolyData()
        return poly

    def footprint_bounds(self) -> tuple[float, float, float, float]:
        return (
            float(np.min(self.vertices[:, 0])),
            float(np.max(self.vertices[:, 0])),
            float(np.min(self.vertices[:, 1])),
            float(np.max(self.vertices[:, 1])),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": "mesh",
            "vertices": self.vertices.tolist(),
        }
        if self.faces is not None:
            payload["faces"] = self.faces.tolist()
            payload["auto_orient"] = bool(self.auto_orient)
        return payload


@dataclass
class ImplicitSlipSurface(SlipSurface3D):
    """General F(x, y, z) = 0 slip manifold sampled by vertical line search.

    ``F < 0`` is treated as the interior / sliding-mass side, so ``n = -∇F / |∇F|``.

    Scripting-only: ``fn`` and ``grad_fn`` are Python callables, so ``to_dict`` records
    only the name and bounds and ``surface_from_dict`` refuses to rebuild the surface. A
    project using one cannot be reopened from its ``.slp3d.json``; export the surface as
    a triangle mesh (``MeshSlipSurface``) when it has to be saved.
    """

    fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    bounds: tuple[float, float, float, float, float, float]
    n_z_scan: int = 48
    name: str = "implicit"
    graph_valued: bool = False

    def footprint_bounds(self) -> tuple[float, float, float, float]:
        return (
            float(self.bounds[0]),
            float(self.bounds[1]),
            float(self.bounds[2]),
            float(self.bounds[3]),
        )

    def z(self, x: Any, y: Any) -> np.ndarray:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return self.sample_base(xa, ya, self.bounds[5], self.bounds[4]).z

    def vertical_hits(
        self, x: Any, y: Any, *, max_hits: int = MAX_VERTICAL_HITS
    ) -> tuple[np.ndarray, np.ndarray]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        x0, x1, y0, y1, z0, z1 = (float(v) for v in self.bounds)
        inside = (xa >= x0) & (xa <= x1) & (ya >= y0) & (ya <= y1)
        zs = np.linspace(z1, z0, max(8, int(self.n_z_scan)))
        layers = [np.asarray(self.fn(xa, ya, np.full(xa.shape, zc)), dtype=float) for zc in zs]
        f_stack = np.stack(layers, axis=-1)
        z_hits, normals = empty_hit_buffers(xa.shape, max_hits=max_hits)
        slot = np.zeros(xa.shape, dtype=int)
        for k in range(len(zs) - 1):
            f0 = f_stack[..., k]
            f1 = f_stack[..., k + 1]
            crossed = (
                inside
                & np.isfinite(f0)
                & np.isfinite(f1)
                & (f0 * f1 <= 0.0)
                & (np.abs(f1 - f0) > 1e-14)
            )
            if not np.any(crossed):
                continue
            z_root = zs[k] - f0 * (zs[k + 1] - zs[k]) / (f1 - f0)
            g = np.asarray(self.grad_fn(xa, ya, z_root), dtype=float)
            if g.shape == xa.shape + (3,):
                grad = g
            else:
                raise ValueError(
                    "ImplicitSlipSurface.grad_fn must return an array of shape (..., 3)."
                )
            ln = np.linalg.norm(grad, axis=-1, keepdims=True)
            n_into = np.where(ln > 1e-14, -grad / np.maximum(ln, 1e-14), np.nan)
            use = crossed & (slot < max_hits)
            if not np.any(use):
                continue
            # Write into the current slot per-point (slots may differ)
            for s in range(max_hits):
                here = use & (slot == s)
                if not np.any(here):
                    continue
                z_hits[..., s] = np.where(here, z_root, z_hits[..., s])
                normals[..., s, :] = np.where(here[..., None], n_into, normals[..., s, :])
            slot = slot + use.astype(int)
        return z_hits, normals

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "implicit",
            "name": self.name,
            "bounds": [float(v) for v in self.bounds],
            "n_z_scan": int(self.n_z_scan),
        }


@dataclass
class WedgePlane3D:
    """Single planar geological discontinuity (joint, bedding, fault, or tension-crack plane)."""

    dip_deg: float
    dip_dir_deg: float
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0
    name: str = "Joint"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.dip_deg) < 89.9:
            raise ValueError("WedgePlane3D dip_deg must lie in [0, 89.9) degrees.")

    def z(self, x: Any, y: Any) -> np.ndarray:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        slope = float(np.tan(np.radians(self.dip_deg)))
        az = float(np.radians(self.dip_dir_deg))
        return self.z0 - slope * ((xa - self.x0) * np.cos(az) + (ya - self.y0) * np.sin(az))

    def slope_xy(self) -> tuple[float, float]:
        slope = float(np.tan(np.radians(self.dip_deg)))
        az = float(np.radians(self.dip_dir_deg))
        return -slope * float(np.cos(az)), -slope * float(np.sin(az))

    def upward_normal(self) -> np.ndarray:
        a = float(np.radians(self.dip_deg))
        b = float(np.radians(self.dip_dir_deg))
        return np.array([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dip_deg": float(self.dip_deg),
            "dip_dir_deg": float(self.dip_dir_deg),
            "x0": float(self.x0),
            "y0": float(self.y0),
            "z0": float(self.z0),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WedgePlane3D:
        return cls(
            dip_deg=float(d["dip_deg"]),
            dip_dir_deg=float(d["dip_dir_deg"]),
            x0=float(d.get("x0", 0.0)),
            y0=float(d.get("y0", 0.0)),
            z0=float(d.get("z0", 0.0)),
            name=str(d.get("name", "Joint")),
        )


@dataclass
class MultiPlaneWedgeSurface(SlipSurface3D):
    """3D rock wedge / fault-controlled slip surface formed by the upper envelope of >= 2 planes.

    Evaluates ``z_b(x, y) = max_k z_k(x, y)`` inside ``bounds = (x_min, x_max, y_min, y_max)``
    with exact piecewise-planar analytic gradients and kinematic intersection-line azimuth.
    """

    planes: list[WedgePlane3D]
    bounds: tuple[float, float, float, float] = (-1.0e4, 1.0e4, -1.0e4, 1.0e4)

    def __post_init__(self) -> None:
        if len(self.planes) < 1:
            raise ValueError("MultiPlaneWedgeSurface requires at least 1 WedgePlane3D.")

    def z(self, x: Any, y: Any) -> np.ndarray:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        x0, x1, y0, y1 = self.bounds
        inside = (xa >= x0) & (xa <= x1) & (ya >= y0) & (ya <= y1)
        z_env = np.full(xa.shape, -np.inf, dtype=float)
        for pl in self.planes:
            z_env = np.maximum(z_env, pl.z(xa, ya))
        return np.where(inside, z_env, np.nan)

    def gradients(self, x: Any, y: Any, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        del eps
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        x0, x1, y0, y1 = self.bounds
        inside = (xa >= x0) & (xa <= x1) & (ya >= y0) & (ya <= y1)
        z_best = np.full(xa.shape, -np.inf, dtype=float)
        gx = np.full(xa.shape, np.nan, dtype=float)
        gy = np.full(xa.shape, np.nan, dtype=float)
        for pl in self.planes:
            zp = pl.z(xa, ya)
            sx, sy = pl.slope_xy()
            win = zp > z_best
            z_best = np.where(win, zp, z_best)
            gx = np.where(win, sx, gx)
            gy = np.where(win, sy, gy)
        return np.where(inside, gx, np.nan), np.where(inside, gy, np.nan)

    def kinematic_azimuth_deg(self) -> float:
        """Downhill plunge azimuth [deg] of the primary two-plane intersection line (or single dip_dir)."""
        if len(self.planes) == 1:
            return float(self.planes[0].dip_dir_deg) % 360.0
        n1 = self.planes[0].upward_normal()
        n2 = self.planes[1].upward_normal()
        line = np.cross(n1, n2)
        horiz = float(np.hypot(line[0], line[1]))
        if horiz < 1e-9:
            return float(self.planes[0].dip_dir_deg) % 360.0
        if line[2] > 0.0:
            line = -line
        elif abs(line[2]) <= 1e-12:
            # Horizontal intersection line: orient along mean dip direction
            d_mean = 0.5 * (n1[:2] + n2[:2])
            if float(np.dot(line[:2], d_mean)) < 0.0:
                line = -line
        return float(np.degrees(np.arctan2(line[1], line[0])) % 360.0)

    def intersection_plunge_deg(self) -> float:
        """Plunge angle [deg] below horizontal of the primary two-plane intersection line."""
        if len(self.planes) == 1:
            return float(self.planes[0].dip_deg)
        n1 = self.planes[0].upward_normal()
        n2 = self.planes[1].upward_normal()
        line = np.cross(n1, n2)
        norm_l = float(np.linalg.norm(line))
        if norm_l < 1e-12:
            return float(self.planes[0].dip_deg)
        if line[2] > 0.0:
            line = -line
        return float(np.degrees(np.arcsin(np.clip(-line[2] / norm_l, 0.0, 1.0))))

    def footprint_bounds(self) -> tuple[float, float, float, float]:
        return (
            float(self.bounds[0]),
            float(self.bounds[1]),
            float(self.bounds[2]),
            float(self.bounds[3]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "multi_plane_wedge",
            "planes": [p.to_dict() for p in self.planes],
            "bounds": [float(v) for v in self.bounds],
        }


def surface_from_dict(d: dict[str, Any]) -> SlipSurface3D:
    """Deserialize a SlipSurface3D instance from a dictionary."""
    kind = str(d.get("kind", "ellipsoid"))
    if kind == "ellipsoid":
        return EllipsoidSurface(
            xc=float(d["xc"]),
            yc=float(d["yc"]),
            zc=float(d["zc"]),
            rx=float(d["rx"]),
            ry=float(d["ry"]),
            rz=float(d["rz"]),
            azimuth_deg=float(d.get("azimuth_deg", 180.0)),
            sheet=str(d.get("sheet", "lower")),
        )
    if kind == "composite_ellipsoid_plane":
        return CompositeEllipsoidPlaneSurface(
            ellipsoid=surface_from_dict(d["ellipsoid"]),  # type: ignore[arg-type]
            plane_x0=float(d.get("plane_x0", 0.0)),
            plane_y0=float(d.get("plane_y0", 0.0)),
            plane_z0=float(d.get("plane_z0", 0.0)),
            dip_deg=float(d.get("dip_deg", 0.0)),
            dip_dir_deg=float(d.get("dip_dir_deg", 180.0)),
        )
    if kind == "weak_layer_clipped":
        from slipsurface3d.geometry.terrain import HorizonSurface

        return WeakLayerClippedSurface(
            base=surface_from_dict(d["base"]),
            layers=[
                (str(item["material"]), HorizonSurface.from_dict(item["horizon"]))
                for item in d.get("layers", [])
            ],
            thicknesses=[float(t) for t in d.get("thicknesses", [])],
        )
    if kind == "spline_patch":
        return SplinePatchSurface(
            base=surface_from_dict(d["base"]),  # type: ignore[arg-type]
            control_dz=np.asarray(d["control_dz"], dtype=float),
        )
    if kind == "multi_plane_wedge":
        b_raw = d.get("bounds", [-1.0e4, 1.0e4, -1.0e4, 1.0e4])
        return MultiPlaneWedgeSurface(
            planes=[WedgePlane3D.from_dict(p) for p in d["planes"]],
            bounds=(float(b_raw[0]), float(b_raw[1]), float(b_raw[2]), float(b_raw[3])),
        )
    if kind == "mesh":
        faces = d.get("faces")
        return MeshSlipSurface(
            vertices=np.asarray(d["vertices"], dtype=float),
            faces=None if faces is None else np.asarray(faces, dtype=np.int64),
            auto_orient=bool(d.get("auto_orient", True)),
        )
    if kind == "implicit":
        raise ValueError("ImplicitSlipSurface cannot be deserialized from JSON (callable F).")
    raise ValueError(f"Unknown 3D slip surface kind: {kind!r}")
