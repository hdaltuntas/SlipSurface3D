"""Three-dimensional slip surface representations.

Supported 3D slip surfaces:
* ``EllipsoidSurface``              : General tri-axial 3D ellipsoid (covers spheres when
                                      ``rx == ry == rz`` and 2D circular cylinders when ``ry`` is large).
* ``CompositeEllipsoidPlaneSurface``: Ellipsoid truncated by a weak geological plane (e.g. Zhang 1988
                                      and Hungr et al. 1989 weak-seam benchmark).
* ``SplinePatchSurface``            : Smooth non-ellipsoidal 3D bowl formed by a tensor-product
                                      elevation perturbation over an ``EllipsoidSurface``.
* ``MeshSlipSurface``               : Arbitrary triangulated / point-cloud 3D slip surface interpolated
                                      linearly over ``(x, y)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline


class SlipSurface3D(ABC):
    """Base class for single-valued 3D basal slip surfaces z_b = f(x, y)."""

    @abstractmethod
    def z(self, x: Any, y: Any) -> np.ndarray:
        """Return basal elevation z_b(x, y), or NaN outside the surface footprint."""

    def gradients(self, x: Any, y: Any, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        """Return partial derivatives (dz_dx, dz_dy) at (x, y)."""
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        dz_dx = (self.z(xa + eps, ya) - self.z(xa - eps, ya)) / (2.0 * eps)
        dz_dy = (self.z(xa, ya + eps) - self.z(xa, ya - eps)) / (2.0 * eps)
        return dz_dx, dz_dy

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

    def __post_init__(self) -> None:
        if self.rx <= 0 or self.ry <= 0 or self.rz <= 0:
            raise ValueError("Ellipsoid semi-axes rx, ry, rz must all be strictly positive.")

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

    def z(self, x: Any, y: Any) -> np.ndarray:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        u, v, _, _ = self._local_uv(xa, ya)
        rho2 = (u / self.rx) ** 2 + (v / self.ry) ** 2
        inside = rho2 < 1.0 - 1e-12
        out = np.full(xa.shape, np.nan, dtype=float)
        if np.any(inside):
            out[inside] = self.zc - self.rz * np.sqrt(np.maximum(1.0 - rho2[inside], 0.0))
        return out

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
    """User-defined / imported 3D slip surface interpolated linearly over scattered (x, y, z) vertices."""

    vertices: np.ndarray
    _interp: LinearNDInterpolator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pts = np.asarray(self.vertices, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3 or not np.all(np.isfinite(pts)):
            raise ValueError("MeshSlipSurface requires at least 3 finite (x, y, z) vertices.")
        self.vertices = pts
        self._interp = LinearNDInterpolator(pts[:, :2], pts[:, 2], fill_value=np.nan)

    def z(self, x: Any, y: Any) -> np.ndarray:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        vals = self._interp(xa.ravel(), ya.ravel())
        return np.asarray(vals, dtype=float).reshape(xa.shape)

    def footprint_bounds(self) -> tuple[float, float, float, float]:
        return (
            float(np.min(self.vertices[:, 0])),
            float(np.max(self.vertices[:, 0])),
            float(np.min(self.vertices[:, 1])),
            float(np.max(self.vertices[:, 1])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "mesh",
            "vertices": self.vertices.tolist(),
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
        return MeshSlipSurface(vertices=np.asarray(d["vertices"], dtype=float))
    raise ValueError(f"Unknown 3D slip surface kind: {kind!r}")
