"""3D Steady-State Anisotropic Seepage Solver (Laplace Total Hydraulic Head H(x, y, z))."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from slipsurface3d.geometry.terrain import HorizonSurface, SlopeModel3D


@dataclass
class SeepageBC3D:
    """Boundary conditions and hydraulic conductivity ratios for 3D steady seepage."""

    head_xmin: float | None = None
    head_xmax: float | None = None
    kx: float = 1.0
    ky: float = 1.0
    kz: float = 1.0
    nx: int = 30
    ny: int = 22
    nz: int = 18

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_xmin": None if self.head_xmin is None else float(self.head_xmin),
            "head_xmax": None if self.head_xmax is None else float(self.head_xmax),
            "kx": float(self.kx),
            "ky": float(self.ky),
            "kz": float(self.kz),
            "nx": int(self.nx),
            "ny": int(self.ny),
            "nz": int(self.nz),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SeepageBC3D:
        return cls(
            head_xmin=None if d.get("head_xmin") is None else float(d["head_xmin"]),
            head_xmax=None if d.get("head_xmax") is None else float(d["head_xmax"]),
            kx=float(d.get("kx", 1.0)),
            ky=float(d.get("ky", 1.0)),
            kz=float(d.get("kz", 1.0)),
            nx=int(d.get("nx", 30)),
            ny=int(d.get("ny", 22)),
            nz=int(d.get("nz", 18)),
        )


@dataclass
class SeepageGrid3D:
    """Solved 3D total hydraulic head field H(x, y, z) [m] and pore-water pressure interpolator."""

    x_coords: np.ndarray
    y_coords: np.ndarray
    z_coords: np.ndarray
    total_head: np.ndarray  # shape (nx, ny, nz)
    gamma_w: float = 9.81
    _interp: RegularGridInterpolator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._interp = RegularGridInterpolator(
            (self.x_coords, self.y_coords, self.z_coords),
            self.total_head,
            bounds_error=False,
            fill_value=None,
        )

    def head_at(self, x: Any, y: Any, z: Any) -> np.ndarray:
        xa, ya, za = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        xc = np.clip(xa.ravel(), self.x_coords[0], self.x_coords[-1])
        yc = np.clip(ya.ravel(), self.y_coords[0], self.y_coords[-1])
        zc = np.clip(za.ravel(), self.z_coords[0], self.z_coords[-1])
        pts = np.column_stack([xc, yc, zc])
        vals = np.asarray(self._interp(pts), dtype=float)
        return vals.reshape(xa.shape)

    def pore_pressure_at(self, x: Any, y: Any, z: Any) -> np.ndarray:
        """Evaluate positive pore-water pressure u = gamma_w * max(H(x,y,z) - z, 0) [kPa]."""
        za = np.asarray(z, dtype=float)
        h_tot = self.head_at(x, y, za)
        return float(self.gamma_w) * np.maximum(h_tot - za, 0.0)

    def phreatic_horizon(self, name: str = "3D Seepage Phreatic Surface") -> HorizonSurface:
        """Extract zero-pressure phreatic surface H(x, y, z) = z across the plan grid."""
        nx, ny, nz = self.total_head.shape
        z_vec = self.z_coords
        z_wt = np.empty((nx, ny), dtype=float)
        for i in range(nx):
            for j in range(ny):
                p_head = self.total_head[i, j, :] - z_vec  # pressure head hp(z) = H - z
                if p_head[0] <= 0.0:
                    z_wt[i, j] = float(z_vec[0])
                elif p_head[-1] >= 0.0:
                    z_wt[i, j] = float(z_vec[-1])
                else:
                    # Locate highest crossing from hp >= 0 to hp < 0
                    z_cross = float(z_vec[0])
                    for k in range(nz - 1):
                        if p_head[k] >= 0.0 and p_head[k + 1] < 0.0:
                            denom = p_head[k] - p_head[k + 1]
                            frac = float(p_head[k] / denom) if abs(denom) > 1e-12 else 0.5
                            z_cross = float(z_vec[k] + frac * (z_vec[k + 1] - z_vec[k]))
                    z_wt[i, j] = z_cross
        return HorizonSurface.from_grid(self.x_coords, self.y_coords, z_wt, name=name)


def solve_seepage_3d(
    model: SlopeModel3D,
    bc: SeepageBC3D | None = None,
    gamma_w: float = 9.81,
) -> SeepageGrid3D:
    """Solve 3D anisotropic steady-state Laplace equation div(K * grad(H)) = 0 with seepage face."""
    cfg = bc or SeepageBC3D()
    nx = max(6, int(cfg.nx))
    ny = max(4, int(cfg.ny))
    nz = max(6, int(cfg.nz))
    kx = max(float(cfg.kx), 1e-9)
    ky = max(float(cfg.ky), 1e-9)
    kz = max(float(cfg.kz), 1e-9)

    xs = np.linspace(model.x_min, model.x_max, nx)
    ys = np.linspace(model.y_min, model.y_max, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zg = model.ground_z(xx, yy)
    z_top_max = float(np.max(zg))
    zs = np.linspace(model.z_min, z_top_max, nz)

    dx = max(float(xs[1] - xs[0]), 1e-6)
    dy = max(float(ys[1] - ys[0]), 1e-6) if ny > 1 else 1.0
    dz = max(float(zs[1] - zs[0]), 1e-6)

    wx = kx / (dx * dx)
    wy = ky / (dy * dy) if ny > 1 else 0.0
    wz = kz / (dz * dz)

    # Default boundary heads from left/right mean ground elevations when unspecified
    h_xmin = cfg.head_xmin
    h_xmax = cfg.head_xmax
    if h_xmin is None and h_xmax is None:
        z_left = float(np.mean(zg[0, :]))
        z_right = float(np.mean(zg[-1, :]))
        if z_right >= z_left:
            h_xmin = z_left
            h_xmax = z_left + 0.80 * (z_right - z_left)
        else:
            h_xmax = z_right
            h_xmin = z_right + 0.80 * (z_left - z_right)

    # Identify highest subterranean node k_top[i, j] for every plan column
    k_top = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            below = np.flatnonzero(zs <= zg[i, j] + 1e-6)
            k_top[i, j] = int(below[-1]) if len(below) > 0 else 0

    n_nodes = nx * ny * nz
    seep_clamped = np.zeros((nx, ny), dtype=bool)
    head_3d: np.ndarray = np.zeros((nx, ny, nz), dtype=float)

    def _nid(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    for _it in range(5):
        mat = lil_matrix((n_nodes, n_nodes), dtype=float)
        rhs = np.zeros(n_nodes, dtype=float)

        for i in range(nx):
            for j in range(ny):
                kt = int(k_top[i, j])
                for k in range(nz):
                    row = _nid(i, j, k)
                    # Air node above ground surface: tie to top ground node kt
                    if k > kt:
                        mat[row, row] = 1.0
                        mat[row, _nid(i, j, kt)] = -1.0
                        rhs[row] = 0.0
                        continue

                    # Lateral Dirichlet head at x = x_min or x = x_max
                    if i == 0 and h_xmin is not None:
                        h_bc = min(float(h_xmin), float(zg[i, j]))
                        mat[row, row] = 1.0
                        rhs[row] = h_bc
                        continue
                    if i == nx - 1 and h_xmax is not None:
                        h_bc = min(float(h_xmax), float(zg[i, j]))
                        mat[row, row] = 1.0
                        rhs[row] = h_bc
                        continue

                    # Atmospheric seepage face at ground surface where head broke out above ground
                    if k == kt and seep_clamped[i, j]:
                        mat[row, row] = 1.0
                        rhs[row] = float(zg[i, j])
                        continue

                    diag = 0.0
                    # X neighbours (mirror Neumann if neighbour is air or exterior)
                    for ni in (i - 1, i + 1):
                        if 0 <= ni < nx and k <= int(k_top[ni, j]):
                            mat[row, _nid(ni, j, k)] = -wx
                            diag += wx
                    # Y neighbours
                    if ny > 1:
                        for nj in (j - 1, j + 1):
                            if 0 <= nj < ny and k <= int(k_top[i, nj]):
                                mat[row, _nid(i, nj, k)] = -wy
                                diag += wy
                    # Z neighbours (impermeable base at k=0, unsaturated no-flux top at k=kt)
                    for nk in (k - 1, k + 1):
                        if 0 <= nk <= kt:
                            mat[row, _nid(i, j, nk)] = -wz
                            diag += wz

                    if diag <= 0.0:
                        mat[row, row] = 1.0
                        rhs[row] = float(zg[i, j])
                    else:
                        mat[row, row] = diag

        sol = np.asarray(spsolve(mat.tocsc(), rhs), dtype=float)
        head_3d = sol.reshape((nx, ny, nz))

        # Check whether any unclamped ground-surface node exceeds ground elevation zg
        newly_seeping = False
        for i in range(1, nx - 1):
            for j in range(ny):
                kt = int(k_top[i, j])
                if not seep_clamped[i, j] and head_3d[i, j, kt] > zg[i, j] + 1e-3:
                    seep_clamped[i, j] = True
                    newly_seeping = True
        if not newly_seeping:
            break

    return SeepageGrid3D(
        x_coords=xs,
        y_coords=ys,
        z_coords=zs,
        total_head=head_3d,
        gamma_w=gamma_w,
    )
