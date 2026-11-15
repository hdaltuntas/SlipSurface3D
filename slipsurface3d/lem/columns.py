"""Three-dimensional Method of Columns discretization engine.

Discretizes the 3D sliding volume bounded above by ``SlopeModel3D.ground`` and
below by ``SlipSurface3D`` into an active grid of vertical prismatic columns,
computing exact 3D basal areas, unit normals, weights, pore pressures, seismic
loads, external surface/point loads, and 3D reinforcement intersections.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from slipsurface3d.geometry.surfaces import MultiPlaneWedgeSurface, SlipSurface3D
from slipsurface3d.geometry.terrain import SlopeModel3D
from slipsurface3d.loads.supports import LoadSet3D
from slipsurface3d.materials.models import Material3D

GAMMA_W = 9.81  # kN/m^3


class ColumnDiscretizationError(ValueError):
    """Raised when a trial 3D slip surface produces no valid/admissible column set."""


@dataclass
class ColumnOptions:
    nx: int = 28
    ny: int = 24
    n_sub: int = 20
    gamma_w: float = GAMMA_W
    pore_mode: str = "water_table"  # water_table | ru | none
    ru: float = 0.0
    kh: float = 0.0
    kv: float = 0.0  # positive upward: Wv = (1 - kv) * W
    sliding_mode: str = "auto"  # auto | right_to_left | left_to_right | custom
    azimuth_deg: float = 180.0
    min_depth: float = 0.05
    min_columns: int = 6
    loads: LoadSet3D | None = None
    pore_pressure_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None


@dataclass
class SupportHit3D:
    """Record of a 3D reinforcement bolt intersecting the basal slip surface."""

    label: str
    hit_xyz: np.ndarray
    force_xyz_kn: np.ndarray  # force vector [kN] exerted BY bolt ON sliding mass
    capacity_kn: float
    application: str  # active | passive
    kind: str
    col_idx: int


@dataclass
class ColumnGrid:
    """Active 3D columns for a given slip surface and sliding azimuth."""

    # Plan coordinates & cell geometry (length = n_cols)
    ix: np.ndarray
    iy: np.ndarray
    nx_grid: int
    ny_grid: int
    dx: float
    dy: float
    x: np.ndarray
    y: np.ndarray
    z_base: np.ndarray
    z_top: np.ndarray
    z_cg: np.ndarray
    h: np.ndarray
    dA: np.ndarray  # dx * dy [m^2]
    area_base: np.ndarray  # true inclined 3D base area dA * J [m^2]

    # Sliding frame (azimuth_deg: +s = sliding direction, +t = transverse, +z = upward)
    azimuth_deg: float
    s: np.ndarray
    t: np.ndarray
    sc: float  # moment axis s-coordinate
    zc: float  # moment axis z-coordinate
    dz_dx: np.ndarray
    dz_dy: np.ndarray
    tan_alpha_s: np.ndarray  # -dz_b / ds (positive dipping downhill in +s)
    tan_alpha_t: np.ndarray  # -dz_b / dt
    alpha_s: np.ndarray  # arctan(tan_alpha_s) [rad]
    sin_alpha_s: np.ndarray
    cos_alpha_s: np.ndarray
    jacobian: np.ndarray  # sqrt(1 + tan^2(alpha_s) + tan^2(alpha_t))

    # Upward unit normal of column base in sliding frame (ns, nt, nz) and world frame (nx, ny, nz)
    ns: np.ndarray
    nt: np.ndarray
    nz: np.ndarray
    normal_xyz: np.ndarray  # (n_cols, 3)

    # Weights, pore pressures, materials
    W: np.ndarray  # total column weight [kN]
    u: np.ndarray  # basal pore-water pressure [kPa]
    u_excess: np.ndarray  # B-bar excess pore pressure [kPa]
    sigma_v_eff: np.ndarray  # effective vertical stress at base [kPa]
    mat_idx: np.ndarray  # index into `materials`
    materials: list[Material3D]
    material_colors: list[str]
    surface: SlipSurface3D

    # Seismic & external loads in sliding frame (+s downhill, +t transverse, +z upward)
    kh: float = 0.0
    kv: float = 0.0
    Qs_active: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Qt_active: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Qz_active: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Ms_active: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )  # CCW driving moment about (sc, zc)
    Qs_passive: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Qt_passive: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Qz_passive: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Ms_passive: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Fx_active: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Fy_active: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Fx_passive: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Fy_passive: np.ndarray = field(default_factory=lambda: np.zeros(0))
    crack_water_force_s: float = 0.0
    crack_water_moment: float = 0.0
    support_hits: list[SupportHit3D] = field(default_factory=list)

    def reproject_azimuth(self, new_azimuth_deg: float) -> ColumnGrid:
        """Re-orient sliding-frame projections ``(s, t, alpha_s, ns, nt, Qs, Qt, Ms)`` to ``new_azimuth_deg``."""
        az = float(new_azimuth_deg) % 360.0
        psi = float(np.radians(az))
        sx, sy = float(np.cos(psi)), float(np.sin(psi))
        tx, ty = -sy, sx

        s = self.x * sx + self.y * sy
        t = self.x * tx + self.y * ty
        tan_as = -(self.dz_dx * sx + self.dz_dy * sy)
        tan_at = -(self.dz_dx * tx + self.dz_dy * ty)
        alpha_s = np.arctan(tan_as)
        sin_as = np.sin(alpha_s)
        cos_as = np.cos(alpha_s)
        ns = tan_as / self.jacobian
        nt = tan_at / self.jacobian

        rot = self.surface.rotation_center_sz(az)
        if rot is not None:
            sc, zc = rot
        else:
            sc = float(0.5 * (np.min(s) + np.max(s)))
            zc = float(np.max(self.z_top) + 0.6 * max(1.0, float(np.max(s) - np.min(s))))

        fx_a = self.Fx_active if len(self.Fx_active) == self.n else np.zeros(self.n)
        fy_a = self.Fy_active if len(self.Fy_active) == self.n else np.zeros(self.n)
        fx_p = self.Fx_passive if len(self.Fx_passive) == self.n else np.zeros(self.n)
        fy_p = self.Fy_passive if len(self.Fy_passive) == self.n else np.zeros(self.n)

        qs_a = fx_a * sx + fy_a * sy
        qt_a = fx_a * tx + fy_a * ty
        qs_p = fx_p * sx + fy_p * sy
        qt_p = fx_p * tx + fy_p * ty
        ms_a = (s - sc) * self.Qz_active - (self.z_top - zc) * qs_a
        ms_p = (s - sc) * self.Qz_passive - (self.z_base - zc) * qs_p

        return ColumnGrid(
            ix=self.ix,
            iy=self.iy,
            nx_grid=self.nx_grid,
            ny_grid=self.ny_grid,
            dx=self.dx,
            dy=self.dy,
            x=self.x,
            y=self.y,
            z_base=self.z_base,
            z_top=self.z_top,
            z_cg=self.z_cg,
            h=self.h,
            dA=self.dA,
            area_base=self.area_base,
            azimuth_deg=az,
            s=s,
            t=t,
            sc=sc,
            zc=zc,
            dz_dx=self.dz_dx,
            dz_dy=self.dz_dy,
            tan_alpha_s=tan_as,
            tan_alpha_t=tan_at,
            alpha_s=alpha_s,
            sin_alpha_s=sin_as,
            cos_alpha_s=cos_as,
            jacobian=self.jacobian,
            ns=ns,
            nt=nt,
            nz=self.nz,
            normal_xyz=self.normal_xyz,
            W=self.W,
            u=self.u,
            u_excess=self.u_excess,
            sigma_v_eff=self.sigma_v_eff,
            mat_idx=self.mat_idx,
            materials=self.materials,
            material_colors=self.material_colors,
            surface=self.surface,
            kh=self.kh,
            kv=self.kv,
            Qs_active=qs_a,
            Qt_active=qt_a,
            Qz_active=self.Qz_active,
            Ms_active=ms_a,
            Qs_passive=qs_p,
            Qt_passive=qt_p,
            Qz_passive=self.Qz_passive,
            Ms_passive=ms_p,
            Fx_active=fx_a,
            Fy_active=fy_a,
            Fx_passive=fx_p,
            Fy_passive=fy_p,
            crack_water_force_s=self.crack_water_force_s,
            crack_water_moment=self.crack_water_moment,
            support_hits=self.support_hits,
        )

    @property
    def n(self) -> int:
        return int(len(self.x))

    @property
    def total_volume(self) -> float:
        return float(np.sum(self.h * self.dA))

    @property
    def total_weight(self) -> float:
        return float(np.sum(self.W))

    @property
    def max_depth(self) -> float:
        return float(np.max(self.h)) if self.n > 0 else 0.0

    @property
    def Wv(self) -> np.ndarray:
        """Downward vertical gravity-seismic force (1 - kv) * W [kN]."""
        return self.W * (1.0 - self.kv)

    @property
    def Fh(self) -> np.ndarray:
        """Horizontal pseudo-static force kh * W [kN] in +s sliding direction."""
        return self.W * self.kh

    def to_hex_mesh(
        self,
        sigma_n_eff: np.ndarray | None = None,
        tau_mob: np.ndarray | None = None,
    ) -> pv.UnstructuredGrid:
        """Build a 3D hexahedral PyVista UnstructuredGrid of all active columns."""
        n = self.n
        if n == 0:
            return pv.UnstructuredGrid()

        hx = 0.5 * self.dx
        hy = 0.5 * self.dy
        # 8 corner points per column (VTK_HEXAHEDRON = cell type 12)
        # Bottom face (0..3) sloped lightly with dz_dx, dz_dy; top face (4..7) at z_top
        cx = np.array([-hx, +hx, +hx, -hx, -hx, +hx, +hx, -hx], dtype=float)
        cy = np.array([-hy, -hy, +hy, +hy, -hy, -hy, +hy, +hy], dtype=float)

        px = self.x[:, None] + cx[None, :]
        py = self.y[:, None] + cy[None, :]

        zb_corners = (
            self.z_base[:, None]
            + self.dz_dx[:, None] * cx[None, :4]
            + self.dz_dy[:, None] * cy[None, :4]
        )
        zt_corners = np.broadcast_to(self.z_top[:, None], (n, 4)).copy()
        zb_corners = np.minimum(zb_corners, zt_corners - 1e-4)
        pz = np.concatenate([zb_corners, zt_corners], axis=1)

        points = np.column_stack([px.ravel(), py.ravel(), pz.ravel()])
        conn = np.arange(8 * n, dtype=np.int64).reshape(n, 8)
        cells = np.hstack([np.full((n, 1), 8, dtype=np.int64), conn]).ravel()
        celltypes = np.full(n, 12, dtype=np.uint8)  # 12 == VTK_HEXAHEDRON

        ugrid = pv.UnstructuredGrid(cells, celltypes, points)
        ugrid.cell_data["ColumnIndex"] = np.arange(n, dtype=np.int32)
        ugrid.cell_data["MaterialID"] = self.mat_idx.astype(np.int32)
        ugrid.cell_data["Height_m"] = self.h
        ugrid.cell_data["Weight_kN"] = self.W
        ugrid.cell_data["PorePressure_kPa"] = self.u
        ugrid.cell_data["SigmaV_Eff_kPa"] = self.sigma_v_eff
        ugrid.cell_data["BaseDip_deg"] = np.degrees(self.alpha_s)
        if sigma_n_eff is not None and len(sigma_n_eff) == n:
            ugrid.cell_data["SigmaN_Eff_kPa"] = np.asarray(sigma_n_eff, dtype=float)
        if tau_mob is not None and len(tau_mob) == n:
            ugrid.cell_data["TauMob_kPa"] = np.asarray(tau_mob, dtype=float)
        return ugrid


def _tight_active_bounds(
    model: SlopeModel3D, surface: SlipSurface3D
) -> tuple[float, float, float, float]:
    """Find tight plan bounding box [x0, x1] x [y0, y1] where z_base < z_ground."""
    fx0, fx1, fy0, fy1 = surface.footprint_bounds()
    x0 = max(model.x_min, fx0)
    x1 = min(model.x_max, fx1)
    y0 = max(model.y_min, fy0)
    y1 = min(model.y_max, fy1)
    if x1 <= x0 or y1 <= y0:
        raise ColumnDiscretizationError("Slip surface footprint lies outside model XY domain.")

    # Coarse-to-fine symmetric probe grid to isolate the daylighting bowl
    nx_p, ny_p = 128, 128
    xs = np.linspace(x0, x1, nx_p)
    ys = np.linspace(y0, y1, ny_p)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zb = surface.z(xx, yy)
    zg = model.ground_z(xx, yy)
    active = np.isfinite(zb) & (zg - zb > 1e-5) & (zb >= model.z_min - 1e-3)
    if not np.any(active):
        raise ColumnDiscretizationError("Slip surface does not intersect below ground surface.")

    act_x = np.any(active, axis=1)
    act_y = np.any(active, axis=0)
    ix_idx = np.flatnonzero(act_x)
    iy_idx = np.flatnonzero(act_y)
    step_x = (x1 - x0) / (nx_p - 1)
    step_y = (y1 - y0) / (ny_p - 1)

    ax0 = max(x0, float(xs[ix_idx[0]]) - step_x)
    ax1 = min(x1, float(xs[ix_idx[-1]]) + step_x)
    ay0 = max(y0, float(ys[iy_idx[0]]) - step_y)
    ay1 = min(y1, float(ys[iy_idx[-1]]) + step_y)

    # Refine exit/entry by 1D bisection along central active section for full-width extruded slopes
    if np.all(act_y) and not np.all(act_x):
        y_mid = 0.5 * (ay0 + ay1)
        x_scan = np.linspace(ax0, ax1, 200)
        y_scan = np.full_like(x_scan, y_mid)
        diff_scan = model.ground_z(x_scan, y_scan) - surface.z(x_scan, y_scan)
        pos = np.flatnonzero(np.isfinite(diff_scan) & (diff_scan > 0.0))
        if len(pos) >= 2:
            i_left, i_right = int(pos[0]), int(pos[-1])
            if i_left > 0 and np.isfinite(diff_scan[i_left - 1]):
                lo, hi = float(x_scan[i_left - 1]), float(x_scan[i_left])
                for _ in range(22):
                    mid = 0.5 * (lo + hi)
                    val = float(model.ground_z(mid, y_mid) - surface.z(mid, y_mid))
                    if np.isfinite(val) and val > 0.0:
                        hi = mid
                    else:
                        lo = mid
                ax0 = hi - 1e-7
            if i_right < len(x_scan) - 1 and np.isfinite(diff_scan[i_right + 1]):
                lo, hi = float(x_scan[i_right]), float(x_scan[i_right + 1])
                for _ in range(22):
                    mid = 0.5 * (lo + hi)
                    val = float(model.ground_z(mid, y_mid) - surface.z(mid, y_mid))
                    if np.isfinite(val) and val > 0.0:
                        lo = mid
                    else:
                        hi = mid
                ax1 = lo + 1e-7
    elif np.all(act_x) and not np.all(act_y):
        x_mid = 0.5 * (ax0 + ax1)
        y_scan = np.linspace(ay0, ay1, 200)
        x_scan = np.full_like(y_scan, x_mid)
        diff_scan = model.ground_z(x_scan, y_scan) - surface.z(x_scan, y_scan)
        pos = np.flatnonzero(np.isfinite(diff_scan) & (diff_scan > 0.0))
        if len(pos) >= 2:
            i_left, i_right = int(pos[0]), int(pos[-1])
            if i_left > 0 and np.isfinite(diff_scan[i_left - 1]):
                lo, hi = float(y_scan[i_left - 1]), float(y_scan[i_left])
                for _ in range(22):
                    mid = 0.5 * (lo + hi)
                    val = float(model.ground_z(x_mid, mid) - surface.z(x_mid, mid))
                    if np.isfinite(val) and val > 0.0:
                        hi = mid
                    else:
                        lo = mid
                ay0 = hi - 1e-7
            if i_right < len(y_scan) - 1 and np.isfinite(diff_scan[i_right + 1]):
                lo, hi = float(y_scan[i_right]), float(y_scan[i_right + 1])
                for _ in range(22):
                    mid = 0.5 * (lo + hi)
                    val = float(model.ground_z(x_mid, mid) - surface.z(x_mid, mid))
                    if np.isfinite(val) and val > 0.0:
                        lo = mid
                    else:
                        hi = mid
                ay1 = lo + 1e-7

    if ax1 - ax0 <= 1e-4 or ay1 - ay0 <= 1e-4:
        raise ColumnDiscretizationError("Active slip footprint has negligible horizontal extent.")
    return ax0, ax1, ay0, ay1


def discretize_columns(
    model: SlopeModel3D,
    surface: SlipSurface3D,
    materials: list[Material3D],
    opts: ColumnOptions | None = None,
    material_colors: list[str] | None = None,
) -> ColumnGrid:
    """Discretize ``(model, surface)`` into an active ``ColumnGrid``."""
    opts = opts or ColumnOptions()
    if not materials:
        raise ValueError("At least one Material3D is required.")
    mat_names = [m.name for m in materials]
    colors = material_colors or ["#d9a441"] * len(materials)

    ax0, ax1, ay0, ay1 = _tight_active_bounds(model, surface)
    nx = max(2, int(opts.nx))
    ny = max(1, int(opts.ny))
    dx = (ax1 - ax0) / nx
    dy = (ay1 - ay0) / ny
    xs_1d = ax0 + (np.arange(nx) + 0.5) * dx
    ys_1d = ay0 + (np.arange(ny) + 0.5) * dy

    ix_grid, iy_grid = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    xx, yy = np.meshgrid(xs_1d, ys_1d, indexing="ij")

    zb_grid = surface.z(xx, yy)
    zg_grid = model.ground_z(xx, yy)
    dz_dx_grid, dz_dy_grid = surface.gradients(xx, yy)

    # Reject if slip surface cuts below the bottom of the geological model domain
    valid_zb = np.isfinite(zb_grid) & np.isfinite(dz_dx_grid) & np.isfinite(dz_dy_grid)
    below_domain = valid_zb & (zb_grid < zg_grid) & (zb_grid < model.z_min - 1e-4)
    if np.any(below_domain):
        raise ColumnDiscretizationError("Slip surface extends below model base elevation z_min.")

    h_grid = zg_grid - zb_grid
    center_active = valid_zb & (h_grid > 1e-5)

    # Symmetric 4x4 sub-cell quadrature on partial daylight-rim cells (preserves 2D parity & 90-deg symmetry)
    xs_c = ax0 + np.arange(nx + 1) * dx
    ys_c = ay0 + np.arange(ny + 1) * dy
    xx_c, yy_c = np.meshgrid(xs_c, ys_c, indexing="ij")
    zb_c = surface.z(xx_c, yy_c)
    zg_c = model.ground_z(xx_c, yy_c)
    act_c = np.isfinite(zb_c) & ((zg_c - zb_c) > 1e-5)
    n_act_c = (
        act_c[:-1, :-1].astype(int)
        + act_c[1:, :-1].astype(int)
        + act_c[:-1, 1:].astype(int)
        + act_c[1:, 1:].astype(int)
    )

    frac_grid = np.where((n_act_c == 4) & center_active, 1.0, 0.0)
    rim_cand = ((n_act_c > 0) & (n_act_c < 4)) | (center_active & (n_act_c < 4))
    if np.any(rim_cand):
        sub_1d = (np.arange(4) + 0.5) / 4.0 - 0.5
        sx_off, sy_off = np.meshgrid(sub_1d * dx, sub_1d * dy, indexing="ij")
        rc_x = xx[rim_cand]
        rc_y = yy[rim_cand]
        n_rc = len(rc_x)
        x_sub = (rc_x[:, None, None] + sx_off[None, :, :]).ravel()
        y_sub = (rc_y[:, None, None] + sy_off[None, :, :]).ravel()
        zb_sub = surface.z(x_sub, y_sub).reshape(n_rc, 4, 4)
        zg_sub = model.ground_z(x_sub, y_sub).reshape(n_rc, 4, 4)
        h_sub = zg_sub - zb_sub
        act_sub = np.isfinite(zb_sub) & (h_sub > 1e-5) & (zb_sub >= model.z_min - 1e-3)
        cnt = np.sum(act_sub, axis=(1, 2))
        has_sub = cnt > 0
        if np.any(has_sub):
            rc_indices = np.argwhere(rim_cand)
            for loc_i in np.flatnonzero(has_sub):
                gi, gj = int(rc_indices[loc_i, 0]), int(rc_indices[loc_i, 1])
                c_num = int(cnt[loc_i])
                if c_num == 16 and center_active[gi, gj]:
                    frac_grid[gi, gj] = 1.0
                else:
                    m_loc = act_sub[loc_i]
                    xb_c = float(np.mean(rc_x[loc_i] + sx_off[m_loc]))
                    yb_c = float(np.mean(rc_y[loc_i] + sy_off[m_loc]))
                    zb_pt = float(np.asarray(surface.z(xb_c, yb_c)).reshape(-1)[0])
                    zg_pt = float(np.asarray(model.ground_z(xb_c, yb_c)).reshape(-1)[0])
                    dzx_pt_a, dzy_pt_a = surface.gradients(xb_c, yb_c)
                    dzx_pt = float(np.asarray(dzx_pt_a).reshape(-1)[0])
                    dzy_pt = float(np.asarray(dzy_pt_a).reshape(-1)[0])
                    h_mean = float(np.mean(h_sub[loc_i][m_loc]))
                    if (
                        np.isfinite(zb_pt)
                        and np.isfinite(dzx_pt)
                        and np.isfinite(dzy_pt)
                        and h_mean > 1e-5
                    ):
                        if zb_pt < model.z_min - 1e-4:
                            raise ColumnDiscretizationError(
                                "Slip surface extends below model base elevation z_min."
                            )
                        xx[gi, gj] = xb_c
                        yy[gi, gj] = yb_c
                        zb_grid[gi, gj] = zb_pt
                        zg_grid[gi, gj] = zg_pt
                        dz_dx_grid[gi, gj] = dzx_pt
                        dz_dy_grid[gi, gj] = dzy_pt
                        h_grid[gi, gj] = h_mean
                        frac_grid[gi, gj] = c_num / 16.0

    active_mask = frac_grid > 0.0
    if int(np.count_nonzero(active_mask)) < opts.min_columns:
        raise ColumnDiscretizationError(
            f"Too few active columns ({int(np.count_nonzero(active_mask))} < {opts.min_columns})."
        )

    # Resolve sliding azimuth first so tension-crack uphill truncation knows +s
    if opts.sliding_mode == "right_to_left":
        az_deg = 180.0
    elif opts.sliding_mode == "left_to_right":
        az_deg = 0.0
    elif opts.sliding_mode == "custom":
        az_deg = float(opts.azimuth_deg) % 360.0
    elif isinstance(surface, MultiPlaneWedgeSurface) and len(surface.planes) >= 2:
        az_deg = surface.kinematic_azimuth_deg()
    else:
        # Auto: area/height-weighted downhill basal gradient -sum(h * grad(z_b))
        w_tmp = h_grid[active_mask]
        gx = -float(np.sum(w_tmp * dz_dx_grid[active_mask]))
        gy = -float(np.sum(w_tmp * dz_dy_grid[active_mask]))
        if np.hypot(gx, gy) < 1e-9:
            az_deg = 180.0
        else:
            az_deg = float(np.degrees(np.arctan2(gy, gx))) % 360.0

    psi = float(np.radians(az_deg))
    sx, sy = float(np.cos(psi)), float(np.sin(psi))
    tx, ty = -sy, sx
    s_grid = xx * sx + yy * sy

    # Optional tension crack truncation on the uphill (-s) half of the sliding mass
    crack = opts.loads.tension_crack if opts.loads is not None else None
    crack_force_s = 0.0
    crack_z_act: list[tuple[float, float]] = []
    if crack is not None and crack.depth_m > 0.0:
        s_mid = float(np.mean(s_grid[active_mask]))
        truncated = active_mask & (s_grid < s_mid) & (h_grid <= crack.depth_m)
        active_mask = active_mask & (~truncated)
        if int(np.count_nonzero(active_mask)) < opts.min_columns:
            raise ColumnDiscretizationError("Tension crack truncated nearly the entire slip mass.")
        if crack.water_fill_fraction > 0.0:
            hw = crack.water_fill_fraction * crack.depth_m
            # Projected width transverse to sliding along exposed rear columns
            # Identify active columns whose uphill neighbour in X or Y was truncated
            step_ix = -1 if sx >= 0 else 1
            step_iy = -1 if sy >= 0 else 1
            for i_c in range(nx):
                for j_c in range(ny):
                    if not active_mask[i_c, j_c]:
                        continue
                    nb_x = i_c + step_ix
                    nb_y = j_c + step_iy
                    exp_x = abs(sx) > 1e-6 and (0 <= nb_x < nx) and truncated[nb_x, j_c]
                    exp_y = abs(sy) > 1e-6 and (0 <= nb_y < ny) and truncated[i_c, nb_y]
                    if exp_x or exp_y:
                        w_perp = (dy * abs(sx) if exp_x else 0.0) + (dx * abs(sy) if exp_y else 0.0)
                        f_c = 0.5 * opts.gamma_w * hw * hw * w_perp
                        crack_force_s += f_c
                        z_act = float(zg_grid[i_c, j_c] - crack.depth_m + hw / 3.0)
                        crack_z_act.append((f_c, z_act))

    if float(np.max(h_grid[active_mask])) < opts.min_depth:
        raise ColumnDiscretizationError(
            f"Maximum slip depth {float(np.max(h_grid[active_mask])):.3f} m < min_depth {opts.min_depth:.3f} m."
        )

    # Extract active 1D arrays
    ix = ix_grid[active_mask]
    iy = iy_grid[active_mask]
    x = xx[active_mask]
    y = yy[active_mask]
    zb = zb_grid[active_mask]
    zt = zg_grid[active_mask]
    h = h_grid[active_mask]
    dz_dx = dz_dx_grid[active_mask]
    dz_dy = dz_dy_grid[active_mask]
    n_cols = len(x)
    dA = frac_grid[active_mask] * (dx * dy)

    # Sliding-frame coordinates & inclinations
    s = x * sx + y * sy
    t = x * tx + y * ty
    tan_as = -(dz_dx * sx + dz_dy * sy)
    tan_at = -(dz_dx * tx + dz_dy * ty)
    alpha_s = np.arctan(tan_as)
    sin_as = np.sin(alpha_s)
    cos_as = np.cos(alpha_s)
    jac = np.sqrt(1.0 + tan_as * tan_as + tan_at * tan_at)
    area_base = dA * jac

    ns = tan_as / jac
    nt = tan_at / jac
    nz = 1.0 / jac
    normal_xyz = np.column_stack([-dz_dx / jac, -dz_dy / jac, nz])

    # Rotation axis (sc, zc) in (s, z) plane
    rot = surface.rotation_center_sz(az_deg)
    if rot is not None:
        sc, zc = rot
    else:
        sc = float(0.5 * (np.min(s) + np.max(s)))
        zc = float(np.max(zt) + 0.6 * max(1.0, float(np.max(s) - np.min(s))))

    # Vertical sub-sampling for exact layered/extruded unit weights & B-bar overburden
    n_sub = max(4, int(opts.n_sub))
    fracs = (np.arange(n_sub) + 0.5) / n_sub
    z_samples = zb[:, None] + h[:, None] * fracs[None, :]  # (n_cols, n_sub)
    mat_sub = model.material_indices_columns(x, y, z_samples, mat_names)

    # Basal material index (just inside the column above z_base)
    mat_idx = model.material_indices_at(x, y, zb + np.minimum(1e-4, 0.02 * h), mat_names)

    z_wt = model.water_z(x, y)
    gamma_dry = np.array([m.gamma for m in materials], dtype=float)
    gamma_sat = np.array([m.gamma_sat for m in materials], dtype=float)
    excess_flags = np.array([m.excess_load for m in materials], dtype=bool)
    use_water = np.array([m.use_water for m in materials], dtype=bool)
    hu_arr = np.array([m.hu for m in materials], dtype=float)
    bbar_arr = np.array([m.bbar for m in materials], dtype=float)
    total_stress_arr = np.array([m.total_stress for m in materials], dtype=bool)

    if z_wt is not None and opts.pore_mode == "water_table":
        submerged = z_samples <= z_wt[:, None]
        gam_samples = np.where(submerged, gamma_sat[mat_sub], gamma_dry[mat_sub])
    else:
        gam_samples = gamma_dry[mat_sub]

    dh = h / n_sub
    sigma_v_soil = np.sum(gam_samples, axis=1) * dh  # [kPa]
    W_soil = sigma_v_soil * dA  # [kN]

    # Column centre of gravity z_cg
    z_cg = np.sum(gam_samples * z_samples, axis=1) * dh / np.maximum(sigma_v_soil, 1e-9)

    # Ponded water above ground surface (when water_table mode & z_wt > z_top)
    W_pond = np.zeros(n_cols, dtype=float)
    if z_wt is not None and opts.pore_mode == "water_table":
        h_pond = np.maximum(z_wt - zt, 0.0)
        W_pond = opts.gamma_w * h_pond * dA

    W = W_soil + W_pond

    # Excess vertical stress from overlying excess_load materials (for B-bar)
    if np.any(excess_flags):
        is_exc = excess_flags[mat_sub]
        if z_wt is not None and opts.pore_mode == "water_table":
            gam_eff_exc = np.where(
                z_samples <= z_wt[:, None], gam_samples - opts.gamma_w, gam_samples
            )
        else:
            gam_eff_exc = gam_samples
        dsig_exc = np.sum(np.where(is_exc, np.maximum(gam_eff_exc, 0.0), 0.0), axis=1) * dh
    else:
        dsig_exc = np.zeros(n_cols, dtype=float)

    # External patch loads & point loads
    Qs_a = np.zeros(n_cols, dtype=float)
    Qt_a = np.zeros(n_cols, dtype=float)
    Qz_a = np.zeros(n_cols, dtype=float)
    Ms_a = np.zeros(n_cols, dtype=float)
    Qs_p = np.zeros(n_cols, dtype=float)
    Qt_p = np.zeros(n_cols, dtype=float)
    Qz_p = np.zeros(n_cols, dtype=float)
    Ms_p = np.zeros(n_cols, dtype=float)
    Fx_a = np.zeros(n_cols, dtype=float)
    Fy_a = np.zeros(n_cols, dtype=float)
    Fx_p = np.zeros(n_cols, dtype=float)
    Fy_p = np.zeros(n_cols, dtype=float)
    surcharge_qz_down = np.zeros(n_cols, dtype=float)

    if opts.loads is not None:
        for pl in opts.loads.patch_loads:
            q_val = pl.pressure_at(x, y)
            if not np.any(q_val != 0.0):
                continue
            uvec = pl.unit_direction_xyz()
            f_mag = q_val * dA  # [kN]
            fx_pl = f_mag * uvec[0]
            fy_pl = f_mag * uvec[1]
            fs = fx_pl * sx + fy_pl * sy
            ft = fx_pl * tx + fy_pl * ty
            fz = f_mag * uvec[2]  # negative for downward load
            Fx_a += fx_pl
            Fy_a += fy_pl
            Qs_a += fs
            Qt_a += ft
            Qz_a += fz
            # CCW driving torque about (sc, zc) applied at ground surface (s_i, z_top_i):
            # tau_ccw = (s_i - s_c) * F_z - (z_app - z_c) * F_s
            Ms_a += (s - sc) * fz - (zt - zc) * fs
            surcharge_qz_down += np.maximum(-fz / dA, 0.0)

        for ptl in opts.loads.point_loads:
            # Assign point load to nearest active column within 1.5 cell widths, or closest column
            dist2 = (x - ptl.x) ** 2 + (y - ptl.y) ** 2
            k_near = int(np.argmin(dist2))
            if dist2[k_near] <= (max(dx, dy) * 2.0) ** 2 or n_cols > 0:
                fxyz = ptl.force_xyz()
                fs_pt = float(fxyz[0] * sx + fxyz[1] * sy)
                ft_pt = float(fxyz[0] * tx + fxyz[1] * ty)
                fz_pt = float(fxyz[2])
                sp_pt = ptl.x * sx + ptl.y * sy
                Fx_a[k_near] += float(fxyz[0])
                Fy_a[k_near] += float(fxyz[1])
                Qs_a[k_near] += fs_pt
                Qt_a[k_near] += ft_pt
                Qz_a[k_near] += fz_pt
                Ms_a[k_near] += (sp_pt - sc) * fz_pt - (ptl.z - zc) * fs_pt

    # Basal pore-water pressure u [kPa]
    u_base = np.zeros(n_cols, dtype=float)
    if opts.pore_pressure_fn is not None and opts.pore_mode in (
        "seepage",
        "seepage_grid",
        "water_table",
    ):
        u_eval = np.maximum(np.asarray(opts.pore_pressure_fn(x, y, zb), dtype=float), 0.0)
        u_base = np.where(use_water[mat_idx], hu_arr[mat_idx] * u_eval, 0.0)
    elif opts.pore_mode == "water_table" and z_wt is not None:
        head = np.maximum(z_wt - zb, 0.0)
        u_base = np.where(use_water[mat_idx], hu_arr[mat_idx] * opts.gamma_w * head, 0.0)
    elif opts.pore_mode == "ru":
        sigma_pre = np.maximum(sigma_v_soil - dsig_exc, 0.0)
        u_base = np.where(use_water[mat_idx], opts.ru * sigma_pre, 0.0)

    u_excess = bbar_arr[mat_idx] * (dsig_exc + surcharge_qz_down)
    u_raw = u_base + u_excess
    sigma_v_eff = np.maximum(W / dA + surcharge_qz_down - u_raw, 0.0)
    u = np.where(total_stress_arr[mat_idx], 0.0, u_raw)

    # Tension crack water driving moment about (sc, zc)
    crack_moment = 0.0
    for f_c, z_act in crack_z_act:
        crack_moment += (zc - z_act) * f_c

    # 3D discrete reinforcement intersections (bolts, tiebacks, nails, micropiles, geotextiles)
    support_hits: list[SupportHit3D] = []
    if opts.loads is not None and opts.loads.bolts:
        for bolt in opts.loads.bolts:
            hit = bolt.intersect_surface(surface)
            if hit is None:
                continue
            hit_xyz, f_xyz_raw, cap_kn = hit
            dist2 = (x - hit_xyz[0]) ** 2 + (y - hit_xyz[1]) ** 2
            k_col = int(np.argmin(dist2))
            if bolt.support_type.kind == "micropile":
                # Mobilized shear opposes +s translation along column base tangent (-cos_as * s_hat + sin_as * z_hat)
                ca = float(cos_as[k_col])
                sa = float(sin_as[k_col])
                f_xyz = cap_kn * np.array([-ca * sx, -ca * sy, sa], dtype=float)
            else:
                f_xyz = f_xyz_raw

            fs_b = float(f_xyz[0] * sx + fxyz_s_helper(f_xyz, sy))
            ft_b = float(f_xyz[0] * tx + f_xyz[1] * ty)
            fz_b = float(f_xyz[2])
            s_hit = float(hit_xyz[0] * sx + hit_xyz[1] * sy)
            z_hit = float(hit_xyz[2])
            mom_b = (s_hit - sc) * fz_b - (z_hit - zc) * fs_b

            if bolt.support_type.application == "passive":
                Fx_p[k_col] += float(f_xyz[0])
                Fy_p[k_col] += float(f_xyz[1])
                Qs_p[k_col] += fs_b
                Qt_p[k_col] += ft_b
                Qz_p[k_col] += fz_b
                Ms_p[k_col] += mom_b
            else:
                Fx_a[k_col] += float(f_xyz[0])
                Fy_a[k_col] += float(f_xyz[1])
                Qs_a[k_col] += fs_b
                Qt_a[k_col] += ft_b
                Qz_a[k_col] += fz_b
                Ms_a[k_col] += mom_b

            support_hits.append(
                SupportHit3D(
                    label=bolt.label,
                    hit_xyz=hit_xyz,
                    force_xyz_kn=f_xyz,
                    capacity_kn=cap_kn,
                    application=bolt.support_type.application,
                    kind=bolt.support_type.kind,
                    col_idx=k_col,
                )
            )

    return ColumnGrid(
        ix=ix,
        iy=iy,
        nx_grid=nx,
        ny_grid=ny,
        dx=dx,
        dy=dy,
        x=x,
        y=y,
        z_base=zb,
        z_top=zt,
        z_cg=z_cg,
        h=h,
        dA=dA,
        area_base=area_base,
        azimuth_deg=az_deg,
        s=s,
        t=t,
        sc=sc,
        zc=zc,
        dz_dx=dz_dx,
        dz_dy=dz_dy,
        tan_alpha_s=tan_as,
        tan_alpha_t=tan_at,
        alpha_s=alpha_s,
        sin_alpha_s=sin_as,
        cos_alpha_s=cos_as,
        jacobian=jac,
        ns=ns,
        nt=nt,
        nz=nz,
        normal_xyz=normal_xyz,
        W=W,
        u=u,
        u_excess=u_excess,
        sigma_v_eff=sigma_v_eff,
        mat_idx=mat_idx,
        materials=materials,
        material_colors=colors,
        surface=surface,
        kh=opts.kh,
        kv=opts.kv,
        Qs_active=Qs_a,
        Qt_active=Qt_a,
        Qz_active=Qz_a,
        Ms_active=Ms_a,
        Qs_passive=Qs_p,
        Qt_passive=Qt_p,
        Qz_passive=Qz_p,
        Ms_passive=Ms_p,
        Fx_active=Fx_a,
        Fy_active=Fy_a,
        Fx_passive=Fx_p,
        Fy_passive=Fy_p,
        crack_water_force_s=crack_force_s,
        crack_water_moment=crack_moment,
        support_hits=support_hits,
    )


def fxyz_s_helper(f_xyz: np.ndarray, sy: float) -> float:
    return float(f_xyz[1] * sy)
