"""Global 3D critical slip surface search (Parallel Cuckoo Search + Nelder-Mead + Multi-Scale SAO).

Supports searching over:
* ``"sphere"``       : 4-parameter spherical slip surfaces ``(xc, yc, zc, r)``
* ``"ellipsoid"``    : 6/7-parameter tri-axial ellipsoids ``(xc, yc, zc, rx, ry, rz[, azimuth])``
* ``"spline_patch"`` : Non-ellipsoidal 3D spline bowls perturbing the critical ellipsoid
* Multi-scale **Surface Altering Optimization (SAO)**: ``3x3 -> 5x5`` Laplacian-regularized
  local ``SplinePatchSurface`` deformation refining any critical sphere or ellipsoid.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

from slipsurface3d.geometry.surfaces import (
    EllipsoidSurface,
    SlipSurface3D,
    SplinePatchSurface,
    sphere_surface,
)
from slipsurface3d.geometry.terrain import SlopeModel3D
from slipsurface3d.lem.columns import (
    ColumnDiscretizationError,
    ColumnGrid,
    ColumnOptions,
    clip_to_weak_layers,
    discretize_columns,
)
from slipsurface3d.lem.solvers import M_ALPHA_MIN_DEFAULT, LEMResult3D, solve_surface_3d
from slipsurface3d.materials.models import Material3D

EFFORT_PRESETS: dict[str, tuple[int, int, int]] = {
    # (n_nests, n_generations, nm_maxfev)
    "fast": (12, 12, 35),
    "normal": (18, 20, 60),
    "fine": (26, 32, 95),
}

PENALTY_FOS = 1.0e6

SEARCH_ALGORITHMS: dict[str, str] = {
    "cuckoo": "Cuckoo Search",
    "pso": "Particle Swarm",
}
# Clerc & Kennedy (2002) constriction coefficients
PSO_INERTIA = 0.7298
PSO_ACCEL = 1.4962

# Local polish: Nelder-Mead starts, restarts from the incumbent, and simplex size
# (fraction of each parameter's search range).
NM_STARTS = 3
NM_RESTARTS = 2
NM_STEP = 0.06


@dataclass
class SearchSettings3D:
    surface_kind: str = "ellipsoid"  # sphere | ellipsoid | spline_patch | user
    effort: str = "normal"  # fast | normal | fine
    primary_method: str = "bishop"
    algorithm: str = "cuckoo"  # cuckoo | pso (global stage; both end with Nelder-Mead polish)
    optimize_azimuth: bool = False
    sao_enabled: bool = False
    n_workers: int = 1  # 1 = serial, 0 = auto multi-core, >=2 = explicit worker threads
    gle_interslice: str = "half-sine"
    check_admissibility: bool = True
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT
    seed: int = 42
    # Plan box (x_min, x_max, y_min, y_max) every sliding column must lie inside.
    search_limits: tuple[float, float, float, float] | None = None
    # Number of distinct critical surfaces to report (1 = global minimum only).
    n_local_minima: int = 1
    y_center_range: tuple[float, float] | None = None
    ry_bounds: tuple[float, float] | None = None


@dataclass
class LocalMinimum3D:
    """One of several distinct critical surfaces found by the multi-modal search."""

    surface: SlipSurface3D  # as analysed (clipped to weak layers when the model has any)
    fos: float
    volume: float
    azimuth_deg: float
    centroid_xy: tuple[float, float]
    source_surface: SlipSurface3D | None = None  # the search surface before clipping


@dataclass
class SearchResult3D:
    critical_surface: SlipSurface3D
    critical_columns: ColumnGrid
    critical_result: LEMResult3D
    trials: list[tuple[SlipSurface3D, float]] = field(default_factory=list)
    local_minima: list[LocalMinimum3D] = field(default_factory=list)
    evaluations: int = 0
    cancelled: bool = False


def _resolve_workers(n_workers: int) -> int:
    if n_workers == 0:
        return max(1, min(os.cpu_count() or 2, 6))
    return max(1, int(n_workers))


def _parameter_bounds(
    model: SlopeModel3D,
    settings: SearchSettings3D,
    default_az: float,
) -> tuple[np.ndarray, np.ndarray]:
    ym = 0.5 * (model.y_min + model.y_max)
    xs = np.linspace(model.x_min, model.x_max, 81)
    zs = model.ground_z(xs, np.full_like(xs, ym))
    z_low = float(np.min(zs))
    z_high = float(np.max(zs))
    h_slope = max(2.0, z_high - z_low)
    span_x = model.span_x
    span_y = model.span_y

    xc_lo = model.x_min + 0.10 * span_x
    xc_hi = model.x_max - 0.10 * span_x
    if settings.y_center_range is not None:
        yc_lo, yc_hi = settings.y_center_range
    else:
        yc_lo = model.y_min + 0.20 * span_y
        yc_hi = model.y_max - 0.20 * span_y
        if yc_hi <= yc_lo:
            yc_lo = yc_hi = 0.5 * (model.y_min + model.y_max)
    if settings.search_limits is not None:
        # The slip mass lies below the centre, so the centre must sit inside the limits.
        lx0, lx1, ly0, ly1 = settings.search_limits
        xc_lo, xc_hi = max(xc_lo, lx0), min(xc_hi, lx1)
        if settings.y_center_range is None:
            yc_lo, yc_hi = max(yc_lo, ly0), min(yc_hi, ly1)
            if yc_hi < yc_lo:
                # Limits lie in the edge band the default range avoids: use their centre.
                yc_lo = yc_hi = float(np.clip(0.5 * (ly0 + ly1), model.y_min, model.y_max))
        if xc_hi <= xc_lo:
            raise ValueError("Search limits leave no room for a slip surface inside the model.")

    zc_lo = z_low + 0.4 * h_slope
    zc_hi = z_high + 2.8 * h_slope
    r_lo = max(1.5, 0.45 * h_slope)
    r_hi = max(r_lo + 2.0, min(0.85 * span_x, 4.2 * h_slope))

    if settings.surface_kind == "sphere":
        lo = np.array([xc_lo, yc_lo, zc_lo, r_lo], dtype=float)
        hi = np.array([xc_hi, yc_hi, zc_hi, r_hi], dtype=float)
        return lo, hi

    if settings.ry_bounds is not None:
        ry_lo, ry_hi = settings.ry_bounds
    else:
        ry_lo = max(1.5, min(0.25 * span_y, 0.5 * h_slope))
        ry_hi = max(ry_lo + 1.0, 1.5 * span_y)

    rz_lo = max(1.5, 0.40 * h_slope)
    rz_hi = max(rz_lo + 2.0, 4.0 * h_slope)

    if settings.optimize_azimuth:
        lo = np.array([xc_lo, yc_lo, zc_lo, r_lo, ry_lo, rz_lo, default_az - 45.0], dtype=float)
        hi = np.array([xc_hi, yc_hi, zc_hi, r_hi, ry_hi, rz_hi, default_az + 45.0], dtype=float)
    else:
        lo = np.array([xc_lo, yc_lo, zc_lo, r_lo, ry_lo, rz_lo], dtype=float)
        hi = np.array([xc_hi, yc_hi, zc_hi, r_hi, ry_hi, rz_hi], dtype=float)
    return lo, hi


def _decode_surface(
    vec: np.ndarray, kind: str, default_az: float, optimize_az: bool
) -> EllipsoidSurface:
    if kind == "sphere":
        return sphere_surface(float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3]))
    az = float(vec[6]) if optimize_az and len(vec) >= 7 else default_az
    return EllipsoidSurface(
        xc=float(vec[0]),
        yc=float(vec[1]),
        zc=float(vec[2]),
        rx=float(vec[3]),
        ry=float(vec[4]),
        rz=float(vec[5]),
        azimuth_deg=az,
    )


def _within_limits(cols: ColumnGrid, limits: tuple[float, float, float, float] | None) -> bool:
    """True when every active column centre lies inside the plan search limits."""
    if limits is None:
        return True
    x0, x1, y0, y1 = limits
    return bool(
        float(np.min(cols.x)) >= x0
        and float(np.max(cols.x)) <= x1
        and float(np.min(cols.y)) >= y0
        and float(np.max(cols.y)) <= y1
    )


def _plan_footprint(
    surface: SlipSurface3D, xx: np.ndarray, yy: np.ndarray, zg: np.ndarray, z_min: float
) -> np.ndarray:
    """Boolean plan mask of where the slip surface lies below ground."""
    zb = surface.sample_base(xx, yy, zg, z_min).z
    return np.isfinite(zb) & ((zg - zb) > 1e-3)


def _footprint_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap of two plan masks as a fraction of the smaller one (1 = one inside the other)."""
    smaller = min(int(np.count_nonzero(a)), int(np.count_nonzero(b)))
    if smaller == 0:
        return 0.0
    return float(np.count_nonzero(a & b)) / smaller


def _encode_ellipsoid(surface: EllipsoidSurface, base_kind: str, optimize_az: bool) -> np.ndarray:
    if base_kind == "sphere":
        return np.array([surface.xc, surface.yc, surface.zc, surface.rx], dtype=float)
    vec = [surface.xc, surface.yc, surface.zc, surface.rx, surface.ry, surface.rz]
    if optimize_az:
        vec.append(surface.azimuth_deg)
    return np.array(vec, dtype=float)


def _polish_mode(
    vec0: np.ndarray,
    seed_mask: np.ndarray,
    kept_masks: list[np.ndarray],
    footprint: Callable[[SlipSurface3D], np.ndarray],
    evaluate: Callable[[np.ndarray], tuple[SlipSurface3D | None, float]],
    bounds: tuple[np.ndarray, np.ndarray],
    nm_maxfev: int,
    max_overlap: float,
    cancel_check: Callable[[], bool] | None,
) -> tuple[SlipSurface3D | None, float, int]:
    """Short Nelder-Mead refinement of one failure mode, confined to its own footprint."""
    lo, hi = bounds
    span = np.maximum(hi - lo, 1e-6)
    best_surface: SlipSurface3D | None = None
    best_fos = PENALTY_FOS
    evals = 0

    def objective(u_norm: np.ndarray) -> float:
        nonlocal best_surface, best_fos, evals
        if cancel_check is not None and cancel_check():
            return PENALTY_FOS
        if np.any(u_norm < 0.0) or np.any(u_norm > 1.0):
            return PENALTY_FOS
        evals += 1
        cand, fos = evaluate(lo + u_norm * span)
        if cand is None or fos >= PENALTY_FOS:
            return PENALTY_FOS
        # Stay in this mode: mostly inside the seed footprint, clear of the kept ones.
        cand_mask = footprint(cand)
        if _footprint_overlap(cand_mask, seed_mask) < 0.6:
            return PENALTY_FOS
        if any(_footprint_overlap(cand_mask, other) >= max_overlap for other in kept_masks):
            return PENALTY_FOS
        if fos < best_fos:
            best_fos = fos
            best_surface = cand
        return fos

    minimize(
        objective,
        np.clip((vec0 - lo) / span, 0.0, 1.0),
        method="Nelder-Mead",
        options={"maxfev": nm_maxfev, "xatol": 2e-3, "fatol": 1e-3},
    )
    return best_surface, best_fos, evals


def _find_local_minima(
    *,
    fast_model: SlopeModel3D,
    trials: list[tuple[SlipSurface3D, float]],
    critical: tuple[SlipSurface3D, ColumnGrid, float],
    n_wanted: int,
    evaluate: Callable[[np.ndarray], tuple[SlipSurface3D | None, float]],
    finalize: Callable[[SlipSurface3D], tuple[ColumnGrid, float] | None],
    bounds: tuple[np.ndarray, np.ndarray],
    optimize_azimuth: bool,
    base_kind: str,
    nm_maxfev: int,
    cancel_check: Callable[[], bool] | None,
    max_overlap: float = 0.5,
    max_candidates: int = 600,
) -> tuple[list[LocalMinimum3D], int]:
    """Pick distinct low-FoS failure modes from the search history and polish each one.

    Trial surfaces are visited from lowest FoS upward. One is a new mode when its plan
    footprint overlaps every mode already kept by less than ``max_overlap`` of the smaller
    footprint. Each new mode is then refined by a short Nelder-Mead run that is confined to
    its own footprint so it cannot slide back into the global minimum. The refinement runs
    on the coarse screening grid; the reported FoS is a final full-resolution solution.

    The global search concentrates its samples around the global minimum, so secondary
    modes are explored less densely: their FoS is an upper bound on the local minimum.
    """
    xs = np.linspace(fast_model.x_min, fast_model.x_max, 56)
    ys = np.linspace(fast_model.y_min, fast_model.y_max, 56)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zg = fast_model.ground_z(xx, yy)

    def footprint(surface: SlipSurface3D) -> np.ndarray:
        return _plan_footprint(surface, xx, yy, zg, fast_model.z_min)

    def describe(surface: SlipSurface3D, cols: ColumnGrid, fos: float) -> LocalMinimum3D:
        weights = cols.h * cols.dA
        total = float(np.sum(weights))
        return LocalMinimum3D(
            surface=cols.surface,
            source_surface=surface,
            fos=fos,
            volume=cols.total_volume,
            azimuth_deg=cols.azimuth_deg,
            centroid_xy=(
                float(np.sum(weights * cols.x) / total),
                float(np.sum(weights * cols.y) / total),
            ),
        )

    crit_surface, crit_cols, crit_fos = critical
    kept_masks = [footprint(crit_surface)]
    minima = [describe(crit_surface, crit_cols, crit_fos)]
    evals = 0

    ellipsoids = [(s, f) for s, f in trials if isinstance(s, EllipsoidSurface) and np.isfinite(f)]
    ellipsoids.sort(key=lambda item: item[1])
    for surf, _screen_fos in ellipsoids[:max_candidates]:
        if len(minima) >= n_wanted or (cancel_check is not None and cancel_check()):
            break
        mask = footprint(surf)
        if not np.any(mask):
            continue
        if any(_footprint_overlap(mask, other) >= max_overlap for other in kept_masks):
            continue
        polished, _fos, n_ev = _polish_mode(
            _encode_ellipsoid(surf, base_kind, optimize_azimuth),
            mask,
            kept_masks,
            footprint,
            evaluate,
            bounds,
            nm_maxfev,
            max_overlap,
            cancel_check,
        )
        evals += n_ev
        if polished is None:
            continue
        final = finalize(polished)
        evals += 1
        if final is None:
            continue
        kept_masks.append(footprint(polished))
        minima.append(describe(polished, final[0], final[1]))

    return minima, evals


def _initial_simplex(u0: np.ndarray, step: float) -> np.ndarray:
    """Nelder-Mead simplex around ``u0`` in the unit box, stepping inward near the walls."""
    dim = len(u0)
    simplex = np.tile(u0, (dim + 1, 1))
    for k in range(dim):
        simplex[k + 1, k] = u0[k] + step if u0[k] + step <= 1.0 else u0[k] - step
    return simplex


def _levy_step(rng: np.random.Generator, dim: int, beta: float = 1.5) -> np.ndarray:
    from math import gamma, pi, sin

    num = gamma(1.0 + beta) * sin(pi * beta / 2.0)
    den = gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (num / den) ** (1.0 / beta)
    u = rng.normal(0.0, sigma_u, size=dim)
    v = rng.normal(0.0, 1.0, size=dim)
    return u / (np.abs(v) ** (1.0 / beta) + 1e-12)


def _laplacian_penalty(ctrl: np.ndarray, rz: float) -> float:
    """Discrete second-difference roughness regularization on a (nu, nv) control grid."""
    pen = 0.0
    if ctrl.shape[0] >= 3:
        d2u = ctrl[:-2, :] - 2.0 * ctrl[1:-1, :] + ctrl[2:, :]
        pen += float(np.mean((d2u / max(rz, 1.0)) ** 2))
    if ctrl.shape[1] >= 3:
        d2v = ctrl[:, :-2] - 2.0 * ctrl[:, 1:-1] + ctrl[:, 2:]
        pen += float(np.mean((d2v / max(rz, 1.0)) ** 2))
    return 2.5e-3 * pen


def run_sao_refinement(
    base_ellipsoid: EllipsoidSurface,
    base_fos: float,
    model: SlopeModel3D,
    materials: list[Material3D],
    col_opts: ColumnOptions,
    method: str = "bishop",
    gle_interslice: str = "half-sine",
    solver_tolerance: float = 1e-4,
    solver_max_iter: int = 60,
    check_admissibility: bool = True,
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT,
    material_colors: list[str] | None = None,
    max_fev_coarse: int = 45,
    max_fev_fine: int = 55,
    cancel_check: Callable[[], bool] | None = None,
    best_surface_cb: Callable[[SlipSurface3D, float], None] | None = None,
) -> tuple[SlipSurface3D, float, list[tuple[SlipSurface3D, float]], int]:
    """Multi-scale Surface Altering Optimization (SAO): 3x3 coarse -> 5x5 fine spline bowl."""
    max_amp = 0.25 * base_ellipsoid.rz
    sao_trials: list[tuple[SlipSurface3D, float]] = []
    evals = 0
    best_surf: SlipSurface3D = base_ellipsoid
    best_f = float(base_fos)

    def eval_ctrl(ctrl_2d: np.ndarray, reg: bool = True) -> float:
        nonlocal evals, best_surf, best_f
        if cancel_check is not None and cancel_check():
            return PENALTY_FOS
        if np.any(np.abs(ctrl_2d) > max_amp):
            return PENALTY_FOS
        evals += 1
        sp_surf = SplinePatchSurface(base=base_ellipsoid, control_dz=ctrl_2d)
        try:
            cols_s = discretize_columns(
                model, sp_surf, materials, opts=col_opts, material_colors=material_colors
            )
            res_s = solve_surface_3d(
                cols_s,
                method=method,
                tolerance=solver_tolerance,
                max_iter=solver_max_iter,
                m_alpha_min=m_alpha_min,
                interslice=gle_interslice,
            )
        except ColumnDiscretizationError:
            return PENALTY_FOS
        if not res_s.ok or (check_admissibility and not res_s.admissible):
            return PENALTY_FOS
        raw_fos = float(res_s.fos)
        sao_trials.append((sp_surf, raw_fos))
        if raw_fos < best_f - 1e-6:
            best_f = raw_fos
            best_surf = sp_surf
            if best_surface_cb is not None:
                best_surface_cb(best_surf, best_f)
        return raw_fos + (_laplacian_penalty(ctrl_2d, base_ellipsoid.rz) if reg else 0.0)

    # Stage 1: Coarse 3x3 control grid (9 DOFs)
    z0_3 = np.zeros(9, dtype=float)
    step_3 = 0.05 * base_ellipsoid.rz
    simplex_3 = np.vstack([z0_3, z0_3 + np.eye(9) * step_3])
    nm_3 = minimize(
        lambda v: eval_ctrl(np.asarray(v, dtype=float).reshape(3, 3), reg=True),
        z0_3,
        method="Nelder-Mead",
        options={"initial_simplex": simplex_3, "maxfev": max(20, max_fev_coarse), "fatol": 4e-4},
    )
    ctrl_3 = (
        np.clip(np.asarray(nm_3.x, dtype=float).reshape(3, 3), -max_amp, max_amp)
        if np.isfinite(nm_3.fun) and nm_3.fun < PENALTY_FOS
        else np.zeros((3, 3), dtype=float)
    )

    if cancel_check is not None and cancel_check():
        return best_surf, best_f, sao_trials, evals

    # Stage 2: Upsample 3x3 -> 5x5 and refine interior 3x3 sub-block (9 localized DOFs on 5x5 patch)
    u3 = np.linspace(-1.0, 1.0, 3)
    u5 = np.linspace(-1.0, 1.0, 5)
    up_sp = RectBivariateSpline(u3, u3, ctrl_3, kx=2, ky=2)
    ctrl_5_base = np.clip(up_sp(u5, u5), -max_amp, max_amp)
    eval_ctrl(ctrl_5_base, reg=False)

    inner_0 = ctrl_5_base[1:4, 1:4].ravel().copy()
    step_5 = 0.03 * base_ellipsoid.rz
    simplex_5 = np.vstack([inner_0, inner_0 + np.eye(9) * step_5])

    def obj_5(inner_vec: np.ndarray) -> float:
        c5 = ctrl_5_base.copy()
        c5[1:4, 1:4] = np.asarray(inner_vec, dtype=float).reshape(3, 3)
        return eval_ctrl(c5, reg=True)

    minimize(
        obj_5,
        inner_0,
        method="Nelder-Mead",
        options={"initial_simplex": simplex_5, "maxfev": max(20, max_fev_fine), "fatol": 3e-4},
    )
    return best_surf, best_f, sao_trials, evals


def search_critical_surface_3d(
    model: SlopeModel3D,
    materials: list[Material3D],
    col_opts: ColumnOptions,
    settings: SearchSettings3D | None = None,
    solver_tolerance: float = 1e-4,
    solver_max_iter: int = 60,
    material_colors: list[str] | None = None,
    progress_cb: Callable[[int, int, float], None] | None = None,
    best_surface_cb: Callable[[SlipSurface3D, float], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> SearchResult3D:
    """Locate the global minimum-FoS 3D slip surface via Parallel Cuckoo Search + NM + SAO."""
    settings = settings or SearchSettings3D()
    n_nests, n_gen, nm_fev = EFFORT_PRESETS.get(settings.effort, EFFORT_PRESETS["normal"])
    rng = np.random.default_rng(settings.seed)
    workers = _resolve_workers(settings.n_workers)

    # Pre-bake scattered RBF horizons onto a dense regular grid for fast exploration
    fast_model = model.bake_horizon_grids(nx=141, ny=141)

    # Determine natural downhill azimuth
    ym = 0.5 * (fast_model.y_min + fast_model.y_max)
    xs_p = np.linspace(fast_model.x_min, fast_model.x_max, 61)
    zs_p = fast_model.ground_z(xs_p, np.full_like(xs_p, ym))
    if col_opts.sliding_mode == "right_to_left":
        default_az = 180.0
    elif col_opts.sliding_mode == "left_to_right":
        default_az = 0.0
    elif col_opts.sliding_mode == "custom":
        default_az = col_opts.azimuth_deg
    else:
        default_az = 180.0 if int(np.argmax(zs_p)) >= int(np.argmin(zs_p)) else 0.0

    base_kind = "sphere" if settings.surface_kind == "sphere" else "ellipsoid"
    lo, hi = _parameter_bounds(fast_model, settings, default_az)
    span = np.maximum(hi - lo, 1e-6)
    dim = len(lo)

    screen_opts = ColumnOptions(
        nx=max(12, min(col_opts.nx, 18)),
        ny=max(10, min(col_opts.ny, 14)),
        n_sub=max(8, min(col_opts.n_sub, 12)),
        gamma_w=col_opts.gamma_w,
        pore_mode=col_opts.pore_mode,
        ru=col_opts.ru,
        kh=col_opts.kh,
        kv=col_opts.kv,
        sliding_mode=col_opts.sliding_mode,
        azimuth_deg=col_opts.azimuth_deg,
        min_depth=max(col_opts.min_depth, 0.25),
        min_columns=col_opts.min_columns,
        loads=col_opts.loads,
        pore_pressure_fn=col_opts.pore_pressure_fn,
    )

    trials: list[tuple[SlipSurface3D, float]] = []
    eval_count = 0

    def _eval_pure(vec: np.ndarray, fine: bool = False) -> tuple[SlipSurface3D | None, float]:
        if np.any(vec < lo) or np.any(vec > hi):
            return None, PENALTY_FOS
        surf = _decode_surface(vec, base_kind, default_az, settings.optimize_azimuth)
        opts_use = col_opts if fine else screen_opts
        mdl_use = model if fine else fast_model
        try:
            cols = discretize_columns(
                mdl_use, surf, materials, opts=opts_use, material_colors=material_colors
            )
            res = solve_surface_3d(
                cols,
                method=settings.primary_method,
                tolerance=solver_tolerance if fine else max(solver_tolerance, 5e-4),
                max_iter=solver_max_iter,
                m_alpha_min=settings.m_alpha_min,
                interslice=settings.gle_interslice,
            )
        except ColumnDiscretizationError:
            return None, PENALTY_FOS
        if not res.ok:
            return None, PENALTY_FOS
        if settings.check_admissibility and not res.admissible:
            return None, PENALTY_FOS
        if not _within_limits(cols, settings.search_limits):
            return None, PENALTY_FOS
        return surf, float(res.fos)

    def eval_vec(vec: np.ndarray, fine: bool = False) -> float:
        nonlocal eval_count
        eval_count += 1
        surf, f_val = _eval_pure(vec, fine=fine)
        if surf is not None and f_val < PENALTY_FOS:
            trials.append((surf, f_val))
        return f_val

    def eval_batch(
        vecs: list[np.ndarray],
        fine: bool = False,
        pool: ThreadPoolExecutor | None = None,
    ) -> list[float]:
        nonlocal eval_count
        eval_count += len(vecs)
        if pool is not None and len(vecs) > 1:
            results = list(pool.map(lambda v: _eval_pure(v, fine=fine), vecs))
        else:
            results = [_eval_pure(v, fine=fine) for v in vecs]
        out_f: list[float] = []
        for surf, f_val in results:
            if surf is not None and f_val < PENALTY_FOS:
                trials.append((surf, f_val))
            out_f.append(f_val)
        return out_f

    # Seed population: heuristic mid-slope spheres/ellipsoids + uniform random samples
    nests = np.empty((n_nests, dim), dtype=float)
    fit = np.full(n_nests, PENALTY_FOS, dtype=float)

    x_mid_slope = float(xs_p[int(0.5 * (np.argmin(zs_p) + np.argmax(zs_p)))])
    z_low, z_high = float(np.min(zs_p)), float(np.max(zs_p))
    h_est = max(2.0, z_high - z_low)
    yc_mid = 0.5 * (lo[1] + hi[1])

    seed_vecs: list[np.ndarray] = []
    for z_mult, r_mult, x_shift in (
        (1.2, 1.5, -0.25),
        (1.6, 1.9, -0.15),
        (1.0, 1.3, 0.0),
        (2.0, 2.3, -0.30),
    ):
        sgn = -1.0 if default_az == 180.0 else 1.0
        xc_s = np.clip(x_mid_slope + sgn * x_shift * h_est, lo[0], hi[0])
        zc_s = np.clip(z_low + z_mult * h_est, lo[2], hi[2])
        rx_s = np.clip(r_mult * h_est, lo[3], hi[3])
        if base_kind == "sphere":
            seed_vecs.append(np.array([xc_s, yc_mid, zc_s, rx_s]))
        else:
            ry_s = np.clip(0.55 * fast_model.span_y, lo[4], hi[4])
            rz_s = np.clip(r_mult * h_est, lo[5], hi[5])
            v = [xc_s, yc_mid, zc_s, rx_s, ry_s, rz_s]
            if settings.optimize_azimuth:
                v.append(default_az)
            seed_vecs.append(np.array(v, dtype=float))

    for i in range(n_nests):
        if i < len(seed_vecs):
            nests[i] = seed_vecs[i]
        else:
            nests[i] = lo + rng.random(dim) * span

    cancelled = False
    pool_ctx = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        fit[:] = eval_batch([nests[i] for i in range(n_nests)], fine=False, pool=pool_ctx)

        if float(np.min(fit)) >= PENALTY_FOS:
            extra_cands = [lo + rng.random(dim) * span for _ in range(4 * n_nests)]
            extra_fits = eval_batch(extra_cands, fine=False, pool=pool_ctx)
            for cand, f_c in zip(extra_cands, extra_fits, strict=False):
                if f_c < PENALTY_FOS:
                    nests[0] = cand
                    fit[0] = f_c
                    break

        pa = 0.25
        best_idx = int(np.argmin(fit))
        best_vec = nests[best_idx].copy()
        best_fos = float(fit[best_idx])
        if best_surface_cb is not None and best_fos < PENALTY_FOS:
            best_surface_cb(
                _decode_surface(best_vec, base_kind, default_az, settings.optimize_azimuth),
                best_fos,
            )

        # Particle Swarm state (Clerc-Kennedy constriction): nests double as personal bests
        use_pso = settings.algorithm == "pso"
        pos = nests.copy()
        vel = (rng.random((n_nests, dim)) - 0.5) * 0.2 * span
        v_max = 0.25 * span

        for gen in range(n_gen):
            if cancel_check is not None and cancel_check():
                cancelled = True
                break

            if use_pso:
                r1 = rng.random((n_nests, dim))
                r2 = rng.random((n_nests, dim))
                vel = PSO_INERTIA * vel + PSO_ACCEL * (
                    r1 * (nests - pos) + r2 * (best_vec[None, :] - pos)
                )
                vel = np.clip(vel, -v_max, v_max)
                pos = pos + vel
                # Reflect at the bounds so particles do not pile up on the walls
                below, above = pos < lo, pos > hi
                pos = np.where(below, 2.0 * lo - pos, np.where(above, 2.0 * hi - pos, pos))
                pos = np.clip(pos, lo, hi)
                vel = np.where(below | above, -0.5 * vel, vel)
                pso_fits = eval_batch([pos[i] for i in range(n_nests)], fine=False, pool=pool_ctx)
                improved_pso = False
                for i, f_c in enumerate(pso_fits):
                    if f_c < fit[i]:
                        nests[i] = pos[i]
                        fit[i] = f_c
                        if f_c < best_fos:
                            best_fos = f_c
                            best_vec = pos[i].copy()
                            best_idx = i
                            improved_pso = True
                # Re-seed particles that have never found a valid surface
                for i in range(n_nests):
                    if fit[i] >= PENALTY_FOS:
                        pos[i] = lo + rng.random(dim) * span
                        vel[i] = (rng.random(dim) - 0.5) * 0.2 * span
                if improved_pso and best_surface_cb is not None and best_fos < PENALTY_FOS:
                    best_surface_cb(
                        _decode_surface(best_vec, base_kind, default_az, settings.optimize_azimuth),
                        best_fos,
                    )
                if progress_cb is not None:
                    progress_cb(gen + 1, n_gen, best_fos)
                continue

            # Lévy flights evaluated in parallel batch
            levy_cands: list[np.ndarray] = []
            levy_targets: list[int] = []
            for i in range(n_nests):
                step = 0.04 * span * _levy_step(rng, dim)
                if i != best_idx:
                    step = step * (0.5 + 0.5 * np.abs(nests[i] - best_vec) / span)
                levy_cands.append(np.clip(nests[i] + step, lo, hi))
                levy_targets.append(int(rng.integers(0, n_nests)))

            levy_fits = eval_batch(levy_cands, fine=False, pool=pool_ctx)
            improved_gen = False
            for cand, j, f_c in zip(levy_cands, levy_targets, levy_fits, strict=False):
                if f_c < fit[j]:
                    nests[j] = cand
                    fit[j] = f_c
                    if f_c < best_fos:
                        best_fos = f_c
                        best_vec = cand.copy()
                        best_idx = j
                        improved_gen = True

            # Abandon fraction pa of worst nests via differential random walk in parallel batch
            worst_order = np.argsort(fit)[::-1]
            n_abandon = max(1, int(pa * n_nests))
            ab_indices = [int(k_idx) for k_idx in worst_order[:n_abandon] if int(k_idx) != best_idx]
            ab_cands: list[np.ndarray] = []
            for k_idx in ab_indices:
                r1, r2 = rng.integers(0, n_nests, size=2)
                ab_cands.append(
                    np.clip(nests[k_idx] + rng.random(dim) * (nests[r1] - nests[r2]), lo, hi)
                )
            if ab_cands:
                ab_fits = eval_batch(ab_cands, fine=False, pool=pool_ctx)
                for k_idx, cand, f_c in zip(ab_indices, ab_cands, ab_fits, strict=False):
                    if f_c < fit[k_idx]:
                        nests[k_idx] = cand
                        fit[k_idx] = f_c
                        if f_c < best_fos:
                            best_fos = f_c
                            best_vec = cand.copy()
                            best_idx = k_idx
                            improved_gen = True

            if improved_gen and best_surface_cb is not None and best_fos < PENALTY_FOS:
                best_surface_cb(
                    _decode_surface(best_vec, base_kind, default_az, settings.optimize_azimuth),
                    best_fos,
                )
            if progress_cb is not None:
                progress_cb(gen + 1, n_gen, best_fos)

        # Re-evaluate top 4 distinct nests on full-resolution column grid
        top_indices = [
            int(idx) for idx in np.argsort(fit)[: min(4, n_nests)] if fit[idx] < PENALTY_FOS
        ]
        fine_vecs = [best_vec.copy()] + [nests[idx].copy() for idx in top_indices]
        fine_fits = eval_batch(fine_vecs, fine=True, pool=pool_ctx)
        best_fine_i = int(np.argmin(fine_fits))
        fine_best_vec = fine_vecs[best_fine_i].copy()
        fine_best_fos = float(fine_fits[best_fine_i])
    finally:
        if pool_ctx is not None:
            pool_ctx.shutdown(wait=True)

    if not cancelled and fine_best_fos < PENALTY_FOS:

        def nm_obj(u_norm: np.ndarray) -> float:
            if cancel_check is not None and cancel_check():
                return PENALTY_FOS
            if np.any(u_norm < 0.0) or np.any(u_norm > 1.0):
                return PENALTY_FOS
            return eval_vec(lo + u_norm * span, fine=True)

        # Local polish: Nelder-Mead from the best full-resolution candidates, then restarts
        # from the incumbent with a fresh, smaller simplex. The FoS valley around the critical
        # surface is flat and slightly noisy (column discretisation), so a single short run
        # stalls at a seed-dependent point; several starts and restarts reduce that scatter.
        ranked = sorted(
            (f, i) for i, f in enumerate(fine_fits) if f < PENALTY_FOS and np.isfinite(f)
        )
        starts: list[np.ndarray] = []
        for _f, i in ranked:
            u_i = np.clip((fine_vecs[i] - lo) / span, 0.0, 1.0)
            if all(float(np.max(np.abs(u_i - u_j))) > 0.02 for u_j in starts):
                starts.append(u_i)
            if len(starts) >= NM_STARTS:
                break
        rounds = [(u_s, NM_STEP) for u_s in starts] + [(None, NM_STEP / 2.0)] * NM_RESTARTS
        for u_start, step in rounds:
            if cancel_check is not None and cancel_check():
                break
            if u_start is None:
                u_start = np.clip((fine_best_vec - lo) / span, 0.0, 1.0)
            nm = minimize(
                nm_obj,
                u_start,
                method="Nelder-Mead",
                options={
                    "maxfev": nm_fev,
                    "xatol": 1e-3,
                    "fatol": 2e-4,
                    "initial_simplex": _initial_simplex(u_start, step),
                },
            )
            if float(nm.fun) < fine_best_fos:
                fine_best_fos = float(nm.fun)
                fine_best_vec = lo + np.clip(nm.x, 0.0, 1.0) * span

    crit_surface: SlipSurface3D = _decode_surface(
        fine_best_vec, base_kind, default_az, settings.optimize_azimuth
    )
    if best_surface_cb is not None and fine_best_fos < PENALTY_FOS:
        best_surface_cb(crit_surface, fine_best_fos)

    # Multi-scale Surface Altering Optimization (SAO: 3x3 -> 5x5) when spline_patch or sao_enabled
    run_sao = (settings.surface_kind == "spline_patch" or settings.sao_enabled) and not cancelled
    if run_sao and fine_best_fos < PENALTY_FOS:
        assert isinstance(crit_surface, EllipsoidSurface)
        sao_surf, sao_fos, sao_trials, sao_evals = run_sao_refinement(
            base_ellipsoid=crit_surface,
            base_fos=fine_best_fos,
            model=model,
            materials=materials,
            col_opts=col_opts,
            method=settings.primary_method,
            gle_interslice=settings.gle_interslice,
            solver_tolerance=solver_tolerance,
            solver_max_iter=solver_max_iter,
            check_admissibility=settings.check_admissibility,
            m_alpha_min=settings.m_alpha_min,
            material_colors=material_colors,
            max_fev_coarse=max(25, nm_fev // 2),
            max_fev_fine=max(30, (2 * nm_fev) // 3),
            cancel_check=cancel_check,
            best_surface_cb=best_surface_cb,
        )
        eval_count += sao_evals
        trials.extend(sao_trials)
        if sao_fos < fine_best_fos:
            crit_surface = sao_surf
            fine_best_fos = sao_fos

    crit_cols = discretize_columns(
        model, crit_surface, materials, opts=col_opts, material_colors=material_colors
    )
    crit_res = solve_surface_3d(
        crit_cols,
        method=settings.primary_method,
        tolerance=solver_tolerance,
        max_iter=solver_max_iter,
        m_alpha_min=settings.m_alpha_min,
        interslice=settings.gle_interslice,
    )
    local_minima: list[LocalMinimum3D] = []
    still_running = not (cancelled or bool(cancel_check is not None and cancel_check()))
    if settings.n_local_minima > 1 and still_running and crit_res.ok:

        def _finalize(surf: SlipSurface3D) -> tuple[ColumnGrid, float] | None:
            try:
                f_cols = discretize_columns(
                    model, surf, materials, opts=col_opts, material_colors=material_colors
                )
            except ColumnDiscretizationError:
                return None
            f_res = solve_surface_3d(
                f_cols,
                method=settings.primary_method,
                tolerance=solver_tolerance,
                max_iter=solver_max_iter,
                m_alpha_min=settings.m_alpha_min,
                interslice=settings.gle_interslice,
            )
            return (f_cols, float(f_res.fos)) if f_res.ok else None

        local_minima, extra_evals = _find_local_minima(
            fast_model=fast_model,
            trials=list(trials),
            critical=(crit_surface, crit_cols, float(crit_res.fos)),
            n_wanted=int(settings.n_local_minima),
            evaluate=lambda v: _eval_pure(v, fine=False),
            finalize=_finalize,
            bounds=(lo, hi),
            optimize_azimuth=settings.optimize_azimuth,
            base_kind=base_kind,
            nm_maxfev=max(90, 3 * nm_fev),
            cancel_check=cancel_check,
        )
        eval_count += extra_evals
        # A refined secondary mode can undercut the main search: promote it to critical.
        local_minima.sort(key=lambda m: m.fos)
        top_source = local_minima[0].source_surface
        if top_source is not None and top_source is not crit_surface:
            promoted = _finalize(top_source)
            if promoted is not None:
                crit_surface = top_source
                crit_cols = promoted[0]
                local_minima[0].surface = crit_cols.surface
                crit_res = solve_surface_3d(
                    crit_cols,
                    method=settings.primary_method,
                    tolerance=solver_tolerance,
                    max_iter=solver_max_iter,
                    m_alpha_min=settings.m_alpha_min,
                    interslice=settings.gle_interslice,
                )

    return SearchResult3D(
        critical_surface=crit_surface,
        critical_columns=crit_cols,
        critical_result=crit_res,
        trials=trials,
        local_minima=local_minima,
        evaluations=eval_count,
        cancelled=cancelled or bool(cancel_check is not None and cancel_check()),
    )


@dataclass
class SafetyMap3D:
    """Slope-wide 3D minimum Factor-of-Safety hazard map projected onto ground topography."""

    x_coords: np.ndarray
    y_coords: np.ndarray
    z_ground: np.ndarray
    min_fos: np.ndarray
    trial_count: np.ndarray

    def to_mesh(self, z_offset: float = 0.04) -> pv.PolyData:
        """Build a clipped PyVista surface mesh over the covered ground cells colored by ``MinFoS``."""
        xx, yy = np.meshgrid(self.x_coords, self.y_coords, indexing="ij")
        zz = self.z_ground + float(z_offset)
        valid = np.isfinite(self.min_fos)
        if not np.any(valid):
            return pv.PolyData()
        fill_v = float(np.nanmax(self.min_fos))
        fos_filled = np.where(valid, self.min_fos, fill_v)
        grid = pv.StructuredGrid(xx, yy, zz)
        grid["MinFoS"] = fos_filled.ravel(order="F")
        grid["TrialCount"] = self.trial_count.ravel(order="F").astype(float)
        grid["CoveredMask"] = valid.ravel(order="F").astype(float)
        surf = grid.extract_surface(algorithm="dataset_surface")
        return surf.threshold(0.5, scalars="CoveredMask", preference="point").extract_surface(
            algorithm="dataset_surface"
        )


def compute_safety_map_3d(
    model: SlopeModel3D,
    trials: list[tuple[SlipSurface3D, float]],
    critical_surface: SlipSurface3D | None = None,
    critical_fos: float | None = None,
    nx: int = 64,
    ny: int = 48,
    min_depth: float = 0.05,
) -> SafetyMap3D:
    """Project all admissible 3D trial slip bowls onto ``(x, y)`` ground grid to map minimum FoS."""
    xs = np.linspace(model.x_min, model.x_max, max(8, nx))
    ys = np.linspace(model.y_min, model.y_max, max(8, ny))
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zg = model.ground_z(xx, yy)

    min_fos = np.full((len(xs), len(ys)), np.nan, dtype=float)
    counts = np.zeros((len(xs), len(ys)), dtype=int)

    all_trials: list[tuple[SlipSurface3D, float]] = [
        (s, float(f)) for s, f in trials if np.isfinite(f) and 0.0 < f < PENALTY_FOS
    ]
    if critical_surface is not None and critical_fos is not None and np.isfinite(critical_fos):
        all_trials.append((critical_surface, float(critical_fos)))

    if not all_trials:
        return SafetyMap3D(
            x_coords=xs, y_coords=ys, z_ground=zg, min_fos=min_fos, trial_count=counts
        )

    all_trials.sort(key=lambda item: item[1])

    for surf, fos_val in all_trials:
        fx0, fx1, fy0, fy1 = surf.footprint_bounds()
        ix0 = max(0, int(np.searchsorted(xs, fx0, side="left")) - 1)
        ix1 = min(len(xs), int(np.searchsorted(xs, fx1, side="right")) + 1)
        iy0 = max(0, int(np.searchsorted(ys, fy0, side="left")) - 1)
        iy1 = min(len(ys), int(np.searchsorted(ys, fy1, side="right")) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        sub_x = xx[ix0:ix1, iy0:iy1]
        sub_y = yy[ix0:ix1, iy0:iy1]
        sub_zg = zg[ix0:ix1, iy0:iy1]
        zb = clip_to_weak_layers(model, surf).sample_base(sub_x, sub_y, sub_zg, model.z_min).z
        active = np.isfinite(zb) & ((sub_zg - zb) >= min_depth)
        if not np.any(active):
            continue
        sub_cnt = counts[ix0:ix1, iy0:iy1]
        sub_cnt[active] += 1
        sub_min = min_fos[ix0:ix1, iy0:iy1]
        upd = active & (~np.isfinite(sub_min) | (fos_val < sub_min))
        sub_min[upd] = fos_val

    return SafetyMap3D(
        x_coords=xs,
        y_coords=ys,
        z_ground=zg,
        min_fos=min_fos,
        trial_count=counts,
    )
