"""Global 3D critical slip surface search (Cuckoo Search + Nelder-Mead local polish).

Supports searching over:
* ``"sphere"``       : 4-parameter spherical slip surfaces ``(xc, yc, zc, r)``
* ``"ellipsoid"``    : 6/7-parameter tri-axial ellipsoids ``(xc, yc, zc, rx, ry, rz[, azimuth])``
* ``"spline_patch"`` : Non-ellipsoidal 3D spline bowls perturbing the critical ellipsoid
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
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


@dataclass
class SearchSettings3D:
    surface_kind: str = "ellipsoid"  # sphere | ellipsoid | spline_patch | user
    effort: str = "normal"  # fast | normal | fine
    primary_method: str = "bishop"
    optimize_azimuth: bool = False
    check_admissibility: bool = True
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT
    seed: int = 42
    x_entry_range: tuple[float, float] | None = None
    y_center_range: tuple[float, float] | None = None
    ry_bounds: tuple[float, float] | None = None


@dataclass
class SearchResult3D:
    critical_surface: SlipSurface3D
    critical_columns: ColumnGrid
    critical_result: LEMResult3D
    trials: list[tuple[SlipSurface3D, float]] = field(default_factory=list)
    evaluations: int = 0


def _detect_slope_crest_toe(model: SlopeModel3D) -> tuple[float, float, float, float, float]:
    """Estimate (x_low, x_high, z_low, z_high, default_azimuth_deg) on central Y profile."""
    ym = 0.5 * (model.y_min + model.y_max)
    xs = np.linspace(model.x_min, model.x_max, 81)
    zs = model.ground_z(xs, np.full_like(xs, ym))
    z_low = float(np.min(zs))
    z_high = float(np.max(zs))
    i_low = int(np.argmin(zs))
    i_high = int(np.argmax(zs))
    x_low = float(xs[i_low])
    x_high = float(xs[i_high])
    az = 180.0 if x_high >= x_low else 0.0
    return x_low, x_high, z_low, z_high, az


def _parameter_bounds(
    model: SlopeModel3D,
    settings: SearchSettings3D,
    default_az: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_low, x_high, z_low, z_high, _ = (
        _detect_slope_crest_toe(model) if False else (0.0, 0.0, 0.0, 0.0, default_az)
    )
    # Inline clean slope geometry statistics
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


def _levy_step(rng: np.random.Generator, dim: int, beta: float = 1.5) -> np.ndarray:
    from math import gamma, pi, sin

    num = gamma(1.0 + beta) * sin(pi * beta / 2.0)
    den = gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (num / den) ** (1.0 / beta)
    u = rng.normal(0.0, sigma_u, size=dim)
    v = rng.normal(0.0, 1.0, size=dim)
    return u / (np.abs(v) ** (1.0 / beta) + 1e-12)


def search_critical_surface_3d(
    model: SlopeModel3D,
    materials: list[Material3D],
    col_opts: ColumnOptions,
    settings: SearchSettings3D | None = None,
    solver_tolerance: float = 1e-4,
    solver_max_iter: int = 60,
    material_colors: list[str] | None = None,
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> SearchResult3D:
    """Locate the global minimum-FoS 3D slip surface via Cuckoo Search + Nelder-Mead."""
    settings = settings or SearchSettings3D()
    n_nests, n_gen, nm_fev = EFFORT_PRESETS.get(settings.effort, EFFORT_PRESETS["normal"])
    rng = np.random.default_rng(settings.seed)

    # Determine natural downhill azimuth
    ym = 0.5 * (model.y_min + model.y_max)
    xs_p = np.linspace(model.x_min, model.x_max, 61)
    zs_p = model.ground_z(xs_p, np.full_like(xs_p, ym))
    if col_opts.sliding_mode == "right_to_left":
        default_az = 180.0
    elif col_opts.sliding_mode == "left_to_right":
        default_az = 0.0
    elif col_opts.sliding_mode == "custom":
        default_az = col_opts.azimuth_deg
    else:
        default_az = 180.0 if int(np.argmax(zs_p)) >= int(np.argmin(zs_p)) else 0.0

    base_kind = "sphere" if settings.surface_kind == "sphere" else "ellipsoid"
    lo, hi = _parameter_bounds(model, settings, default_az)
    span = np.maximum(hi - lo, 1e-6)
    dim = len(lo)

    # Coarser column grid during global exploration for speed, exact col_opts during polish
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
    )

    trials: list[tuple[SlipSurface3D, float]] = []
    eval_count = 0

    def eval_vec(vec: np.ndarray, fine: bool = False) -> float:
        nonlocal eval_count
        if np.any(vec < lo) or np.any(vec > hi):
            return PENALTY_FOS
        eval_count += 1
        surf = _decode_surface(vec, base_kind, default_az, settings.optimize_azimuth)
        opts_use = col_opts if fine else screen_opts
        try:
            cols = discretize_columns(
                model, surf, materials, opts=opts_use, material_colors=material_colors
            )
            res = solve_surface_3d(
                cols,
                method=settings.primary_method,
                tolerance=solver_tolerance if fine else max(solver_tolerance, 5e-4),
                max_iter=solver_max_iter,
                m_alpha_min=settings.m_alpha_min,
            )
        except ColumnDiscretizationError:
            return PENALTY_FOS
        if not res.ok:
            return PENALTY_FOS
        if settings.check_admissibility and not res.admissible:
            return PENALTY_FOS
        trials.append((surf, float(res.fos)))
        return float(res.fos)

    # Seed population: heuristic mid-slope spheres/ellipsoids + uniform Latin-like samples
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
            ry_s = np.clip(0.55 * model.span_y, lo[4], hi[4])
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
        fit[i] = eval_vec(nests[i], fine=False)

    # Ensure at least one valid starting nest if domain is narrow
    if float(np.min(fit)) >= PENALTY_FOS:
        for _ in range(4 * n_nests):
            cand = lo + rng.random(dim) * span
            f_c = eval_vec(cand, fine=False)
            if f_c < PENALTY_FOS:
                nests[0] = cand
                fit[0] = f_c
                break

    pa = 0.25
    best_idx = int(np.argmin(fit))
    best_vec = nests[best_idx].copy()
    best_fos = float(fit[best_idx])

    for gen in range(n_gen):
        # Lévy flights toward global best
        for i in range(n_nests):
            step = 0.04 * span * _levy_step(rng, dim)
            if i != best_idx:
                step = step * (0.5 + 0.5 * np.abs(nests[i] - best_vec) / span)
            cand = np.clip(nests[i] + step, lo, hi)
            j = int(rng.integers(0, n_nests))
            f_c = eval_vec(cand, fine=False)
            if f_c < fit[j]:
                nests[j] = cand
                fit[j] = f_c
                if f_c < best_fos:
                    best_fos = f_c
                    best_vec = cand.copy()
                    best_idx = j

        # Abandon fraction pa of worst nests via differential random walk
        worst_order = np.argsort(fit)[::-1]
        n_abandon = max(1, int(pa * n_nests))
        for k_idx in worst_order[:n_abandon]:
            if k_idx == best_idx:
                continue
            r1, r2 = rng.integers(0, n_nests, size=2)
            cand = np.clip(nests[k_idx] + rng.random(dim) * (nests[r1] - nests[r2]), lo, hi)
            f_c = eval_vec(cand, fine=False)
            if f_c < fit[k_idx]:
                nests[k_idx] = cand
                fit[k_idx] = f_c
                if f_c < best_fos:
                    best_fos = f_c
                    best_vec = cand.copy()
                    best_idx = int(k_idx)

        if progress_cb is not None:
            progress_cb(gen + 1, n_gen, best_fos)

    # Re-evaluate top 4 distinct nests on full-resolution column grid & Nelder-Mead polish
    top_indices = np.argsort(fit)[: min(4, n_nests)]
    fine_best_vec = best_vec.copy()
    fine_best_fos = eval_vec(fine_best_vec, fine=True)
    for idx in top_indices:
        if fit[idx] >= PENALTY_FOS:
            continue
        f_val = eval_vec(nests[idx], fine=True)
        if f_val < fine_best_fos:
            fine_best_fos = f_val
            fine_best_vec = nests[idx].copy()

    if fine_best_fos < PENALTY_FOS:
        # Normalized Nelder-Mead polish on [0, 1]^dim
        u0 = np.clip((fine_best_vec - lo) / span, 0.0, 1.0)

        def nm_obj(u_norm: np.ndarray) -> float:
            if np.any(u_norm < 0.0) or np.any(u_norm > 1.0):
                return PENALTY_FOS
            return eval_vec(lo + u_norm * span, fine=True)

        nm = minimize(
            nm_obj,
            u0,
            method="Nelder-Mead",
            options={"maxfev": nm_fev, "xatol": 1e-3, "fatol": 5e-4},
        )
        if float(nm.fun) < fine_best_fos:
            fine_best_fos = float(nm.fun)
            fine_best_vec = lo + np.clip(nm.x, 0.0, 1.0) * span

    crit_surface: SlipSurface3D = _decode_surface(
        fine_best_vec, base_kind, default_az, settings.optimize_azimuth
    )

    # Optional non-ellipsoidal 3x3 spline patch deformation stage
    if settings.surface_kind == "spline_patch" and fine_best_fos < PENALTY_FOS:
        assert isinstance(crit_surface, EllipsoidSurface)
        base_ell = crit_surface
        max_amp = 0.20 * base_ell.rz

        def spline_obj(dz_flat: np.ndarray) -> float:
            nonlocal eval_count
            if np.any(np.abs(dz_flat) > max_amp):
                return PENALTY_FOS
            eval_count += 1
            sp_surf = SplinePatchSurface(base=base_ell, control_dz=dz_flat.reshape(3, 3))
            try:
                cols_s = discretize_columns(
                    model, sp_surf, materials, opts=col_opts, material_colors=material_colors
                )
                res_s = solve_surface_3d(
                    cols_s,
                    method=settings.primary_method,
                    tolerance=solver_tolerance,
                    max_iter=solver_max_iter,
                    m_alpha_min=settings.m_alpha_min,
                )
            except ColumnDiscretizationError:
                return PENALTY_FOS
            if not res_s.ok or (settings.check_admissibility and not res_s.admissible):
                return PENALTY_FOS
            trials.append((sp_surf, float(res_s.fos)))
            return float(res_s.fos)

        z0_patch = np.zeros(9, dtype=float)
        init_simplex = np.vstack([z0_patch, z0_patch + np.eye(9) * (0.04 * base_ell.rz)])
        nm_sp = minimize(
            spline_obj,
            z0_patch,
            method="Nelder-Mead",
            options={
                "initial_simplex": init_simplex,
                "maxfev": max(30, nm_fev // 2),
                "fatol": 5e-4,
            },
        )
        if float(nm_sp.fun) < fine_best_fos:
            crit_surface = SplinePatchSurface(
                base=base_ell, control_dz=np.asarray(nm_sp.x, dtype=float).reshape(3, 3)
            )

    crit_cols = discretize_columns(
        model, crit_surface, materials, opts=col_opts, material_colors=material_colors
    )
    crit_res = solve_surface_3d(
        crit_cols,
        method=settings.primary_method,
        tolerance=solver_tolerance,
        max_iter=solver_max_iter,
        m_alpha_min=settings.m_alpha_min,
    )
    return SearchResult3D(
        critical_surface=crit_surface,
        critical_columns=crit_cols,
        critical_result=crit_res,
        trials=trials,
        evaluations=eval_count,
    )
