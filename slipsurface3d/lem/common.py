"""Shared 3D LEM result type and column-state helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from slipsurface3d.lem.columns import ColumnGrid

M_ALPHA_MIN_DEFAULT = 0.2


@dataclass
class LEMResult3D:
    fos: float
    method: str
    converged: bool
    iterations: int
    azimuth_deg: float = 180.0
    m_alpha_min: float = np.nan
    lam: float = np.nan
    janbu_f0: float = np.nan
    force_residual: float = np.nan
    moment_residual: float = np.nan
    transverse_residual: float = 0.0
    interslice_func: str = "constant"
    admissible: bool = True
    issue: str = ""
    message: str = ""
    base_normal_eff: np.ndarray | None = None  # N'_i [kN]
    sigma_n_eff: np.ndarray | None = None  # sigma'_n,i [kPa]
    base_shear_mob: np.ndarray | None = None  # S_mob,i [kN]
    tau_mob: np.ndarray | None = None  # tau_mob,i [kPa]

    @property
    def ok(self) -> bool:
        return bool(self.converged and np.isfinite(self.fos) and self.fos > 0.0)


def _evaluate_strength(cols: ColumnGrid, sigma_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate equivalent (c, tan_phi) on each active column base."""
    c_arr = np.empty(cols.n, dtype=float)
    tp_arr = np.empty(cols.n, dtype=float)
    sn_clamped = np.maximum(sigma_n, 0.0)
    for idx, mat in enumerate(cols.materials):
        mask = cols.mat_idx == idx
        if not np.any(mask):
            continue
        c_sub, tp_sub = mat.strength(
            sn_clamped[mask],
            z=cols.z_base[mask] if mat.uses_elevation else None,
            sigma_v=cols.sigma_v_eff[mask] if mat.uses_vertical_stress else None,
            normal_xyz=cols.normal_xyz[mask] if mat.uses_normal_vector else None,
        )
        c_arr[mask] = c_sub
        tp_arr[mask] = tp_sub
    return c_arr, tp_arr


def _has_nonlinear_material(cols: ColumnGrid) -> bool:
    from slipsurface3d.materials.models import (
        BartonBandis3D,
        HoekBrown3D,
        HoekBrownDirect3D,
        PowerCurve3D,
        PowerLaw3D,
    )

    nl_types = (HoekBrown3D, HoekBrownDirect3D, BartonBandis3D, PowerCurve3D, PowerLaw3D)
    present = set(np.unique(cols.mat_idx).tolist())
    return any(isinstance(cols.materials[i], nl_types) for i in present)


def _initial_sigma_n(cols: ColumnGrid) -> np.ndarray:
    return np.maximum(cols.Wv / cols.dA - cols.u, 0.0) * (cols.nz**2)


def _check_admissibility_3d(
    res: LEMResult3D,
    cols: ColumnGrid,
    ma: np.ndarray,
    base_strength: np.ndarray,
    n_eff: np.ndarray,
    c_force: np.ndarray,
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT,
) -> LEMResult3D:
    issues: list[str] = []
    min_ma = float(np.min(ma)) if len(ma) else np.nan
    res.m_alpha_min = min_ma
    if np.isfinite(min_ma) and min_ma < m_alpha_min:
        issues.append(f"m_alpha,min = {min_ma:.3f} < {m_alpha_min:g}")
    w_mean = float(np.mean(cols.W)) if cols.n > 0 else 1.0
    if float(np.min(base_strength)) < -1e-3 * w_mean:
        issues.append("negative basal shear resistance")
    # Flag widespread basal effective tension exceeding cohesive capacity
    tension = n_eff < -0.5 * np.maximum(c_force, 1e-6)
    if int(np.count_nonzero(tension)) >= max(3, cols.n // 10):
        issues.append(f"basal tension on {int(np.count_nonzero(tension))}/{cols.n} columns")
    res.admissible = not issues
    res.issue = "; ".join(issues)
    return res


def _column_state(
    cols: ColumnGrid,
    fos: float,
    lam: float,
    c_arr: np.ndarray,
    tp_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compute (N_eff, S_mob, m_alpha, sigma_n_eff, force_res, moment_res) for (F, lambda)."""
    inv_f = 1.0 / fos
    v_down = cols.Wv - (cols.Qz_active + cols.Qz_passive * inv_f)
    h_down = cols.Fh + cols.Qs_active + cols.Qs_passive * inv_f
    u_force = cols.u * cols.area_base
    c_force = c_arr * cols.area_base

    nz_lam = cols.nz - lam * cols.ns
    sa_lam = cols.sin_alpha_s + lam * cols.cos_alpha_s
    ma = nz_lam + sa_lam * tp_arr * inv_f

    # Guard against near-zero denominator singularities during root bracketing
    if np.any(np.abs(ma) < 1e-9):
        return (
            np.full(cols.n, np.nan),
            np.full(cols.n, np.nan),
            ma,
            np.full(cols.n, np.nan),
            np.nan,
            np.nan,
        )

    n_eff = (v_down + lam * h_down - u_force * nz_lam - c_force * sa_lam * inv_f) / ma
    n_tot = n_eff + u_force
    s_avail = c_force + n_eff * tp_arr
    s_mob = s_avail * inv_f
    sigma_n_eff = n_eff / cols.area_base

    # Global horizontal force balance (resisting - driving) normalized by total weight
    scale_f = max(cols.total_weight, 1e-9)
    e_net = float(
        np.sum(s_mob * cols.cos_alpha_s - n_tot * cols.ns - h_down) - cols.crack_water_force_s
    )
    force_res = e_net * float(np.hypot(1.0, lam)) / scale_f

    # Global moment balance about (sc, zc) normalized by W_total * L_char
    ds = cols.sc - cols.s
    dz = cols.zc - cols.z_base
    r_arm = dz * cols.cos_alpha_s + ds * cols.sin_alpha_s
    e_norm = dz * cols.ns - ds * cols.nz

    m_resist = float(np.sum(s_mob * r_arm))
    m_drive = (
        float(np.sum(cols.Wv * ds))
        + float(np.sum(cols.Fh * (cols.zc - cols.z_cg)))
        + float(np.sum(cols.Ms_active + cols.Ms_passive * inv_f))
        + cols.crack_water_moment
        + float(np.sum(n_tot * e_norm))
    )
    l_char = max(1.0, float(np.max(cols.s) - np.min(cols.s)), float(np.mean(np.abs(r_arm))))
    moment_res = (m_resist - m_drive) / (scale_f * l_char)
    return n_eff, s_mob, ma, sigma_n_eff, force_res, moment_res


def _solve_1d_root(
    residual_fn: Callable[[float], float],
    f_guess: float,
    tol: float = 1e-6,
    valid_fn: Callable[[float], bool] | None = None,
) -> tuple[float, int]:
    """Bracket and solve residual_fn(F) = 0 on a geometric grid around f_guess."""
    g0 = float(np.clip(f_guess if np.isfinite(f_guess) and f_guess > 0 else 1.2, 0.05, 50.0))
    grid = np.unique(
        np.concatenate(
            [
                g0 * np.geomspace(0.15, 6.0, 29),
                np.geomspace(0.05, 40.0, 25),
            ]
        )
    )
    vals = np.array([residual_fn(float(f)) for f in grid], dtype=float)
    candidates: list[tuple[float, float, float]] = []
    for k in range(len(grid) - 1):
        v0, v1 = vals[k], vals[k + 1]
        if np.isfinite(v0) and abs(v0) < tol:
            candidates.append((abs(np.log(grid[k] / g0)), float(grid[k]), float(grid[k])))
        elif np.isfinite(v0) and np.isfinite(v1) and v0 * v1 < 0.0:
            mid = float(np.sqrt(grid[k] * grid[k + 1]))
            candidates.append((abs(np.log(mid / g0)), float(grid[k]), float(grid[k + 1])))

    total_iters = 0
    for _, a, b in sorted(candidates, key=lambda item: item[0]):
        if a == b:
            r = a
            it = 1
        else:
            try:
                r, info = brentq(
                    residual_fn, a, b, xtol=1e-10, rtol=1e-10, maxiter=80, full_output=True
                )
                it = int(info.iterations)
            except (ValueError, RuntimeError):
                continue
        total_iters += it
        rv = residual_fn(float(r))
        if (
            np.isfinite(rv)
            and abs(rv) <= max(tol, 1e-5)
            and (valid_fn is None or valid_fn(float(r)))
        ):
            return float(r), total_iters
    return np.nan, total_iters
