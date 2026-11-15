"""Three-dimensional Limit Equilibrium solvers (Method of Columns).

Implemented methods:
* ``ordinary_3d``         : 3D Ordinary / Fellenius moment equilibrium (Hovland 1977; Hungr 1987)
* ``bishop_3d``           : 3D Simplified Bishop moment equilibrium (Hungr 1987; Hungr et al. 1989)
* ``janbu_simplified_3d`` : 3D Simplified Janbu horizontal force equilibrium (Hungr et al. 1989)
* ``janbu_corrected_3d``  : 3D Corrected Janbu (f0 * F_Janbu)
* ``spencer_3d``          : 3D Rigorous Spencer coupled force + moment equilibrium
                            (Lam & Fredlund 1993; Cheng & Yip 2007)
* ``gle_mp_3d``           : 3D General Limit Equilibrium / Morgenstern-Price with selectable
                            intercolumn inclination functions (constant, half-sine, clipped-sine, trapezoidal)
* ``cheng_yip_3d``        : 3D Asymmetric Cheng & Yip (2007) simultaneous longitudinal force,
                            transverse force, and moment equilibrium solving true sliding azimuth
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, root

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


def ordinary_3d(
    cols: ColumnGrid,
    tolerance: float = 1e-4,
    max_iter: int = 40,
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT,
) -> LEMResult3D:
    """3D Ordinary / Fellenius moment-equilibrium method."""
    sigma_n = _initial_sigma_n(cols)
    nonlinear = _has_nonlinear_material(cols)

    ds = cols.sc - cols.s
    dz = cols.zc - cols.z_base
    r_arm = dz * cols.cos_alpha_s + ds * cols.sin_alpha_s
    e_norm = dz * cols.ns - ds * cols.nz

    fos = 1.0
    iters = 0
    converged = False
    n_eff = np.zeros(cols.n)
    c_arr = np.zeros(cols.n)
    tp_arr = np.zeros(cols.n)

    for it in range(1, (max_iter if nonlinear else 6) + 1):
        iters = it
        c_arr, tp_arr = _evaluate_strength(cols, sigma_n)

        def mom_res(f_trial: float, _c: np.ndarray = c_arr, _tp: np.ndarray = tp_arr) -> float:
            inv_f = 1.0 / f_trial
            v_down = cols.Wv - (cols.Qz_active + cols.Qz_passive * inv_f)
            h_down = cols.Fh + cols.Qs_active + cols.Qs_passive * inv_f
            n_tot_loc = v_down * cols.nz - h_down * cols.ns
            n_eff_loc = np.maximum(n_tot_loc - cols.u * cols.area_base, 0.0)
            s_mob_loc = (_c * cols.area_base + n_eff_loc * _tp) * inv_f
            m_res = float(np.sum(s_mob_loc * r_arm))
            m_drv = (
                float(np.sum(cols.Wv * ds))
                + float(np.sum(cols.Fh * (cols.zc - cols.z_cg)))
                + float(np.sum(cols.Ms_active + cols.Ms_passive * inv_f))
                + cols.crack_water_moment
                + float(np.sum((n_eff_loc + cols.u * cols.area_base) * e_norm))
            )
            scale = max(cols.total_weight, 1e-9) * max(1.0, float(np.mean(np.abs(r_arm))))
            return (m_res - m_drv) / scale

        f_new, sub_it = _solve_1d_root(mom_res, fos, tol=max(tolerance * 0.1, 1e-6))
        iters += sub_it
        if not np.isfinite(f_new) or f_new <= 0:
            return LEMResult3D(
                fos=np.nan,
                method="ordinary",
                converged=False,
                iterations=iters,
                azimuth_deg=cols.azimuth_deg,
                message="3D Ordinary moment balance did not converge to a positive root.",
            )

        inv_f = 1.0 / f_new
        v_down = cols.Wv - (cols.Qz_active + cols.Qz_passive * inv_f)
        h_down = cols.Fh + cols.Qs_active + cols.Qs_passive * inv_f
        n_tot = v_down * cols.nz - h_down * cols.ns
        n_eff = np.maximum(n_tot - cols.u * cols.area_base, 0.0)
        sn_new = n_eff / cols.area_base
        fos = f_new
        if not nonlinear or float(np.max(np.abs(sn_new - sigma_n))) <= tolerance * max(
            1.0, float(np.mean(sn_new))
        ):
            sigma_n = sn_new
            converged = True
            break
        sigma_n = 0.5 * sigma_n + 0.5 * sn_new

    if not converged:
        return LEMResult3D(
            fos=np.nan,
            method="ordinary",
            converged=False,
            iterations=iters,
            azimuth_deg=cols.azimuth_deg,
            message="Nonlinear strength iteration did not converge in 3D Ordinary.",
        )

    s_avail = c_arr * cols.area_base + n_eff * tp_arr
    s_mob = s_avail / fos
    res = LEMResult3D(
        fos=float(fos),
        method="ordinary",
        converged=True,
        iterations=iters,
        azimuth_deg=cols.azimuth_deg,
        m_alpha_min=float(np.min(cols.nz)),
        lam=0.0,
        moment_residual=0.0,
        base_normal_eff=n_eff,
        sigma_n_eff=sigma_n,
        base_shear_mob=s_mob,
        tau_mob=s_mob / cols.area_base,
    )
    return _check_admissibility_3d(
        res, cols, cols.nz, s_avail, n_eff, c_arr * cols.area_base, m_alpha_min=m_alpha_min
    )


def bishop_3d(
    cols: ColumnGrid,
    tolerance: float = 1e-4,
    max_iter: int = 75,
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT,
) -> LEMResult3D:
    """3D Simplified Bishop moment-equilibrium solver (Hungr 1987; Hungr et al. 1989)."""
    ord_res = ordinary_3d(cols, tolerance=tolerance, m_alpha_min=m_alpha_min)
    f_guess = ord_res.fos if ord_res.ok else 1.2
    sigma_n = (
        ord_res.sigma_n_eff.copy() if ord_res.sigma_n_eff is not None else _initial_sigma_n(cols)
    )
    nonlinear = _has_nonlinear_material(cols)

    fos = f_guess
    total_iters = 0
    converged = False
    c_arr, tp_arr = _evaluate_strength(cols, sigma_n)

    for outer in range(1, (max_iter if nonlinear else 2) + 1):
        c_arr, tp_arr = _evaluate_strength(cols, sigma_n)

        def mom_fn(f_trial: float, _c: np.ndarray = c_arr, _tp: np.ndarray = tp_arr) -> float:
            _, _, ma_loc, _, _, m_res = _column_state(cols, f_trial, 0.0, _c, _tp)
            if not np.isfinite(m_res) or float(np.min(ma_loc)) <= 0.02:
                return np.nan
            return m_res

        def valid_mom(f_val: float, _tp: np.ndarray = tp_arr) -> bool:
            return float(np.min(cols.nz + cols.sin_alpha_s * _tp / f_val)) > 0.02

        f_new, sub_it = _solve_1d_root(
            mom_fn,
            fos,
            tol=min(tolerance * 0.1, 1e-5),
            valid_fn=valid_mom,
        )
        total_iters += max(1, sub_it)
        if not np.isfinite(f_new) or f_new <= 0:
            return LEMResult3D(
                fos=np.nan,
                method="bishop",
                converged=False,
                iterations=total_iters,
                azimuth_deg=cols.azimuth_deg,
                message="3D Bishop moment equilibrium failed to converge.",
            )

        n_eff, s_mob, ma, sn_new, f_res, m_res = _column_state(cols, f_new, 0.0, c_arr, tp_arr)
        sn_pos = np.maximum(sn_new, 0.0)
        prev_fos = fos
        fos = f_new
        sn_tol = max(1e-2, tolerance * float(np.mean(sn_pos)))
        if not nonlinear or (
            outer >= 2
            and abs(fos - prev_fos) <= tolerance
            and float(np.max(np.abs(sn_pos - sigma_n))) <= sn_tol
        ):
            sigma_n = sn_pos
            converged = True
            break
        sigma_n = 0.5 * sigma_n + 0.5 * sn_pos

    if not converged:
        return LEMResult3D(
            fos=np.nan,
            method="bishop",
            converged=False,
            iterations=total_iters,
            azimuth_deg=cols.azimuth_deg,
            message="3D Bishop nonlinear strength loop exceeded max_iter.",
        )

    n_eff, s_mob, ma, sn_final, f_res, m_res = _column_state(cols, fos, 0.0, c_arr, tp_arr)
    s_avail = c_arr * cols.area_base + n_eff * tp_arr
    res = LEMResult3D(
        fos=float(fos),
        method="bishop",
        converged=True,
        iterations=total_iters,
        azimuth_deg=cols.azimuth_deg,
        lam=0.0,
        force_residual=float(f_res),
        moment_residual=float(m_res),
        base_normal_eff=n_eff,
        sigma_n_eff=sn_final,
        base_shear_mob=s_mob,
        tau_mob=s_mob / cols.area_base,
    )
    return _check_admissibility_3d(
        res, cols, ma, s_avail, n_eff, c_arr * cols.area_base, m_alpha_min=m_alpha_min
    )


def janbu_simplified_3d(
    cols: ColumnGrid,
    tolerance: float = 1e-4,
    max_iter: int = 75,
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT,
) -> LEMResult3D:
    """3D Simplified Janbu horizontal force-equilibrium solver (Hungr et al. 1989)."""
    sigma_n = _initial_sigma_n(cols)
    nonlinear = _has_nonlinear_material(cols)
    fos = 1.2
    total_iters = 0
    converged = False
    c_arr, tp_arr = _evaluate_strength(cols, sigma_n)

    for outer in range(1, (max_iter if nonlinear else 2) + 1):
        c_arr, tp_arr = _evaluate_strength(cols, sigma_n)

        def force_fn(f_trial: float, _c: np.ndarray = c_arr, _tp: np.ndarray = tp_arr) -> float:
            _, _, ma_loc, _, f_res, _ = _column_state(cols, f_trial, 0.0, _c, _tp)
            if not np.isfinite(f_res) or float(np.min(ma_loc)) <= 0.02:
                return np.nan
            return f_res

        def valid_force(f_val: float, _tp: np.ndarray = tp_arr) -> bool:
            return float(np.min(cols.nz + cols.sin_alpha_s * _tp / f_val)) > 0.02

        f_new, sub_it = _solve_1d_root(
            force_fn,
            fos,
            tol=min(tolerance * 0.1, 1e-5),
            valid_fn=valid_force,
        )
        total_iters += max(1, sub_it)
        if not np.isfinite(f_new) or f_new <= 0:
            return LEMResult3D(
                fos=np.nan,
                method="janbu_simplified",
                converged=False,
                iterations=total_iters,
                azimuth_deg=cols.azimuth_deg,
                message="3D Janbu force equilibrium failed to converge.",
            )

        _, _, _, sn_new, _, _ = _column_state(cols, f_new, 0.0, c_arr, tp_arr)
        sn_pos = np.maximum(sn_new, 0.0)
        prev_fos = fos
        fos = f_new
        sn_tol = max(1e-2, tolerance * float(np.mean(sn_pos)))
        if not nonlinear or (
            outer >= 2
            and abs(fos - prev_fos) <= tolerance
            and float(np.max(np.abs(sn_pos - sigma_n))) <= sn_tol
        ):
            sigma_n = sn_pos
            converged = True
            break
        sigma_n = 0.5 * sigma_n + 0.5 * sn_pos

    if not converged:
        return LEMResult3D(
            fos=np.nan,
            method="janbu_simplified",
            converged=False,
            iterations=total_iters,
            azimuth_deg=cols.azimuth_deg,
            message="3D Janbu nonlinear strength loop exceeded max_iter.",
        )

    n_eff, s_mob, ma, sn_final, f_res, m_res = _column_state(cols, fos, 0.0, c_arr, tp_arr)
    s_avail = c_arr * cols.area_base + n_eff * tp_arr
    res = LEMResult3D(
        fos=float(fos),
        method="janbu_simplified",
        converged=True,
        iterations=total_iters,
        azimuth_deg=cols.azimuth_deg,
        lam=0.0,
        force_residual=float(f_res),
        moment_residual=float(m_res),
        base_normal_eff=n_eff,
        sigma_n_eff=sn_final,
        base_shear_mob=s_mob,
        tau_mob=s_mob / cols.area_base,
    )
    return _check_admissibility_3d(
        res, cols, ma, s_avail, n_eff, c_arr * cols.area_base, m_alpha_min=m_alpha_min
    )


def _janbu_f0_factor(cols: ColumnGrid, c_arr: np.ndarray, tp_arr: np.ndarray) -> float:
    """Empirical Janbu (1973) correction factor f0 = 1 + b1 * (d/L - 1.4*(d/L)^2) along central strip."""
    t_mid = float(0.5 * (np.min(cols.t) + np.max(cols.t)))
    strip = np.abs(cols.t - t_mid) <= max(cols.dy * 0.75, 1e-6)
    if int(np.count_nonzero(strip)) < 2:
        strip = np.ones(cols.n, dtype=bool)
    order = np.argsort(cols.s[strip])
    s_line = cols.s[strip][order]
    zb_line = cols.z_base[strip][order]
    zt_line = cols.z_top[strip][order]
    # Chord from uphill entry (s_min, z_top) to downhill exit (s_max, z_top)
    p0 = np.array([float(s_line[0]) - 0.5 * cols.dx, float(zt_line[0])], dtype=float)
    p1 = np.array([float(s_line[-1]) + 0.5 * cols.dx, float(zt_line[-1])], dtype=float)
    chord = p1 - p0
    length_l = float(np.hypot(chord[0], chord[1]))
    if length_l <= 1e-6:
        return 1.0
    # Upward unit normal to chord (from uphill s_min to downhill s_max, chord[0] > 0)
    n_up = np.array([-chord[1], chord[0]], dtype=float) / length_l
    pts = np.column_stack([s_line, zb_line])
    d_max = max(0.0, -float(np.min((pts - p0[None, :]) @ n_up)))
    ratio = min(d_max / length_l, 1.0 / 2.8)

    has_c = bool(np.any(c_arr > 1e-6))
    has_phi = bool(np.any(tp_arr > 1e-6))
    if has_c and not has_phi:
        b1 = 0.69
    elif has_phi and not has_c:
        b1 = 0.31
    else:
        b1 = 0.50
    return float(max(1.0, 1.0 + b1 * (ratio - 1.4 * ratio * ratio)))


def janbu_corrected_3d(
    cols: ColumnGrid,
    tolerance: float = 1e-4,
    max_iter: int = 75,
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT,
) -> LEMResult3D:
    """3D Corrected Janbu solver (f0 * F_Janbu_Simplified)."""
    base = janbu_simplified_3d(
        cols, tolerance=tolerance, max_iter=max_iter, m_alpha_min=m_alpha_min
    )
    if not base.ok or base.sigma_n_eff is None:
        base.method = "janbu_corrected"
        return base
    c_arr, tp_arr = _evaluate_strength(cols, base.sigma_n_eff)
    f0 = _janbu_f0_factor(cols, c_arr, tp_arr)
    base.method = "janbu_corrected"
    base.janbu_f0 = f0
    base.fos = float(base.fos * f0)
    if base.base_shear_mob is not None:
        base.base_shear_mob = base.base_shear_mob / f0
    if base.tau_mob is not None:
        base.tau_mob = base.tau_mob / f0
    return base


def spencer_3d(
    cols: ColumnGrid,
    tolerance: float = 1e-4,
    max_iter: int = 75,
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT,
) -> LEMResult3D:
    """3D Rigorous Spencer coupled force & moment equilibrium solver (constant intercolumn lambda)."""
    b_res = bishop_3d(cols, tolerance=tolerance, max_iter=max_iter, m_alpha_min=m_alpha_min)
    j_res = janbu_simplified_3d(
        cols, tolerance=tolerance, max_iter=max_iter, m_alpha_min=m_alpha_min
    )

    if b_res.ok and b_res.sigma_n_eff is not None:
        f_seed = b_res.fos
        sigma_n = b_res.sigma_n_eff.copy()
    elif j_res.ok and j_res.sigma_n_eff is not None:
        f_seed = j_res.fos
        sigma_n = j_res.sigma_n_eff.copy()
    else:
        f_seed = 1.2
        sigma_n = _initial_sigma_n(cols)

    nonlinear = _has_nonlinear_material(cols)
    fos = f_seed
    lam = 0.0
    total_iters = 0
    converged = False
    res_tol = min(max(tolerance * 0.05, 1e-6), 1e-4)
    c_arr, tp_arr = _evaluate_strength(cols, sigma_n)

    for outer in range(1, (max_iter if nonlinear else 2) + 1):
        c_arr, tp_arr = _evaluate_strength(cols, sigma_n)

        def residuals_vec(
            vars_fl: np.ndarray, _c: np.ndarray = c_arr, _tp: np.ndarray = tp_arr
        ) -> np.ndarray:
            f_v, l_v = float(vars_fl[0]), float(vars_fl[1])
            if f_v <= 0.02 or abs(l_v) > 2.5:
                return np.array([1e3, 1e3], dtype=float)
            _, _, ma_loc, _, fr_loc, mr_loc = _column_state(cols, f_v, l_v, _c, _tp)
            if not (np.isfinite(fr_loc) and np.isfinite(mr_loc)) or float(np.min(ma_loc)) <= 0.02:
                return np.array([1e3, 1e3], dtype=float)
            return np.array([fr_loc, mr_loc], dtype=float)

        # Stage 1: Fast Newton-Powell / hybr from current (fos, lam) and nearby lambda seeds
        solved_pair: tuple[float, float] | None = None
        for lam_init in (lam, 0.08, -0.08, 0.20, -0.20, 0.40, -0.40):
            sol = root(
                residuals_vec, np.array([fos, lam_init], dtype=float), method="hybr", tol=1e-9
            )
            total_iters += int(getattr(sol, "nfev", 5))
            f_cand, l_cand = float(sol.x[0]), float(sol.x[1])
            if f_cand > 0.02 and abs(l_cand) <= 2.0:
                _, _, ma_c, _, fr_c, mr_c = _column_state(cols, f_cand, l_cand, c_arr, tp_arr)
                if (
                    np.isfinite(fr_c)
                    and np.isfinite(mr_c)
                    and max(abs(fr_c), abs(mr_c)) <= res_tol
                    and float(np.min(ma_c)) > 0.02
                ):
                    solved_pair = (f_cand, l_cand)
                    break

        # Stage 2: 1D continuation scan over lambda in [-0.85, 0.85] solving Fm(lambda) then rf=0
        if solved_pair is None:
            lam_grid = np.linspace(-0.85, 0.85, 45)
            fm_vals = np.full_like(lam_grid, np.nan)
            rf_vals = np.full_like(lam_grid, np.nan)
            for idx_l, l_try in enumerate(lam_grid):

                def m_only(
                    f_try: float,
                    _l: float = float(l_try),
                    _c: np.ndarray = c_arr,
                    _tp: np.ndarray = tp_arr,
                ) -> float:
                    _, _, ma_l, _, _, mr_l = _column_state(cols, f_try, _l, _c, _tp)
                    if not np.isfinite(mr_l) or float(np.min(ma_l)) <= 0.02:
                        return np.nan
                    return mr_l

                fm_k, it_k = _solve_1d_root(m_only, fos, tol=1e-5)
                total_iters += it_k
                if np.isfinite(fm_k):
                    _, _, ma_k, _, fr_k, _ = _column_state(cols, fm_k, float(l_try), c_arr, tp_arr)
                    if np.isfinite(fr_k) and float(np.min(ma_k)) > 0.02:
                        fm_vals[idx_l] = fm_k
                        rf_vals[idx_l] = fr_k

            # Rank sign-change brackets closest to lambda = 0
            brackets: list[tuple[float, int]] = []
            for k in range(len(lam_grid) - 1):
                if (
                    np.isfinite(rf_vals[k])
                    and np.isfinite(rf_vals[k + 1])
                    and rf_vals[k] * rf_vals[k + 1] <= 0.0
                ):
                    brackets.append((abs(0.5 * (lam_grid[k] + lam_grid[k + 1])), k))

            for _, k in sorted(brackets):
                l_mid = float(0.5 * (lam_grid[k] + lam_grid[k + 1]))
                f_mid = float(0.5 * (fm_vals[k] + fm_vals[k + 1]))
                sol = root(
                    residuals_vec, np.array([f_mid, l_mid], dtype=float), method="hybr", tol=1e-10
                )
                total_iters += int(getattr(sol, "nfev", 5))
                f_cand, l_cand = float(sol.x[0]), float(sol.x[1])
                if f_cand > 0.02 and abs(l_cand) <= 2.0:
                    _, _, ma_c, _, fr_c, mr_c = _column_state(cols, f_cand, l_cand, c_arr, tp_arr)
                    if (
                        np.isfinite(fr_c)
                        and np.isfinite(mr_c)
                        and max(abs(fr_c), abs(mr_c)) <= res_tol
                        and float(np.min(ma_c)) > 0.02
                    ):
                        solved_pair = (f_cand, l_cand)
                        break

        if solved_pair is None:
            return LEMResult3D(
                fos=np.nan,
                method="spencer",
                converged=False,
                iterations=total_iters,
                azimuth_deg=cols.azimuth_deg,
                message="3D Spencer coupled force-moment equilibrium did not converge.",
            )

        f_new, l_new = solved_pair
        _, _, _, sn_new, _, _ = _column_state(cols, f_new, l_new, c_arr, tp_arr)
        sn_pos = np.maximum(sn_new, 0.0)
        prev_fos = fos
        fos, lam = f_new, l_new
        sn_tol = max(1e-2, tolerance * float(np.mean(sn_pos)))
        if not nonlinear or (
            outer >= 2
            and abs(fos - prev_fos) <= tolerance
            and float(np.max(np.abs(sn_pos - sigma_n))) <= sn_tol
        ):
            sigma_n = sn_pos
            converged = True
            break
        sigma_n = 0.5 * sigma_n + 0.5 * sn_pos

    if not converged:
        return LEMResult3D(
            fos=np.nan,
            method="spencer",
            converged=False,
            iterations=total_iters,
            azimuth_deg=cols.azimuth_deg,
            message="3D Spencer nonlinear strength loop exceeded max_iter.",
        )

    n_eff, s_mob, ma, sn_final, f_res, m_res = _column_state(cols, fos, lam, c_arr, tp_arr)
    s_avail = c_arr * cols.area_base + n_eff * tp_arr
    res = LEMResult3D(
        fos=float(fos),
        method="spencer",
        converged=True,
        iterations=total_iters,
        azimuth_deg=cols.azimuth_deg,
        lam=float(-lam),  # Standard Slide2/GeoStudio intercolumn inclination sign
        force_residual=float(f_res),
        moment_residual=float(m_res),
        transverse_residual=_transverse_force_residual(cols, n_eff, fos),
        base_normal_eff=n_eff,
        sigma_n_eff=sn_final,
        base_shear_mob=s_mob,
        tau_mob=s_mob / cols.area_base,
    )
    return _check_admissibility_3d(
        res, cols, ma, s_avail, n_eff, c_arr * cols.area_base, m_alpha_min=m_alpha_min
    )


INTERCOLUMN_FUNCTIONS = ("constant", "half-sine", "clipped-sine", "trapezoidal")


def intercolumn_function_3d(
    kind: str,
    xi: np.ndarray,
    start: float = 0.5,
    end: float = 0.5,
) -> np.ndarray:
    """Evaluate intercolumn shear-to-normal function f(xi) on normalized sliding coordinate xi in [0, 1]."""
    t = np.clip(np.asarray(xi, dtype=float), 0.0, 1.0)
    if kind == "constant":
        return np.ones_like(t)
    if kind == "half-sine":
        return np.sin(np.pi * t)
    if kind == "clipped-sine":
        a0 = float(np.arcsin(np.clip(start, 0.0, 1.0)))
        a1 = float(np.pi - np.arcsin(np.clip(end, 0.0, 1.0)))
        return np.sin(a0 + (a1 - a0) * t)
    if kind == "trapezoidal":
        return np.clip(np.minimum(t, 1.0 - t) / 0.25, 0.0, 1.0)
    raise ValueError(
        f"Unknown intercolumn function {kind!r} (expected {list(INTERCOLUMN_FUNCTIONS)})."
    )


def _transverse_force_residual(cols: ColumnGrid, n_eff: np.ndarray, fos: float) -> float:
    """Normalized net transverse horizontal force sum(F_t) / W_total in +t direction."""
    scale_f = max(cols.total_weight, 1e-9)
    n_tot = n_eff + cols.u * cols.area_base
    qt_a = cols.Qt_active if len(cols.Qt_active) == cols.n else np.zeros(cols.n)
    qt_p = cols.Qt_passive if len(cols.Qt_passive) == cols.n else np.zeros(cols.n)
    ft_net = float(np.sum(n_tot * cols.nt + qt_a + qt_p / max(fos, 1e-6)))
    return ft_net / scale_f


def _build_gle_strips(
    cols: ColumnGrid,
    interslice: str,
    f_start: float = 0.5,
    f_end: float = 0.5,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Partition active columns into longitudinal strips ordered uphill->downhill and compute (f_L, f_R)."""
    psi = float(np.radians(cols.azimuth_deg))
    sx, sy = abs(float(np.cos(psi))), abs(float(np.sin(psi)))
    ds_col = max(cols.dx * sx + cols.dy * sy, 1e-6)
    s_up = float(np.min(cols.s) - 0.5 * ds_col)
    s_dn = float(np.max(cols.s) + 0.5 * ds_col)
    s_span = max(s_dn - s_up, 1e-6)

    xi_l = np.clip((cols.s - 0.5 * ds_col - s_up) / s_span, 0.0, 1.0)
    xi_r = np.clip((cols.s + 0.5 * ds_col - s_up) / s_span, 0.0, 1.0)
    f_l = intercolumn_function_3d(interslice, xi_l, start=f_start, end=f_end)
    f_r = intercolumn_function_3d(interslice, xi_r, start=f_start, end=f_end)

    if sy < 1e-6:
        strip_key = cols.iy
    elif sx < 1e-6:
        strip_key = cols.ix
    else:
        dt_bin = max(float(np.hypot(cols.dx * sy, cols.dy * sx)), 1e-6)
        strip_key = np.round((cols.t - float(np.min(cols.t))) / dt_bin).astype(int)

    strips: list[np.ndarray] = []
    for key in np.unique(strip_key):
        idx = np.flatnonzero(strip_key == key)
        order = idx[np.argsort(cols.s[idx], kind="mergesort")]
        strips.append(order)
    return strips, f_l, f_r


def _gle_column_state(
    cols: ColumnGrid,
    fos: float,
    lam: float,
    c_arr: np.ndarray,
    tp_arr: np.ndarray,
    strips: list[np.ndarray],
    f_l: np.ndarray,
    f_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Strip-wise 3D GLE state (N_eff, S_mob, m_alpha, sigma_n_eff, force_res, moment_res)."""
    inv_f = 1.0 / fos
    v_down = cols.Wv - (cols.Qz_active + cols.Qz_passive * inv_f)
    h_down = cols.Fh + cols.Qs_active + cols.Qs_passive * inv_f
    u_force = cols.u * cols.area_base
    c_force = c_arr * cols.area_base

    ma0 = cols.nz + cols.sin_alpha_s * tp_arr * inv_f
    a_l = lam * f_l
    a_r = lam * f_r
    ma_eff = (cols.nz - a_r * cols.ns) + (
        cols.sin_alpha_s + a_r * cols.cos_alpha_s
    ) * tp_arr * inv_f
    if np.any(np.abs(ma0) < 1e-9) or np.any(np.abs(ma_eff) < 1e-9):
        nan_v = np.full(cols.n, np.nan)
        return nan_v, nan_v, ma_eff, nan_v, np.nan, np.nan

    k_fac = tp_arr * cols.cos_alpha_s * inv_f - cols.ns
    p0 = (v_down - u_force * cols.nz - c_force * cols.sin_alpha_s * inv_f) / ma0
    den = 1.0 + k_fac * a_r / ma0
    if np.any(np.abs(den) < 1e-9):
        nan_v = np.full(cols.n, np.nan)
        return nan_v, nan_v, ma_eff, nan_v, np.nan, np.nan

    rec_a = (1.0 + k_fac * a_l / ma0) / den
    rec_b = (c_force * cols.cos_alpha_s * inv_f - u_force * cols.ns - h_down + k_fac * p0) / den

    n_eff = np.empty(cols.n, dtype=float)
    e_free_sum = 0.0
    n_strips = max(1, len(strips))
    e_uphill_per_strip = -cols.crack_water_force_s / n_strips

    ra_list = rec_a.tolist()
    rb_list = rec_b.tolist()
    al_list = a_l.tolist()
    ar_list = a_r.tolist()
    p0_list = p0.tolist()
    ma0_list = ma0.tolist()

    for strip_idx in strips:
        e_prev = e_uphill_per_strip
        for idx_c in strip_idx.tolist():
            e_cur = ra_list[idx_c] * e_prev + rb_list[idx_c]
            n_eff[idx_c] = (
                p0_list[idx_c]
                + (al_list[idx_c] * e_prev - ar_list[idx_c] * e_cur) / ma0_list[idx_c]
            )
            e_prev = e_cur
        e_free_sum += e_prev

    if not np.all(np.isfinite(n_eff)):
        nan_v = np.full(cols.n, np.nan)
        return nan_v, nan_v, ma_eff, nan_v, np.nan, np.nan

    n_tot = n_eff + u_force
    s_avail = c_force + n_eff * tp_arr
    s_mob = s_avail * inv_f
    sigma_n_eff = n_eff / cols.area_base

    scale_f = max(cols.total_weight, 1e-9)
    force_res = e_free_sum * float(np.hypot(1.0, lam)) / scale_f

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
    return n_eff, s_mob, ma_eff, sigma_n_eff, force_res, moment_res


def gle_mp_3d(
    cols: ColumnGrid,
    tolerance: float = 1e-4,
    max_iter: int = 75,
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT,
    interslice: str = "half-sine",
    f_start: float = 0.5,
    f_end: float = 0.5,
) -> LEMResult3D:
    """3D General Limit Equilibrium / Morgenstern-Price coupled force & moment solver."""
    if interslice == "constant":
        sp = spencer_3d(cols, tolerance=tolerance, max_iter=max_iter, m_alpha_min=m_alpha_min)
        sp.method = "gle_mp"
        sp.interslice_func = "constant"
        return sp

    strips, f_l, f_r = _build_gle_strips(cols, interslice=interslice, f_start=f_start, f_end=f_end)
    b_res = bishop_3d(cols, tolerance=tolerance, max_iter=max_iter, m_alpha_min=m_alpha_min)
    j_res = janbu_simplified_3d(
        cols, tolerance=tolerance, max_iter=max_iter, m_alpha_min=m_alpha_min
    )
    if b_res.ok and b_res.sigma_n_eff is not None:
        f_seed = b_res.fos
        sigma_n = b_res.sigma_n_eff.copy()
    elif j_res.ok and j_res.sigma_n_eff is not None:
        f_seed = j_res.fos
        sigma_n = j_res.sigma_n_eff.copy()
    else:
        f_seed = 1.2
        sigma_n = _initial_sigma_n(cols)

    nonlinear = _has_nonlinear_material(cols)
    fos = f_seed
    lam = 0.0
    total_iters = 0
    converged = False
    res_tol = min(max(tolerance * 0.05, 1e-6), 1e-4)
    c_arr, tp_arr = _evaluate_strength(cols, sigma_n)

    for outer in range(1, (max_iter if nonlinear else 2) + 1):
        c_arr, tp_arr = _evaluate_strength(cols, sigma_n)

        def residuals_vec(
            vars_fl: np.ndarray, _c: np.ndarray = c_arr, _tp: np.ndarray = tp_arr
        ) -> np.ndarray:
            f_v, l_v = float(vars_fl[0]), float(vars_fl[1])
            if f_v <= 0.02 or abs(l_v) > 3.0:
                return np.array([1e3, 1e3], dtype=float)
            _, _, ma_loc, _, fr_loc, mr_loc = _gle_column_state(
                cols, f_v, l_v, _c, _tp, strips, f_l, f_r
            )
            if not (np.isfinite(fr_loc) and np.isfinite(mr_loc)) or float(np.min(ma_loc)) <= 0.02:
                return np.array([1e3, 1e3], dtype=float)
            return np.array([fr_loc, mr_loc], dtype=float)

        solved_pair: tuple[float, float] | None = None
        for lam_init in (lam, 0.10, -0.10, 0.25, -0.25, 0.50, -0.50):
            sol = root(
                residuals_vec, np.array([fos, lam_init], dtype=float), method="hybr", tol=1e-9
            )
            total_iters += int(getattr(sol, "nfev", 5))
            f_cand, l_cand = float(sol.x[0]), float(sol.x[1])
            if f_cand > 0.02 and abs(l_cand) <= 2.5:
                _, _, ma_c, _, fr_c, mr_c = _gle_column_state(
                    cols, f_cand, l_cand, c_arr, tp_arr, strips, f_l, f_r
                )
                if (
                    np.isfinite(fr_c)
                    and np.isfinite(mr_c)
                    and max(abs(fr_c), abs(mr_c)) <= res_tol
                    and float(np.min(ma_c)) > 0.02
                ):
                    solved_pair = (f_cand, l_cand)
                    break

        if solved_pair is None:
            lam_grid = np.linspace(-1.1, 1.1, 45)
            fm_vals = np.full_like(lam_grid, np.nan)
            rf_vals = np.full_like(lam_grid, np.nan)
            for idx_l, l_try in enumerate(lam_grid):

                def m_only(
                    f_try: float,
                    _l: float = float(l_try),
                    _c: np.ndarray = c_arr,
                    _tp: np.ndarray = tp_arr,
                ) -> float:
                    _, _, ma_l, _, _, mr_l = _gle_column_state(
                        cols, f_try, _l, _c, _tp, strips, f_l, f_r
                    )
                    if not np.isfinite(mr_l) or float(np.min(ma_l)) <= 0.02:
                        return np.nan
                    return mr_l

                fm_k, it_k = _solve_1d_root(m_only, fos, tol=1e-5)
                total_iters += it_k
                if np.isfinite(fm_k):
                    _, _, ma_k, _, fr_k, _ = _gle_column_state(
                        cols, fm_k, float(l_try), c_arr, tp_arr, strips, f_l, f_r
                    )
                    if np.isfinite(fr_k) and float(np.min(ma_k)) > 0.02:
                        fm_vals[idx_l] = fm_k
                        rf_vals[idx_l] = fr_k

            brackets: list[tuple[float, int]] = []
            for k in range(len(lam_grid) - 1):
                if (
                    np.isfinite(rf_vals[k])
                    and np.isfinite(rf_vals[k + 1])
                    and rf_vals[k] * rf_vals[k + 1] <= 0.0
                ):
                    brackets.append((abs(0.5 * (lam_grid[k] + lam_grid[k + 1])), k))

            for _, k in sorted(brackets):
                l_mid = float(0.5 * (lam_grid[k] + lam_grid[k + 1]))
                f_mid = float(0.5 * (fm_vals[k] + fm_vals[k + 1]))
                sol = root(
                    residuals_vec, np.array([f_mid, l_mid], dtype=float), method="hybr", tol=1e-10
                )
                total_iters += int(getattr(sol, "nfev", 5))
                f_cand, l_cand = float(sol.x[0]), float(sol.x[1])
                if f_cand > 0.02 and abs(l_cand) <= 2.5:
                    _, _, ma_c, _, fr_c, mr_c = _gle_column_state(
                        cols, f_cand, l_cand, c_arr, tp_arr, strips, f_l, f_r
                    )
                    if (
                        np.isfinite(fr_c)
                        and np.isfinite(mr_c)
                        and max(abs(fr_c), abs(mr_c)) <= res_tol
                        and float(np.min(ma_c)) > 0.02
                    ):
                        solved_pair = (f_cand, l_cand)
                        break

        if solved_pair is None:
            return LEMResult3D(
                fos=np.nan,
                method="gle_mp",
                converged=False,
                iterations=total_iters,
                azimuth_deg=cols.azimuth_deg,
                interslice_func=interslice,
                message=f"3D GLE ({interslice}) coupled force-moment equilibrium did not converge.",
            )

        f_new, l_new = solved_pair
        _, _, _, sn_new, _, _ = _gle_column_state(
            cols, f_new, l_new, c_arr, tp_arr, strips, f_l, f_r
        )
        sn_pos = np.maximum(sn_new, 0.0)
        prev_fos = fos
        fos, lam = f_new, l_new
        sn_tol = max(1e-2, tolerance * float(np.mean(sn_pos)))
        if not nonlinear or (
            outer >= 2
            and abs(fos - prev_fos) <= tolerance
            and float(np.max(np.abs(sn_pos - sigma_n))) <= sn_tol
        ):
            sigma_n = sn_pos
            converged = True
            break
        sigma_n = 0.5 * sigma_n + 0.5 * sn_pos

    if not converged:
        return LEMResult3D(
            fos=np.nan,
            method="gle_mp",
            converged=False,
            iterations=total_iters,
            azimuth_deg=cols.azimuth_deg,
            interslice_func=interslice,
            message=f"3D GLE ({interslice}) nonlinear strength loop exceeded max_iter.",
        )

    n_eff, s_mob, ma, sn_final, f_res, m_res = _gle_column_state(
        cols, fos, lam, c_arr, tp_arr, strips, f_l, f_r
    )
    s_avail = c_arr * cols.area_base + n_eff * tp_arr
    res = LEMResult3D(
        fos=float(fos),
        method="gle_mp",
        converged=True,
        iterations=total_iters,
        azimuth_deg=cols.azimuth_deg,
        lam=float(-lam),
        force_residual=float(f_res),
        moment_residual=float(m_res),
        transverse_residual=_transverse_force_residual(cols, n_eff, fos),
        interslice_func=interslice,
        base_normal_eff=n_eff,
        sigma_n_eff=sn_final,
        base_shear_mob=s_mob,
        tau_mob=s_mob / cols.area_base,
    )
    return _check_admissibility_3d(
        res, cols, ma, s_avail, n_eff, c_arr * cols.area_base, m_alpha_min=m_alpha_min
    )


def cheng_yip_3d(
    cols: ColumnGrid,
    tolerance: float = 1e-4,
    max_iter: int = 75,
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT,
    interslice: str = "constant",
) -> LEMResult3D:
    """3D Asymmetric Cheng & Yip (2007) solver finding equilibrium sliding azimuth where sum(F_t) = 0."""
    base_az = cols.azimuth_deg
    r0 = gle_mp_3d(
        cols,
        tolerance=tolerance,
        max_iter=max_iter,
        m_alpha_min=m_alpha_min,
        interslice=interslice,
    )
    if r0.ok and abs(r0.transverse_residual) <= max(tolerance, 5e-4):
        r0.method = "cheng_yip"
        return r0

    total_iters = r0.iterations
    best_eval: tuple[float, LEMResult3D] | None = (
        (abs(r0.transverse_residual), r0) if r0.ok else None
    )

    def eval_delta(d_az: float) -> float:
        nonlocal total_iters, best_eval
        c_rot = cols if abs(d_az) < 1e-9 else cols.reproject_azimuth(base_az + d_az)
        res_d = gle_mp_3d(
            c_rot,
            tolerance=tolerance,
            max_iter=max_iter,
            m_alpha_min=m_alpha_min,
            interslice=interslice,
        )
        total_iters += max(1, res_d.iterations)
        if not res_d.ok:
            return np.nan
        rt = res_d.transverse_residual
        if best_eval is None or abs(rt) < best_eval[0]:
            best_eval = (abs(rt), res_d)
        return rt

    deltas = np.linspace(-75.0, 75.0, 21)
    rt_vals = np.array([eval_delta(float(d)) for d in deltas], dtype=float)
    brackets: list[tuple[float, int]] = []
    for k in range(len(deltas) - 1):
        if (
            np.isfinite(rt_vals[k])
            and np.isfinite(rt_vals[k + 1])
            and rt_vals[k] * rt_vals[k + 1] <= 0.0
        ):
            brackets.append((abs(0.5 * (deltas[k] + deltas[k + 1])), k))

    for _, k in sorted(brackets):
        try:
            d_star = float(
                brentq(eval_delta, float(deltas[k]), float(deltas[k + 1]), xtol=1e-3, maxiter=25)
            )
        except (ValueError, RuntimeError):
            continue
        c_final = cols.reproject_azimuth(base_az + d_star)
        r_final = gle_mp_3d(
            c_final,
            tolerance=tolerance,
            max_iter=max_iter,
            m_alpha_min=m_alpha_min,
            interslice=interslice,
        )
        r_final.method = "cheng_yip"
        r_final.iterations = total_iters + r_final.iterations
        if r_final.ok and abs(r_final.transverse_residual) <= 5e-3:
            return r_final

    if best_eval is not None and best_eval[0] <= 1e-2:
        out = best_eval[1]
        out.method = "cheng_yip"
        out.iterations = total_iters
        return out

    return LEMResult3D(
        fos=np.nan,
        method="cheng_yip",
        converged=False,
        iterations=total_iters,
        azimuth_deg=base_az,
        interslice_func=interslice,
        message="3D Cheng & Yip transverse + longitudinal force-moment equilibrium did not converge.",
    )


METHODS_3D: dict[str, tuple[str, Callable[..., LEMResult3D]]] = {
    "ordinary": ("3D Ordinary (Fellenius)", ordinary_3d),
    "bishop": ("3D Bishop Simplified", bishop_3d),
    "janbu_simplified": ("3D Janbu Simplified", janbu_simplified_3d),
    "janbu_corrected": ("3D Janbu Corrected", janbu_corrected_3d),
    "spencer": ("3D Spencer", spencer_3d),
    "gle_mp": ("3D GLE / Morgenstern-Price", gle_mp_3d),
    "cheng_yip": ("3D Cheng & Yip Asymmetric", cheng_yip_3d),
}


def solve_surface_3d(
    cols: ColumnGrid,
    method: str = "bishop",
    tolerance: float = 1e-4,
    max_iter: int = 75,
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT,
    interslice: str = "half-sine",
) -> LEMResult3D:
    if method not in METHODS_3D:
        raise ValueError(f"Unknown 3D LEM method {method!r} (expected one of {list(METHODS_3D)}).")
    _, fn = METHODS_3D[method]
    if method in ("gle_mp", "cheng_yip"):
        use_fn = (
            interslice
            if method == "gle_mp"
            else (interslice if interslice in INTERCOLUMN_FUNCTIONS else "constant")
        )
        return fn(
            cols,
            tolerance=tolerance,
            max_iter=max_iter,
            m_alpha_min=m_alpha_min,
            interslice=use_fn,
        )
    return fn(cols, tolerance=tolerance, max_iter=max_iter, m_alpha_min=m_alpha_min)
