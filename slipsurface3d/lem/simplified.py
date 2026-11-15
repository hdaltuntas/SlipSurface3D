"""Simplified 3D Method of Columns solvers (Ordinary, Bishop, Janbu)."""

from __future__ import annotations

import numpy as np

from slipsurface3d.lem.columns import ColumnGrid
from slipsurface3d.lem.common import (
    M_ALPHA_MIN_DEFAULT,
    LEMResult3D,
    _check_admissibility_3d,
    _column_state,
    _evaluate_strength,
    _has_nonlinear_material,
    _initial_sigma_n,
    _solve_1d_root,
)


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
