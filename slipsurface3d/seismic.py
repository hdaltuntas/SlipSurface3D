"""3D Pseudo-Static Critical Yield Acceleration (k_y) and Newmark Permanent Displacement."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from slipsurface3d.analysis import run_analysis_3d
from slipsurface3d.geometry.surfaces import SlipSurface3D
from slipsurface3d.lem.columns import discretize_columns
from slipsurface3d.lem.solvers import solve_surface_3d
from slipsurface3d.project import Project3D

G_ACCEL_MPS2 = 9.80665


@dataclass
class YieldAccelerationResult3D:
    """Result of 3D critical seismic yield acceleration search (FoS(k_y) = 1.0)."""

    ky_g: float
    static_fos: float
    fos_at_ky: float
    converged: bool
    iterations: int
    method: str
    message: str = ""


@dataclass
class NewmarkResult3D:
    """Empirical and/or time-history Newmark permanent 3D block displacement [cm]."""

    ky_g: float
    pga_g: float
    mw: float
    jibson_2007_median_cm: float
    jibson_2007_p16_cm: float
    jibson_2007_p84_cm: float
    ambraseys_menu_cm: float
    time_history_cm: float | None = None


def find_yield_acceleration_3d(
    project: Project3D,
    surface: SlipSurface3D | None = None,
    method: str | None = None,
    kh_max: float = 1.5,
    tol: float = 1e-4,
) -> YieldAccelerationResult3D:
    """Solve for horizontal seismic coefficient ``k_y`` where 3D Factor of Safety equals 1.0."""
    project.validate()
    an = project.analysis
    use_method = method or (
        project.surface_config.primary_method
        if project.surface_config.primary_method in an.methods
        else an.methods[0]
    )

    p_stat = project.copy()
    p_stat.analysis.kh = 0.0
    if surface is None:
        det = run_analysis_3d(p_stat)
        crit_surf = det.critical_surface
    else:
        crit_surf = surface

    mdl = p_stat.build_model()
    base_cols = discretize_columns(
        mdl,
        crit_surf,
        p_stat.build_materials(),
        opts=p_stat.build_column_options(kv_override=0.0),
        material_colors=p_stat.material_colors(),
    )

    evals = 0

    def fos_at(kh_val: float) -> float:
        nonlocal evals
        evals += 1
        c_loc = copy.copy(base_cols)
        c_loc.kh = float(kh_val)
        res = solve_surface_3d(
            c_loc,
            method=use_method,
            tolerance=min(an.tolerance, tol * 0.5),
            max_iter=an.max_iter,
            m_alpha_min=an.m_alpha_min,
            interslice=an.gle_interslice,
        )
        return float(res.fos) if res.ok else np.nan

    f0 = fos_at(0.0)
    if not np.isfinite(f0):
        return YieldAccelerationResult3D(
            ky_g=np.nan,
            static_fos=np.nan,
            fos_at_ky=np.nan,
            converged=False,
            iterations=evals,
            method=use_method,
            message="Static (kh=0) 3D equilibrium did not converge.",
        )
    if f0 <= 1.0:
        return YieldAccelerationResult3D(
            ky_g=0.0,
            static_fos=f0,
            fos_at_ky=f0,
            converged=True,
            iterations=evals,
            method=use_method,
            message="Slope is already at or below limit equilibrium under static conditions (FoS <= 1.0).",
        )

    # Bracket k_y where FoS(k_h) crosses below 1.0
    kh_grid = np.concatenate([np.linspace(0.02, 0.50, 13), np.linspace(0.60, kh_max, 10)])
    lo_kh = 0.0
    hi_kh: float | None = None
    last_valid_kh = 0.0
    for kh_try in kh_grid:
        fk = fos_at(float(kh_try))
        if np.isfinite(fk):
            if fk <= 1.0:
                hi_kh = float(kh_try)
                break
            lo_kh = float(kh_try)
            last_valid_kh = float(kh_try)
        else:
            # Solver hit non-convergence above limit load; probe finer between last_valid_kh and kh_try
            for sub_kh in np.linspace(last_valid_kh, float(kh_try), 12)[1:]:
                f_sub = fos_at(float(sub_kh))
                if np.isfinite(f_sub):
                    if f_sub <= 1.0:
                        hi_kh = float(sub_kh)
                        break
                    lo_kh = float(sub_kh)
            if hi_kh is not None:
                break

    if hi_kh is None:
        return YieldAccelerationResult3D(
            ky_g=np.nan,
            static_fos=f0,
            fos_at_ky=np.nan,
            converged=False,
            iterations=evals,
            method=use_method,
            message=f"FoS remained > 1.0 up to kh_max = {kh_max:.2f}g (or solver stalled).",
        )

    def root_fn(k: float) -> float:
        fv = fos_at(k)
        return (fv - 1.0) if np.isfinite(fv) else -0.25

    ky_star = float(brentq(root_fn, lo_kh, hi_kh, xtol=tol, maxiter=40))
    f_star = fos_at(ky_star)
    return YieldAccelerationResult3D(
        ky_g=ky_star,
        static_fos=f0,
        fos_at_ky=f_star,
        converged=bool(np.isfinite(f_star) and abs(f_star - 1.0) <= 5e-3),
        iterations=evals,
        method=use_method,
        message="Converged.",
    )


def integrate_newmark_accelerogram(
    time_s: np.ndarray,
    accel_g: np.ndarray,
    ky_g: float,
    both_polarities: bool = True,
) -> float:
    """Exact rigid-block Newmark sliding-block integration returning permanent displacement [cm]."""
    t = np.asarray(time_s, dtype=float).ravel()
    a = np.asarray(accel_g, dtype=float).ravel()
    if len(t) != len(a) or len(t) < 2:
        raise ValueError(
            "Accelerogram requires matching 1D time_s and accel_g arrays of length >= 2."
        )
    if ky_g <= 0.0:
        return float("inf")

    def _integrate_one(acc_g: np.ndarray) -> float:
        vel = 0.0  # [m/s]
        disp = 0.0  # [m]
        for i in range(len(t) - 1):
            dt = float(t[i + 1] - t[i])
            if dt <= 0.0:
                continue
            a_net_0 = (float(acc_g[i]) - ky_g) * G_ACCEL_MPS2
            a_net_1 = (float(acc_g[i + 1]) - ky_g) * G_ACCEL_MPS2
            a_avg = 0.5 * (a_net_0 + a_net_1)
            if vel <= 0.0 and a_net_0 <= 0.0 and a_net_1 <= 0.0:
                continue
            v_next = vel + a_avg * dt
            if v_next > 0.0:
                disp += max(0.0, vel * dt + 0.5 * a_avg * dt * dt)
                vel = v_next
            else:
                # Sliding block comes to rest during step
                if vel > 0.0 and a_avg < 0.0:
                    dt_stop = min(dt, -vel / a_avg)
                    disp += max(0.0, 0.5 * vel * dt_stop)
                vel = 0.0
        return disp * 100.0  # [cm]

    d_pos = _integrate_one(a)
    if not both_polarities:
        return d_pos
    return max(d_pos, _integrate_one(-a))


def estimate_newmark_displacement_3d(
    ky_g: float,
    pga_g: float,
    mw: float = 6.5,
    accel_time_s: np.ndarray | None = None,
    accel_g: np.ndarray | None = None,
) -> NewmarkResult3D:
    """Compute empirical Jibson (2007) & Ambraseys-Menu (1988) Newmark displacements (+ optional time history)."""
    if pga_g <= 0.0:
        raise ValueError("Require peak ground acceleration pga_g > 0.")
    if ky_g <= 0.0:
        return NewmarkResult3D(
            ky_g=ky_g,
            pga_g=pga_g,
            mw=mw,
            jibson_2007_median_cm=float("inf"),
            jibson_2007_p16_cm=float("inf"),
            jibson_2007_p84_cm=float("inf"),
            ambraseys_menu_cm=float("inf"),
            time_history_cm=float("inf") if accel_time_s is not None else None,
        )

    th_cm: float | None = None
    if accel_time_s is not None and accel_g is not None:
        th_cm = integrate_newmark_accelerogram(accel_time_s, accel_g, ky_g, both_polarities=True)

    ratio = ky_g / pga_g
    if ratio >= 1.0:
        return NewmarkResult3D(
            ky_g=ky_g,
            pga_g=pga_g,
            mw=mw,
            jibson_2007_median_cm=0.0,
            jibson_2007_p16_cm=0.0,
            jibson_2007_p84_cm=0.0,
            ambraseys_menu_cm=0.0,
            time_history_cm=th_cm if th_cm is not None else 0.0,
        )

    # Jibson (2007) Eq. (7): log10(Dn [cm]) = -2.710 + log10((1 - r)^2.335 * r^-1.478) + 0.424 * Mw +- 0.454
    log_jib = -2.710 + np.log10(((1.0 - ratio) ** 2.335) * (ratio**-1.478)) + 0.424 * float(mw)
    sigma_jib = 0.454
    jib_med = float(10.0**log_jib)
    jib_p16 = float(10.0 ** (log_jib - sigma_jib))
    jib_p84 = float(10.0 ** (log_jib + sigma_jib))

    # Ambraseys & Menu (1988): log10(Dn [cm]) = 0.90 + log10((1 - r)^2.53 * r^-1.09)
    log_amb = 0.90 + np.log10(((1.0 - ratio) ** 2.53) * (ratio**-1.09))
    amb_med = float(10.0**log_amb)

    return NewmarkResult3D(
        ky_g=float(ky_g),
        pga_g=float(pga_g),
        mw=float(mw),
        jibson_2007_median_cm=jib_med,
        jibson_2007_p16_cm=jib_p16,
        jibson_2007_p84_cm=jib_p84,
        ambraseys_menu_cm=amb_med,
        time_history_cm=th_cm,
    )
