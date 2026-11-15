"""Verification tests for Stage 3D-2: Probabilistic 3D LHS/MC, Tornado Sensitivity, Seismic ky/Newmark, and 3D Safety Map."""

from __future__ import annotations

import numpy as np
import pytest

from slipsurface3d.analysis import run_analysis_3d
from slipsurface3d.probabilistic import (
    RandomVarSpec3D,
    run_probabilistic_3d,
    run_sensitivity_3d,
)
from slipsurface3d.project import (
    extruded_acads_1a_project,
    zhang_1988_homog_ellipsoid_project,
)
from slipsurface3d.seismic import (
    G_ACCEL_MPS2,
    estimate_newmark_displacement_3d,
    find_yield_acceleration_3d,
    integrate_newmark_accelerogram,
)


def test_probabilistic_3d_lhs_and_tornado_sensitivity() -> None:
    """3D LHS reliability analysis produces consistent mean/Pf/beta and ordered tornado sensitivity."""
    proj = zhang_1988_homog_ellipsoid_project()
    mat_name = proj.materials[0].name

    specs = [
        RandomVarSpec3D(param="c", material_name=mat_name, distribution="lognormal", cov=0.20),
        RandomVarSpec3D(param="phi", material_name=mat_name, distribution="normal", std=3.0),
        RandomVarSpec3D(
            param="ru",
            material_name=None,
            distribution="triangular",
            min_val=0.0,
            mode_val=0.05,
            max_val=0.20,
        ),
    ]
    proj.analysis.pore_mode = "ru"
    proj.analysis.ru = 0.05

    prob = run_probabilistic_3d(proj, specs, n_samples=120, sampling="lhs", fixed_surface=True)
    assert prob.n_converged == 120
    assert prob.mean_fos == pytest.approx(prob.deterministic_fos, rel=0.08)
    assert prob.std_fos > 0.05
    assert 0.0 <= prob.pf <= 1.0
    assert prob.p5 < prob.p50 < prob.p95
    assert np.isfinite(prob.beta_normal) and np.isfinite(prob.beta_lognormal)

    curves = run_sensitivity_3d(proj, n_points=5)
    assert len(curves) >= 3
    # Curves are sorted descending by tornado swing delta_fos
    deltas = [c.delta_fos for c in curves]
    assert deltas == sorted(deltas, reverse=True)
    assert deltas[0] > 0.15


def test_seismic_yield_acceleration_and_newmark_displacement() -> None:
    """find_yield_acceleration_3d finds exact ky where FoS(ky) == 1.0, and Newmark matches pulse analytical disp."""
    proj = zhang_1988_homog_ellipsoid_project()
    ky_res = find_yield_acceleration_3d(proj, method="bishop", tol=1e-4)
    assert ky_res.converged
    assert 0.20 <= ky_res.ky_g <= 0.70
    assert ky_res.fos_at_ky == pytest.approx(1.0, abs=2e-3)

    # Empirical Newmark regressions: zero when PGA <= ky, strictly positive when PGA > ky
    nm_zero = estimate_newmark_displacement_3d(ky_g=0.35, pga_g=0.30, mw=6.5)
    assert nm_zero.jibson_2007_median_cm == 0.0
    assert nm_zero.ambraseys_menu_cm == 0.0

    nm_strong = estimate_newmark_displacement_3d(ky_g=0.15, pga_g=0.45, mw=7.0)
    assert (
        nm_strong.jibson_2007_p16_cm
        < nm_strong.jibson_2007_median_cm
        < nm_strong.jibson_2007_p84_cm
    )
    assert nm_strong.ambraseys_menu_cm > 1.0

    # Analytical rectangular acceleration pulse check:
    # Block subject to constant a = 0.30g against ky = 0.10g for T = 0.50 s, then a = 0.0g until arrest:
    # During pulse: v_peak = (0.20 * g) * 0.50 = 0.10 g; d1 = 0.5 * (0.20 g) * 0.25 = 0.025 g
    # Deceleration under -0.10 g takes t_stop = 1.00 s; d2 = 0.5 * v_peak * 1.00 = 0.050 g
    # Total exact displacement = 0.075 * g [m] = 7.5 * g [cm]
    dt = 0.001
    t_arr = np.arange(0.0, 2.0 + dt, dt)
    a_arr = np.where(t_arr <= 0.50, 0.30, 0.0)
    d_cm = integrate_newmark_accelerogram(t_arr, a_arr, ky_g=0.10, both_polarities=False)
    exact_cm = 0.075 * G_ACCEL_MPS2 * 100.0
    assert d_cm == pytest.approx(exact_cm, rel=1e-2)


def test_3d_slope_safety_map_generation() -> None:
    """run_analysis_3d populates a 3D SafetyMap3D whose minimum matches critical_fos and generates a valid mesh."""
    proj = extruded_acads_1a_project()
    proj.surface_config.effort = "fast"
    res = run_analysis_3d(proj)
    assert res.safety_map is not None
    smap = res.safety_map
    assert np.nanmin(smap.min_fos) == pytest.approx(res.critical_fos, abs=5e-2)
    assert int(np.max(smap.trial_count)) >= 5
    mesh = smap.to_mesh()
    assert mesh.n_points > 20
    assert "MinFoS" in mesh.point_data
