"""Verification tests for Stage 3D-1: 3D GLE/M-P, Cheng & Yip Asymmetric 3D, SAO, and Parallel Search."""

from __future__ import annotations

import numpy as np
import pytest

from slipsurface3d.analysis import run_analysis_3d
from slipsurface3d.geometry.surfaces import (
    EllipsoidSurface,
    SplinePatchSurface,
    cylindrical_surface,
)
from slipsurface3d.geometry.terrain import BoreholeLayer, BoreholeLog, HorizonSurface, SlopeModel3D
from slipsurface3d.lem.columns import ColumnOptions, discretize_columns
from slipsurface3d.lem.solvers import (
    INTERCOLUMN_FUNCTIONS,
    cheng_yip_3d,
    gle_mp_3d,
    spencer_3d,
)
from slipsurface3d.materials.models import MohrCoulomb3D
from slipsurface3d.project import (
    MaterialSpec3D,
    Project3D,
    extruded_acads_1a_project,
    zhang_1988_homog_ellipsoid_project,
    zhang_1988_weak_seam_project,
)


def test_gle_mp_3d_interslice_functions_and_2d_extrusion_invariance() -> None:
    """3D GLE/M-P converges for all 4 interslice functions and is width-invariant on extruded 2D."""
    proj_1 = extruded_acads_1a_project()
    proj_1.analysis.ny = 1
    proj_1.analysis.nx = 40
    mdl = proj_1.build_model()
    mats = proj_1.build_materials()
    # Cylindrical ACADS 1(a) critical circle (toe=15, crest=35, H=10)
    surf = cylindrical_surface(xc=20.45, zc=16.20, radius=17.15, y_center=20.0)

    cols_1 = discretize_columns(mdl, surf, mats, opts=proj_1.build_column_options())
    cols_12_opts = proj_1.build_column_options()
    cols_12_opts.ny = 12
    cols_12 = discretize_columns(mdl, surf, mats, opts=cols_12_opts)

    sp_res = spencer_3d(cols_1)
    gle_const = gle_mp_3d(cols_1, interslice="constant")
    assert gle_const.ok
    assert gle_const.fos == pytest.approx(sp_res.fos, abs=1e-9)

    for fn_name in INTERCOLUMN_FUNCTIONS:
        r1 = gle_mp_3d(cols_1, interslice=fn_name)
        r12 = gle_mp_3d(cols_12, interslice=fn_name)
        assert r1.ok and r12.ok, f"GLE ({fn_name}) failed to converge"
        assert abs(r1.force_residual) < 1e-4
        assert abs(r1.moment_residual) < 1e-4
        # Exact 2D plane-strain width invariance across ny=1 vs ny=12
        assert r1.fos == pytest.approx(r12.fos, abs=1e-10)
        assert r1.fos == pytest.approx(sp_res.fos, rel=0.015)

    # True 3D ellipsoid (Zhang 1988 Ex. 1)
    pz = zhang_1988_homog_ellipsoid_project()
    cols_z = discretize_columns(
        pz.build_model(),
        pz.user_slip_surface(),  # type: ignore[arg-type]
        pz.build_materials(),
        opts=pz.build_column_options(),
    )
    gle_hs = gle_mp_3d(cols_z, interslice="half-sine")
    sp_z = spencer_3d(cols_z)
    assert gle_hs.ok and gle_hs.admissible
    assert abs(gle_hs.force_residual) < 1e-4
    assert abs(gle_hs.moment_residual) < 1e-4
    assert gle_hs.fos == pytest.approx(sp_z.fos, rel=0.025)


def test_cheng_yip_3d_symmetric_and_oblique_asymmetric_equilibrium() -> None:
    """Cheng & Yip matches Spencer on symmetric slopes and finds sum(Ft)=0 azimuth on oblique slopes."""
    # 1. Symmetric slope: transverse residual is already zero at 180 deg
    pz = zhang_1988_homog_ellipsoid_project()
    cols_sym = discretize_columns(
        pz.build_model(),
        pz.user_slip_surface(),  # type: ignore[arg-type]
        pz.build_materials(),
        opts=pz.build_column_options(),
    )
    sp_sym = spencer_3d(cols_sym)
    cy_sym = cheng_yip_3d(cols_sym)
    assert cy_sym.ok
    assert abs(cy_sym.transverse_residual) < 1e-6
    assert cy_sym.fos == pytest.approx(sp_sym.fos, abs=1e-6)
    assert cy_sym.azimuth_deg == pytest.approx(180.0, abs=1e-4)

    # 2. Oblique 3D slope dipping toward azimuth ~206.6 deg (downhill in -x and -y)
    xs = np.linspace(0.0, 50.0, 51)
    ys = np.linspace(0.0, 50.0, 51)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz_ground = 10.0 + 0.36 * xx + 0.18 * yy
    ground = HorizonSurface.from_grid(xs, ys, zz_ground, name="ObliqueGround")
    mdl = SlopeModel3D(
        x_min=0.0,
        x_max=50.0,
        y_min=0.0,
        y_max=50.0,
        z_min=-15.0,
        ground=ground,
        layers=[("Soil", ground)],
    )
    mats = [MohrCoulomb3D("Soil", gamma=19.0, c=18.0, phi=26.0)]
    surf = EllipsoidSurface(xc=18.0, yc=19.0, zc=34.0, rx=22.0, ry=22.0, rz=22.0, azimuth_deg=180.0)
    opts = ColumnOptions(nx=28, ny=28, sliding_mode="custom", azimuth_deg=180.0)
    cols_180 = discretize_columns(mdl, surf, mats, opts=opts)

    sp_180 = spencer_3d(cols_180)
    assert sp_180.ok
    assert abs(sp_180.transverse_residual) > 0.02

    cy_oblique = cheng_yip_3d(cols_180)
    assert cy_oblique.ok
    assert abs(cy_oblique.transverse_residual) < 5e-3
    assert abs(cy_oblique.force_residual) < 1e-4
    assert abs(cy_oblique.moment_residual) < 1e-4
    assert 198.0 <= cy_oblique.azimuth_deg <= 215.0


def test_spline_patch_analytic_gradients_match_finite_differences() -> None:
    """SplinePatchSurface.gradients matches central finite differences to < 2e-5."""
    base = EllipsoidSurface(xc=20.0, yc=22.0, zc=24.0, rx=18.0, ry=16.0, rz=15.0, azimuth_deg=195.0)
    rng = np.random.default_rng(7)
    ctrl = rng.uniform(-1.5, 1.5, size=(5, 5))
    sp = SplinePatchSurface(base=base, control_dz=ctrl)

    xs = np.linspace(12.0, 28.0, 9)
    ys = np.linspace(14.0, 30.0, 9)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    dz_dx, dz_dy = sp.gradients(xx, yy)
    eps = 1e-6
    fd_x = (sp.z(xx + eps, yy) - sp.z(xx - eps, yy)) / (2.0 * eps)
    fd_y = (sp.z(xx, yy + eps) - sp.z(xx, yy - eps)) / (2.0 * eps)

    valid = np.isfinite(dz_dx) & np.isfinite(fd_x)
    assert int(np.count_nonzero(valid)) > 50
    np.testing.assert_allclose(dz_dx[valid], fd_x[valid], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(dz_dy[valid], fd_y[valid], atol=2e-5, rtol=2e-5)


def test_sao_refinement_and_parallel_search_callbacks() -> None:
    """SAO spline refinement improves or preserves FoS; parallel workers, baking & cancel work."""
    proj = zhang_1988_weak_seam_project()
    proj.surface_config.mode = "search"
    proj.surface_config.effort = "fast"
    proj.surface_config.sao_enabled = False
    proj.analysis.methods = ["bishop"]
    proj.analysis.n_workers = 2
    res_ellip = run_analysis_3d(proj)
    assert res_ellip.primary_result.ok

    proj_sao = proj.copy()
    proj_sao.surface_config.sao_enabled = True
    proj_sao.analysis.n_workers = 4
    best_updates: list[float] = []
    res_sao = run_analysis_3d(
        proj_sao,
        best_surface_cb=lambda _s, f: best_updates.append(f),
    )
    assert res_sao.primary_result.ok
    assert res_sao.primary_result.admissible
    assert res_sao.critical_fos <= res_ellip.critical_fos + 1e-3
    assert len(best_updates) >= 1

    # Borehole RBF horizon baking + early cancellation check
    bhs = [
        BoreholeLog("BH-1", x=5.0, y=5.0, collar_z=0.0, layers=[BoreholeLayer("Soil", 0.0, -10.0)]),
        BoreholeLog(
            "BH-2", x=5.0, y=35.0, collar_z=0.0, layers=[BoreholeLayer("Soil", 0.0, -10.0)]
        ),
        BoreholeLog(
            "BH-3", x=25.0, y=20.0, collar_z=6.0, layers=[BoreholeLayer("Soil", 6.0, -10.0)]
        ),
        BoreholeLog(
            "BH-4", x=45.0, y=5.0, collar_z=12.0, layers=[BoreholeLayer("Soil", 12.0, -10.0)]
        ),
        BoreholeLog(
            "BH-5", x=45.0, y=35.0, collar_z=12.0, layers=[BoreholeLayer("Soil", 12.0, -10.0)]
        ),
    ]
    pit = Project3D(
        materials=[MaterialSpec3D("Soil", c=15.0, phi=25.0, gamma=19.0)],
        x_min=0.0,
        x_max=50.0,
        y_min=0.0,
        y_max=40.0,
        z_min=-10.0,
        boreholes=bhs,
    )
    pit.rebuild_from_boreholes()
    pit.surface_config.effort = "normal"
    pit.analysis.n_workers = 2
    calls = 0

    def cancel_soon() -> bool:
        nonlocal calls
        calls += 1
        return calls > 3

    res_cancel = run_analysis_3d(pit, cancel_check=cancel_soon)
    assert res_cancel.cancelled is True
    assert np.isfinite(res_cancel.critical_fos)
