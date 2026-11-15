"""Tests for design-standard partial factors, tabulated / infinite-strength materials,
plan search limits, and the multi-modal (distinct critical surfaces) search."""

from __future__ import annotations

import numpy as np
import pytest

from slipsurface3d.analysis import report_text_3d, run_analysis_3d
from slipsurface3d.design import DESIGN_PRESETS, DesignFactors3D, design_preset
from slipsurface3d.geometry import EllipsoidSurface
from slipsurface3d.lem.columns import ColumnDiscretizationError
from slipsurface3d.loads.supports import PatchLoad3D
from slipsurface3d.materials.models import ShearNormalFunction3D
from slipsurface3d.optimizer.search import _footprint_overlap, _plan_footprint
from slipsurface3d.project import Project3D
from slipsurface3d.scripting import create_simple_slope_project


def _fixed_surface_project(**kwargs: float) -> Project3D:
    proj = create_simple_slope_project(methods=("bishop", "spencer"), **kwargs)
    proj.surface_config.mode = "user"
    proj.surface_config.user_surface = EllipsoidSurface(
        xc=0.45 * (proj.x_min + proj.x_max), yc=20.0, zc=22.0, rx=16.0, ry=14.0, rz=16.0
    ).to_dict()
    return proj


def test_equal_material_factors_divide_the_factor_of_safety_exactly():
    proj = _fixed_surface_project()
    base = run_analysis_3d(proj)
    assert base.result_label == "FoS" and base.design_summary == ""

    proj.design = design_preset("ec7_da3")
    da3 = run_analysis_3d(proj)
    assert da3.result_label == "ODF"
    for method in ("bishop", "spencer"):
        assert da3.method_results[method].fos == pytest.approx(
            base.method_results[method].fos / 1.25, rel=1e-6
        )

    # DA2: gamma_R;e = 1.1 divides the whole resistance. The 1.35 weight factor scales the
    # driving force and the frictional resistance but not the cohesive part, so ODF < FoS / 1.1.
    proj.design = design_preset("ec7_da2")
    da2 = run_analysis_3d(proj)
    assert da2.method_results["bishop"].fos < base.method_results["bishop"].fos / 1.1
    assert "Design Standard" in report_text_3d(proj, da2)


def test_design_factors_scale_loads_supports_and_round_trip():
    proj = _fixed_surface_project()
    proj.patch_loads = [PatchLoad3D(x1=30.0, x2=45.0, y1=10.0, y2=30.0, q1=20.0)]
    proj.design = design_preset("ec7_da1_c1")
    loads = proj.build_loads()
    assert loads.patch_loads[0].q1 == pytest.approx(30.0)
    assert proj.patch_loads[0].q1 == pytest.approx(20.0)  # project data is untouched
    mats = proj.build_materials()
    assert mats[0].gamma == pytest.approx(19.0 * 1.35)
    assert mats[0].pf_c == pytest.approx(1.0)

    restored = Project3D.from_dict(proj.to_dict())
    assert restored.design == proj.design
    assert not DESIGN_PRESETS["none"].active
    with pytest.raises(ValueError):
        DesignFactors3D(friction=0.0)
    with pytest.raises(ValueError):
        design_preset("no_such_standard")


def test_undrained_material_uses_the_undrained_factor():
    proj = _fixed_surface_project()
    proj.materials[0].model = "UD"
    proj.materials[0].su = 60.0
    base = run_analysis_3d(proj).method_results["bishop"].fos
    proj.design = design_preset("ec7_da1_c2")
    assert run_analysis_3d(proj).method_results["bishop"].fos == pytest.approx(base / 1.4, rel=1e-6)


def test_linear_shear_normal_table_reproduces_mohr_coulomb():
    proj = _fixed_surface_project(c=15.0, phi=26.0)
    mc = run_analysis_3d(proj)
    tan_phi = float(np.tan(np.radians(26.0)))
    proj.materials[0].model = "SN"
    proj.materials[0].sn_table = [[0.0, 15.0], [1000.0, 15.0 + 1000.0 * tan_phi]]
    sn = run_analysis_3d(proj)
    for method in ("bishop", "spencer"):
        assert sn.method_results[method].fos == pytest.approx(
            mc.method_results[method].fos, rel=1e-6
        )
    assert Project3D.from_dict(proj.to_dict()).materials[0].sn_table == proj.materials[0].sn_table


def test_shear_normal_function_is_piecewise_linear_and_validated():
    mat = ShearNormalFunction3D(name="Joint", table=[(100.0, 60.0), (0.0, 10.0), (300.0, 100.0)])
    c_eq, tan_phi = mat.strength(np.array([50.0, 200.0, 500.0]))
    assert np.allclose(c_eq + np.array([50.0, 200.0, 500.0]) * tan_phi, [35.0, 80.0, 140.0])
    assert np.allclose(tan_phi, [0.5, 0.2, 0.2])
    with pytest.raises(ValueError):
        ShearNormalFunction3D(name="Bad", table=[(0.0, 10.0)])
    with pytest.raises(ValueError):
        ShearNormalFunction3D(name="Bad", table=[(0.0, 10.0), (100.0, 5.0)])


def test_infinite_strength_material_rejects_the_slip_surface():
    proj = _fixed_surface_project()
    proj.materials[0].model = "INF"
    with pytest.raises(ColumnDiscretizationError, match="infinite-strength"):
        run_analysis_3d(proj)


def test_search_limits_confine_the_sliding_mass():
    proj = create_simple_slope_project(methods=("bishop",), width_y=120.0, effort="fast")
    proj.surface_config.ry_bounds = [6.0, 18.0]
    proj.surface_config.search_limits = [proj.x_min, proj.x_max, 60.0, 120.0]
    res = run_analysis_3d(proj)
    assert float(np.min(res.columns.y)) >= 60.0
    assert float(np.max(res.columns.y)) <= 120.0
    assert np.isfinite(res.critical_fos)

    proj.surface_config.search_limits = [0.0, 10.0, 500.0, 600.0]
    with pytest.raises(ValueError, match="outside the model domain"):
        proj.validate()


def test_multi_modal_search_returns_distinct_ranked_surfaces():
    proj = create_simple_slope_project(methods=("bishop",), width_y=120.0, effort="fast")
    proj.surface_config.ry_bounds = [6.0, 18.0]
    proj.surface_config.n_local_minima = 3
    res = run_analysis_3d(proj)
    minima = res.local_minima
    assert len(minima) >= 2
    assert [m.fos for m in minima] == sorted(m.fos for m in minima)
    # The first entry is the critical surface that the columns and FoS table describe.
    assert minima[0].surface is res.critical_surface
    assert minima[0].fos == pytest.approx(res.critical_fos, rel=1e-6)

    model = res.model
    xs = np.linspace(model.x_min, model.x_max, 56)
    ys = np.linspace(model.y_min, model.y_max, 56)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zg = model.ground_z(xx, yy)
    masks = [_plan_footprint(m.surface, xx, yy, zg, model.z_min) for m in minima]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            assert _footprint_overlap(masks[i], masks[j]) < 0.5
    assert "Distinct critical surfaces" in report_text_3d(proj, res)

    proj.surface_config.n_local_minima = 1
    assert run_analysis_3d(proj).local_minima == []
