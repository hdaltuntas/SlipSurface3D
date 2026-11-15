"""Tests for areal supports (geotextile sheets, pile rows), construction stages,
particle-swarm search, finite-thickness weak layers and pore-water consistency."""

from __future__ import annotations

import numpy as np
import pytest

from slipsurface3d.analysis import report_text_3d, run_analysis_3d
from slipsurface3d.geometry import EllipsoidSurface, HorizonSurface, WeakLayer3D
from slipsurface3d.loads.supports import GeotextileSheet3D, PileRow3D
from slipsurface3d.project import (
    MaterialSpec3D,
    Project3D,
    Stage3D,
    zhang_1988_homog_ellipsoid_project,
    zhang_1988_weak_seam_project,
)
from slipsurface3d.scripting import create_simple_slope_project
from slipsurface3d.stages import run_staged_analysis_3d, stage_project


def _fixed(**kwargs: float) -> Project3D:
    proj = create_simple_slope_project(methods=("bishop", "spencer"), **kwargs)
    proj.surface_config.mode = "user"
    proj.surface_config.user_surface = EllipsoidSurface(
        xc=0.45 * (proj.x_min + proj.x_max), yc=20.0, zc=22.0, rx=16.0, ry=14.0, rz=16.0
    ).to_dict()
    return proj


def test_geotextile_sheet_is_cut_strip_by_strip_and_raises_fos():
    proj = _fixed()
    base = run_analysis_3d(proj)
    sheet = GeotextileSheet3D(
        x1=5.0, x2=50.0, y1=10.0, y2=30.0, z=8.0, tensile_kn_m=60.0, pullout_kpa=30.0
    )
    proj.geotextiles = [sheet]
    res = run_analysis_3d(proj)
    hits = res.columns.support_hits
    # Only the strips under the 3D bowl are cut; the sheet is wider than the slip mass.
    assert 0 < len(hits) < len(sheet.to_bolts(proj.build_model().ground_z))
    assert {h.kind for h in hits} == {"geotextile"}
    for h in hits:
        assert h.hit_xyz[2] == pytest.approx(8.0, abs=1e-6)
        assert h.capacity_kn <= 60.0 + 1e-9  # 1 m strips: tensile strength governs at most
        assert h.force_xyz_kn[0] > 0.0  # pulls the mass back into the slope (+X)
    assert res.critical_fos > base.critical_fos

    # No anchorage at the face: a sheet barely entering the mass is limited by pullout
    weak_bond = GeotextileSheet3D(
        x1=5.0, x2=50.0, y1=10.0, y2=30.0, z=8.0, tensile_kn_m=60.0, pullout_kpa=0.5
    )
    proj.geotextiles = [weak_bond]
    limited = run_analysis_3d(proj)
    assert max(h.capacity_kn for h in limited.columns.support_hits) < 60.0
    assert Project3D.from_dict(proj.to_dict()).geotextiles[0] == weak_bond
    with pytest.raises(ValueError):
        GeotextileSheet3D(direction="z")


def test_pile_row_resists_only_where_piles_pass_through_the_slip_surface():
    proj = _fixed()
    base = run_analysis_3d(proj)
    row = PileRow3D(x1=32.0, y1=4.0, x2=32.0, y2=36.0, spacing=2.0, length_m=15.0, shear_kn=200.0)
    assert len(row.pile_positions()) == 17
    proj.pile_rows = [row]
    res = run_analysis_3d(proj)
    hits = res.columns.support_hits
    assert 0 < len(hits) < 17
    assert all(h.kind == "micropile" and h.capacity_kn == 200.0 for h in hits)
    assert res.critical_fos > base.critical_fos

    # Piles too short to reach the slip surface do nothing
    proj.pile_rows = [PileRow3D(x1=32.0, y1=4.0, x2=32.0, y2=36.0, length_m=0.5, shear_kn=200.0)]
    short = run_analysis_3d(proj)
    assert short.columns.support_hits == []
    assert short.critical_fos == pytest.approx(base.critical_fos, rel=1e-9)
    assert Project3D.from_dict(proj.to_dict()).pile_rows == proj.pile_rows


def test_construction_stages_reshape_ground_and_stratigraphy():
    proj = create_simple_slope_project(methods=("bishop",), effort="fast")
    proj.materials.append(MaterialSpec3D(name="Fill", c=5.0, phi=32.0, gamma=20.0, use_water=False))
    proj.stages = [
        Stage3D(
            name="Toe cut", kind="excavation", x1=0, x2=14, y1=10, y2=30, level=-3.0, batter=1.0
        ),
        Stage3D(
            name="Crest fill",
            kind="fill",
            x1=45,
            x2=59,
            y1=5,
            y2=35,
            level=15.0,
            batter=1.5,
            fill_material="Fill",
        ),
    ]
    proj.validate()
    assert Project3D.from_dict(proj.to_dict()).stages == proj.stages

    cut = stage_project(proj, 1).build_model()
    assert float(cut.ground_z(7.0, 20.0)) == pytest.approx(-3.0, abs=1e-6)
    assert float(cut.ground_z(17.0, 20.0)) == pytest.approx(0.0, abs=0.05)  # battered back
    assert float(cut.ground_z(7.0, 2.0)) == pytest.approx(0.0, abs=1e-6)  # outside the box in Y

    both = stage_project(proj, 2)
    model = both.build_model()
    assert float(model.ground_z(52.0, 20.0)) == pytest.approx(15.0, abs=1e-6)
    names = both.material_names()

    def at(z: float) -> int:
        idx = model.material_indices_at(np.array([52.0]), np.array([20.0]), np.array([z]), names)
        return int(idx[0])

    assert names[at(14.0)] == "Fill"
    assert names[at(5.0)] == "Slope Soil"

    staged = run_staged_analysis_3d(proj)
    assert [s.name for s in staged] == ["Initial ground", "Toe cut", "Crest fill"]
    assert all(s.results is not None and np.isfinite(s.fos) for s in staged)
    # Removing the toe and loading the crest each make the slope less stable
    assert staged[1].fos < staged[0].fos
    assert staged[2].fos < staged[0].fos
    assert proj.stages and len(staged[0].project.stages) == 0  # original project untouched

    proj.stages[1].fill_material = "Missing"
    with pytest.raises(ValueError, match="fill material"):
        proj.validate()


def test_particle_swarm_search_finds_a_comparable_critical_surface():
    cuckoo = create_simple_slope_project(methods=("bishop",))
    pso = create_simple_slope_project(methods=("bishop",))
    pso.surface_config.algorithm = "pso"
    f_c = run_analysis_3d(cuckoo).critical_fos
    f_p = run_analysis_3d(pso).critical_fos
    assert np.isfinite(f_p)
    assert f_p == pytest.approx(f_c, rel=0.05)
    pso.surface_config.algorithm = "annealing"
    with pytest.raises(ValueError, match="search algorithm"):
        pso.validate()


def test_weak_layer_thickness_reproduces_the_thick_seam_benchmark():
    ref = zhang_1988_weak_seam_project()
    ref.analysis.nx = ref.analysis.ny = 48
    seam = ref.surface_config.user_surface
    assert seam is not None
    expected = run_analysis_3d(ref)

    proj = zhang_1988_homog_ellipsoid_project()
    proj.analysis.nx = proj.analysis.ny = 48
    proj.analysis.methods = list(ref.analysis.methods)
    proj.materials.append(ref.materials[-1])
    plane = HorizonSurface.from_plane(
        (proj.x_min, proj.x_max), (proj.y_min, proj.y_max), float(seam["plane_z0"])
    )
    proj.weak_layers = [WeakLayer3D("Seam", ref.materials[-1].name, plane, thickness=0.15)]
    proj.surface_config.mode = "user"
    proj.surface_config.user_surface = seam["ellipsoid"]
    thick = run_analysis_3d(proj)
    for method in ("bishop", "spencer"):
        assert thick.method_results[method].fos == pytest.approx(
            expected.method_results[method].fos, rel=3e-3
        )
    proj.weak_layers[0].thickness = 0.0
    thin = run_analysis_3d(proj)
    assert thin.method_results["bishop"].fos > thick.method_results["bishop"].fos
    assert Project3D.from_dict(proj.to_dict()).weak_layers[0].thickness == 0.0
    with pytest.raises(ValueError):
        WeakLayer3D("Bad", "Seam", plane, thickness=-1.0)


def test_piezometric_surface_matches_an_identical_water_table_exactly():
    proj = _fixed()
    proj.materials[0].use_water = True
    level = HorizonSurface.from_plane((proj.x_min, proj.x_max), (proj.y_min, proj.y_max), 10.0)
    proj.water_table = level
    by_table = run_analysis_3d(proj)
    proj.water_table = None
    proj.piezo_surfaces = [("P", level)]
    proj.materials[0].piezo = "P"
    by_piezo = run_analysis_3d(proj)
    # Same pore pressure and the same saturated weights, so the same answer
    assert np.allclose(by_piezo.columns.W, by_table.columns.W)
    assert by_piezo.critical_fos == pytest.approx(by_table.critical_fos, rel=1e-12)


def test_pore_grid_saturates_below_water_and_warns_outside_the_samples():
    proj = _fixed()
    proj.materials[0].use_water = True
    # A water table parallel to the 2H:1V slope face, about 0.75 m below it (no ponding)
    x_mid = 0.5 * (proj.x_min + proj.x_max)
    table = HorizonSurface.from_plane(
        (proj.x_min, proj.x_max),
        (proj.y_min, proj.y_max),
        6.5,
        dip_deg=float(np.degrees(np.arctan(0.5))),
        dip_dir_deg=180.0,
    )
    proj.water_table = table
    by_table = run_analysis_3d(proj)
    assert float(np.max(by_table.columns.u)) > 5.0

    def samples(x_hi: float) -> list[list[float]]:
        xs = np.linspace(proj.x_min, x_hi, 9)
        ys = np.linspace(proj.y_min, proj.y_max, 5)
        zs = np.linspace(proj.z_min, 14.0, 13)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        # Signed hydrostatic pressures (negative above the phreatic surface) keep the
        # zero-pressure line exact under linear interpolation.
        u = proj.analysis.gamma_w * (6.5 + 0.5 * (gx - x_mid) - gz)
        return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel(), u.ravel()]).tolist()

    proj.water_table = None
    proj.analysis.pore_mode = "grid"
    proj.pore_grid = samples(proj.x_max)
    full = run_analysis_3d(proj)
    assert full.warnings == []
    # Same pore pressures and the soil below the phreatic surface is saturated in both
    assert np.allclose(full.columns.u, by_table.columns.u, atol=1e-6)
    assert np.allclose(full.columns.W, by_table.columns.W, rtol=1e-9)
    assert full.critical_fos == pytest.approx(by_table.critical_fos, rel=1e-6)

    proj.pore_grid = samples(30.0)  # samples stop short of the slip mass
    partial = run_analysis_3d(proj)
    assert len(partial.warnings) == 1 and "outside the pore-pressure samples" in partial.warnings[0]
    assert "WARNING" in report_text_3d(proj, partial)
