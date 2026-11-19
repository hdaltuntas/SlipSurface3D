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
        GeotextileSheet3D(dip_deg=85.0)
    # Files written before arbitrary directions existed used direction "x" / "y"
    assert GeotextileSheet3D.from_dict({"direction": "y"}).azimuth_deg == 90.0


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


def test_inclined_and_rotated_geotextile_sheets():
    proj = _fixed()
    ground = proj.build_model().ground_z
    sheet = GeotextileSheet3D(
        x1=10.0, x2=50.0, y1=10.0, y2=30.0, z=8.0, dip_deg=10.0, dip_dir_deg=180.0
    )
    # Dips towards -X: lower at small x, higher at large x, level across Y
    corners = sheet.corners()
    assert corners[0, 2] == pytest.approx(8.0 - 20.0 * np.tan(np.radians(10.0)))
    assert corners[1, 2] == pytest.approx(8.0 + 20.0 * np.tan(np.radians(10.0)))
    assert corners[0, 2] == pytest.approx(corners[3, 2])
    for a, b, _w in sheet.strip_segments():
        assert a[2] == pytest.approx(float(sheet.z_at(a[0], a[1])))
        assert b[2] > a[2]
    proj.geotextiles = [sheet]
    res = run_analysis_3d(proj)
    hits = res.columns.support_hits
    assert hits
    for h in hits:  # every hit lies on the inclined plane and pulls up-dip along it
        assert h.hit_xyz[2] == pytest.approx(
            float(sheet.z_at(h.hit_xyz[0], h.hit_xyz[1])), abs=1e-6
        )
        assert h.force_xyz_kn[0] > 0.0 and h.force_xyz_kn[2] > 0.0

    # Tension at 45 deg in plan: strips are clipped to the box and keep their spacing
    diag = GeotextileSheet3D(x1=0.0, x2=10.0, y1=0.0, y2=10.0, azimuth_deg=45.0, strip_spacing=1.0)
    segs = diag.strip_segments()
    assert len(segs) == 14
    total_area = sum(float(np.linalg.norm(b - a)) * w for a, b, w in segs)
    assert total_area == pytest.approx(100.0, rel=0.02)  # strips tile the whole sheet
    for a, b, _w in segs:
        d = (b - a)[:2] / np.linalg.norm((b - a)[:2])
        assert np.allclose(np.abs(d), np.sqrt(0.5))
        for pt in (a, b):
            assert -1e-9 <= pt[0] <= 10.0 + 1e-9 and -1e-9 <= pt[1] <= 10.0 + 1e-9
    assert len(diag.to_bolts(ground)) == 14


def test_stage_water_level_and_staged_activation_of_loads_and_supports():
    from slipsurface3d.loads.supports import PatchLoad3D

    proj = _fixed()
    proj.materials[0].use_water = True
    proj.patch_loads = [
        PatchLoad3D(x1=30.0, x2=45.0, y1=10.0, y2=30.0, q1=40.0, name="Plant", from_stage=1)
    ]
    proj.geotextiles = [GeotextileSheet3D(x1=5.0, x2=50.0, y1=10.0, y2=30.0, z=8.0, from_stage=2)]
    proj.stages = [
        Stage3D(name="Load on", x1=0.0, x2=1.0, y1=0.0, y2=1.0, level=0.0, water_level=None),
        Stage3D(
            name="Reinforce + flood", x1=0.0, x2=1.0, y1=0.0, y2=1.0, level=0.0, water_level=9.0
        ),
    ]
    assert Project3D.from_dict(proj.to_dict()).stages == proj.stages
    assert Stage3D(water_level="  ").water_level is None  # blank table cell
    assert Stage3D(water_level="7,5").water_level == 7.5

    # A plain analysis is the initial state: nothing installed yet
    assert proj.build_loads().patch_loads == [] and proj.build_loads().bolts == []
    s1, s2 = stage_project(proj, 1), stage_project(proj, 2)
    assert len(s1.build_loads().patch_loads) == 1 and s1.build_loads().bolts == []
    assert len(s2.build_loads().bolts) > 0
    assert s1.water_table is None and s2.water_table is not None
    assert float(s2.water_table.z(20.0, 20.0)) == 9.0

    staged = run_staged_analysis_3d(proj)
    f0, f1, f2 = (s.fos for s in staged)
    assert f1 < f0  # surcharge on the crest
    assert staged[1].results.columns.support_hits == []
    assert len(staged[2].results.columns.support_hits) > 0
    assert float(np.max(staged[2].results.columns.u)) > 0.0
    assert float(np.max(staged[1].results.columns.u)) == 0.0
    assert np.isfinite(f2)


def test_fill_stage_on_an_extruded_2d_region_model():
    from shapely.geometry import Polygon

    from slipsurface3d.geometry.terrain import ExtrudedProfileRegion, HorizonSurface
    from slipsurface3d.project import AnalysisSettings3D, Project3D, SurfaceConfig3D

    profile = np.array([[0.0, 10.0], [10.0, 10.0], [26.0, 18.0], [44.0, 18.0]])
    ext_poly = Polygon(
        [(0.0, 0.0), (44.0, 0.0), (44.0, 18.0), (26.0, 18.0), (10.0, 10.0), (0.0, 10.0)]
    )
    proj = Project3D(
        materials=[
            MaterialSpec3D(name="Base Soil", c=12.0, phi=24.0, gamma=19.0, use_water=False),
            MaterialSpec3D(name="Fill", c=5.0, phi=32.0, gamma=20.0, use_water=False),
        ],
        x_min=0.0,
        x_max=44.0,
        y_min=0.0,
        y_max=30.0,
        z_min=0.0,
        ground=HorizonSurface.from_extruded(profile, name="Ground"),
        profile_regions=[ExtrudedProfileRegion(polygon=ext_poly, material="Base Soil")],
        analysis=AnalysisSettings3D(methods=["bishop"], nx=24, ny=20),
        surface_config=SurfaceConfig3D(
            mode="search", surface_kind="ellipsoid", effort="fast", primary_method="bishop"
        ),
    )
    assert proj.profile_regions
    base = proj.build_model()
    xc = 32.0
    ym = 0.5 * (proj.y_min + proj.y_max)
    crest = float(base.ground_z(xc, ym))
    proj.stages = [
        Stage3D(
            name="Embankment",
            kind="fill",
            x1=xc - 4.0,
            x2=xc + 4.0,
            y1=proj.y_min,
            y2=proj.y_max,
            level=crest + 3.0,
            batter=1.5,
            fill_material="Fill",
        )
    ]
    proj.validate()
    staged_proj = stage_project(proj, 1)
    model = staged_proj.build_model()
    assert float(model.ground_z(xc, ym)) == pytest.approx(crest + 3.0, abs=1e-6)
    names = staged_proj.material_names()

    def mat(z: float) -> str:
        return names[
            int(model.material_indices_at(np.array([xc]), np.array([ym]), np.array([z]), names)[0])
        ]

    assert mat(crest + 1.5) == "Fill"
    assert mat(crest - 0.5) != "Fill"
    assert {name for name, _ in model.layer_solid_meshes()} >= {"Fill"}

    staged = run_staged_analysis_3d(proj)
    assert all(s.results is not None for s in staged)
    assert staged[1].fos < staged[0].fos  # the embankment loads the crest


def test_clamped_pore_grid_recovers_the_phreatic_surface_and_ponds_water():
    from slipsurface3d.porewater import PorePressureGrid3D, restore_suction

    # Zero-clamped hydrostatic strings: the water level (z = 3.3) lies between samples
    zs = np.arange(0.0, 8.5, 1.0)
    rows = [
        [x, y, z, 9.81 * max(3.3 - z, 0.0)] for x in (0.0, 10.0) for y in (0.0, 10.0) for z in zs
    ]
    fixed, n_changed = restore_suction(np.asarray(rows), gamma_w=9.81)
    assert n_changed == 4 * 5
    assert np.allclose(fixed[:, 3], 9.81 * (3.3 - fixed[:, 2]))
    grid = PorePressureGrid3D(rows)
    assert grid.n_restored == 20
    assert float(grid(5.0, 5.0, 3.2)) == pytest.approx(0.981, rel=1e-6)
    assert float(grid(5.0, 5.0, 3.4)) == 0.0
    assert grid.to_list() == [[float(v) for v in r] for r in rows]  # the file keeps the raw data
    signed = [[r[0], r[1], r[2], 9.81 * (3.3 - r[2])] for r in rows]
    assert restore_suction(np.asarray(signed))[1] == 0  # already signed: untouched

    # In an analysis: clamped samples now match the equivalent water table closely, and a
    # level above the slope face ponds water on it just as a water table does.
    proj = _fixed()
    proj.materials[0].use_water = True
    level = 10.0
    proj.water_table = HorizonSurface.from_plane(
        (proj.x_min, proj.x_max), (proj.y_min, proj.y_max), level
    )
    by_table = run_analysis_3d(proj)
    xs = np.linspace(proj.x_min, proj.x_max, 9)
    ys = np.linspace(proj.y_min, proj.y_max, 5)
    zg = np.linspace(proj.z_min, 14.5, 36)  # 0.7 m spacing: z = 10 is not a sample
    gx, gy, gz = np.meshgrid(xs, ys, zg, indexing="ij")
    u = proj.analysis.gamma_w * np.maximum(level - gz, 0.0)
    proj.water_table = None
    proj.analysis.pore_mode = "grid"
    proj.pore_grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel(), u.ravel()]).tolist()
    by_grid = run_analysis_3d(proj)
    assert np.allclose(by_grid.columns.u, by_table.columns.u, atol=1e-6)
    assert np.allclose(by_grid.columns.W, by_table.columns.W, rtol=1e-9)  # saturation + ponding
    assert by_grid.critical_fos == pytest.approx(by_table.critical_fos, rel=1e-6)
