"""Verification tests for Stage 3D-3: Hoek & Bray 3D Rock Wedge, 3D Seepage, Bolt Pattern, and STL/DEM/CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slipsurface3d.analysis import run_analysis_3d
from slipsurface3d.exchange import (
    export_esri_ascii_dem,
    export_stl_mesh,
    export_xyz_csv_horizon,
    import_esri_ascii_dem,
    import_stl_horizon,
    import_stl_slip_surface,
    import_xyz_csv_horizon,
)
from slipsurface3d.geometry.surfaces import MultiPlaneWedgeSurface
from slipsurface3d.loads.supports import BoltPatternSpec3D, SupportType3D, generate_bolt_pattern_3d
from slipsurface3d.project import (
    Project3D,
    rock_wedge_3d_project,
    zhang_1988_homog_ellipsoid_project,
)
from slipsurface3d.seepage import SeepageBC3D, solve_seepage_3d


def test_hoek_bray_1981_symmetric_3d_rock_wedge_analytical_benchmark(tmp_path: Path) -> None:
    """Symmetric 3D rock wedge matches closed-form Hoek & Bray (1981) FoS = tan(phi)/(tan(psi_i)*cos(beta))."""
    proj = rock_wedge_3d_project()
    surf = proj.user_slip_surface()
    assert isinstance(surf, MultiPlaneWedgeSurface)
    assert surf.kinematic_azimuth_deg() == pytest.approx(180.0, abs=1e-6)
    assert surf.intersection_plunge_deg() == pytest.approx(30.0, abs=1e-6)

    exact_fos = float(
        np.tan(np.radians(38.0)) / (np.tan(np.radians(30.0)) * np.cos(np.radians(35.0)))
    )
    res = run_analysis_3d(proj)
    for meth in ("janbu_simplified", "spencer", "gle_mp", "cheng_yip"):
        mr = res.method_results[meth]
        assert mr.ok and mr.admissible, f"{meth} failed on 3D rock wedge"
        assert mr.fos == pytest.approx(exact_fos, abs=1e-9)

    # JSON save/load round-trip preserves MultiPlaneWedgeSurface
    saved = proj.save(tmp_path / "wedge.slp3d.json")
    loaded = Project3D.load(saved)
    assert isinstance(loaded.user_slip_surface(), MultiPlaneWedgeSurface)


def test_3d_steady_seepage_solver_and_project_integration() -> None:
    """solve_seepage_3d satisfies boundary heads & seepage-face cap, and lowers FoS vs dry slope."""
    proj_dry = zhang_1988_homog_ellipsoid_project()
    proj_dry.materials[0].use_water = True
    proj_dry.materials[0].gamma_sat = 20.0
    proj_dry.analysis.pore_mode = "none"
    res_dry = run_analysis_3d(proj_dry)

    proj_seep = proj_dry.copy()
    proj_seep.analysis.pore_mode = "seepage"
    proj_seep.seepage_bc = SeepageBC3D(
        head_xmin=0.0, head_xmax=11.0, kx=2.0, ky=1.0, kz=1.0, nx=24, ny=16, nz=14
    )

    mdl = proj_seep.build_model()
    grid = solve_seepage_3d(mdl, bc=proj_seep.seepage_bc)
    # Uphill boundary head near 11 m, downhill toe near 0 m, positive pore pressure inside slope
    assert float(np.mean(grid.total_head[-1, :, 0])) == pytest.approx(11.0, abs=0.2)
    assert float(np.mean(grid.total_head[0, :, 0])) == pytest.approx(0.0, abs=0.2)
    u_mid = float(grid.pore_pressure_at(35.0, 70.0, 0.0))
    assert u_mid > 25.0

    phreatic = grid.phreatic_horizon()
    assert float(phreatic.z(50.0, 70.0)) > float(phreatic.z(10.0, 70.0))

    res_seep = run_analysis_3d(proj_seep)
    assert res_seep.primary_result.ok
    assert res_seep.critical_fos < res_dry.critical_fos - 0.15


def test_automatic_3d_bolt_pattern_generator_increases_fos() -> None:
    """generate_bolt_pattern_3d collars bolts onto terrain and increases 3D Factor of Safety."""
    proj = zhang_1988_homog_ellipsoid_project()
    res_unreinf = run_analysis_3d(proj)

    st = SupportType3D(
        name="Pattern Nail 180 kN",
        kind="soil_nail",
        tensile_kn=180.0,
        plate_kn=150.0,
        bond_strength_kn_m=60.0,
        application="active",
    )
    proj.support_types = [st]
    mdl = proj.build_model()
    spec = BoltPatternSpec3D(
        x_min=20.0,
        x_max=34.0,
        y_min=55.0,
        y_max=85.0,
        spacing_x=4.5,
        spacing_y=7.5,
        length_m=18.0,
        plunge_deg=15.0,
        trend_azimuth_deg=0.0,
        support_type_name=st.name,
        stagger_rows=True,
    )
    bolts = generate_bolt_pattern_3d(mdl.ground_z, spec)
    assert len(bolts) >= 12
    # Every bolt head sits directly on the ground surface
    for b in bolts:
        hx, hy, hz = b["head"]
        assert hz == pytest.approx(float(mdl.ground_z(hx, hy)), abs=1e-6)

    proj.bolts = bolts
    res_reinf = run_analysis_3d(proj)
    assert res_reinf.columns.support_hits
    assert res_reinf.critical_fos > res_unreinf.critical_fos + 0.04


def test_stl_binary_ascii_dem_and_xyz_csv_round_trips(tmp_path: Path) -> None:
    """Binary STL, ASCII STL, ESRI ASCII DEM (.asc), and XYZ/CSV point clouds round-trip accurately."""
    proj = zhang_1988_homog_ellipsoid_project()
    mdl = proj.build_model()
    assert mdl.ground is not None
    g_mesh = mdl.ground_mesh(nx=24, ny=20)

    # 1. Binary & ASCII STL round-trips
    stl_bin = export_stl_mesh(tmp_path / "ground_bin.stl", g_mesh, binary=True)
    stl_asc = export_stl_mesh(tmp_path / "ground_asc.stl", g_mesh, binary=False)
    h_bin = import_stl_horizon(stl_bin)
    h_asc = import_stl_horizon(stl_asc)
    assert float(h_bin.z(25.0, 70.0)) == pytest.approx(float(mdl.ground_z(25.0, 70.0)), abs=0.25)
    assert float(h_asc.z(25.0, 70.0)) == pytest.approx(float(mdl.ground_z(25.0, 70.0)), abs=0.25)

    slip_surf = proj.user_slip_surface()
    assert slip_surf is not None
    s_mesh = slip_surf.to_mesh(
        (mdl.x_min, mdl.x_max), (mdl.y_min, mdl.y_max), ground_fn=mdl.ground_z
    )
    stl_slip = export_stl_mesh(tmp_path / "slip.stl", s_mesh, binary=True)
    imp_slip = import_stl_slip_surface(stl_slip)
    assert float(imp_slip.z(24.0, 70.0)) == pytest.approx(float(slip_surf.z(24.0, 70.0)), abs=0.25)

    # 2. ESRI ASCII DEM (.asc) round-trip
    dem_file = export_esri_ascii_dem(
        tmp_path / "terrain.asc",
        mdl.ground,
        (mdl.x_min, mdl.x_max),
        (mdl.y_min, mdl.y_max),
        cellsize=2.0,
    )
    h_dem = import_esri_ascii_dem(dem_file)
    assert float(h_dem.z(27.0, 70.0)) == pytest.approx(float(mdl.ground_z(27.0, 70.0)), abs=0.30)

    # 3. XYZ / CSV point-cloud round-trip
    csv_file = export_xyz_csv_horizon(
        tmp_path / "survey.csv",
        mdl.ground,
        (mdl.x_min, mdl.x_max),
        (mdl.y_min, mdl.y_max),
        nx=25,
        ny=21,
    )
    h_csv = import_xyz_csv_horizon(csv_file)
    assert float(h_csv.z(27.0, 70.0)) == pytest.approx(float(mdl.ground_z(27.0, 70.0)), abs=0.25)
