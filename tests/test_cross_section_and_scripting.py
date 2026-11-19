"""Verification tests for Stage 3D-4: 2D Cross-Section F3D/F2D Comparison, .slp.json Export, and Scripting API."""

from __future__ import annotations

from pathlib import Path

import pytest

from slipsurface3d import scripting
from slipsurface3d.cross_section import compare_2d_cross_section, export_2d_slp_json
from slipsurface3d.geometry.surfaces import cylindrical_surface
from slipsurface3d.project import (
    extruded_acads_1a_project,
    import_2d_project,
    rock_wedge_3d_project,
    zhang_1988_homog_ellipsoid_project,
)


def test_cross_section_2d_3d_end_effect_and_slp_json_round_trip(tmp_path: Path) -> None:
    """F_3D / F_2D == 1.000 on extruded cylinders and > 1.0 on finite 3D ellipsoids & rock wedges."""
    # 1. Extruded 2D cylinder -> exact F_3D / F_2D == 1.0000
    p_cyl = extruded_acads_1a_project()
    p_cyl.surface_config.mode = "user"
    p_cyl.surface_config.user_surface = cylindrical_surface(
        20.0, 16.0, 17.0, y_center=20.0
    ).to_dict()
    p_cyl.analysis.nx = 50
    cmp_cyl = compare_2d_cross_section(p_cyl, y0=20.0, n_slices_2d=50)
    assert cmp_cyl.primary_ratio_3d_2d == pytest.approx(1.0, abs=1e-6)

    # 2. Zhang (1988) Ex. 1 finite 3D ellipsoid -> F_3D > F_2D (~1.02 - 1.05 on wide 200 ft bowl)
    p_z1 = zhang_1988_homog_ellipsoid_project()
    cmp_z1 = compare_2d_cross_section(p_z1, y0=70.0, n_slices_2d=48)
    assert 2.02 <= cmp_z1.primary_fos_2d <= 2.11
    assert cmp_z1.primary_fos_3d > cmp_z1.primary_fos_2d
    assert 1.01 <= cmp_z1.primary_ratio_3d_2d <= 1.08

    # 3. Hoek & Bray 3D Rock Wedge -> F_3D / F_2D == 1 / cos(35 deg) = 1.2208 along apex section y=25
    p_w = rock_wedge_3d_project()
    cmp_w = compare_2d_cross_section(p_w, y0=25.0, n_slices_2d=40)
    assert cmp_w.primary_ratio_3d_2d == pytest.approx(1.0 / 0.819152044, rel=2e-2)

    # 4. Exported 2D .slp.json re-imports cleanly into Project3D via import_2d_project
    slp_path = export_2d_slp_json(tmp_path / "extracted_section.slp.json", p_z1, y0=70.0)
    assert slp_path.is_file()
    reimported = import_2d_project(slp_path, y_min=0.0, y_max=20.0)
    reimported.validate()
    assert len(reimported.materials) == len(p_z1.materials)


def test_python_scripting_and_batch_api(tmp_path: Path) -> None:
    """Ergonomic scripting API creates, solves, runs probabilistic/ky/2D-section, and batch-solves."""
    proj = scripting.create_simple_slope_project(
        title="API Test Slope",
        height=10.0,
        ratio_h_to_v=2.0,
        c=12.0,
        phi=25.0,
        effort="fast",
    )
    saved = scripting.save_project(proj, tmp_path / "api_slope.slp3d.json")
    loaded = scripting.load_project(saved)
    res = scripting.solve(loaded, methods=["bishop", "gle_mp"], effort="fast", n_workers=2)
    assert res.primary_result.ok
    assert "gle_mp" in res.method_results
    assert "API Test Slope" in scripting.report_text_3d(loaded, res)

    prob = scripting.run_probabilistic(
        loaded,
        [scripting.RandomVarSpec3D("c", "Slope Soil", "lognormal", cov=0.15)],
        n_samples=30,
    )
    assert prob.n_converged == 30

    sens = scripting.run_sensitivity(loaded, n_points=5)
    assert len(sens) >= 2

    ky_res, nm_res = scripting.find_ky(loaded, pga_g=0.40, mw=6.5)
    assert ky_res.converged
    assert nm_res is not None

    sec = scripting.compare_2d_section(loaded, res)
    assert sec.primary_ratio_3d_2d > 0.95

    batch_out = scripting.batch_solve(
        [scripting.load_example("hungr_1987_hemisphere"), scripting.load_example("rock_wedge_3d")],
        n_workers=2,
    )
    assert len(batch_out) == 2
    assert all(r.primary_result.ok for _, r in batch_out)


def test_design_factors_consistency_across_modules_and_pdf_render(tmp_path: Path) -> None:
    """EC7 partial factors (incl. earth_resistance) stay consistent across 2D, probabilistic, seismic & PDF."""
    from slipsurface3d.design import design_preset
    from slipsurface3d.geometry.terrain import HorizonSurface, WeakLayer3D
    from slipsurface3d.loads.supports import GeotextileSheet3D, PileRow3D
    from slipsurface3d.probabilistic import (
        RandomVarSpec3D,
        run_probabilistic_3d,
        run_sensitivity_3d,
    )
    from slipsurface3d.project import MaterialSpec3D
    from slipsurface3d.report.pdf import render_offscreen_scene_png
    from slipsurface3d.seismic import find_yield_acceleration_3d

    # Extruded cylinder under EC7 DA2 (gamma_G=1.35, gamma_R;e=1.1): F_3D/F_2D must remain 1.000
    p_ec7 = extruded_acads_1a_project()
    p_ec7.design = design_preset("ec7_da2")
    p_ec7.surface_config.mode = "user"
    p_ec7.surface_config.user_surface = cylindrical_surface(
        20.0, 16.0, 17.0, y_center=20.0
    ).to_dict()
    p_ec7.analysis.methods = ["bishop"]
    p_ec7.surface_config.primary_method = "bishop"
    p_ec7.analysis.nx = 40

    res_ec7 = scripting.solve(p_ec7)
    cmp_ec7 = compare_2d_cross_section(p_ec7, results=res_ec7, y0=20.0, n_slices_2d=40)
    assert cmp_ec7.primary_ratio_3d_2d == pytest.approx(1.0, abs=1e-4)

    # Probabilistic fixed-surface mean with tiny CoV must match deterministic EC7 ODF
    prob = run_probabilistic_3d(
        p_ec7,
        [RandomVarSpec3D("phi", p_ec7.materials[0].name, "normal", cov=0.001)],
        n_samples=8,
        fixed_surface=True,
    )
    assert prob.mean_fos == pytest.approx(res_ec7.critical_fos, rel=5e-3)

    # Sensitivity midpoint at base phi must match deterministic EC7 ODF
    base_phi = p_ec7.materials[0].phi
    curves = run_sensitivity_3d(
        p_ec7,
        specs=[
            scripting.SensitivityParamSpec3D(
                "phi", p_ec7.materials[0].name, base_phi - 2.0, base_phi + 2.0
            )
        ],
        n_points=3,
    )
    assert curves[0].fos_values[1] == pytest.approx(res_ec7.critical_fos, rel=1e-3)

    # Seismic yield acceleration static_fos must match deterministic EC7 ODF
    ky_res = find_yield_acceleration_3d(p_ec7)
    assert ky_res.static_fos == pytest.approx(res_ec7.critical_fos, rel=1e-3)

    # Scripting overrides + weak layer, geotextile, pile row rendered in PDF offscreen scene
    p_rich = scripting.create_simple_slope_project(methods=("bishop",), effort="fast")
    p_rich.materials.append(MaterialSpec3D(name="Weak", c=2.0, phi=12.0, use_water=False))
    p_rich.weak_layers = [
        WeakLayer3D(
            name="Seam",
            material="Weak",
            horizon=HorizonSurface.from_plane(
                (p_rich.x_min, p_rich.x_max), (p_rich.y_min, p_rich.y_max), 2.0, name="Seam"
            ),
        )
    ]
    p_rich.geotextiles = [GeotextileSheet3D(x1=20.0, x2=42.0, y1=10.0, y2=30.0, z=6.0)]
    p_rich.pile_rows = [PileRow3D(x1=22.0, y1=12.0, x2=22.0, y2=28.0, spacing=4.0, length_m=8.0)]
    res_pso = scripting.solve(p_rich, algorithm="pso", design="ec7_da2", n_local_minima=2)
    assert res_pso.primary_result.ok
    png = render_offscreen_scene_png(p_rich, res_pso, size=(640, 420))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
