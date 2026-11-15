"""Tests for global 3D critical surface search, PDF report generation, and 3D DXF/OBJ exchange."""

from __future__ import annotations

from pathlib import Path

from slipsurface3d.analysis import report_text_3d, run_analysis_3d
from slipsurface3d.exchange import (
    export_dxf_3d,
    export_obj_mesh,
    import_dxf_3d_vertices,
    import_obj_horizon,
    import_obj_slip_surface,
)
from slipsurface3d.project import extruded_acads_1a_project, reinforced_3d_valley_project
from slipsurface3d.report import export_pdf_report_3d, render_offscreen_scene_png


def test_3d_global_search_sphere_ellipsoid_and_spline():
    proj = extruded_acads_1a_project()
    proj.analysis.nx = 18
    proj.analysis.ny = 14
    proj.surface_config.effort = "fast"

    # Sphere search
    proj.surface_config.surface_kind = "sphere"
    res_sph = run_analysis_3d(proj)
    assert res_sph.primary_result.ok
    assert 0.95 <= res_sph.critical_fos <= 1.35

    # Ellipsoid search
    proj.surface_config.surface_kind = "ellipsoid"
    res_ell = run_analysis_3d(proj)
    assert res_ell.primary_result.ok
    assert 0.95 <= res_ell.critical_fos <= 1.30

    # Spline patch search
    proj.surface_config.surface_kind = "spline_patch"
    res_sp = run_analysis_3d(proj)
    assert res_sp.primary_result.ok
    txt = report_text_3d(proj, res_sp)
    assert "3D Bishop Simplified" in txt


def test_reinforced_valley_pdf_report_and_cad_exchange(tmp_path: Path):
    proj = reinforced_3d_valley_project()
    proj.analysis.nx = 20
    proj.analysis.ny = 16
    res = run_analysis_3d(proj)
    assert res.primary_result.ok

    # Offscreen PyVista PNG & ReportLab PDF
    png = render_offscreen_scene_png(proj, res, camera_view="iso", size=(640, 420))
    assert png.startswith(b"\x89PNG")

    pdf_path = export_pdf_report_3d(tmp_path / "valley_report.pdf", proj, res)
    assert pdf_path.is_file() and pdf_path.stat().st_size > 15_000
    assert pdf_path.read_bytes().startswith(b"%PDF")

    # DXF 3DFACE roundtrip
    model = res.model
    crit_mesh = res.critical_surface.to_mesh(
        model.x_bounds, model.y_bounds, ground_fn=model.ground_z, nx=24, ny=24
    )
    dxf_path = export_dxf_3d(
        tmp_path / "valley_3d.dxf", model, critical_mesh=crit_mesh, nx=16, ny=12
    )
    pts_dxf = import_dxf_3d_vertices(dxf_path, layer="GROUND_3D")
    assert pts_dxf.shape[1] == 3 and len(pts_dxf) >= 50

    # OBJ mesh roundtrip
    obj_path = export_obj_mesh(tmp_path / "slip.obj", [("CriticalSlip", crit_mesh)])
    hor_obj = import_obj_horizon(obj_path)
    slip_obj = import_obj_slip_surface(obj_path)
    assert hor_obj.points is not None and len(hor_obj.points) >= 10
    assert len(slip_obj.vertices) >= 10
