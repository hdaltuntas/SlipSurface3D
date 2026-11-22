"""Offscreen PySide6 + PyVista 3D Desktop UI tests."""

from __future__ import annotations

from pathlib import Path

from slipsurface3d.project import Project3D
from slipsurface3d.ui import configure_qt_vtk_env
from slipsurface3d.ui.dialogs import (
    AnalysisSurfaceDialog,
    BoreholesDialog,
    ColumnDataDialog,
    CrossSectionComparisonDialog,
    LoadsSupports3DDialog,
    Materials3DDialog,
    ParametricSlopeDialog,
    ProbabilisticSensitivity3DDialog,
    SeismicNewmark3DDialog,
)
from slipsurface3d.ui.main_window import MainWindow3D
from slipsurface3d.ui.preferences import ABOUT_HTML, apply_theme


def test_main_window_3d_compute_views_undo_and_exports(qapp, tmp_path: Path):
    win = MainWindow3D()
    win.show()
    qapp.processEvents()

    assert win.viewport.is_offscreen
    assert win.project.ground is None
    assert win.project.summary.title == "Untitled 3D Project"
    win.load_builtin_example("zhang_1988_homog")
    win.compute_analysis(synchronous=True)
    assert win.results is not None
    assert win.results.primary_result.ok

    # Pick a column, switch scalar fields & camera views, toggle clipping & theme
    win.viewport.select_column(0)
    assert "Selected Column #0" in win.col_info.toPlainText()

    win.viewport.set_scalar_field("TauMob_kPa")
    win.viewport.set_camera_view("xy")
    win.viewport.set_layer_visibility(clip_mid_y=True, safety_map=True, force_vectors=True)
    win.viewport.preview_trial_surface(win.results.critical_surface)
    win.toggle_theme()
    apply_theme(qapp, "dark")

    # Save & reopen .slp3d.json
    saved_file = tmp_path / "ui_test.slp3d.json"
    win.project_path = saved_file
    win.save_project()
    assert saved_file.is_file()

    win.load_builtin_example("hungr_1987_hemisphere")
    win.undo()
    assert "Zhang" in win.project.summary.title
    win.redo()
    assert "Hungr" in win.project.summary.title

    win.open_file(saved_file)
    assert Project3D.load(saved_file).summary.title == win.project.summary.title

    shot_path = tmp_path / "shot.png"
    win.viewport.export_screenshot(str(shot_path), width=480, height=320)
    assert shot_path.is_file() and shot_path.stat().st_size > 1000

    win.close()


def test_dialogs_construct_and_apply_cleanly(qapp):
    win = MainWindow3D()
    win.load_builtin_example("zhang_1988_homog")
    win.compute_analysis(synchronous=True)
    assert win.results is not None

    # ParametricSlopeDialog
    d_slope = ParametricSlopeDialog(win.project, win)
    d_slope.height_sp.setValue(14.0)
    d_slope.wt_chk.setChecked(True)
    d_slope.apply_to_project(win.project)
    assert win.project.water_table is not None
    d_slope.close()

    # Materials3DDialog
    d_mat = Materials3DDialog(win.project.materials, win)
    d_mat._add_mat()
    d_mat._on_accept()
    assert len(d_mat.materials) == 2
    d_mat.close()

    # BoreholesDialog
    win.project.materials = d_mat.materials
    d_bh = BoreholesDialog(win.project, win)
    d_bh._seed_four(win.project)
    d_bh._on_accept()
    assert len(d_bh.boreholes) == 4
    d_bh.close()

    # LoadsSupports3DDialog
    d_ld = LoadsSupports3DDialog(win.project, win)
    d_ld._add_patch()
    d_ld._add_bolt()
    d_ld._on_accept()
    assert len(d_ld.patch_loads) == 1 and len(d_ld.bolts) == 1
    d_ld.close()

    # AnalysisSurfaceDialog — keep primary method when extra solvers are ticked
    d_an = AnalysisSurfaceDialog(win.project, win)
    assert win.project.surface_config.primary_method == "bishop"
    d_an.nx_sp.setValue(22)
    d_an.meth_checks["ordinary"].setChecked(True)
    d_an._on_accept()
    assert win.project.analysis.nx == 22
    assert "ordinary" in win.project.analysis.methods
    assert win.project.surface_config.primary_method == "bishop"
    d_an.close()

    # ColumnDataDialog
    d_col = ColumnDataDialog(win.results, "bishop", win)
    assert d_col.table.rowCount() == win.results.columns.n
    d_col.close()

    # Advanced Stage 3D-4 Dialogs (Probabilistic, Seismic ky/Newmark, 2D Cross-Section)
    d_prob = ProbabilisticSensitivity3DDialog(win.project, win.results, win)
    d_prob.samples_sp.setValue(24)
    d_prob.execute_probabilistic()
    assert "Mean FoS" in d_prob.summary_lbl.text()
    d_prob.execute_sensitivity()
    assert d_prob.table.rowCount() >= 2
    d_prob.close()

    d_seis = SeismicNewmark3DDialog(win.project, win.results, win)
    d_seis.execute_seismic()
    assert "Critical Yield Acceleration" in d_seis.out_lbl.text()
    d_seis.close()

    d_sec = CrossSectionComparisonDialog(win.project, win.results, win)
    d_sec.evaluate_section()
    assert d_sec.table.rowCount() >= 1
    d_sec.close()

    win.close()


def test_about_dialog_agpl_notice(qapp):
    assert "GNU AGPL" in ABOUT_HTML
    assert "Hasan Deniz Altuntaş" in ABOUT_HTML
    win = MainWindow3D()
    help_titles = [a.text() for a in win.menuBar().actions()]
    assert any("Help" in t for t in help_titles)
    assert win.act_about.text().startswith("About ")
    win.close()


def test_wayland_and_c_locale_env_guard():
    env_wayland = {
        "XDG_SESSION_TYPE": "wayland",
        "WAYLAND_DISPLAY": "wayland-1",
        "DISPLAY": ":0",
        "LC_ALL": "C",
        "LANG": "POSIX",
    }
    configure_qt_vtk_env(env=env_wayland, platform_str="linux")
    assert env_wayland["QT_QPA_PLATFORM"] == "xcb"
    assert env_wayland["LC_ALL"] == "C.UTF-8"
    assert env_wayland["LANG"] == "C.UTF-8"
    assert env_wayland["QT_API"] == "pyside6"

    # Explicit QT_QPA_PLATFORM=offscreen must never be overridden
    env_off = {
        "QT_QPA_PLATFORM": "offscreen",
        "WAYLAND_DISPLAY": "wayland-1",
        "DISPLAY": ":0",
    }
    configure_qt_vtk_env(env=env_off, platform_str="linux")
    assert env_off["QT_QPA_PLATFORM"] == "offscreen"
    assert env_off["LC_NUMERIC"] == "C.UTF-8"


def test_design_standard_multimodal_and_new_material_pages(qapp, tmp_path: Path):
    from slipsurface3d.report.pdf import export_pdf_report_3d
    from slipsurface3d.scripting import create_simple_slope_project

    proj = create_simple_slope_project(methods=("bishop",), width_y=120.0, effort="fast")
    proj.surface_config.ry_bounds = [6.0, 18.0]

    # Materials: generic pages are editable, the shear/normal table parses, bad input is refused
    mdlg = Materials3DDialog(proj.materials)
    assert mdlg.stack.count() == mdlg.model_cb.count()
    mdlg.model_cb.setCurrentIndex(mdlg.model_cb.findData("BB"))
    dict(mdlg._generic_spins["BB"])["bb_jrc"].setValue(7.5)
    mdlg._save_current()
    assert mdlg.materials[0].model == "BB" and mdlg.materials[0].bb_jrc == 7.5
    mdlg.model_cb.setCurrentIndex(mdlg.model_cb.findData("SN"))
    mdlg.sn_edit.setPlainText("0, 5\n200; 120\n")
    mdlg._save_current()
    assert mdlg.materials[0].sn_table == [[0.0, 5.0], [200.0, 120.0]]
    mdlg.sn_edit.setPlainText("not a table")
    assert mdlg._parse_sn_table() is None
    mdlg.done(0)

    # Analysis settings: design standard preset, distinct surfaces and plan limits
    adlg = AnalysisSurfaceDialog(proj)
    adlg.design_cb.setCurrentIndex(adlg.design_cb.findData("ec7_da1_c2"))
    assert adlg.design_spins["friction"].value() == 1.25
    assert not adlg.design_spins["friction"].isEnabled()
    adlg.design_cb.setCurrentIndex(adlg.design_cb.findData("user"))
    assert adlg.design_spins["friction"].isEnabled()
    adlg.design_spins["friction"].setValue(1.3)
    adlg.n_minima_sp.setValue(3)
    adlg.limits_chk.setChecked(True)
    adlg.lim_y0.setValue(10.0)
    adlg.lim_y1.setValue(110.0)
    adlg._on_accept()
    assert proj.design.key == "user" and proj.design.friction == 1.3
    assert proj.design.cohesion == 1.25  # carried over from the preset that was showing
    assert proj.surface_config.n_local_minima == 3
    assert proj.surface_config.search_limits == [proj.x_min, proj.x_max, 10.0, 110.0]

    win = MainWindow3D(proj)
    win.show()
    qapp.processEvents()
    win.compute_analysis(synchronous=True)
    assert win.results is not None and len(win.results.local_minima) >= 2
    assert "Critical ODF" in win.fos_banner.text()
    win.chk_local_minima.setChecked(False)
    win.chk_local_minima.setChecked(True)
    shot = tmp_path / "minima.png"
    win.viewport.export_screenshot(str(shot))
    assert shot.stat().st_size > 2000
    pdf = export_pdf_report_3d(tmp_path / "design.pdf", win.project, win.results)
    assert pdf.stat().st_size > 5000
    win.viewport.cleanup()
    win.close()


def test_weak_layers_piezo_dialog_and_viewport(qapp, tmp_path: Path):
    import numpy as np

    from slipsurface3d.project import MaterialSpec3D
    from slipsurface3d.scripting import create_simple_slope_project
    from slipsurface3d.ui.dialogs import WeakLayersPiezoDialog

    proj = create_simple_slope_project(methods=("bishop",), effort="fast")
    proj.materials[0].use_water = True
    proj.materials.append(MaterialSpec3D(name="Slick", c=0.0, phi=12.0, color="#f43f5e"))

    dlg = WeakLayersPiezoDialog(proj)
    dlg.add_planar_weak_layer(4.0, 5.0, 180.0, material="Slick")
    name = dlg.add_planar_piezo_surface(9.0)
    dlg.assign_piezo("Slope Soil", name)
    dlg.pz_table.item(0, 0).setText("Perched")
    csv_path = tmp_path / "u.csv"
    csv_path.write_text("x,y,z,u\n0,0,0,50\n10,0,0,40\n0,10,0,45\n0,0,5,5\n10,10,5,0\n")
    from slipsurface3d.porewater import read_pore_pressure_csv

    dlg.set_pore_grid(read_pore_pressure_csv(csv_path))
    assert "5 samples" in dlg.grid_label.text()
    dlg._on_accept()
    dlg.apply_to_project(proj)
    assert proj.weak_layers[0].material == "Slick"
    assert proj.piezo_surfaces[0][0] == "Perched"
    assert proj.materials[0].piezo == "Perched"  # assignment follows the rename
    assert len(proj.pore_grid) == 5
    proj.validate()

    # Removing the surface clears the assignment that pointed at it
    dlg2 = WeakLayersPiezoDialog(proj)
    dlg2.pz_table.setCurrentCell(0, 0)
    dlg2._remove_piezo()
    dlg2._on_accept()
    assert dlg2.material_piezo["Slope Soil"] == ""
    dlg2.done(0)

    win = MainWindow3D(proj)
    win.show()
    qapp.processEvents()
    win.compute_analysis(synchronous=True)
    assert win.results is not None and win.results.primary_result.ok
    assert float(np.max(win.results.columns.u)) > 0.0
    shot = tmp_path / "layers.png"
    win.viewport.export_screenshot(str(shot))
    assert shot.stat().st_size > 2000
    saved = tmp_path / "layers.slp3d.json"
    win.project.save(saved)
    assert len(Project3D.load(saved).weak_layers) == 1
    win.viewport.cleanup()
    win.close()


def test_supports_stages_dialogs_and_stage_display(qapp, tmp_path: Path):
    from slipsurface3d.project import MaterialSpec3D
    from slipsurface3d.scripting import create_simple_slope_project
    from slipsurface3d.ui.dialogs import (
        StagedAnalysisDialog,
        SupportsStagesDialog,
        WeakLayersPiezoDialog,
    )

    proj = create_simple_slope_project(methods=("bishop",), effort="fast")
    proj.materials.append(MaterialSpec3D(name="Fill", c=5.0, phi=32.0, use_water=False))

    dlg = SupportsStagesDialog(proj)
    dlg._add_geotextile()
    dlg._add_pile_row()
    dlg._add_stage()
    dlg._add_stage()
    # Second stage: a crest fill in the "Fill" material
    kind_cb = dlg.stage_table.cellWidget(1, 1)
    kind_cb.setCurrentIndex(kind_cb.findText("fill"))
    mat_cb = dlg.stage_table.cellWidget(1, 8)
    mat_cb.setCurrentIndex(mat_cb.findText("Fill"))
    for col, text in ((2, "45"), (3, "59"), (6, "15")):
        dlg.stage_table.item(1, col).setText(text)
    from slipsurface3d.loads.supports import GeotextileSheet3D

    dlg.geo_table.item(0, 9).setText("not a number")
    try:
        dlg.geo_table.records(GeotextileSheet3D, GeotextileSheet3D())
        raise AssertionError("a non-numeric cell must be rejected")
    except ValueError as exc:
        assert "Row 1" in str(exc) and "Tensile" in str(exc)
    dlg.geo_table.item(0, 9).setText("75,5")  # decimal comma is accepted
    dlg._on_accept()
    dlg.apply_to_project(proj)
    assert proj.geotextiles[0].tensile_kn_m == 75.5
    assert len(proj.pile_rows) == 1
    assert [s.kind for s in proj.stages] == ["excavation", "fill"]
    assert proj.stages[1].fill_material == "Fill" and proj.stages[1].level == 15.0
    proj.validate()

    # Weak-layer seam thickness is editable in its table
    ldlg = WeakLayersPiezoDialog(proj)
    ldlg.add_planar_weak_layer(3.0, material="Fill")
    ldlg.wl_table.item(0, 2).setText("0.25")
    ldlg._on_accept()
    assert ldlg.weak_layers[0].thickness == 0.25
    ldlg.done(0)

    win = MainWindow3D(proj)
    win.show()
    qapp.processEvents()
    sdlg = StagedAnalysisDialog(proj, win)
    sdlg.run_stages()
    assert sdlg.table.rowCount() == 3
    assert all(sr.results is not None for sr in sdlg.stage_results)
    sdlg.table.selectRow(2)
    sdlg._show_selected()
    assert sdlg.selected is not None and sdlg.selected.name == "Stage 2"
    win.show_stage_result(sdlg.selected)
    assert win.results is sdlg.selected.results
    assert float(win.results.model.ground_z(52.0, 20.0)) == 15.0
    shot = tmp_path / "stage.png"
    win.viewport.export_screenshot(str(shot))
    assert shot.stat().st_size > 2000
    assert len(win.project.stages) == 2  # showing a stage does not alter the project
    win.viewport.cleanup()
    win.close()
