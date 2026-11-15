"""Offscreen PySide6 + PyVista 3D Desktop UI tests."""

from __future__ import annotations

from pathlib import Path

from slipsurface3d.project import Project3D
from slipsurface3d.ui.dialogs import (
    AnalysisSurfaceDialog,
    BoreholesDialog,
    ColumnDataDialog,
    LoadsSupports3DDialog,
    Materials3DDialog,
    ParametricSlopeDialog,
)
from slipsurface3d.ui.main_window import MainWindow3D
from slipsurface3d.ui.preferences import apply_theme


def test_main_window_3d_compute_views_undo_and_exports(qapp, tmp_path: Path):
    win = MainWindow3D()
    win.show()
    qapp.processEvents()

    assert win.viewport.is_offscreen
    win.load_builtin_example("zhang_1988_homog")
    win.compute_analysis(synchronous=True)
    assert win.results is not None
    assert win.results.primary_result.ok

    # Pick a column, switch scalar fields & camera views, toggle clipping & theme
    win.viewport.select_column(0)
    assert "Selected Column #0" in win.col_info.toPlainText()

    win.viewport.set_scalar_field("TauMob_kPa")
    win.viewport.set_camera_view("xy")
    win.viewport.set_layer_visibility(clip_mid_y=True)
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

    # AnalysisSurfaceDialog
    d_an = AnalysisSurfaceDialog(win.project, win)
    d_an.nx_sp.setValue(22)
    d_an._on_accept()
    assert win.project.analysis.nx == 22
    d_an.close()

    # ColumnDataDialog
    d_col = ColumnDataDialog(win.results, "bishop", win)
    assert d_col.table.rowCount() == win.results.columns.n
    d_col.close()
    win.close()
