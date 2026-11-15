"""Main 3D Modeler & Interpret Desktop Window for SlipSurfaceIQ-3D."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from slipsurface3d import APP_NAME, __version__
from slipsurface3d.analysis import AnalysisResults3D, report_text_3d, run_analysis_3d
from slipsurface3d.exchange import (
    export_dxf_3d,
    export_esri_ascii_dem,
    export_obj_mesh,
    export_stl_mesh,
    import_horizon_file,
    import_obj_slip_surface,
    import_stl_slip_surface,
)
from slipsurface3d.lem.solvers import METHODS_3D
from slipsurface3d.project import (
    EXAMPLE_PROJECTS_3D,
    Project3D,
    extruded_acads_1a_project,
    import_2d_project,
)
from slipsurface3d.report.pdf import export_pdf_report_3d
from slipsurface3d.ui.dialogs import (
    AnalysisSurfaceDialog,
    BoreholesDialog,
    ColumnDataDialog,
    CrossSectionComparisonDialog,
    Import2DExtrusionDialog,
    LoadsSupports3DDialog,
    Materials3DDialog,
    ParametricSlopeDialog,
    ProbabilisticSensitivity3DDialog,
    SeismicNewmark3DDialog,
    StagedAnalysisDialog,
    SupportsStagesDialog,
    WeakLayersPiezoDialog,
)
from slipsurface3d.ui.preferences import ABOUT_HTML, apply_theme, current_theme, push_recent_file
from slipsurface3d.ui.viewport3d import SCALAR_FIELD_OPTIONS, Viewport3D


class _ComputeWorker3D(QThread):
    progress = Signal(int, int, float)
    best_surface = Signal(object, float)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, project: Project3D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project = project.copy()
        self._cancelled = False

    def request_cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            res = run_analysis_3d(
                self._project,
                progress_cb=lambda g, n, f: self.progress.emit(g, n, f),
                best_surface_cb=lambda s, f: self.best_surface.emit(s, f),
                cancel_check=lambda: self._cancelled,
            )
            self.finished_ok.emit(res)
        except Exception as exc:
            self.failed.emit(str(exc))


def _safe_icon(name: str, color: str) -> QIcon | None:
    if QApplication.instance() is None:
        return None
    try:
        import qtawesome as qta

        ic = qta.icon(name, color=color)
        return ic if isinstance(ic, QIcon) else None
    except Exception:
        return None


class MainWindow3D(QMainWindow):
    """Unified 3D Modeler + Interpret Main Window."""

    def __init__(self, project: Project3D | None = None) -> None:
        super().__init__()
        self.project: Project3D = project or extruded_acads_1a_project()
        self.results: AnalysisResults3D | None = None
        self.project_path: Path | None = None
        self._undo_stack: list[dict[str, Any]] = []
        self._redo_stack: list[dict[str, Any]] = []
        self._worker: _ComputeWorker3D | None = None

        self.setWindowTitle(f"{APP_NAME} {__version__} — {self.project.summary.title}")
        self.resize(1380, 860)

        # Central 3D PyVista/VTK viewport
        self.viewport = Viewport3D(self)
        self.viewport.set_theme(current_theme())
        self.viewport.columnSelected.connect(self._on_column_picked)
        self.setCentralWidget(self.viewport)

        self._build_actions_and_menus()
        self._build_toolbar()
        self._build_docks()
        self._build_statusbar()

        self._refresh_all(reset_camera=True)

    # --------------------------------------------------------- UI Construction
    def _icon_fg(self) -> str:
        return "#e2e8f0" if current_theme() == "dark" else "#1e293b"

    def _build_actions_and_menus(self) -> None:
        fg = self._icon_fg()
        mb = self.menuBar()

        # File Menu
        file_m = mb.addMenu("&File")
        self.act_new = QAction("New Parametric 3D Slope...", self)
        self.act_new.setShortcut(QKeySequence.StandardKey.New)
        self.act_new.triggered.connect(self.new_parametric_slope)
        file_m.addAction(self.act_new)

        self.act_open = QAction("Open 3D / 2D Project...", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.open_project_dialog)
        file_m.addAction(self.act_open)

        self.act_import2d = QAction("Import & Extrude 2D (.slp.json)...", self)
        self.act_import2d.triggered.connect(self.import_2d_dialog)
        file_m.addAction(self.act_import2d)

        ex_menu = file_m.addMenu("Built-in 3D Examples & Benchmarks")
        for key, (label, _) in EXAMPLE_PROJECTS_3D.items():
            act = QAction(label, self)
            act.triggered.connect(lambda _chk=False, k=key: self.load_builtin_example(k))
            ex_menu.addAction(act)

        file_m.addSeparator()
        self.act_save = QAction("Save Project", self)
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_save.triggered.connect(self.save_project)
        file_m.addAction(self.act_save)

        self.act_save_as = QAction("Save Project As...", self)
        self.act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.act_save_as.triggered.connect(self.save_project_as)
        file_m.addAction(self.act_save_as)

        file_m.addSeparator()
        cad_m = file_m.addMenu("3D CAD / GIS / Mesh Exchange")
        act_exp_dxf = QAction("Export 3D DXF (3DFACE)...", self)
        act_exp_dxf.triggered.connect(self.export_dxf_dialog)
        cad_m.addAction(act_exp_dxf)

        act_exp_obj = QAction("Export Wavefront OBJ Mesh...", self)
        act_exp_obj.triggered.connect(self.export_obj_dialog)
        cad_m.addAction(act_exp_obj)

        act_exp_stl = QAction("Export Binary STL Mesh...", self)
        act_exp_stl.triggered.connect(self.export_stl_dialog)
        cad_m.addAction(act_exp_stl)

        act_exp_dem = QAction("Export Ground ESRI ASCII DEM (.asc)...", self)
        act_exp_dem.triggered.connect(self.export_dem_dialog)
        cad_m.addAction(act_exp_dem)

        cad_m.addSeparator()
        act_imp_terr = QAction("Import Terrain Horizon (DXF / OBJ / STL / DEM / CSV)...", self)
        act_imp_terr.triggered.connect(self.import_terrain_mesh_dialog)
        cad_m.addAction(act_imp_terr)

        act_imp_slip = QAction("Import User 3D Slip Surface (OBJ / STL)...", self)
        act_imp_slip.triggered.connect(self.import_slip_obj_dialog)
        cad_m.addAction(act_imp_slip)

        file_m.addSeparator()
        self.act_pdf = QAction("Export Engineering PDF Report...", self)
        self.act_pdf.triggered.connect(self.export_pdf_dialog)
        file_m.addAction(self.act_pdf)

        act_shot = QAction("Save 3D Viewport Screenshot (PNG)...", self)
        act_shot.triggered.connect(self.export_screenshot_dialog)
        file_m.addAction(act_shot)

        file_m.addSeparator()
        act_quit = QAction("Exit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_m.addAction(act_quit)

        # Edit Menu
        edit_m = mb.addMenu("&Edit")
        self.act_undo = QAction("Undo", self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.triggered.connect(self.undo)
        edit_m.addAction(self.act_undo)

        self.act_redo = QAction("Redo", self)
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.act_redo.triggered.connect(self.redo)
        edit_m.addAction(self.act_redo)

        # Model Menu
        mod_m = mb.addMenu("&Model")
        self.act_materials = QAction("Materials & Constitutive Models...", self)
        self.act_materials.triggered.connect(self.edit_materials)
        mod_m.addAction(self.act_materials)

        self.act_boreholes = QAction("3D Boreholes & RBF Horizons...", self)
        self.act_boreholes.triggered.connect(self.edit_boreholes)
        mod_m.addAction(self.act_boreholes)

        self.act_loads = QAction("3D Loads, Reinforcement Bolts & Crack...", self)
        self.act_loads.triggered.connect(self.edit_loads_supports)
        mod_m.addAction(self.act_loads)

        self.act_layers = QAction("Weak Layers, Piezometric Surfaces & Pore Pressure Grid...", self)
        self.act_layers.triggered.connect(self.edit_weak_layers_piezo)
        mod_m.addAction(self.act_layers)

        self.act_supports_stages = QAction("Geotextiles, Pile Rows & Construction Stages...", self)
        self.act_supports_stages.triggered.connect(self.edit_supports_stages)
        mod_m.addAction(self.act_supports_stages)

        # Analysis Menu
        an_m = mb.addMenu("&Analysis")
        self.act_settings = QAction("Solvers, Columns & Slip Surface Settings...", self)
        self.act_settings.triggered.connect(self.edit_analysis_settings)
        an_m.addAction(self.act_settings)

        self.act_compute = QAction("Compute 3D Stability (F9)", self)
        self.act_compute.setShortcut(Qt.Key.Key_F9)
        ic_play = _safe_icon("fa5s.play", "#22c55e")
        if ic_play is not None:
            self.act_compute.setIcon(ic_play)
        self.act_compute.triggered.connect(self.compute_analysis)
        an_m.addAction(self.act_compute)

        self.act_cancel = QAction("Stop Search", self)
        self.act_cancel.setEnabled(False)
        ic_stop = _safe_icon("fa5s.stop", "#ef4444")
        if ic_stop is not None:
            self.act_cancel.setIcon(ic_stop)
        self.act_cancel.triggered.connect(self.cancel_analysis)
        an_m.addAction(self.act_cancel)

        self.act_col_table = QAction("Inspect 3D Column Table...", self)
        self.act_col_table.triggered.connect(self.open_column_table)
        an_m.addAction(self.act_col_table)

        an_m.addSeparator()
        self.act_prob = QAction("Probabilistic & Tornado Sensitivity...", self)
        self.act_prob.triggered.connect(self.open_probabilistic_dialog)
        an_m.addAction(self.act_prob)

        self.act_seismic = QAction("Seismic ky & Newmark Displacement...", self)
        self.act_seismic.triggered.connect(self.open_seismic_dialog)
        an_m.addAction(self.act_seismic)

        self.act_stages = QAction("Run Construction Stages...", self)
        self.act_stages.triggered.connect(self.open_staged_analysis_dialog)
        an_m.addAction(self.act_stages)

        self.act_section2d = QAction("2D Cross-Section & F3D/F2D Comparison...", self)
        self.act_section2d.triggered.connect(self.open_cross_section_dialog)
        an_m.addAction(self.act_section2d)

        # View Menu
        view_m = mb.addMenu("&View")
        for label, preset in (
            ("Isometric 3D View", "iso"),
            ("Plan View (XY)", "xy"),
            ("Front Section View (XZ)", "xz"),
            ("Side View (YZ)", "yz"),
        ):
            act_v = QAction(label, self)
            act_v.triggered.connect(lambda _c=False, pr=preset: self.viewport.set_camera_view(pr))
            view_m.addAction(act_v)

        view_m.addSeparator()
        self.act_theme = QAction("Toggle Dark / Light Theme", self)
        self.act_theme.triggered.connect(self.toggle_theme)
        view_m.addAction(self.act_theme)

        help_m = mb.addMenu("&Help")
        self.act_about = QAction(f"About {APP_NAME}...", self)
        self.act_about.triggered.connect(self.show_about)
        help_m.addAction(self.act_about)

        # Assign subtle toolbar icons
        for act, ic_name in (
            (self.act_new, "fa5s.cube"),
            (self.act_open, "fa5s.folder-open"),
            (self.act_import2d, "fa5s.layer-group"),
            (self.act_save, "fa5s.save"),
            (self.act_materials, "fa5s.sliders-h"),
            (self.act_boreholes, "fa5s.map-marker-alt"),
            (self.act_loads, "fa5s.Compress-arrows-alt".lower()),
            (self.act_settings, "fa5s.cogs"),
            (self.act_pdf, "fa5s.file-pdf"),
        ):
            ic = _safe_icon(ic_name, fg)
            if ic is not None:
                act.setIcon(ic)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main 3D Toolbar", self)
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        tb.addAction(self.act_new)
        tb.addAction(self.act_open)
        tb.addAction(self.act_import2d)
        tb.addAction(self.act_save)
        tb.addSeparator()
        tb.addAction(self.act_materials)
        tb.addAction(self.act_boreholes)
        tb.addAction(self.act_loads)
        tb.addAction(self.act_settings)
        tb.addSeparator()
        tb.addAction(self.act_compute)
        tb.addAction(self.act_cancel)
        tb.addSeparator()
        tb.addAction(self.act_prob)
        tb.addAction(self.act_seismic)
        tb.addAction(self.act_section2d)
        tb.addSeparator()
        tb.addAction(self.act_pdf)

    def _build_docks(self) -> None:
        # Left Dock: 3D Scene & Display Controls
        ctrl_dock = QDockWidget("3D Scene & Display Controls", self)
        ctrl_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        w_ctrl = QWidget()
        vc = QVBoxLayout(w_ctrl)

        cam_box = QGroupBox("Camera Presets")
        cam_row = QHBoxLayout(cam_box)
        for txt, code in (("Iso", "iso"), ("Plan XY", "xy"), ("Front XZ", "xz"), ("Side YZ", "yz")):
            b = QPushButton(txt)
            b.clicked.connect(lambda _c=False, cd=code: self.viewport.set_camera_view(cd))
            cam_row.addWidget(b)
        vc.addWidget(cam_box)

        vis_box = QGroupBox("3D Layers & Contours")
        fvis = QFormLayout(vis_box)
        self.method_combo = QComboBox()
        self.method_combo.currentIndexChanged.connect(self._on_method_combo_changed)
        fvis.addRow("Active LEM Method:", self.method_combo)

        self.scalar_combo = QComboBox()
        for code, label in SCALAR_FIELD_OPTIONS:
            self.scalar_combo.addItem(label, code)
        self.scalar_combo.currentIndexChanged.connect(self._on_scalar_combo_changed)
        fvis.addRow("Column Scalar Map:", self.scalar_combo)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(45)
        self.opacity_slider.valueChanged.connect(
            lambda val: self.viewport.set_strata_opacity(val / 100.0)
        )
        fvis.addRow("Strata Opacity:", self.opacity_slider)

        self.chk_cols = QCheckBox("Show 3D Active Columns")
        self.chk_cols.setChecked(True)
        self.chk_cols.toggled.connect(lambda v: self.viewport.set_layer_visibility(columns=v))
        fvis.addRow("", self.chk_cols)

        self.chk_slip = QCheckBox("Show Critical Slip Surface")
        self.chk_slip.setChecked(True)
        self.chk_slip.toggled.connect(lambda v: self.viewport.set_layer_visibility(slip_surface=v))
        fvis.addRow("", self.chk_slip)

        self.chk_water = QCheckBox("Show Water Table Surface")
        self.chk_water.setChecked(True)
        self.chk_water.toggled.connect(lambda v: self.viewport.set_layer_visibility(water=v))
        fvis.addRow("", self.chk_water)

        self.chk_clip = QCheckBox("Section Cut at Mid-Y (y = y_mid)")
        self.chk_clip.setChecked(False)
        self.chk_clip.toggled.connect(lambda v: self.viewport.set_layer_visibility(clip_mid_y=v))
        fvis.addRow("", self.chk_clip)

        self.chk_safety_map = QCheckBox("Show 3D Slope Safety Map")
        self.chk_safety_map.setChecked(False)
        self.chk_safety_map.toggled.connect(
            lambda v: self.viewport.set_layer_visibility(safety_map=v)
        )
        fvis.addRow("", self.chk_safety_map)

        self.chk_local_minima = QCheckBox("Show Secondary Critical Surfaces")
        self.chk_local_minima.setChecked(True)
        self.chk_local_minima.toggled.connect(
            lambda v: self.viewport.set_layer_visibility(local_minima=v)
        )
        fvis.addRow("", self.chk_local_minima)

        self.chk_forces = QCheckBox("Show Picked Column 3D Force Arrows")
        self.chk_forces.setChecked(True)
        self.chk_forces.toggled.connect(
            lambda v: self.viewport.set_layer_visibility(force_vectors=v)
        )
        fvis.addRow("", self.chk_forces)

        vc.addWidget(vis_box)

        mat_box = QGroupBox("Project Materials")
        vm = QVBoxLayout(mat_box)
        self.mat_summary_lbl = QLabel()
        self.mat_summary_lbl.setWordWrap(True)
        vm.addWidget(self.mat_summary_lbl)
        vc.addWidget(mat_box)
        vc.addStretch(1)

        ctrl_dock.setWidget(w_ctrl)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, ctrl_dock)

        # Right Dock: 3D Results & Picked Column Inspector
        res_dock = QDockWidget("3D Stability Results & Column Inspector", self)
        res_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        w_res = QWidget()
        vr = QVBoxLayout(w_res)

        self.fos_banner = QLabel("Click 'Compute 3D Stability (F9)' to solve.")
        self.fos_banner.setWordWrap(True)
        self.fos_banner.setStyleSheet("font-size: 14px; font-weight: bold; padding: 6px;")
        vr.addWidget(self.fos_banner)

        self.res_table = QTableWidget(0, 5)
        self.res_table.setHorizontalHeaderLabels(["Method", "FoS", "Adm", "λ", "m_α,min"])
        self.res_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.res_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.res_table.setMaximumHeight(195)
        vr.addWidget(self.res_table)

        col_box = QGroupBox("Selected 3D Column Details")
        vcol = QVBoxLayout(col_box)
        self.col_info = QTextEdit()
        self.col_info.setReadOnly(True)
        vcol.addWidget(self.col_info)
        btn_tbl = QPushButton("Open Full 3D Column Table...")
        btn_tbl.clicked.connect(self.open_column_table)
        vcol.addWidget(btn_tbl)
        vr.addWidget(col_box, 1)

        res_dock.setWidget(w_res)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, res_dock)

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(220)
        self.progress_bar.setVisible(False)
        sb.addPermanentWidget(self.progress_bar)
        sb.showMessage("Ready — 3D Limit Equilibrium Modeler & Interpret")

    # ------------------------------------------------------- Undo / State Sync
    def _push_undo(self) -> None:
        self._undo_stack.append(self.project.to_dict())
        if len(self._undo_stack) > 30:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self.project.to_dict())
        prev = self._undo_stack.pop()
        self.project = Project3D.from_dict(prev)
        self.results = None
        self._refresh_all(reset_camera=False)

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self.project.to_dict())
        nxt = self._redo_stack.pop()
        self.project = Project3D.from_dict(nxt)
        self.results = None
        self._refresh_all(reset_camera=False)

    def _refresh_all(self, reset_camera: bool = False) -> None:
        self.setWindowTitle(f"{APP_NAME} {__version__} — {self.project.summary.title}")
        lines = []
        for m in self.project.materials:
            lines.append(
                f"<span style='color:{m.color};'>&#9632;</span> <b>{m.name}</b> ({m.model}, γ={m.gamma:g} kN/m³)"
            )
        self.mat_summary_lbl.setText("<br>".join(lines))

        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        for meth in self.project.analysis.methods:
            self.method_combo.addItem(METHODS_3D[meth][0], meth)
        self.method_combo.blockSignals(False)

        self._populate_results_panel()
        active_m = str(self.method_combo.currentData() or self.project.analysis.methods[0])
        self.viewport.set_scene_data(
            self.project, self.results, active_method=active_m, reset_camera=reset_camera
        )

    def _populate_results_panel(self) -> None:
        self.res_table.setRowCount(0)
        if self.results is None:
            self.fos_banner.setText("Model ready — press F9 to compute 3D FoS.")
            self.col_info.setPlainText("Run 3D stability analysis to inspect active columns.")
            return

        prim = self.results.primary_result
        fos_txt = f"{prim.fos:.4f}" if np.isfinite(prim.fos) else "N/A (Not Converged)"
        self.fos_banner.setText(
            f"Critical {self.results.result_label} "
            f"({METHODS_3D[self.results.primary_method][0]}): {fos_txt}\n"
            f"Volume: {self.results.columns.total_volume:.1f} m³ | "
            f"Columns: {self.results.columns.n} | Azimuth: {self.results.columns.azimuth_deg:.1f}°"
        )

        for r_idx, (key, res) in enumerate(self.results.method_results.items()):
            self.res_table.insertRow(r_idx)
            vals = [
                METHODS_3D[key][0],
                f"{res.fos:.4f}" if np.isfinite(res.fos) else "N/A",
                "OK" if res.admissible else "Warn",
                f"{res.lam:.3f}" if np.isfinite(res.lam) else "—",
                f"{res.m_alpha_min:.3f}" if np.isfinite(res.m_alpha_min) else "—",
            ]
            for c_idx, txt in enumerate(vals):
                self.res_table.setItem(r_idx, c_idx, QTableWidgetItem(txt))

        self.col_info.setPlainText(report_text_3d(self.project, self.results))

    def _on_method_combo_changed(self, _idx: int) -> None:
        meth = self.method_combo.currentData()
        if meth:
            self.viewport.set_active_method(str(meth))

    def _on_scalar_combo_changed(self, _idx: int) -> None:
        sc = self.scalar_combo.currentData()
        if sc:
            self.viewport.set_scalar_field(str(sc))

    def _on_column_picked(self, col_idx: int) -> None:
        if self.results is None or not (0 <= col_idx < self.results.columns.n):
            return
        cols = self.results.columns
        meth = str(self.method_combo.currentData() or self.results.primary_method)
        mres = self.results.method_results.get(meth, self.results.primary_result)
        sn = float(mres.sigma_n_eff[col_idx]) if mres.sigma_n_eff is not None else np.nan
        tm = float(mres.tau_mob[col_idx]) if mres.tau_mob is not None else np.nan
        mat = cols.materials[int(cols.mat_idx[col_idx])].name
        self.col_info.setPlainText(
            f"Selected Column #{col_idx}\n"
            f"------------------------------\n"
            f"Plan Centre (X, Y) : ({cols.x[col_idx]:.3f}, {cols.y[col_idx]:.3f}) m\n"
            f"Elevation Base/Top : {cols.z_base[col_idx]:.3f} -> {cols.z_top[col_idx]:.3f} m\n"
            f"Column Height h    : {cols.h[col_idx]:.3f} m\n"
            f"3D Base Area A_b   : {cols.area_base[col_idx]:.3f} m² (dA = {cols.dA[col_idx]:.3f} m²)\n"
            f"Base Dip α_s       : {np.degrees(cols.alpha_s[col_idx]):.2f}°\n"
            f"Base Material      : {mat}\n"
            f"Column Weight W    : {cols.W[col_idx]:.2f} kN\n"
            f"Pore Pressure u    : {cols.u[col_idx]:.2f} kPa\n"
            f"Eff. Normal σ'n    : {sn:.2f} kPa\n"
            f"Mobilized τ_mob    : {tm:.2f} kPa\n"
        )

    # ------------------------------------------------------- Actions & Dialogs
    def new_parametric_slope(self) -> None:
        dlg = ParametricSlopeDialog(self.project, self)
        if dlg.exec():
            self._push_undo()
            dlg.apply_to_project(self.project)
            self.results = None
            self.project_path = None
            self._refresh_all(reset_camera=True)

    def load_builtin_example(self, key: str) -> None:
        if key not in EXAMPLE_PROJECTS_3D:
            return
        self._push_undo()
        _, factory = EXAMPLE_PROJECTS_3D[key]
        self.project = factory()
        self.results = None
        self.project_path = None
        self._refresh_all(reset_camera=True)
        self.statusBar().showMessage(f"Loaded 3D example: {self.project.summary.title}")

    def open_project_dialog(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Open 3D or 2D Slope Project",
            "",
            "Slope Projects (*.slp3d.json *.slp.json *.json);;All Files (*)",
        )
        if fn:
            self.open_file(fn)

    def open_file(self, path: str | Path) -> None:
        try:
            proj = Project3D.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open Project Failed", str(exc))
            return
        self._push_undo()
        self.project = proj
        self.results = None
        p = Path(path)
        self.project_path = p if p.name.endswith(".slp3d.json") else None
        push_recent_file(str(p))
        self._refresh_all(reset_camera=True)
        self.statusBar().showMessage(f"Opened {p.name}")

    def import_2d_dialog(self) -> None:
        dlg = Import2DExtrusionDialog(parent=self)
        if dlg.exec():
            try:
                proj = import_2d_project(
                    dlg.path_edit.text().strip(),
                    y_min=dlg.ymin_sp.value(),
                    y_max=dlg.ymax_sp.value(),
                    extrude_surface_as_cylinder=dlg.cyl_chk.isChecked(),
                )
            except Exception as exc:
                QMessageBox.critical(self, "2D Import Failed", str(exc))
                return
            self._push_undo()
            self.project = proj
            self.results = None
            self.project_path = None
            self._refresh_all(reset_camera=True)
            self.statusBar().showMessage("Imported & extruded 2D project into 3D.")

    def save_project(self) -> None:
        if self.project_path is None:
            self.save_project_as()
            return
        try:
            self.project.save(self.project_path)
            push_recent_file(str(self.project_path))
            self.statusBar().showMessage(f"Saved {self.project_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    def save_project_as(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(
            self,
            "Save 3D Project",
            "model.slp3d.json",
            "3D Slope Projects (*.slp3d.json);;JSON Files (*.json)",
        )
        if not fn:
            return
        if not fn.endswith(".json"):
            fn += ".slp3d.json"
        self.project_path = Path(fn)
        self.save_project()

    def edit_materials(self) -> None:
        dlg = Materials3DDialog(self.project.materials, self)
        if dlg.exec():
            self._push_undo()
            old_first = self.project.materials[0].name
            self.project.materials = dlg.materials
            new_first = self.project.materials[0].name
            valid_set = set(self.project.material_names())
            self.project.layers = [
                (m if m in valid_set else new_first, h) for m, h in self.project.layers
            ]
            for reg in self.project.profile_regions:
                if reg.material not in valid_set:
                    reg.material = new_first if old_first not in valid_set else reg.material
            self.results = None
            self._refresh_all(reset_camera=False)

    def edit_boreholes(self) -> None:
        dlg = BoreholesDialog(self.project, self)
        if dlg.exec() and dlg.boreholes:
            self._push_undo()
            self.project.boreholes = dlg.boreholes
            self.project.rebuild_from_boreholes()
            self.results = None
            self._refresh_all(reset_camera=True)

    def edit_loads_supports(self) -> None:
        dlg = LoadsSupports3DDialog(self.project, self)
        if dlg.exec():
            self._push_undo()
            self.project.patch_loads = dlg.patch_loads
            self.project.support_types = dlg.support_types
            self.project.bolts = dlg.bolts
            self.project.tension_crack = dlg.tension_crack
            self.results = None
            self._refresh_all(reset_camera=False)

    def edit_weak_layers_piezo(self) -> None:
        dlg = WeakLayersPiezoDialog(self.project, self)
        if dlg.exec():
            self._push_undo()
            dlg.apply_to_project(self.project)
            if not self.project.pore_grid and self.project.analysis.pore_mode == "grid":
                self.project.analysis.pore_mode = "water_table"
            self.results = None
            self._refresh_all(reset_camera=False)

    def edit_supports_stages(self) -> None:
        dlg = SupportsStagesDialog(self.project, self)
        if dlg.exec():
            self._push_undo()
            dlg.apply_to_project(self.project)
            self.results = None
            self._refresh_all(reset_camera=False)

    def open_staged_analysis_dialog(self) -> None:
        if not self.project.stages:
            QMessageBox.information(
                self,
                "Construction Stages",
                "Define stages first under Model → Geotextiles, Pile Rows & Construction Stages.",
            )
            return
        dlg = StagedAnalysisDialog(self.project, self)
        if dlg.exec() and dlg.selected is not None:
            self.show_stage_result(dlg.selected)

    def show_stage_result(self, stage: Any) -> None:
        """Display one stage's geometry and results; the project itself is not changed."""
        self.results = stage.results
        self._refresh_all(reset_camera=False)
        self.statusBar().showMessage(
            f"Showing stage '{stage.name}'. Recompute (F9) to return to the initial ground."
        )

    def edit_analysis_settings(self) -> None:
        dlg = AnalysisSurfaceDialog(self.project, self)
        if dlg.exec():
            self._push_undo()
            self.results = None
            self._refresh_all(reset_camera=False)

    def compute_analysis(self, synchronous: bool = False) -> None:
        if synchronous:
            self.results = run_analysis_3d(self.project)
            self._refresh_all(reset_camera=False)
            return

        if self._worker is not None and self._worker.isRunning():
            return
        self.act_compute.setEnabled(False)
        self.act_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.statusBar().showMessage("Computing 3D Limit Equilibrium analysis...")

        worker = _ComputeWorker3D(self.project, self)
        worker.progress.connect(self._on_worker_progress)
        worker.best_surface.connect(self._on_worker_best_surface)
        worker.finished_ok.connect(self._on_worker_done)
        worker.failed.connect(self._on_worker_failed)
        self._worker = worker
        worker.start()

    def cancel_analysis(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self.statusBar().showMessage("Stopping 3D search after current evaluation...")

    def _on_worker_best_surface(self, surf_obj: Any, best_fos: float) -> None:
        if self.results is None:
            self.viewport.preview_trial_surface(surf_obj)
            self.statusBar().showMessage(f"Live 3D search best trial FoS = {best_fos:.4f}")

    def _on_worker_progress(self, gen: int, total: int, best_fos: float) -> None:
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(gen)
        if best_fos < 1e5:
            self.statusBar().showMessage(
                f"3D Cuckoo Search generation {gen}/{total} — current min FoS = {best_fos:.4f}"
            )

    def _on_worker_done(self, res_obj: Any) -> None:
        self.act_compute.setEnabled(True)
        self.act_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.results = res_obj
        self._refresh_all(reset_camera=False)
        assert self.results is not None
        tag = " (Stopped early)" if self.results.cancelled else ""
        self.statusBar().showMessage(
            f"3D Analysis complete{tag} in {self.results.elapsed_s:.2f} s — "
            f"Critical FoS = {self.results.critical_fos:.4f}"
        )

    def _on_worker_failed(self, err_msg: str) -> None:
        self.act_compute.setEnabled(True)
        self.act_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.viewport.preview_trial_surface(None)
        self.statusBar().showMessage("3D Analysis failed.")
        QMessageBox.critical(self, "3D Analysis Error", err_msg)

    def open_column_table(self) -> None:
        if self.results is None:
            QMessageBox.information(self, "Column Data", "Compute 3D analysis first (F9).")
            return
        meth = str(self.method_combo.currentData() or self.results.primary_method)
        dlg = ColumnDataDialog(self.results, meth, self)
        dlg.exec()

    def open_probabilistic_dialog(self) -> None:
        dlg = ProbabilisticSensitivity3DDialog(self.project, self.results, self)
        dlg.exec()

    def open_seismic_dialog(self) -> None:
        dlg = SeismicNewmark3DDialog(self.project, self.results, self)
        dlg.exec()

    def open_cross_section_dialog(self) -> None:
        dlg = CrossSectionComparisonDialog(self.project, self.results, self)
        dlg.exec()

    # ------------------------------------------------------- Exports / Imports
    def export_dxf_dialog(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export 3D DXF", "slope_3d.dxf", "DXF Files (*.dxf)"
        )
        if not fn:
            return
        model = self.results.model if self.results is not None else self.project.build_model()
        crit_mesh = (
            self.results.critical_surface.to_mesh(
                model.x_bounds, model.y_bounds, ground_fn=model.ground_z
            )
            if self.results is not None
            else None
        )
        export_dxf_3d(fn, model, critical_mesh=crit_mesh)
        self.statusBar().showMessage(f"Exported 3D DXF to {fn}")

    def export_obj_dialog(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export Wavefront OBJ", "slope_3d.obj", "OBJ Files (*.obj)"
        )
        if not fn:
            return
        model = self.results.model if self.results is not None else self.project.build_model()
        meshes: list[tuple[str, Any]] = [("Ground_3D", model.ground_mesh())]
        if self.results is not None:
            meshes.append(
                (
                    "Critical_Slip_3D",
                    self.results.critical_surface.to_mesh(
                        model.x_bounds, model.y_bounds, ground_fn=model.ground_z
                    ),
                )
            )
        export_obj_mesh(fn, meshes)
        self.statusBar().showMessage(f"Exported OBJ mesh to {fn}")

    def export_stl_dialog(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export Binary STL", "slope_3d.stl", "STL Files (*.stl)"
        )
        if not fn:
            return
        model = self.results.model if self.results is not None else self.project.build_model()
        meshes: list[tuple[str, Any]] = [("Ground_3D", model.ground_mesh())]
        if self.results is not None:
            meshes.append(
                (
                    "Critical_Slip_3D",
                    self.results.critical_surface.to_mesh(
                        model.x_bounds, model.y_bounds, ground_fn=model.ground_z
                    ),
                )
            )
        export_stl_mesh(fn, meshes, binary=True)
        self.statusBar().showMessage(f"Exported Binary STL mesh to {fn}")

    def export_dem_dialog(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export ESRI ASCII DEM", "terrain.asc", "ESRI ASCII Grid (*.asc)"
        )
        if not fn:
            return
        model = self.results.model if self.results is not None else self.project.build_model()
        export_esri_ascii_dem(fn, model.ground, model.x_bounds, model.y_bounds, cellsize=1.0)
        self.statusBar().showMessage(f"Exported ESRI ASCII DEM to {fn}")

    def import_terrain_mesh_dialog(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Import 3D Terrain Horizon",
            "",
            "3D Terrain / GIS (*.obj *.dxf *.stl *.asc *.csv *.xyz)",
        )
        if not fn:
            return
        try:
            hor = import_horizon_file(fn)

            if hor.x_coords is not None and hor.y_coords is not None and hor.z_grid is not None:
                x0, x1 = float(hor.x_coords[0]), float(hor.x_coords[-1])
                y0, y1 = float(hor.y_coords[0]), float(hor.y_coords[-1])
                zmin = float(np.min(hor.z_grid)) - 10.0
            else:
                assert hor.points is not None
                x0, x1 = float(np.min(hor.points[:, 0])), float(np.max(hor.points[:, 0]))
                y0, y1 = float(np.min(hor.points[:, 1])), float(np.max(hor.points[:, 1]))
                zmin = float(np.min(hor.points[:, 2])) - 10.0

            self._push_undo()
            self.project.x_min = x0
            self.project.x_max = x1
            self.project.y_min = y0
            self.project.y_max = y1
            self.project.z_min = zmin
            self.project.ground = hor
            self.project.layers = [(self.project.materials[0].name, hor)]
            self.project.profile_regions = []
            self.results = None
            self._refresh_all(reset_camera=True)
        except Exception as exc:
            QMessageBox.critical(self, "Import Terrain Failed", str(exc))

    def import_slip_obj_dialog(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Import User 3D Slip Surface",
            "",
            "3D Mesh Files (*.obj *.stl)",
        )
        if not fn:
            return
        try:
            p = Path(fn)
            surf = (
                import_stl_slip_surface(p)
                if p.suffix.lower() == ".stl"
                else import_obj_slip_surface(p)
            )
            self._push_undo()
            self.project.surface_config.mode = "user"
            self.project.surface_config.user_surface = surf.to_dict()
            self.results = None
            self._refresh_all(reset_camera=False)
        except Exception as exc:
            QMessageBox.critical(self, "Import Slip Surface Failed", str(exc))

    def export_pdf_dialog(self) -> None:
        if self.results is None:
            QMessageBox.information(self, "Export PDF Report", "Compute 3D stability first (F9).")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export 3D Engineering PDF Report", "report_3d.pdf", "PDF Files (*.pdf)"
        )
        if not fn:
            return
        try:
            export_pdf_report_3d(fn, self.project, self.results)
            self.statusBar().showMessage(f"Saved PDF report to {fn}")
        except Exception as exc:
            QMessageBox.critical(self, "PDF Export Failed", str(exc))

    def export_screenshot_dialog(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save 3D Screenshot", "viewport_3d.png", "PNG Images (*.png)"
        )
        if fn:
            self.viewport.export_screenshot(fn)
            self.statusBar().showMessage(f"Saved screenshot to {fn}")

    def show_about(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle(f"About {APP_NAME}")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        box.setText(ABOUT_HTML)
        box.setStandardButtons(QMessageBox.StandardButton.Close)
        box.exec()

    def toggle_theme(self) -> None:
        app = QApplication.instance()
        if isinstance(app, QApplication):
            nxt = "light" if current_theme() == "dark" else "dark"
            apply_theme(app, nxt)

    def on_theme_changed(self, theme: str) -> None:
        self.viewport.set_theme(theme)

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        self.viewport.cleanup()
        super().closeEvent(event)
