"""Analysis dialogs: solvers/columns/slip surface settings and column table."""

from __future__ import annotations

import csv

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from slipsurface3d.analysis import AnalysisResults3D
from slipsurface3d.design import (
    DESIGN_PRESETS,
    USER_DESIGN_KEY,
    USER_DESIGN_LABEL,
    DesignFactors3D,
    design_preset,
)
from slipsurface3d.geometry.surfaces import (
    EllipsoidSurface,
    MultiPlaneWedgeSurface,
    WedgePlane3D,
)
from slipsurface3d.lem.solvers import INTERCOLUMN_FUNCTIONS, METHODS_3D
from slipsurface3d.optimizer.search import SEARCH_ALGORITHMS
from slipsurface3d.project import Project3D
from slipsurface3d.seepage import SeepageBC3D
from slipsurface3d.ui.forms import _dspin


class AnalysisSurfaceDialog(QDialog):
    """Configure 3D LEM solvers, column discretization, seismic/groundwater, and slip search."""

    def __init__(self, project: Project3D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D Analysis & Slip Surface Settings")
        self.resize(560, 520)
        self._project = project

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # Tab 1: Methods & Columns
        w_m = QWidget()
        fm = QFormLayout(w_m)
        an = project.analysis
        self.meth_checks: dict[str, QCheckBox] = {}
        for code, (label, _) in METHODS_3D.items():
            chk = QCheckBox(label)
            chk.setChecked(code in an.methods)
            self.meth_checks[code] = chk
            fm.addRow("", chk)

        self.primary_cb = QComboBox()
        for chk in self.meth_checks.values():
            chk.toggled.connect(self._refresh_primary_methods)
        self._refresh_primary_methods()

        self.nx_sp = QSpinBox()
        self.nx_sp.setRange(4, 150)
        self.nx_sp.setValue(an.nx)
        self.nx_sp.setToolTip(
            "Interactive default is 28. Published verification cases use 64–72 for FoS accuracy."
        )
        self.ny_sp = QSpinBox()
        self.ny_sp.setRange(1, 150)
        self.ny_sp.setValue(an.ny)
        self.ny_sp.setToolTip(
            "Interactive default is 24. Published verification cases use 64–72 for FoS accuracy."
        )
        self.slide_cb = QComboBox()
        self.slide_cb.addItems(["auto", "right_to_left", "left_to_right", "custom"])
        self.slide_cb.setCurrentText(an.sliding_mode)
        self.az_sp = _dspin(0.0, 360.0, an.azimuth_deg, 1, 5.0, "deg")
        self.adm_chk = QCheckBox("Enforce physical admissibility (Whitman & Bailey m_α)")
        self.adm_chk.setChecked(an.check_admissibility)
        self.gle_cb = QComboBox()
        self.gle_cb.addItems(list(INTERCOLUMN_FUNCTIONS))
        self.gle_cb.setCurrentText(an.gle_interslice)
        self.workers_sp = QSpinBox()
        self.workers_sp.setRange(0, 64)
        self.workers_sp.setValue(an.n_workers)
        self.workers_sp.setSpecialValueText("Auto (Multi-Core)")

        fm.addRow("Primary Search Method:", self.primary_cb)
        fm.addRow("Columns along X (nx):", self.nx_sp)
        fm.addRow("Columns along Y (ny):", self.ny_sp)
        grid_hint = QLabel(
            "Default 28 × 24 is for interactive search. Verification benchmarks use 64–72."
        )
        grid_hint.setWordWrap(True)
        fm.addRow("", grid_hint)
        fm.addRow("GLE Interslice f(ξ):", self.gle_cb)
        fm.addRow("Search Threads:", self.workers_sp)
        fm.addRow("Sliding Direction:", self.slide_cb)
        fm.addRow("Custom Azimuth:", self.az_sp)
        fm.addRow("", self.adm_chk)
        tabs.addTab(w_m, "Solvers & Columns")

        # Tab 2: Groundwater, 3D Seepage & Seismic
        w_gs = QWidget()
        fgs = QFormLayout(w_gs)
        self.pore_cb = QComboBox()
        self.pore_cb.addItems(["water_table", "ru", "seepage", "grid", "none"])
        self.pore_cb.setCurrentText(an.pore_mode)
        self.ru_sp = _dspin(0.0, 1.0, an.ru, 3, 0.05)
        sbc = project.seepage_bc or SeepageBC3D()
        self.seep_hmin_sp = _dspin(
            -500.0, 2000.0, 0.0 if sbc.head_xmin is None else sbc.head_xmin, 2, 0.5, "m"
        )
        self.seep_hmax_sp = _dspin(
            -500.0, 2000.0, 10.0 if sbc.head_xmax is None else sbc.head_xmax, 2, 0.5, "m"
        )
        self.kh_sp = _dspin(0.0, 1.5, an.kh, 3, 0.02, "g")
        self.kv_sp = _dspin(-0.8, 0.8, an.kv, 3, 0.02, "g")
        self.both_kv_chk = QCheckBox("Evaluate ±kv and retain governing FoS")
        self.both_kv_chk.setChecked(an.both_kv_signs)
        fgs.addRow("Pore Pressure Mode:", self.pore_cb)
        fgs.addRow("Ru Coefficient:", self.ru_sp)
        fgs.addRow("3D Seepage Head @ X_min:", self.seep_hmin_sp)
        fgs.addRow("3D Seepage Head @ X_max:", self.seep_hmax_sp)
        fgs.addRow("Horizontal Seismic kh:", self.kh_sp)
        fgs.addRow("Vertical Seismic kv (+up):", self.kv_sp)
        fgs.addRow("", self.both_kv_chk)
        tabs.addTab(w_gs, "Water & Seismic")

        # Tab 3: 3D Slip Surface Search / User Ellipsoid / Rock Wedge
        w_sf = QWidget()
        fsf = QFormLayout(w_sf)
        sc = project.surface_config
        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["search", "user"])
        self.mode_cb.setCurrentText(sc.mode)
        self.kind_cb = QComboBox()
        self.kind_cb.addItems(["ellipsoid", "sphere", "spline_patch", "wedge"])
        self.kind_cb.setCurrentText(sc.surface_kind)
        self.effort_cb = QComboBox()
        self.effort_cb.addItems(["fast", "normal", "fine"])
        self.effort_cb.setCurrentText(sc.effort)
        self.algo_cb = QComboBox()
        for a_key, a_label in SEARCH_ALGORITHMS.items():
            self.algo_cb.addItem(a_label, a_key)
        self.algo_cb.setCurrentIndex(max(0, self.algo_cb.findData(sc.algorithm)))
        self.opt_az_chk = QCheckBox("Co-optimize ellipsoid horizontal orientation")
        self.opt_az_chk.setChecked(sc.optimize_azimuth)
        self.sao_chk = QCheckBox("Enable Multi-Scale SAO (3×3 -> 5×5 spline refinement)")
        self.sao_chk.setChecked(sc.sao_enabled)

        us = sc.user_surface or {}
        if us.get("kind") == "composite_ellipsoid_plane":
            us = us.get("ellipsoid", {})
        self.uxc = _dspin(-1000.0, 2000.0, float(us.get("xc", 20.0)), 2, 1.0, "m")
        self.uyc = _dspin(-1000.0, 2000.0, float(us.get("yc", 20.0)), 2, 1.0, "m")
        self.uzc = _dspin(-1000.0, 2000.0, float(us.get("zc", 18.0)), 2, 1.0, "m")
        self.urx = _dspin(0.5, 1e10, float(us.get("rx", 18.0)), 2, 1.0, "m")
        self.ury = _dspin(0.5, 1e10, float(us.get("ry", 24.0)), 2, 1.0, "m")
        self.urz = _dspin(0.5, 1e10, float(us.get("rz", 18.0)), 2, 1.0, "m")

        planes_raw = us.get("planes", []) if us.get("kind") == "multi_plane_wedge" else []
        p1_d = planes_raw[0] if len(planes_raw) >= 1 else {"dip_deg": 45.0, "dip_dir_deg": 230.0}
        p2_d = planes_raw[1] if len(planes_raw) >= 2 else {"dip_deg": 45.0, "dip_dir_deg": 130.0}
        self.w_dip1 = _dspin(1.0, 88.0, float(p1_d.get("dip_deg", 45.0)), 1, 2.0, "deg")
        self.w_dd1 = _dspin(0.0, 360.0, float(p1_d.get("dip_dir_deg", 230.0)), 1, 5.0, "deg")
        self.w_dip2 = _dspin(1.0, 88.0, float(p2_d.get("dip_deg", 45.0)), 1, 2.0, "deg")
        self.w_dd2 = _dspin(0.0, 360.0, float(p2_d.get("dip_dir_deg", 130.0)), 1, 5.0, "deg")

        self.n_minima_sp = QSpinBox()
        self.n_minima_sp.setRange(1, 12)
        self.n_minima_sp.setValue(sc.n_local_minima)
        self.n_minima_sp.setToolTip(
            "Multi-modal search: report this many distinct failure modes (plan footprints "
            "overlapping by less than half). 1 = global minimum only."
        )
        lim = sc.search_limits or [project.x_min, project.x_max, project.y_min, project.y_max]
        self.limits_chk = QCheckBox("Restrict sliding mass to plan limits")
        self.limits_chk.setChecked(sc.search_limits is not None)
        self.lim_x0 = _dspin(-1e6, 1e6, float(lim[0]), 2, 1.0, "m")
        self.lim_x1 = _dspin(-1e6, 1e6, float(lim[1]), 2, 1.0, "m")
        self.lim_y0 = _dspin(-1e6, 1e6, float(lim[2]), 2, 1.0, "m")
        self.lim_y1 = _dspin(-1e6, 1e6, float(lim[3]), 2, 1.0, "m")
        for w_lim in (self.lim_x0, self.lim_x1, self.lim_y0, self.lim_y1):
            w_lim.setEnabled(self.limits_chk.isChecked())
            self.limits_chk.toggled.connect(w_lim.setEnabled)

        fsf.addRow("Surface Mode:", self.mode_cb)
        fsf.addRow("Surface / Search Geometry:", self.kind_cb)
        fsf.addRow("Search Effort:", self.effort_cb)
        fsf.addRow("Global Search Algorithm:", self.algo_cb)
        fsf.addRow("", self.opt_az_chk)
        fsf.addRow("", self.sao_chk)
        fsf.addRow("Distinct Critical Surfaces:", self.n_minima_sp)
        fsf.addRow("", self.limits_chk)
        fsf.addRow("Limits X min / max:", self._row2(self.lim_x0, self.lim_x1))
        fsf.addRow("Limits Y min / max:", self._row2(self.lim_y0, self.lim_y1))
        fsf.addRow(QLabel("<b>User Ellipsoid (Mode=user, Geometry=ellipsoid/sphere):</b>"))
        fsf.addRow("Centre Xc / Yc / Zc:", self._row3(self.uxc, self.uyc, self.uzc))
        fsf.addRow("Semi-Axes Rx / Ry / Rz:", self._row3(self.urx, self.ury, self.urz))
        fsf.addRow(QLabel("<b>User 3D Rock Wedge (Mode=user, Geometry=wedge):</b>"))
        fsf.addRow("Joint A Dip / Dip-Dir:", self._row2(self.w_dip1, self.w_dd1))
        fsf.addRow("Joint B Dip / Dip-Dir:", self._row2(self.w_dip2, self.w_dd2))
        tabs.addTab(w_sf, "3D Slip Surface")

        # Tab 4: Design standard (partial factors)
        w_ds = QWidget()
        fds = QFormLayout(w_ds)
        ds = project.design
        self.design_cb = QComboBox()
        for key, preset in DESIGN_PRESETS.items():
            self.design_cb.addItem(preset.label, key)
        self.design_cb.addItem(USER_DESIGN_LABEL, USER_DESIGN_KEY)
        self.design_cb.setCurrentIndex(max(0, self.design_cb.findData(ds.key)))
        self.design_spins: dict[str, QDoubleSpinBox] = {}
        fds.addRow("Design Standard:", self.design_cb)
        for attr, label in (
            ("permanent_action", "Permanent actions γG (soil unit weight):"),
            ("variable_action", "Variable actions γQ (patch / point loads):"),
            ("cohesion", "Effective cohesion γc':"),
            ("friction", "Friction γφ' (on tan φ'):"),
            ("undrained", "Undrained strength γcu:"),
            ("support_resistance", "Support resistance γa:"),
            ("earth_resistance", "Earth resistance γR;e:"),
        ):
            sp = _dspin(0.1, 10.0, float(getattr(ds, attr)), 3, 0.05)
            self.design_spins[attr] = sp
            fds.addRow(label, sp)
        ds_hint = QLabel(
            "With a design standard active, results are over-design factors (ODF ≥ 1.0 "
            "satisfies the limit state). Eurocode 7 presets use the recommended values of "
            "EN 1997-1 Annex A — check your National Annex. Pore pressures are not factored."
        )
        ds_hint.setWordWrap(True)
        fds.addRow(ds_hint)
        self.design_cb.currentIndexChanged.connect(self._on_design_changed)
        self._on_design_changed(load_preset=False)
        tabs.addTab(w_ds, "Design Standard")

        layout.addWidget(tabs, 1)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_design_changed(self, _idx: int = 0, load_preset: bool = True) -> None:
        key = str(self.design_cb.currentData())
        is_user = key == USER_DESIGN_KEY
        if load_preset and not is_user:
            preset = design_preset(key)
            for attr, sp in self.design_spins.items():
                sp.setValue(float(getattr(preset, attr)))
        for sp in self.design_spins.values():
            sp.setEnabled(is_user)

    def _refresh_primary_methods(self) -> None:
        selected = [k for k, chk in self.meth_checks.items() if chk.isChecked()]
        keep = self.primary_cb.currentData() or self._project.surface_config.primary_method
        self.primary_cb.blockSignals(True)
        self.primary_cb.clear()
        for code in selected:
            self.primary_cb.addItem(METHODS_3D[code][0], code)
        if keep in selected:
            self.primary_cb.setCurrentIndex(selected.index(keep))
        self.primary_cb.blockSignals(False)

    @staticmethod
    def _row2(w1: QWidget, w2: QWidget) -> QWidget:
        box = QWidget()
        hl = QHBoxLayout(box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(w1)
        hl.addWidget(w2)
        return box

    @staticmethod
    def _row3(w1: QWidget, w2: QWidget, w3: QWidget) -> QWidget:
        box = QWidget()
        hl = QHBoxLayout(box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(w1)
        hl.addWidget(w2)
        hl.addWidget(w3)
        return box

    def _on_accept(self) -> None:
        selected = [k for k, chk in self.meth_checks.items() if chk.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Analysis Settings", "Select at least one 3D LEM method.")
            return
        if self.limits_chk.isChecked() and (
            self.lim_x1.value() <= self.lim_x0.value() or self.lim_y1.value() <= self.lim_y0.value()
        ):
            QMessageBox.warning(self, "Analysis Settings", "Search limits need max > min.")
            return
        d_key = str(self.design_cb.currentData())
        base = design_preset(d_key)
        self._project.design = DesignFactors3D(
            key=base.key,
            label=base.label,
            **{attr: sp.value() for attr, sp in self.design_spins.items()},
        )
        an = self._project.analysis
        an.methods = selected
        an.nx = self.nx_sp.value()
        an.ny = self.ny_sp.value()
        an.gle_interslice = self.gle_cb.currentText()
        an.n_workers = self.workers_sp.value()
        an.sliding_mode = self.slide_cb.currentText()
        an.azimuth_deg = self.az_sp.value()
        an.check_admissibility = self.adm_chk.isChecked()
        if self.pore_cb.currentText() == "grid" and not self._project.pore_grid:
            QMessageBox.warning(
                self,
                "Analysis Settings",
                "Pore mode 'grid' needs pore-pressure samples. Import them under "
                "Model → Weak Layers, Piezometric Surfaces & Pore Pressure Grid.",
            )
            return
        an.pore_mode = self.pore_cb.currentText()
        an.ru = self.ru_sp.value()
        an.kh = self.kh_sp.value()
        an.kv = self.kv_sp.value()
        an.both_kv_signs = self.both_kv_chk.isChecked()
        if an.pore_mode == "seepage":
            self._project.seepage_bc = SeepageBC3D(
                head_xmin=self.seep_hmin_sp.value(),
                head_xmax=self.seep_hmax_sp.value(),
            )

        sc = self._project.surface_config
        sc.mode = self.mode_cb.currentText()
        sc.surface_kind = self.kind_cb.currentText()
        if sc.surface_kind == "wedge":
            sc.mode = "user"
        sc.effort = self.effort_cb.currentText()
        sc.algorithm = str(self.algo_cb.currentData())
        sc.primary_method = str(self.primary_cb.currentData() or selected[0])
        sc.optimize_azimuth = self.opt_az_chk.isChecked()
        sc.sao_enabled = self.sao_chk.isChecked()
        sc.n_local_minima = self.n_minima_sp.value()
        sc.search_limits = (
            [
                self.lim_x0.value(),
                self.lim_x1.value(),
                self.lim_y0.value(),
                self.lim_y1.value(),
            ]
            if self.limits_chk.isChecked()
            else None
        )
        if sc.mode == "user":
            if sc.surface_kind == "wedge":
                p = self._project
                ym = 0.5 * (p.y_min + p.y_max)
                x0 = p.x_min + 0.20 * (p.x_max - p.x_min)
                z0 = float(p.build_model().ground_z(x0, ym))
                sc.user_surface = MultiPlaneWedgeSurface(
                    planes=[
                        WedgePlane3D(
                            self.w_dip1.value(), self.w_dd1.value(), x0, ym, z0, "Joint A"
                        ),
                        WedgePlane3D(
                            self.w_dip2.value(), self.w_dd2.value(), x0, ym, z0, "Joint B"
                        ),
                    ],
                    bounds=(p.x_min, p.x_max, p.y_min, p.y_max),
                ).to_dict()
            elif sc.user_surface is None or sc.user_surface.get("kind") == "ellipsoid":
                sc.user_surface = EllipsoidSurface(
                    xc=self.uxc.value(),
                    yc=self.uyc.value(),
                    zc=self.uzc.value(),
                    rx=self.urx.value(),
                    ry=self.ury.value(),
                    rz=self.urz.value(),
                    azimuth_deg=an.azimuth_deg if an.sliding_mode == "custom" else 180.0,
                ).to_dict()
        self.accept()


class ColumnDataDialog(QDialog):
    """Tabular inspector of all active 3D columns with CSV export."""

    def __init__(
        self, results: AnalysisResults3D, method: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"3D Column Data Inspector — {METHODS_3D[method][0]}")
        self.resize(940, 520)
        self._results = results
        self._method = method

        layout = QVBoxLayout(self)
        cols = results.columns
        mres = results.method_results.get(method, results.primary_result)
        headers = [
            "Col #",
            "X [m]",
            "Y [m]",
            "z_base [m]",
            "z_top [m]",
            "h [m]",
            "α_s [deg]",
            "A_base [m²]",
            "W [kN]",
            "u [kPa]",
            "σ'n [kPa]",
            "τ_mob [kPa]",
            "Material",
        ]
        self.table = QTableWidget(cols.n, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        sn = mres.sigma_n_eff if mres.sigma_n_eff is not None else np.full(cols.n, np.nan)
        tm = mres.tau_mob if mres.tau_mob is not None else np.full(cols.n, np.nan)
        for i in range(cols.n):
            mat_name = cols.materials[int(cols.mat_idx[i])].name
            vals = [
                str(i),
                f"{cols.x[i]:.2f}",
                f"{cols.y[i]:.2f}",
                f"{cols.z_base[i]:.2f}",
                f"{cols.z_top[i]:.2f}",
                f"{cols.h[i]:.2f}",
                f"{np.degrees(cols.alpha_s[i]):.2f}",
                f"{cols.area_base[i]:.3f}",
                f"{cols.W[i]:.2f}",
                f"{cols.u[i]:.2f}",
                f"{sn[i]:.2f}" if np.isfinite(sn[i]) else "—",
                f"{tm[i]:.2f}" if np.isfinite(tm[i]) else "—",
                mat_name,
            ]
            for c_idx, txt in enumerate(vals):
                it = QTableWidgetItem(txt)
                if c_idx < len(headers) - 1:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, c_idx, it)

        layout.addWidget(self.table, 1)
        bot = QHBoxLayout()
        csv_btn = QPushButton("Export CSV...")
        csv_btn.clicked.connect(self._export_csv)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bot.addWidget(csv_btn)
        bot.addStretch(1)
        bot.addWidget(close_btn)
        layout.addLayout(bot)

    def _export_csv(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export 3D Column Table CSV", "columns_3d.csv", "CSV Files (*.csv)"
        )
        if not fn:
            return
        with open(fn, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            headers = [
                self.table.horizontalHeaderItem(c).text() for c in range(self.table.columnCount())
            ]
            writer.writerow(headers)
            for r in range(self.table.rowCount()):
                writer.writerow(
                    [
                        self.table.item(r, c).text() if self.table.item(r, c) else ""
                        for c in range(self.table.columnCount())
                    ]
                )
