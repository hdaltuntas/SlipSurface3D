"""Engineering dialogs for SlipSurfaceIQ-3D Modeler & Interpret workflows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from slipsurface3d.analysis import AnalysisResults3D
from slipsurface3d.geometry.surfaces import EllipsoidSurface
from slipsurface3d.geometry.terrain import (
    BoreholeLayer,
    BoreholeLog,
    simple_3d_slope,
)
from slipsurface3d.lem.solvers import METHODS_3D
from slipsurface3d.loads.supports import (
    SUPPORT_KINDS_3D,
    PatchLoad3D,
    SupportType3D,
    TensionCrack3D,
)
from slipsurface3d.materials.models import MODEL_LABELS
from slipsurface3d.project import PALETTE_3D, MaterialSpec3D, Project3D


def _dspin(
    lo: float, hi: float, val: float, decimals: int = 2, step: float = 1.0, suffix: str = ""
) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(decimals)
    sp.setSingleStep(step)
    sp.setValue(val)
    if suffix:
        sp.setSuffix(f" {suffix}")
    return sp


class ParametricSlopeDialog(QDialog):
    """Build a clean parametric 3D slope extruded along Y."""

    def __init__(self, project: Project3D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Parametric 3D Slope Builder")
        self.resize(420, 340)
        self._project = project

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.height_sp = _dspin(1.0, 500.0, 10.0, 2, 1.0, "m")
        self.ratio_sp = _dspin(0.2, 10.0, 2.0, 2, 0.25, "H:1V")
        self.toe_sp = _dspin(-500.0, 500.0, 15.0, 2, 1.0, "m")
        self.left_sp = _dspin(2.0, 500.0, 15.0, 2, 1.0, "m")
        self.right_sp = _dspin(2.0, 500.0, 20.0, 2, 1.0, "m")
        self.depth_sp = _dspin(2.0, 500.0, 10.0, 2, 1.0, "m")
        self.width_y_sp = _dspin(4.0, 2000.0, max(10.0, project.y_max - project.y_min), 2, 5.0, "m")
        self.wt_chk = QCheckBox("Include Water Table")
        self.wt_crest_depth_sp = _dspin(0.0, 200.0, 2.5, 2, 0.5, "m below crest")

        form.addRow("Slope Height H:", self.height_sp)
        form.addRow("Slope Ratio (H:1V):", self.ratio_sp)
        form.addRow("Toe X Coordinate:", self.toe_sp)
        form.addRow("Toe Bench Length:", self.left_sp)
        form.addRow("Crest Bench Length:", self.right_sp)
        form.addRow("Foundation Depth:", self.depth_sp)
        form.addRow("3D Transverse Width Y:", self.width_y_sp)
        form.addRow(self.wt_chk, self.wt_crest_depth_sp)
        layout.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def apply_to_project(self, project: Project3D) -> None:
        h = self.height_sp.value()
        ratio = self.ratio_sp.value()
        toe = self.toe_sp.value()
        mat_name = project.materials[0].name if project.materials else "Soil 1"
        wt_prof = None
        if self.wt_chk.isChecked():
            crest_x = toe + ratio * h
            x0 = toe - self.left_sp.value()
            x1 = crest_x + self.right_sp.value()
            zw_crest = max(0.0, h - self.wt_crest_depth_sp.value())
            wt_prof = [[x0, 0.0], [toe, 0.0], [crest_x, zw_crest], [x1, zw_crest]]

        m3d = simple_3d_slope(
            height=h,
            ratio_h_to_v=ratio,
            toe_x=toe,
            left_bench=self.left_sp.value(),
            right_bench=self.right_sp.value(),
            depth_below_toe=self.depth_sp.value(),
            width_y=self.width_y_sp.value(),
            material=mat_name,
            water_table_profile=wt_prof,
        )
        project.x_min, project.x_max = m3d.x_min, m3d.x_max
        project.y_min, project.y_max = m3d.y_min, m3d.y_max
        project.z_min = m3d.z_min
        project.ground = m3d.ground
        project.layers = m3d.layers
        project.profile_regions = []
        project.boreholes = []
        project.water_table = m3d.water_table
        if wt_prof is not None:
            project.analysis.pore_mode = "water_table"


class Import2DExtrusionDialog(QDialog):
    """Select a 2D SlipSurfaceIQ-LE (.slp.json) file and extrude into 3D."""

    def __init__(self, initial_path: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import & Extrude 2D Project (.slp.json)")
        self.resize(480, 220)

        layout = QVBoxLayout(self)
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(initial_path)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        form = QFormLayout()
        self.ymin_sp = _dspin(-1000.0, 1000.0, 0.0, 2, 5.0, "m")
        self.ymax_sp = _dspin(-1000.0, 5000.0, 40.0, 2, 5.0, "m")
        self.cyl_chk = QCheckBox("Extrude fixed 2D circle as infinite cylinder (2D parity check)")
        form.addRow("Extrusion Y Min:", self.ymin_sp)
        form.addRow("Extrusion Y Max:", self.ymax_sp)
        form.addRow("", self.cyl_chk)
        layout.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Select 2D SlipSurfaceIQ-LE Project",
            self.path_edit.text(),
            "2D Slope Projects (*.slp.json *.json);;All Files (*)",
        )
        if fn:
            self.path_edit.setText(fn)

    def _validate_and_accept(self) -> None:
        p = Path(self.path_edit.text().strip())
        if not p.is_file():
            QMessageBox.warning(
                self, "Import 2D Project", "Please select an existing .slp.json file."
            )
            return
        if self.ymax_sp.value() <= self.ymin_sp.value():
            QMessageBox.warning(self, "Import 2D Project", "Y Max must be greater than Y Min.")
            return
        self.accept()


class Materials3DDialog(QDialog):
    """Create, edit, recolor, and configure 3D constitutive material models."""

    def __init__(self, materials: list[MaterialSpec3D], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D Geotechnical Materials")
        self.resize(700, 460)
        self.materials: list[MaterialSpec3D] = [
            MaterialSpec3D.from_dict(m.__dict__.copy()) for m in materials
        ]
        self._idx = 0
        self._updating = False

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.list_w = QListWidget()
        self.list_w.currentRowChanged.connect(self._on_row_changed)
        left.addWidget(self.list_w, 1)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add")
        del_btn = QPushButton("− Remove")
        add_btn.clicked.connect(self._add_mat)
        del_btn.clicked.connect(self._del_mat)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(del_btn)
        left.addLayout(btn_row)
        root.addLayout(left, 1)

        right = QVBoxLayout()
        common_box = QGroupBox("General & Unit Weights")
        cform = QFormLayout(common_box)
        self.name_ed = QLineEdit()
        self.model_cb = QComboBox()
        for code, label in MODEL_LABELS.items():
            self.model_cb.addItem(f"{label} ({code})", code)
        self.gamma_sp = _dspin(1.0, 50.0, 19.0, 2, 0.5, "kN/m³")
        self.gsat_sp = _dspin(1.0, 50.0, 20.0, 2, 0.5, "kN/m³")
        self.water_chk = QCheckBox("Pore pressure acts on base")
        self.hu_sp = _dspin(0.0, 1.5, 1.0, 2, 0.1)
        self.bbar_sp = _dspin(0.0, 1.0, 0.0, 2, 0.05)
        self.color_btn = QPushButton("Pick Color")
        self.color_btn.clicked.connect(self._pick_color)

        cform.addRow("Name:", self.name_ed)
        cform.addRow("Constitutive Model:", self.model_cb)
        cform.addRow("Unsaturated γ:", self.gamma_sp)
        cform.addRow("Saturated γ_sat:", self.gsat_sp)
        cform.addRow("", self.water_chk)
        cform.addRow("Water Factor Hu:", self.hu_sp)
        cform.addRow("Excess Pore B-bar:", self.bbar_sp)
        cform.addRow("Display Color:", self.color_btn)
        right.addWidget(common_box)

        # Stacked constitutive pages
        self.stack = QStackedWidget()
        self._model_codes = list(MODEL_LABELS.keys())

        # 0: MC
        w_mc = QGroupBox("Mohr-Coulomb Parameters")
        f_mc = QFormLayout(w_mc)
        self.c_sp = _dspin(0.0, 10_000.0, 10.0, 2, 1.0, "kPa")
        self.phi_sp = _dspin(0.0, 85.0, 30.0, 2, 1.0, "deg")
        f_mc.addRow("Cohesion c':", self.c_sp)
        f_mc.addRow("Friction Angle φ':", self.phi_sp)
        self.stack.addWidget(w_mc)

        # 1: UD
        w_ud = QGroupBox("Undrained (φ = 0)")
        f_ud = QFormLayout(w_ud)
        self.su_sp = _dspin(0.0, 10_000.0, 40.0, 2, 2.0, "kPa")
        f_ud.addRow("Undrained Shear Strength su:", self.su_sp)
        self.stack.addWidget(w_ud)

        # 2: UDG
        w_udg = QGroupBox("Depth-Linear Undrained su(z)")
        f_udg = QFormLayout(w_udg)
        self.sut_sp = _dspin(0.0, 10_000.0, 25.0, 2, 2.0, "kPa")
        self.sub_sp = _dspin(0.0, 10_000.0, 55.0, 2, 2.0, "kPa")
        self.zt_sp = _dspin(-500.0, 1000.0, 10.0, 2, 1.0, "m")
        self.zb_sp = _dspin(-500.0, 1000.0, 0.0, 2, 1.0, "m")
        f_udg.addRow("su at Top:", self.sut_sp)
        f_udg.addRow("su at Bottom:", self.sub_sp)
        f_udg.addRow("Top Elevation z_top:", self.zt_sp)
        f_udg.addRow("Bottom Elevation z_bot:", self.zb_sp)
        self.stack.addWidget(w_udg)

        # 3: HB
        w_hb = QGroupBox("Generalized Hoek-Brown (GSI, mi, D)")
        f_hb = QFormLayout(w_hb)
        self.sci_sp = _dspin(10.0, 500_000.0, 25_000.0, 1, 1000.0, "kPa")
        self.gsi_sp = _dspin(1.0, 100.0, 50.0, 1, 5.0)
        self.mi_sp = _dspin(1.0, 50.0, 10.0, 1, 1.0)
        self.d_sp = _dspin(0.0, 1.0, 0.0, 2, 0.1)
        f_hb.addRow("Intact UCS σ_ci:", self.sci_sp)
        f_hb.addRow("GSI:", self.gsi_sp)
        f_hb.addRow("mi:", self.mi_sp)
        f_hb.addRow("Disturbance D:", self.d_sp)
        self.stack.addWidget(w_hb)

        # 4..8: HBD, BB, PC, PL, SH compact generic pages + 9: ANISO3D
        for code in ("HBD", "BB", "PC", "PL", "SH"):
            w_gen = QGroupBox(f"{MODEL_LABELS[code]} Parameters")
            fl = QFormLayout(w_gen)
            fl.addRow(QLabel("Configured parameters preserved in project JSON."))
            self.stack.addWidget(w_gen)

        w_an = QGroupBox("3D Anisotropic Bedding (Dip / Dip-Direction)")
        f_an = QFormLayout(w_an)
        self.cmass_sp = _dspin(0.0, 10_000.0, 40.0, 2, 2.0, "kPa")
        self.phimass_sp = _dspin(0.0, 85.0, 35.0, 2, 1.0, "deg")
        self.cbed_sp = _dspin(0.0, 10_000.0, 5.0, 2, 1.0, "kPa")
        self.phibed_sp = _dspin(0.0, 85.0, 18.0, 2, 1.0, "deg")
        self.dip_sp = _dspin(0.0, 89.0, 20.0, 1, 2.0, "deg")
        self.dipdir_sp = _dspin(0.0, 360.0, 180.0, 1, 5.0, "deg")
        f_an.addRow("Rock Mass c':", self.cmass_sp)
        f_an.addRow("Rock Mass φ':", self.phimass_sp)
        f_an.addRow("Bedding Joint c'_j:", self.cbed_sp)
        f_an.addRow("Bedding Joint φ'_j:", self.phibed_sp)
        f_an.addRow("Bedding Dip:", self.dip_sp)
        f_an.addRow("Bedding Dip-Direction:", self.dipdir_sp)
        self.stack.addWidget(w_an)

        right.addWidget(self.stack)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        right.addWidget(btns)
        root.addLayout(right, 2)

        self.model_cb.currentIndexChanged.connect(self._on_model_combo)
        self._refresh_list()
        self.list_w.setCurrentRow(0)

    def _refresh_list(self) -> None:
        self._updating = True
        self.list_w.clear()
        for m in self.materials:
            self.list_w.addItem(f"{m.name} [{m.model}]")
        self._updating = False

    def _save_current(self) -> None:
        if self._updating or not (0 <= self._idx < len(self.materials)):
            return
        m = self.materials[self._idx]
        m.name = self.name_ed.text().strip() or f"Material {self._idx + 1}"
        m.model = str(self.model_cb.currentData())
        m.gamma = self.gamma_sp.value()
        m.gamma_sat = self.gsat_sp.value()
        m.use_water = self.water_chk.isChecked()
        m.hu = self.hu_sp.value()
        m.bbar = self.bbar_sp.value()
        m.c = self.c_sp.value()
        m.phi = self.phi_sp.value()
        m.su = self.su_sp.value()
        m.su_top = self.sut_sp.value()
        m.su_bottom = self.sub_sp.value()
        m.z_top = self.zt_sp.value()
        m.z_bottom = self.zb_sp.value()
        m.sigma_ci = self.sci_sp.value()
        m.gsi = self.gsi_sp.value()
        m.mi = self.mi_sp.value()
        m.d = self.d_sp.value()
        m.c_mass = self.cmass_sp.value()
        m.phi_mass = self.phimass_sp.value()
        m.c_bedding = self.cbed_sp.value()
        m.phi_bedding = self.phibed_sp.value()
        m.dip_deg = self.dip_sp.value()
        m.dip_dir_deg = self.dipdir_sp.value()

    def _load_current(self) -> None:
        if not (0 <= self._idx < len(self.materials)):
            return
        self._updating = True
        m = self.materials[self._idx]
        self.name_ed.setText(m.name)
        idx_c = max(0, self.model_cb.findData(m.model))
        self.model_cb.setCurrentIndex(idx_c)
        self.stack.setCurrentIndex(idx_c)
        self.gamma_sp.setValue(m.gamma)
        self.gsat_sp.setValue(m.gamma_sat)
        self.water_chk.setChecked(m.use_water)
        self.hu_sp.setValue(m.hu)
        self.bbar_sp.setValue(m.bbar)
        self.color_btn.setStyleSheet(
            f"background-color: {m.color}; color: #0f172a; font-weight: bold;"
        )
        self.c_sp.setValue(m.c)
        self.phi_sp.setValue(m.phi)
        self.su_sp.setValue(m.su)
        self.sut_sp.setValue(m.su_top)
        self.sub_sp.setValue(m.su_bottom)
        self.zt_sp.setValue(m.z_top)
        self.zb_sp.setValue(m.z_bottom)
        self.sci_sp.setValue(m.sigma_ci)
        self.gsi_sp.setValue(m.gsi)
        self.mi_sp.setValue(m.mi)
        self.d_sp.setValue(m.d)
        self.cmass_sp.setValue(m.c_mass)
        self.phimass_sp.setValue(m.phi_mass)
        self.cbed_sp.setValue(m.c_bedding)
        self.phibed_sp.setValue(m.phi_bedding)
        self.dip_sp.setValue(m.dip_deg)
        self.dipdir_sp.setValue(m.dip_dir_deg)
        self._updating = False

    def _on_row_changed(self, row: int) -> None:
        if self._updating or row < 0:
            return
        self._save_current()
        self._idx = row
        self._load_current()

    def _on_model_combo(self, idx: int) -> None:
        if idx >= 0:
            self.stack.setCurrentIndex(idx)

    def _pick_color(self) -> None:
        m = self.materials[self._idx]
        col = QColorDialog.getColor(QColor(m.color), self, "Select Material Color")
        if col.isValid():
            m.color = col.name()
            self.color_btn.setStyleSheet(
                f"background-color: {m.color}; color: #0f172a; font-weight: bold;"
            )

    def _add_mat(self) -> None:
        self._save_current()
        k = len(self.materials) + 1
        col = PALETTE_3D[(k - 1) % len(PALETTE_3D)]
        self.materials.append(MaterialSpec3D(name=f"Soil {k}", color=col))
        self._refresh_list()
        self.list_w.setCurrentRow(len(self.materials) - 1)

    def _del_mat(self) -> None:
        if len(self.materials) <= 1:
            QMessageBox.information(self, "Materials", "At least one material must remain.")
            return
        del self.materials[self._idx]
        self._idx = max(0, self._idx - 1)
        self._refresh_list()
        self.list_w.setCurrentRow(self._idx)
        self._load_current()

    def _on_accept(self) -> None:
        self._save_current()
        names = [m.name for m in self.materials]
        if len(set(names)) != len(names):
            QMessageBox.warning(self, "Materials", "Material names must be unique.")
            return
        self.accept()


class BoreholesDialog(QDialog):
    """Define or edit 3D borehole collar coordinates & layer contacts for RBF horizon interpolation."""

    def __init__(self, project: Project3D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D Borehole Stratigraphy Editor")
        self.resize(720, 420)
        self._mat_names = project.material_names()
        self.boreholes: list[BoreholeLog] = [
            BoreholeLog.from_dict(b.to_dict()) for b in project.boreholes
        ]

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Define 3D boreholes (X, Y, Collar Z, Water Z, Primary Layer Bottom Z). "
                "Accepting interpolates smooth 3D RBF horizons across the model domain."
            )
        )
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "Name",
                "X [m]",
                "Y [m]",
                "Collar Z [m]",
                "Water Z (opt)",
                "Upper Mat",
                "Contact Z (opt)",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        row_btns = QHBoxLayout()
        add_btn = QPushButton("+ Add Borehole")
        seed_btn = QPushButton("Generate 4 Sample Boreholes")
        del_btn = QPushButton("− Delete Selected")
        add_btn.clicked.connect(self._add_row)
        seed_btn.clicked.connect(lambda: self._seed_four(project))
        del_btn.clicked.connect(self._del_row)
        row_btns.addWidget(add_btn)
        row_btns.addWidget(seed_btn)
        row_btns.addWidget(del_btn)
        row_btns.addStretch(1)
        layout.addLayout(row_btns)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        for bh in self.boreholes:
            m0 = bh.layers[0].material if bh.layers else self._mat_names[0]
            zc = f"{bh.layers[0].z_bottom:.2f}" if len(bh.layers) >= 2 else ""
            wz = "" if bh.water_z is None else f"{bh.water_z:.2f}"
            self._append_row_values(bh.name, bh.x, bh.y, bh.collar_z, wz, m0, zc)

    def _append_row_values(
        self, name: str, x: float, y: float, cz: float, wz: str, mat: str, contact_z: str
    ) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        vals = [name, f"{x:.2f}", f"{y:.2f}", f"{cz:.2f}", wz, mat, contact_z]
        for c, txt in enumerate(vals):
            self.table.setItem(r, c, QTableWidgetItem(txt))

    def _add_row(self) -> None:
        r = self.table.rowCount() + 1
        self._append_row_values(f"BH-{r}", 10.0 * r, 15.0, 8.0, "", self._mat_names[0], "")

    def _seed_four(self, proj: Project3D) -> None:
        self.table.setRowCount(0)
        m0 = self._mat_names[0]
        c_z = "2.0" if len(self._mat_names) >= 2 else ""
        pts = [
            ("BH-1", proj.x_min + 5.0, proj.y_min + 5.0, 0.5, "0.0"),
            ("BH-2", proj.x_min + 5.0, proj.y_max - 5.0, 0.8, "0.2"),
            ("BH-3", proj.x_max - 5.0, proj.y_min + 5.0, 11.5, "7.0"),
            ("BH-4", proj.x_max - 5.0, proj.y_max - 5.0, 12.2, "7.5"),
        ]
        for nm, x, y, cz, wz in pts:
            self._append_row_values(nm, x, y, cz, wz, m0, c_z)

    def _del_row(self) -> None:
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)

    def _on_accept(self) -> None:
        out: list[BoreholeLog] = []
        try:
            for r in range(self.table.rowCount()):
                nm = (
                    self.table.item(r, 0).text() if self.table.item(r, 0) else f"BH-{r + 1}"
                ).strip()
                x = float(self.table.item(r, 1).text())
                y = float(self.table.item(r, 2).text())
                cz = float(self.table.item(r, 3).text())
                wz_txt = (self.table.item(r, 4).text() if self.table.item(r, 4) else "").strip()
                wz = float(wz_txt) if wz_txt else None
                m0 = (
                    self.table.item(r, 5).text() if self.table.item(r, 5) else self._mat_names[0]
                ).strip()
                if m0 not in self._mat_names:
                    m0 = self._mat_names[0]
                cz_txt = (self.table.item(r, 6).text() if self.table.item(r, 6) else "").strip()
                layers: list[BoreholeLayer] = []
                if cz_txt and len(self._mat_names) >= 2:
                    z_int = float(cz_txt)
                    layers.append(BoreholeLayer(material=m0, z_top=cz, z_bottom=z_int))
                    m1 = self._mat_names[1] if self._mat_names[0] == m0 else self._mat_names[0]
                    layers.append(BoreholeLayer(material=m1, z_top=z_int, z_bottom=z_int - 12.0))
                else:
                    layers.append(BoreholeLayer(material=m0, z_top=cz, z_bottom=cz - 15.0))
                out.append(BoreholeLog(name=nm, x=x, y=y, collar_z=cz, layers=layers, water_z=wz))
        except (ValueError, AttributeError) as exc:
            QMessageBox.warning(self, "Boreholes", f"Invalid numeric entry: {exc}")
            return
        self.boreholes = out
        self.accept()


class LoadsSupports3DDialog(QDialog):
    """Manage 3D patch loads, point loads, reinforcement bolts, and tension cracks."""

    def __init__(self, project: Project3D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D Loads, Reinforcement Bolts & Tension Crack")
        self.resize(740, 460)
        self._proj_copy = project.copy()

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # Tab 1: Patch Loads
        w_patch = QWidget()
        vp = QVBoxLayout(w_patch)
        self.patch_tbl = QTableWidget(0, 6)
        self.patch_tbl.setHorizontalHeaderLabels(
            ["Name", "X1 [m]", "X2 [m]", "Y1 [m]", "Y2 [m]", "q [kPa]"]
        )
        self.patch_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        vp.addWidget(self.patch_tbl, 1)
        pb_row = QHBoxLayout()
        b_add_p = QPushButton("+ Add Patch Load")
        b_del_p = QPushButton("− Remove")
        b_add_p.clicked.connect(self._add_patch)
        b_del_p.clicked.connect(lambda: self.patch_tbl.removeRow(self.patch_tbl.currentRow()))
        pb_row.addWidget(b_add_p)
        pb_row.addWidget(b_del_p)
        pb_row.addStretch(1)
        vp.addLayout(pb_row)
        tabs.addTab(w_patch, "Patch Surcharges")

        # Tab 2: 3D Bolts
        w_bolt = QWidget()
        vb = QVBoxLayout(w_bolt)
        st_box = QGroupBox("Default Support Type")
        st_form = QFormLayout(st_box)
        st0 = self._proj_copy.support_types[0] if self._proj_copy.support_types else SupportType3D()
        self.st_kind_cb = QComboBox()
        for k in SUPPORT_KINDS_3D:
            self.st_kind_cb.addItem(k, k)
        self.st_kind_cb.setCurrentIndex(max(0, self.st_kind_cb.findData(st0.kind)))
        self.st_ten_sp = _dspin(1.0, 10_000.0, st0.tensile_kn, 1, 25.0, "kN")
        self.st_bond_sp = _dspin(1.0, 5_000.0, st0.bond_strength_kn_m, 1, 10.0, "kN/m")
        self.st_app_cb = QComboBox()
        self.st_app_cb.addItems(["active", "passive"])
        self.st_app_cb.setCurrentText(st0.application)
        st_form.addRow("Support Kind:", self.st_kind_cb)
        st_form.addRow("Tensile Capacity:", self.st_ten_sp)
        st_form.addRow("Bond Pullout Strength:", self.st_bond_sp)
        st_form.addRow("Force Application:", self.st_app_cb)
        vb.addWidget(st_box)

        self.bolt_tbl = QTableWidget(0, 7)
        self.bolt_tbl.setHorizontalHeaderLabels(
            ["Label", "Head X", "Head Y", "Head Z", "Tip X", "Tip Y", "Tip Z"]
        )
        self.bolt_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        vb.addWidget(self.bolt_tbl, 1)
        bb_row = QHBoxLayout()
        b_add_b = QPushButton("+ Add 3D Bolt")
        b_del_b = QPushButton("− Remove")
        b_add_b.clicked.connect(self._add_bolt)
        b_del_b.clicked.connect(lambda: self.bolt_tbl.removeRow(self.bolt_tbl.currentRow()))
        bb_row.addWidget(b_add_b)
        bb_row.addWidget(b_del_b)
        bb_row.addStretch(1)
        vb.addLayout(bb_row)
        tabs.addTab(w_bolt, "3D Reinforcement Bolts")

        # Tab 3: Tension Crack
        w_tc = QWidget()
        ftc = QFormLayout(w_tc)
        tc0 = self._proj_copy.tension_crack or TensionCrack3D(0.0, 0.0)
        self.tc_depth_sp = _dspin(0.0, 50.0, tc0.depth_m, 2, 0.25, "m")
        self.tc_water_sp = _dspin(0.0, 1.0, tc0.water_fill_fraction, 2, 0.1)
        ftc.addRow("Tension Crack Depth (0 = none):", self.tc_depth_sp)
        ftc.addRow("Water-Filled Fraction [0..1]:", self.tc_water_sp)
        tabs.addTab(w_tc, "Tension Crack")

        layout.addWidget(tabs, 1)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        for pl in self._proj_copy.patch_loads:
            self._insert_patch_row(pl.name, pl.x1, pl.x2, pl.y1, pl.y2, pl.q1)
        for b in self._proj_copy.bolts:
            hd, tp = b["head"], b["tip"]
            self._insert_bolt_row(b.get("label", "Bolt"), hd[0], hd[1], hd[2], tp[0], tp[1], tp[2])

    def _insert_patch_row(
        self, nm: str, x1: float, x2: float, y1: float, y2: float, q: float
    ) -> None:
        r = self.patch_tbl.rowCount()
        self.patch_tbl.insertRow(r)
        for c, v in enumerate([nm, f"{x1:.2f}", f"{x2:.2f}", f"{y1:.2f}", f"{y2:.2f}", f"{q:.2f}"]):
            self.patch_tbl.setItem(r, c, QTableWidgetItem(v))

    def _add_patch(self) -> None:
        p = self._proj_copy
        ym = 0.5 * (p.y_min + p.y_max)
        self._insert_patch_row(
            "Crest Surcharge", p.x_max - 15.0, p.x_max - 5.0, ym - 8.0, ym + 8.0, 20.0
        )

    def _insert_bolt_row(
        self, lbl: str, hx: float, hy: float, hz: float, tx: float, ty: float, tz: float
    ) -> None:
        r = self.bolt_tbl.rowCount()
        self.bolt_tbl.insertRow(r)
        for c, v in enumerate(
            [lbl, f"{hx:.2f}", f"{hy:.2f}", f"{hz:.2f}", f"{tx:.2f}", f"{ty:.2f}", f"{tz:.2f}"]
        ):
            self.bolt_tbl.setItem(r, c, QTableWidgetItem(v))

    def _add_bolt(self) -> None:
        p = self._proj_copy
        ym = 0.5 * (p.y_min + p.y_max)
        xm = 0.5 * (p.x_min + p.x_max)
        self._insert_bolt_row(
            f"Bolt {self.bolt_tbl.rowCount() + 1}", xm, ym, 5.0, xm + 12.0, ym, 1.5
        )

    def _on_accept(self) -> None:
        try:
            patches: list[PatchLoad3D] = []
            for r in range(self.patch_tbl.rowCount()):
                nm = self.patch_tbl.item(r, 0).text().strip() or f"Patch {r + 1}"
                x1 = float(self.patch_tbl.item(r, 1).text())
                x2 = float(self.patch_tbl.item(r, 2).text())
                y1 = float(self.patch_tbl.item(r, 3).text())
                y2 = float(self.patch_tbl.item(r, 4).text())
                q = float(self.patch_tbl.item(r, 5).text())
                patches.append(PatchLoad3D(x1=x1, x2=x2, y1=y1, y2=y2, q1=q, q2=q, name=nm))

            st = SupportType3D(
                name="Primary Support",
                kind=str(self.st_kind_cb.currentData()),
                tensile_kn=self.st_ten_sp.value(),
                plate_kn=self.st_ten_sp.value(),
                bond_strength_kn_m=self.st_bond_sp.value(),
                application=self.st_app_cb.currentText(),
            )
            bolts: list[dict[str, Any]] = []
            for r in range(self.bolt_tbl.rowCount()):
                lbl = self.bolt_tbl.item(r, 0).text().strip() or f"Bolt {r + 1}"
                hx = float(self.bolt_tbl.item(r, 1).text())
                hy = float(self.bolt_tbl.item(r, 2).text())
                hz = float(self.bolt_tbl.item(r, 3).text())
                tx = float(self.bolt_tbl.item(r, 4).text())
                ty = float(self.bolt_tbl.item(r, 5).text())
                tz = float(self.bolt_tbl.item(r, 6).text())
                bolts.append(
                    {"head": [hx, hy, hz], "tip": [tx, ty, tz], "type": st.name, "label": lbl}
                )

            self.patch_loads = patches
            self.support_types = [st] if bolts else []
            self.bolts = bolts
            self.tension_crack = (
                TensionCrack3D(self.tc_depth_sp.value(), self.tc_water_sp.value())
                if self.tc_depth_sp.value() > 0.0
                else None
            )
        except (ValueError, AttributeError) as exc:
            QMessageBox.warning(self, "3D Loads & Bolts", f"Invalid entry: {exc}")
            return
        self.accept()


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

        self.nx_sp = QSpinBox()
        self.nx_sp.setRange(4, 150)
        self.nx_sp.setValue(an.nx)
        self.ny_sp = QSpinBox()
        self.ny_sp.setRange(1, 150)
        self.ny_sp.setValue(an.ny)
        self.slide_cb = QComboBox()
        self.slide_cb.addItems(["auto", "right_to_left", "left_to_right", "custom"])
        self.slide_cb.setCurrentText(an.sliding_mode)
        self.az_sp = _dspin(0.0, 360.0, an.azimuth_deg, 1, 5.0, "deg")
        self.adm_chk = QCheckBox("Enforce physical admissibility (Whitman & Bailey m_α)")
        self.adm_chk.setChecked(an.check_admissibility)

        fm.addRow("Columns along X (nx):", self.nx_sp)
        fm.addRow("Columns along Y (ny):", self.ny_sp)
        fm.addRow("Sliding Direction:", self.slide_cb)
        fm.addRow("Custom Azimuth:", self.az_sp)
        fm.addRow("", self.adm_chk)
        tabs.addTab(w_m, "Solvers & Columns")

        # Tab 2: Groundwater & Seismic
        w_gs = QWidget()
        fgs = QFormLayout(w_gs)
        self.pore_cb = QComboBox()
        self.pore_cb.addItems(["water_table", "ru", "none"])
        self.pore_cb.setCurrentText(an.pore_mode)
        self.ru_sp = _dspin(0.0, 1.0, an.ru, 3, 0.05)
        self.kh_sp = _dspin(0.0, 1.5, an.kh, 3, 0.02, "g")
        self.kv_sp = _dspin(-0.8, 0.8, an.kv, 3, 0.02, "g")
        self.both_kv_chk = QCheckBox("Evaluate ±kv and retain governing FoS")
        self.both_kv_chk.setChecked(an.both_kv_signs)
        fgs.addRow("Pore Pressure Mode:", self.pore_cb)
        fgs.addRow("Ru Coefficient:", self.ru_sp)
        fgs.addRow("Horizontal Seismic kh:", self.kh_sp)
        fgs.addRow("Vertical Seismic kv (+up):", self.kv_sp)
        fgs.addRow("", self.both_kv_chk)
        tabs.addTab(w_gs, "Water & Seismic")

        # Tab 3: 3D Slip Surface Search / User Ellipsoid
        w_sf = QWidget()
        fsf = QFormLayout(w_sf)
        sc = project.surface_config
        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["search", "user"])
        self.mode_cb.setCurrentText(sc.mode)
        self.kind_cb = QComboBox()
        self.kind_cb.addItems(["ellipsoid", "sphere", "spline_patch"])
        self.kind_cb.setCurrentText(sc.surface_kind)
        self.effort_cb = QComboBox()
        self.effort_cb.addItems(["fast", "normal", "fine"])
        self.effort_cb.setCurrentText(sc.effort)
        self.opt_az_chk = QCheckBox("Co-optimize ellipsoid horizontal orientation")
        self.opt_az_chk.setChecked(sc.optimize_azimuth)

        us = sc.user_surface or {}
        if us.get("kind") == "composite_ellipsoid_plane":
            us = us.get("ellipsoid", {})
        self.uxc = _dspin(-1000.0, 2000.0, float(us.get("xc", 20.0)), 2, 1.0, "m")
        self.uyc = _dspin(-1000.0, 2000.0, float(us.get("yc", 20.0)), 2, 1.0, "m")
        self.uzc = _dspin(-1000.0, 2000.0, float(us.get("zc", 18.0)), 2, 1.0, "m")
        self.urx = _dspin(0.5, 1e10, float(us.get("rx", 18.0)), 2, 1.0, "m")
        self.ury = _dspin(0.5, 1e10, float(us.get("ry", 24.0)), 2, 1.0, "m")
        self.urz = _dspin(0.5, 1e10, float(us.get("rz", 18.0)), 2, 1.0, "m")

        fsf.addRow("Surface Mode:", self.mode_cb)
        fsf.addRow("Search Surface Geometry:", self.kind_cb)
        fsf.addRow("Search Effort:", self.effort_cb)
        fsf.addRow("", self.opt_az_chk)
        fsf.addRow(QLabel("<b>User Ellipsoid (when Mode = user):</b>"))
        fsf.addRow("Centre Xc:", self.uxc)
        fsf.addRow("Centre Yc:", self.uyc)
        fsf.addRow("Centre Zc:", self.uzc)
        fsf.addRow("Semi-Axis Rx:", self.urx)
        fsf.addRow("Semi-Axis Ry:", self.ury)
        fsf.addRow("Semi-Axis Rz:", self.urz)
        tabs.addTab(w_sf, "3D Slip Surface")

        layout.addWidget(tabs, 1)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_accept(self) -> None:
        selected = [k for k, chk in self.meth_checks.items() if chk.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Analysis Settings", "Select at least one 3D LEM method.")
            return
        an = self._project.analysis
        an.methods = selected
        an.nx = self.nx_sp.value()
        an.ny = self.ny_sp.value()
        an.sliding_mode = self.slide_cb.currentText()
        an.azimuth_deg = self.az_sp.value()
        an.check_admissibility = self.adm_chk.isChecked()
        an.pore_mode = self.pore_cb.currentText()
        an.ru = self.ru_sp.value()
        an.kh = self.kh_sp.value()
        an.kv = self.kv_sp.value()
        an.both_kv_signs = self.both_kv_chk.isChecked()

        sc = self._project.surface_config
        sc.mode = self.mode_cb.currentText()
        sc.surface_kind = self.kind_cb.currentText()
        sc.effort = self.effort_cb.currentText()
        sc.primary_method = selected[0]
        sc.optimize_azimuth = self.opt_az_chk.isChecked()
        if sc.mode == "user" and (
            sc.user_surface is None or sc.user_surface.get("kind") == "ellipsoid"
        ):
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
