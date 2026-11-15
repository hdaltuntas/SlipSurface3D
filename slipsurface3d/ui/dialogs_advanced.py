"""Advanced analysis dialogs: probabilistic, seismic ky/Newmark, 2D cross-section."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from slipsurface3d.analysis import AnalysisResults3D
from slipsurface3d.cross_section import compare_2d_cross_section, export_2d_slp_json
from slipsurface3d.lem.solvers import METHODS_3D
from slipsurface3d.probabilistic import (
    RandomVarSpec3D,
    run_probabilistic_3d,
    run_sensitivity_3d,
)
from slipsurface3d.project import Project3D
from slipsurface3d.seismic import (
    estimate_newmark_displacement_3d,
    find_yield_acceleration_3d,
)
from slipsurface3d.ui.forms import _dspin


class ProbabilisticSensitivity3DDialog(QDialog):
    """3D Latin Hypercube / Monte Carlo Reliability & Tornado Sensitivity Dialog."""

    def __init__(
        self,
        project: Project3D,
        results: AnalysisResults3D | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D Probabilistic Reliability & Tornado Sensitivity")
        self.resize(760, 500)
        self._project = project
        self._results = results

        layout = QVBoxLayout(self)
        cfg_box = QGroupBox("Probabilistic Simulation Settings")
        fl = QFormLayout(cfg_box)
        self.samples_sp = QSpinBox()
        self.samples_sp.setRange(20, 5000)
        self.samples_sp.setValue(200)
        self.samp_cb = QComboBox()
        self.samp_cb.addItems(["lhs", "mc"])
        self.cov_c_sp = _dspin(0.02, 0.80, 0.20, 2, 0.05)
        self.std_phi_sp = _dspin(0.2, 15.0, 2.5, 1, 0.5, "deg")
        fl.addRow("Number of Realizations:", self.samples_sp)
        fl.addRow("Sampling Scheme:", self.samp_cb)
        fl.addRow("Cohesion CoV (Lognormal):", self.cov_c_sp)
        fl.addRow("Friction Angle σ_φ (Normal):", self.std_phi_sp)
        layout.addWidget(cfg_box)

        run_row = QHBoxLayout()
        self.btn_prob = QPushButton("▶ Run 3D LHS / Monte Carlo")
        self.btn_sens = QPushButton("▶ Run 3D Tornado Sensitivity")
        self.btn_prob.clicked.connect(self.execute_probabilistic)
        self.btn_sens.clicked.connect(self.execute_sensitivity)
        run_row.addWidget(self.btn_prob)
        run_row.addWidget(self.btn_sens)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        self.summary_lbl = QLabel("Ready — click Run to evaluate 3D reliability or Tornado swings.")
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self.summary_lbl)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Parameter", "Base", "Min FoS", "Max FoS", "ΔFoS Swing"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def execute_probabilistic(self) -> None:
        specs: list[RandomVarSpec3D] = []
        for m in self._project.materials:
            if m.model in ("MC", "AN"):
                if m.c > 0.0:
                    specs.append(
                        RandomVarSpec3D("c", m.name, "lognormal", cov=self.cov_c_sp.value())
                    )
                if m.phi > 0.0:
                    specs.append(
                        RandomVarSpec3D("phi", m.name, "normal", std=self.std_phi_sp.value())
                    )
            elif m.model == "UD" and m.su > 0.0:
                specs.append(RandomVarSpec3D("su", m.name, "lognormal", cov=self.cov_c_sp.value()))
        if not specs:
            specs.append(
                RandomVarSpec3D("gamma", self._project.materials[0].name, "normal", cov=0.06)
            )
        surf = self._results.critical_surface if self._results is not None else None
        res = run_probabilistic_3d(
            self._project,
            specs,
            n_samples=self.samples_sp.value(),
            sampling=self.samp_cb.currentText(),
            fixed_surface=True,
            surface=surf,
        )
        self.summary_lbl.setText(
            f"Deterministic FoS = {res.deterministic_fos:.4f} | Mean FoS = {res.mean_fos:.4f} ± {res.std_fos:.4f} | "
            f"Pf = {100.0 * res.pf:.2f}% | β_N = {res.beta_normal:.2f} | β_LN = {res.beta_lognormal:.2f} | "
            f"[P5={res.p5:.3f}, P50={res.p50:.3f}, P95={res.p95:.3f}]"
        )

    def execute_sensitivity(self) -> None:
        surf = self._results.critical_surface if self._results is not None else None
        curves = run_sensitivity_3d(self._project, n_points=7, surface=surf)
        self.table.setRowCount(len(curves))
        for r, c in enumerate(curves):
            vals = [
                c.label,
                f"{c.base_value:.3g}",
                f"{c.min_fos:.4f}",
                f"{c.max_fos:.4f}",
                f"{c.delta_fos:.4f}",
            ]
            for col_i, txt in enumerate(vals):
                self.table.setItem(r, col_i, QTableWidgetItem(txt))
        if curves:
            self.summary_lbl.setText(
                f"Top governing parameter: {curves[0].label} (Tornado ΔFoS = {curves[0].delta_fos:.4f})"
            )


class SeismicNewmark3DDialog(QDialog):
    """3D Critical Yield Acceleration (k_y) & Empirical Newmark Permanent Displacement Dialog."""

    def __init__(
        self,
        project: Project3D,
        results: AnalysisResults3D | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D Seismic Yield Acceleration (ky) & Newmark Displacement")
        self.resize(560, 340)
        self._project = project
        self._results = results

        layout = QVBoxLayout(self)
        fl = QFormLayout()
        self.pga_sp = _dspin(0.02, 2.50, 0.35, 3, 0.05, "g")
        self.mw_sp = _dspin(4.5, 9.2, 6.8, 1, 0.2)
        fl.addRow("Design Peak Ground Accel (PGA):", self.pga_sp)
        fl.addRow("Moment Magnitude (Mw):", self.mw_sp)
        layout.addLayout(fl)

        self.run_btn = QPushButton("▶ Compute 3D ky & Newmark Displacements")
        self.run_btn.clicked.connect(self.execute_seismic)
        layout.addWidget(self.run_btn)

        self.out_lbl = QLabel("Press Compute to solve for FoS(ky) = 1.000.")
        self.out_lbl.setWordWrap(True)
        self.out_lbl.setStyleSheet("padding: 8px; font-size: 13px;")
        layout.addWidget(self.out_lbl, 1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def execute_seismic(self) -> None:
        surf = self._results.critical_surface if self._results is not None else None
        ky_res = find_yield_acceleration_3d(self._project, surface=surf)
        if not ky_res.converged:
            self.out_lbl.setText(f"ky Search Status: {ky_res.message}")
            return
        nm = estimate_newmark_displacement_3d(
            ky_g=ky_res.ky_g, pga_g=self.pga_sp.value(), mw=self.mw_sp.value()
        )
        self.out_lbl.setText(
            f"<b>Static 3D FoS (kh = 0):</b> {ky_res.static_fos:.4f}<br>"
            f"<b>Critical Yield Acceleration k_y:</b> {ky_res.ky_g:.4f} g "
            f"(FoS @ ky = {ky_res.fos_at_ky:.4f})<br><br>"
            f"<b>Jibson (2007) Median Displacement:</b> {nm.jibson_2007_median_cm:.2f} cm "
            f"(16–84%: {nm.jibson_2007_p16_cm:.2f} – {nm.jibson_2007_p84_cm:.2f} cm)<br>"
            f"<b>Ambraseys &amp; Menu (1988) Displacement:</b> {nm.ambraseys_menu_cm:.2f} cm"
        )


class CrossSectionComparisonDialog(QDialog):
    """Interactive 2D Cross-Section Slicer & F_3D / F_2D End-Effect Analyzer."""

    def __init__(
        self,
        project: Project3D,
        results: AnalysisResults3D | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("2D Cross-Section Extraction & F_3D / F_2D End-Effect Comparison")
        self.resize(680, 420)
        self._project = project
        self._results = results

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        ym = 0.5 * (project.y_min + project.y_max)
        self.y_sp = _dspin(project.y_min, project.y_max, ym, 2, 1.0, "m")
        btn_eval = QPushButton("▶ Compare 3D vs 2D at y = y0")
        btn_exp = QPushButton("Export 2D .slp.json...")
        btn_eval.clicked.connect(self.evaluate_section)
        btn_exp.clicked.connect(self.export_section_dialog)
        top.addWidget(QLabel("Section Plane y0:"))
        top.addWidget(self.y_sp)
        top.addWidget(btn_eval)
        top.addWidget(btn_exp)
        layout.addLayout(top)

        self.banner = QLabel("Click Compare to evaluate 2D plane-strain FoS vs 3D FoS.")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("font-weight: bold; padding: 6px;")
        layout.addWidget(self.banner)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["LEM Method", "F_3D", "F_2D (Section)", "F_3D / F_2D Ratio"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def evaluate_section(self) -> None:
        try:
            cmp_res = compare_2d_cross_section(
                self._project, results=self._results, y0=self.y_sp.value()
            )
        except Exception as exc:
            QMessageBox.warning(self, "2D Section Comparison", str(exc))
            return
        self.banner.setText(
            f"Section y0 = {cmp_res.y0:.2f} m | Primary F_3D = {cmp_res.primary_fos_3d:.4f} vs "
            f"F_2D = {cmp_res.primary_fos_2d:.4f} -> 3D End-Effect Ratio = {cmp_res.primary_ratio_3d_2d:.4f}"
        )
        meths = list(cmp_res.fos_3d_by_method.keys())
        self.table.setRowCount(len(meths))
        for r, m in enumerate(meths):
            f3 = cmp_res.fos_3d_by_method[m]
            f2 = cmp_res.fos_2d_by_method[m]
            rt = cmp_res.end_effect_ratios[m]
            for c_i, txt in enumerate(
                [
                    METHODS_3D[m][0],
                    f"{f3:.4f}" if np.isfinite(f3) else "N/A",
                    f"{f2:.4f}" if np.isfinite(f2) else "N/A",
                    f"{rt:.4f}" if np.isfinite(rt) else "N/A",
                ]
            ):
                self.table.setItem(r, c_i, QTableWidgetItem(txt))

    def export_section_dialog(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(
            self,
            "Export Extracted 2D Section (.slp.json)",
            f"section_y{self.y_sp.value():.1f}.slp.json",
            "2D SlipSurface Projects (*.slp.json);;JSON Files (*.json)",
        )
        if not fn:
            return
        export_2d_slp_json(fn, self._project, y0=self.y_sp.value())
