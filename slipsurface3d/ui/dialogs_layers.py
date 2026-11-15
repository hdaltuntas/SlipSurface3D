"""Model dialog for weak layers, piezometric surfaces and pore-pressure grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from slipsurface3d.exchange import HORIZON_FILE_FILTER, import_horizon_file
from slipsurface3d.geometry.terrain import HorizonSurface, WeakLayer3D
from slipsurface3d.porewater import PorePressureGrid3D, read_pore_pressure_csv
from slipsurface3d.project import Project3D
from slipsurface3d.ui.forms import _dspin

WATER_TABLE_LABEL = "(global water table)"


def _describe(horizon: HorizonSurface, project: Project3D) -> str:
    xs = np.linspace(project.x_min, project.x_max, 9)
    ys = np.linspace(project.y_min, project.y_max, 9)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = np.asarray(horizon.z(xx, yy), dtype=float)
    if not np.any(np.isfinite(zz)):
        return f"{horizon.kind} (outside model)"
    return f"{horizon.kind}, z from {np.nanmin(zz):.2f} to {np.nanmax(zz):.2f} m"


class _PlaneDialog(QDialog):
    """Ask for a plane through the model midpoint: elevation, dip and dip direction."""

    def __init__(self, title: str, z0: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        form = QFormLayout(self)
        self.z0_sp = _dspin(-10_000.0, 10_000.0, z0, 2, 0.5, "m")
        self.dip_sp = _dspin(0.0, 85.0, 0.0, 1, 1.0, "deg")
        self.dir_sp = _dspin(0.0, 360.0, 180.0, 1, 5.0, "deg")
        form.addRow("Elevation at model midpoint:", self.z0_sp)
        form.addRow("Dip:", self.dip_sp)
        form.addRow("Dip direction (0° = +X, 90° = +Y):", self.dir_sp)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)


class WeakLayersPiezoDialog(QDialog):
    """Edit weak layers, per-material piezometric surfaces and the pore-pressure grid."""

    def __init__(self, project: Project3D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Weak Layers, Piezometric Surfaces & Pore Pressure Grid")
        self.resize(680, 480)
        self._project = project
        self.weak_layers: list[WeakLayer3D] = list(project.weak_layers)
        self.piezo_surfaces: list[tuple[str, HorizonSurface]] = list(project.piezo_surfaces)
        self.material_piezo: dict[str, str] = {m.name: m.piezo for m in project.materials}
        self.pore_grid: list[list[float]] = [list(r) for r in project.pore_grid]

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # --- Weak layers
        w_wl = QWidget()
        v_wl = QVBoxLayout(w_wl)
        hint = QLabel(
            "A slip surface that would pass below a weak layer is clipped to run along it, "
            "and takes the layer's material strength there. A seam thickness > 0 also applies "
            "that strength to slip surfaces passing through the band above the layer surface."
        )
        hint.setWordWrap(True)
        v_wl.addWidget(hint)
        self.wl_table = QTableWidget(0, 4)
        self.wl_table.setHorizontalHeaderLabels(
            ["Name", "Strength Material", "Seam Thickness [m]", "Surface"]
        )
        self.wl_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v_wl.addWidget(self.wl_table, 1)
        v_wl.addLayout(
            self._buttons(self._prompt_weak_plane, self._prompt_weak_file, self._remove_weak)
        )
        tabs.addTab(w_wl, "Weak Layers")

        # --- Piezometric surfaces
        w_pz = QWidget()
        v_pz = QVBoxLayout(w_pz)
        self.pz_table = QTableWidget(0, 2)
        self.pz_table.setHorizontalHeaderLabels(["Name", "Surface"])
        self.pz_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v_pz.addWidget(self.pz_table, 1)
        v_pz.addLayout(
            self._buttons(self._prompt_piezo_plane, self._prompt_piezo_file, self._remove_piezo)
        )
        pz_hint = QLabel(
            "Pore pressure at a column base uses the piezometric surface assigned to the base "
            "material (water-table pore mode). Saturated unit weights still follow the global "
            "water table."
        )
        pz_hint.setWordWrap(True)
        v_pz.addWidget(pz_hint)
        self.assign_table = QTableWidget(0, 2)
        self.assign_table.setHorizontalHeaderLabels(["Material", "Pore Pressure From"])
        self.assign_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v_pz.addWidget(self.assign_table, 1)
        tabs.addTab(w_pz, "Piezometric Surfaces")

        # --- Pore pressure grid
        w_pg = QWidget()
        v_pg = QVBoxLayout(w_pg)
        pg_hint = QLabel(
            "Scattered pore-pressure samples (x, y, z, u [kPa]) from an external seepage "
            "analysis or piezometers. Select pore mode 'grid' in the analysis settings to use "
            "them. Linear interpolation inside the samples, nearest sample outside."
        )
        pg_hint.setWordWrap(True)
        v_pg.addWidget(pg_hint)
        self.grid_label = QLabel()
        v_pg.addWidget(self.grid_label)
        row = QHBoxLayout()
        imp_btn = QPushButton("Import CSV (x, y, z, u)...")
        imp_btn.clicked.connect(self._prompt_pore_grid)
        clr_btn = QPushButton("Clear")
        clr_btn.clicked.connect(lambda: self.set_pore_grid([]))
        row.addWidget(imp_btn)
        row.addWidget(clr_btn)
        row.addStretch(1)
        v_pg.addLayout(row)
        v_pg.addStretch(1)
        tabs.addTab(w_pg, "Pore Pressure Grid")

        layout.addWidget(tabs, 1)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._refresh_weak()
        self._refresh_piezo()
        self.set_pore_grid(self.pore_grid)

    @staticmethod
    def _buttons(on_plane: object, on_file: object, on_remove: object) -> QHBoxLayout:
        row = QHBoxLayout()
        for text, slot in (
            ("+ Add Plane...", on_plane),
            ("+ Import Surface...", on_file),
            ("− Remove", on_remove),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(slot)  # type: ignore[arg-type]
            row.addWidget(btn)
        row.addStretch(1)
        return row

    def _plane(self, z0: float, dip_deg: float, dip_dir_deg: float, name: str) -> HorizonSurface:
        p = self._project
        return HorizonSurface.from_plane(
            (p.x_min, p.x_max), (p.y_min, p.y_max), z0, dip_deg, dip_dir_deg, name=name
        )

    # ------------------------------------------------------------------ weak layers
    def add_weak_layer(self, horizon: HorizonSurface, material: str | None = None) -> None:
        self._read_tables()
        name = f"Weak Layer {len(self.weak_layers) + 1}"
        mat = material or self._project.materials[-1].name
        self.weak_layers.append(WeakLayer3D(name=name, material=mat, horizon=horizon))
        self._refresh_weak()

    def add_planar_weak_layer(
        self,
        z0: float,
        dip_deg: float = 0.0,
        dip_dir_deg: float = 180.0,
        material: str | None = None,
    ) -> None:
        self.add_weak_layer(self._plane(z0, dip_deg, dip_dir_deg, "Weak Layer Plane"), material)

    def _prompt_weak_plane(self) -> None:
        p = self._project
        z_ground = float(
            p.build_model().ground_z(0.5 * (p.x_min + p.x_max), 0.5 * (p.y_min + p.y_max))
        )
        dlg = _PlaneDialog("Planar Weak Layer", 0.5 * (p.z_min + z_ground), self)
        if dlg.exec():
            self.add_planar_weak_layer(dlg.z0_sp.value(), dlg.dip_sp.value(), dlg.dir_sp.value())

    def _prompt_weak_file(self) -> None:
        hor = self._prompt_horizon_file("Import Weak Layer Surface")
        if hor is not None:
            self.add_weak_layer(hor)

    def _remove_weak(self) -> None:
        row = self.wl_table.currentRow()
        if 0 <= row < len(self.weak_layers):
            self._read_tables()
            del self.weak_layers[row]
            self._refresh_weak()

    def _refresh_weak(self) -> None:
        names = [m.name for m in self._project.materials]
        self.wl_table.setRowCount(len(self.weak_layers))
        for r, wl in enumerate(self.weak_layers):
            self.wl_table.setItem(r, 0, QTableWidgetItem(wl.name))
            cb = QComboBox()
            cb.addItems(names)
            cb.setCurrentIndex(max(0, cb.findText(wl.material)))
            self.wl_table.setCellWidget(r, 1, cb)
            self.wl_table.setItem(r, 2, QTableWidgetItem(f"{wl.thickness:g}"))
            desc = QTableWidgetItem(_describe(wl.horizon, self._project))
            desc.setFlags(desc.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.wl_table.setItem(r, 3, desc)

    # ------------------------------------------------------------------ piezometric surfaces
    def add_piezo_surface(self, horizon: HorizonSurface) -> str:
        self._read_tables()
        existing = {n for n, _ in self.piezo_surfaces}
        k = len(self.piezo_surfaces) + 1
        while f"Piezo {k}" in existing:
            k += 1
        name = f"Piezo {k}"
        self.piezo_surfaces.append((name, horizon))
        self._refresh_piezo()
        return name

    def add_planar_piezo_surface(
        self, z0: float, dip_deg: float = 0.0, dip_dir_deg: float = 180.0
    ) -> str:
        return self.add_piezo_surface(self._plane(z0, dip_deg, dip_dir_deg, "Piezometric Plane"))

    def assign_piezo(self, material: str, piezo_name: str) -> None:
        self._read_tables()
        self.material_piezo[material] = piezo_name
        self._refresh_piezo()

    def _prompt_piezo_plane(self) -> None:
        dlg = _PlaneDialog("Planar Piezometric Surface", 0.0, self)
        if dlg.exec():
            self.add_planar_piezo_surface(dlg.z0_sp.value(), dlg.dip_sp.value(), dlg.dir_sp.value())

    def _prompt_piezo_file(self) -> None:
        hor = self._prompt_horizon_file("Import Piezometric Surface")
        if hor is not None:
            self.add_piezo_surface(hor)

    def _remove_piezo(self) -> None:
        row = self.pz_table.currentRow()
        if 0 <= row < len(self.piezo_surfaces):
            self._read_tables()
            removed = self.piezo_surfaces.pop(row)[0]
            for mat, pz in self.material_piezo.items():
                if pz == removed:
                    self.material_piezo[mat] = ""
            self._refresh_piezo()

    def _refresh_piezo(self) -> None:
        self.pz_table.setRowCount(len(self.piezo_surfaces))
        for r, (name, hor) in enumerate(self.piezo_surfaces):
            self.pz_table.setItem(r, 0, QTableWidgetItem(name))
            desc = QTableWidgetItem(_describe(hor, self._project))
            desc.setFlags(desc.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.pz_table.setItem(r, 1, desc)
        options = [WATER_TABLE_LABEL] + [n for n, _ in self.piezo_surfaces]
        self.assign_table.setRowCount(len(self._project.materials))
        for r, m in enumerate(self._project.materials):
            item = QTableWidgetItem(m.name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.assign_table.setItem(r, 0, item)
            cb = QComboBox()
            cb.addItems(options)
            cb.setCurrentIndex(max(0, cb.findText(self.material_piezo.get(m.name, ""))))
            self.assign_table.setCellWidget(r, 1, cb)

    # ------------------------------------------------------------------ pore grid
    def set_pore_grid(self, rows: list[list[float]]) -> None:
        self.pore_grid = [list(r) for r in rows]
        if not self.pore_grid:
            self.grid_label.setText("No pore-pressure samples loaded.")
            return
        arr = np.asarray(self.pore_grid, dtype=float)
        self.grid_label.setText(
            f"{len(arr)} samples — u from {arr[:, 3].min():.1f} to {arr[:, 3].max():.1f} kPa, "
            f"z from {arr[:, 2].min():.2f} to {arr[:, 2].max():.2f} m."
        )

    def _prompt_pore_grid(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(
            self, "Import Pore Pressure Grid", "", "CSV / Text (*.csv *.txt *.xyz *.dat)"
        )
        if not fn:
            return
        try:
            rows = read_pore_pressure_csv(fn)
            PorePressureGrid3D(rows)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Import Pore Pressure Grid Failed", str(exc))
            return
        self.set_pore_grid(rows)

    # ------------------------------------------------------------------ shared
    def _prompt_horizon_file(self, title: str) -> HorizonSurface | None:
        fn, _ = QFileDialog.getOpenFileName(self, title, "", HORIZON_FILE_FILTER)
        if not fn:
            return None
        try:
            return import_horizon_file(fn, name=Path(fn).stem)
        except Exception as exc:  # file parsers raise many error types
            QMessageBox.critical(self, f"{title} Failed", str(exc))
            return None

    def _read_tables(self) -> None:
        """Pull edited names and combo selections back from the tables."""
        for r, wl in enumerate(self.weak_layers):
            item = self.wl_table.item(r, 0)
            cb = self.wl_table.cellWidget(r, 1)
            if item is not None and item.text().strip():
                wl.name = item.text().strip()
            if isinstance(cb, QComboBox):
                wl.material = cb.currentText()
            t_item = self.wl_table.item(r, 2)
            if t_item is not None:
                try:
                    wl.thickness = max(0.0, float(t_item.text().replace(",", ".")))
                except ValueError:
                    pass  # keep the previous thickness when the cell is not a number
        renamed: dict[str, str] = {}
        for r, (old_name, hor) in enumerate(self.piezo_surfaces):
            item = self.pz_table.item(r, 0)
            new_name = item.text().strip() if item is not None and item.text().strip() else old_name
            renamed[old_name] = new_name
            self.piezo_surfaces[r] = (new_name, hor)
        for r, m in enumerate(self._project.materials):
            cb = self.assign_table.cellWidget(r, 1)
            if isinstance(cb, QComboBox):
                choice = cb.currentText()
                self.material_piezo[m.name] = (
                    "" if choice == WATER_TABLE_LABEL else renamed.get(choice, choice)
                )

    def _on_accept(self) -> None:
        self._read_tables()
        names = [n for n, _ in self.piezo_surfaces]
        if len(set(names)) != len(names):
            QMessageBox.warning(self, "Piezometric Surfaces", "Surface names must be unique.")
            return
        self.accept()

    def apply_to_project(self, project: Project3D) -> None:
        project.weak_layers = list(self.weak_layers)
        project.piezo_surfaces = list(self.piezo_surfaces)
        project.pore_grid = [list(r) for r in self.pore_grid]
        for m in project.materials:
            m.piezo = self.material_piezo.get(m.name, "")
