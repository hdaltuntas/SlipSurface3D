"""Dialogs for areal supports (geotextiles, pile rows) and construction stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
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

from slipsurface3d.loads.supports import GeotextileSheet3D, PileRow3D
from slipsurface3d.project import Project3D, Stage3D
from slipsurface3d.stages import StageResult3D, run_staged_analysis_3d


@dataclass
class _Column:
    attr: str
    header: str
    choices: list[str] | None = None  # None = free text / number


class _RecordTable(QTableWidget):
    """Editable table bound to a list of dataclass records, one row each."""

    def __init__(self, columns: list[_Column], parent: QWidget | None = None) -> None:
        super().__init__(0, len(columns), parent)
        self._columns = columns
        self.setHorizontalHeaderLabels([c.header for c in columns])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def add_record(self, record: Any) -> None:
        row = self.rowCount()
        self.insertRow(row)
        for c_idx, col in enumerate(self._columns):
            value = getattr(record, col.attr)
            if col.choices is not None:
                cb = QComboBox()
                cb.addItems(col.choices)
                cb.setCurrentIndex(max(0, cb.findText(str(value))))
                self.setCellWidget(row, c_idx, cb)
            else:
                if value is None:
                    text = ""
                else:
                    text = f"{value:g}" if isinstance(value, float) else str(value)
                self.setItem(row, c_idx, QTableWidgetItem(text))

    def remove_current(self) -> None:
        if self.currentRow() >= 0:
            self.removeRow(self.currentRow())

    def records(self, factory: Callable[..., Any], template: Any) -> list[Any]:
        """Rebuild records; raises ``ValueError`` naming the offending row and column."""
        out: list[Any] = []
        for row in range(self.rowCount()):
            kwargs: dict[str, Any] = {}
            for c_idx, col in enumerate(self._columns):
                widget = self.cellWidget(row, c_idx)
                if isinstance(widget, QComboBox):
                    kwargs[col.attr] = widget.currentText()
                    continue
                item = self.item(row, c_idx)
                text = item.text().strip() if item is not None else ""
                sample = getattr(template, col.attr)
                if isinstance(sample, (int, float)) and not isinstance(sample, bool):
                    try:
                        number = float(text.replace(",", "."))
                        if isinstance(sample, int) and number != int(number):
                            raise ValueError
                        kwargs[col.attr] = int(number) if isinstance(sample, int) else number
                    except ValueError as exc:
                        kind = "a whole number" if isinstance(sample, int) else "a number"
                        raise ValueError(
                            f"Row {row + 1}, '{col.header}': '{text}' is not {kind}."
                        ) from exc
                else:
                    kwargs[col.attr] = text
            try:
                out.append(factory(**kwargs))
            except ValueError as exc:
                raise ValueError(f"Row {row + 1}: {exc}") from exc
        return out


class SupportsStagesDialog(QDialog):
    """Edit geotextile sheets, pile rows and construction stages."""

    def __init__(self, project: Project3D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Geotextiles, Pile Rows & Construction Stages")
        self.resize(1180, 460)
        self._project = project
        self.geotextiles: list[GeotextileSheet3D] = list(project.geotextiles)
        self.pile_rows: list[PileRow3D] = list(project.pile_rows)
        self.stages: list[Stage3D] = list(project.stages)
        mats = [m.name for m in project.materials]
        apps = ["passive", "active"]

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        self.geo_table = _RecordTable(
            [
                _Column("name", "Name"),
                _Column("x1", "X min [m]"),
                _Column("x2", "X max [m]"),
                _Column("y1", "Y min [m]"),
                _Column("y2", "Y max [m]"),
                _Column("z", "Centre Elev. [m]"),
                _Column("dip_deg", "Dip [°]"),
                _Column("dip_dir_deg", "Dip Dir [°]"),
                _Column("azimuth_deg", "Tension Dir [°]"),
                _Column("tensile_kn_m", "Tensile [kN/m]"),
                _Column("pullout_kpa", "Pullout [kPa]"),
                _Column("anchorage_kn_m", "Face Anch. [kN/m]"),
                _Column("strip_spacing", "Strip Width [m]"),
                _Column("application", "Application", apps),
                _Column("from_stage", "From Stage"),
            ]
        )
        tabs.addTab(
            self._page(
                self.geo_table,
                "Planar reinforcement sheets, horizontal or dipping. Directions are in plan: "
                "0° = +X, 90° = +Y. Each sheet is cut strip by strip where the slip surface "
                "crosses it: T = min(tensile, pullout × anchored length, face anchorage + "
                "pullout × length in the sliding mass) per metre width. 'From Stage' is the "
                "construction stage at which it is installed (0 = from the start).",
                self._add_geotextile,
            ),
            "Geotextiles / Geogrids",
        )

        self.pile_table = _RecordTable(
            [
                _Column("name", "Name"),
                _Column("x1", "X start [m]"),
                _Column("y1", "Y start [m]"),
                _Column("x2", "X end [m]"),
                _Column("y2", "Y end [m]"),
                _Column("spacing", "Spacing [m]"),
                _Column("length_m", "Pile Length [m]"),
                _Column("shear_kn", "Shear per Pile [kN]"),
                _Column("application", "Application", apps),
                _Column("from_stage", "From Stage"),
            ]
        )
        tabs.addTab(
            self._page(
                self.pile_table,
                "Rows of vertical piles from the ground surface. A pile that passes through "
                "the slip surface resists sliding with its shear capacity.",
                self._add_pile_row,
            ),
            "Pile Rows",
        )

        self.stage_table = _RecordTable(
            [
                _Column("name", "Name"),
                _Column("kind", "Type", ["excavation", "fill"]),
                _Column("x1", "X min [m]"),
                _Column("x2", "X max [m]"),
                _Column("y1", "Y min [m]"),
                _Column("y2", "Y max [m]"),
                _Column("level", "Floor / Crest Level [m]"),
                _Column("batter", "Batter H:V"),
                _Column("fill_material", "Fill Material", ["", *mats]),
                _Column("water_level", "Water Level [m]"),
            ]
        )
        tabs.addTab(
            self._page(
                self.stage_table,
                "Stages are built in order. Inside the box the ground is cut down to "
                "(excavation) or built up to (fill) the level, then battered back to the "
                "existing ground. 'Water Level' sets a flat water table from that stage on "
                "(blank = unchanged). Run them with Analysis → Run Construction Stages.",
                self._add_stage,
            ),
            "Construction Stages",
        )

        for rec in self.geotextiles:
            self.geo_table.add_record(rec)
        for rec in self.pile_rows:
            self.pile_table.add_record(rec)
        for rec in self.stages:
            self.stage_table.add_record(rec)

        layout.addWidget(tabs, 1)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    @staticmethod
    def _page(table: _RecordTable, hint: str, on_add: Callable[[], None]) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        lbl = QLabel(hint)
        lbl.setWordWrap(True)
        v.addWidget(lbl)
        v.addWidget(table, 1)
        row = QHBoxLayout()
        add_btn = QPushButton("+ Add")
        add_btn.clicked.connect(on_add)
        del_btn = QPushButton("− Remove")
        del_btn.clicked.connect(table.remove_current)
        row.addWidget(add_btn)
        row.addWidget(del_btn)
        row.addStretch(1)
        v.addLayout(row)
        return page

    def _mid_box(self) -> tuple[float, float, float, float, float]:
        """A box over the middle third of the model and the ground level at its centre."""
        p = self._project
        x1 = p.x_min + (p.x_max - p.x_min) / 3.0
        x2 = p.x_max - (p.x_max - p.x_min) / 3.0
        y1 = p.y_min + (p.y_max - p.y_min) / 4.0
        y2 = p.y_max - (p.y_max - p.y_min) / 4.0
        z_mid = float(p.build_model().ground_z(0.5 * (x1 + x2), 0.5 * (y1 + y2)))
        return round(x1, 2), round(x2, 2), round(y1, 2), round(y2, 2), z_mid

    def _add_geotextile(self) -> None:
        x1, x2, y1, y2, z_mid = self._mid_box()
        self.geo_table.add_record(
            GeotextileSheet3D(
                name=f"Geotextile {self.geo_table.rowCount() + 1}",
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
                z=round(z_mid - 2.0, 2),
            )
        )

    def _add_pile_row(self) -> None:
        x1, x2, y1, y2, _z = self._mid_box()
        xm = round(0.5 * (x1 + x2), 2)
        self.pile_table.add_record(
            PileRow3D(name=f"Pile Row {self.pile_table.rowCount() + 1}", x1=xm, y1=y1, x2=xm, y2=y2)
        )

    def _add_stage(self) -> None:
        x1, x2, y1, y2, z_mid = self._mid_box()
        self.stage_table.add_record(
            Stage3D(
                name=f"Stage {self.stage_table.rowCount() + 1}",
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
                level=round(z_mid - 3.0, 2),
            )
        )

    def _on_accept(self) -> None:
        try:
            self.geotextiles = self.geo_table.records(GeotextileSheet3D, GeotextileSheet3D())
            self.pile_rows = self.pile_table.records(PileRow3D, PileRow3D())
            self.stages = self.stage_table.records(Stage3D, Stage3D())
        except ValueError as exc:
            QMessageBox.warning(self, "Supports & Stages", str(exc))
            return
        self.accept()

    def apply_to_project(self, project: Project3D) -> None:
        project.geotextiles = list(self.geotextiles)
        project.pile_rows = list(self.pile_rows)
        project.stages = list(self.stages)


class StagedAnalysisDialog(QDialog):
    """Run every construction stage and list the factor of safety after each one."""

    def __init__(self, project: Project3D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Construction Stage Analysis")
        self.resize(620, 360)
        self._project = project
        self.stage_results: list[StageResult3D] = []
        self.selected: StageResult3D | None = None

        layout = QVBoxLayout(self)
        note = QLabel(
            "Each stage is solved for the ground, water level and the loads and supports in "
            "force at that stage (limit equilibrium carries no stress history)."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Stage", "Critical", "Volume [m³]", "Note"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.run_btn = QPushButton("Run All Stages")
        self.run_btn.clicked.connect(self.run_stages)
        self.show_btn = QPushButton("Show Selected Stage in Viewport")
        self.show_btn.clicked.connect(self._show_selected)
        self.show_btn.setEnabled(False)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        row.addWidget(self.run_btn)
        row.addWidget(self.show_btn)
        row.addStretch(1)
        row.addWidget(close_btn)
        layout.addLayout(row)

    def run_stages(self) -> None:
        self.run_btn.setEnabled(False)
        try:
            self.stage_results = run_staged_analysis_3d(self._project)
        except ValueError as exc:
            QMessageBox.critical(self, "Construction Stage Analysis", str(exc))
            return
        finally:
            self.run_btn.setEnabled(True)
        self.table.setRowCount(len(self.stage_results))
        for r, sr in enumerate(self.stage_results):
            res = sr.results
            label = res.result_label if res is not None else "FoS"
            fos = f"{label} {sr.fos:.4f}" if np.isfinite(sr.fos) else "—"
            vol = f"{res.columns.total_volume:.1f}" if res is not None else "—"
            for c, txt in enumerate([sr.name, fos, vol, sr.error]):
                self.table.setItem(r, c, QTableWidgetItem(txt))
        self.show_btn.setEnabled(any(sr.results is not None for sr in self.stage_results))
        if self.stage_results:
            self.table.selectRow(0)

    def _show_selected(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self.stage_results) and self.stage_results[row].results is not None:
            self.selected = self.stage_results[row]
            self.accept()
