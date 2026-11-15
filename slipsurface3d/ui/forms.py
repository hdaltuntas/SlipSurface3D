"""Shared Qt form helpers for SlipSurfaceIQ-3D dialogs."""

from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox


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
