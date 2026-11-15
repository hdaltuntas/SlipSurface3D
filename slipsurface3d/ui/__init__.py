"""Desktop UI package for SlipSurfaceIQ-3D."""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")

from slipsurface3d.ui.main_window import MainWindow3D  # noqa: E402
from slipsurface3d.ui.viewport3d import Viewport3D  # noqa: E402

__all__ = ["MainWindow3D", "Viewport3D"]
