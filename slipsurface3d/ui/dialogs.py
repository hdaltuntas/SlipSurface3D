"""Engineering dialogs for SlipSurfaceIQ-3D Modeler & Interpret workflows."""

from __future__ import annotations

from slipsurface3d.ui.dialogs_advanced import (
    CrossSectionComparisonDialog,
    ProbabilisticSensitivity3DDialog,
    SeismicNewmark3DDialog,
)
from slipsurface3d.ui.dialogs_analysis import AnalysisSurfaceDialog, ColumnDataDialog
from slipsurface3d.ui.dialogs_layers import WeakLayersPiezoDialog
from slipsurface3d.ui.dialogs_model import (
    BoreholesDialog,
    Import2DExtrusionDialog,
    LoadsSupports3DDialog,
    Materials3DDialog,
    ParametricSlopeDialog,
)
from slipsurface3d.ui.dialogs_stages import StagedAnalysisDialog, SupportsStagesDialog
from slipsurface3d.ui.forms import _dspin

__all__ = [
    "AnalysisSurfaceDialog",
    "BoreholesDialog",
    "ColumnDataDialog",
    "CrossSectionComparisonDialog",
    "Import2DExtrusionDialog",
    "LoadsSupports3DDialog",
    "Materials3DDialog",
    "ParametricSlopeDialog",
    "ProbabilisticSensitivity3DDialog",
    "SeismicNewmark3DDialog",
    "StagedAnalysisDialog",
    "SupportsStagesDialog",
    "WeakLayersPiezoDialog",
    "_dspin",
]
