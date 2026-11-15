"""3D Project model (.slp3d.json), validation, 2D .slp.json importer, and 3D benchmarks."""

from __future__ import annotations

from slipsurface3d.project_examples import (
    EXAMPLE_PROJECTS_3D,
    extruded_acads_1a_project,
    hungr_1987_hemisphere_project,
    reinforced_3d_valley_project,
    rock_wedge_3d_project,
    zhang_1988_homog_ellipsoid_project,
    zhang_1988_weak_seam_project,
)
from slipsurface3d.project_import import import_2d_project
from slipsurface3d.project_model import (
    FORMAT_3D,
    PALETTE_3D,
    VERSION_3D,
    AnalysisSettings3D,
    MaterialSpec3D,
    Project3D,
    ProjectSummary3D,
    SurfaceConfig3D,
)

__all__ = [
    "EXAMPLE_PROJECTS_3D",
    "FORMAT_3D",
    "PALETTE_3D",
    "VERSION_3D",
    "AnalysisSettings3D",
    "MaterialSpec3D",
    "Project3D",
    "ProjectSummary3D",
    "SurfaceConfig3D",
    "extruded_acads_1a_project",
    "hungr_1987_hemisphere_project",
    "import_2d_project",
    "reinforced_3d_valley_project",
    "rock_wedge_3d_project",
    "zhang_1988_homog_ellipsoid_project",
    "zhang_1988_weak_seam_project",
]
