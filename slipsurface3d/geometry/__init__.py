"""3D Geometry package for SlipSurfaceIQ-3D."""

from __future__ import annotations

from slipsurface3d.geometry.intersection import BaseSample
from slipsurface3d.geometry.surfaces import (
    CompositeEllipsoidPlaneSurface,
    EllipsoidSurface,
    ImplicitSlipSurface,
    MeshSlipSurface,
    MultiPlaneWedgeSurface,
    SlipSurface3D,
    SplinePatchSurface,
    WeakLayerClippedSurface,
    WedgePlane3D,
    cylindrical_surface,
    sphere_surface,
    surface_from_dict,
)
from slipsurface3d.geometry.terrain import (
    BoreholeLayer,
    BoreholeLog,
    ExtrudedProfileRegion,
    HorizonSurface,
    SlopeModel3D,
    WeakLayer3D,
    build_2d_profile_regions,
    horizons_from_boreholes,
    simple_3d_slope,
)

__all__ = [
    "BaseSample",
    "BoreholeLayer",
    "BoreholeLog",
    "CompositeEllipsoidPlaneSurface",
    "EllipsoidSurface",
    "ImplicitSlipSurface",
    "ExtrudedProfileRegion",
    "HorizonSurface",
    "MeshSlipSurface",
    "MultiPlaneWedgeSurface",
    "SlipSurface3D",
    "SlopeModel3D",
    "WeakLayer3D",
    "WeakLayerClippedSurface",
    "SplinePatchSurface",
    "WedgePlane3D",
    "build_2d_profile_regions",
    "cylindrical_surface",
    "horizons_from_boreholes",
    "simple_3d_slope",
    "sphere_surface",
    "surface_from_dict",
]
