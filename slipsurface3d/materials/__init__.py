"""3D Geotechnical material models package."""

from __future__ import annotations

from slipsurface3d.materials.models import (
    MODEL_LABELS,
    AnisotropicBedding3D,
    BartonBandis3D,
    HoekBrown3D,
    HoekBrownDirect3D,
    Material3D,
    MohrCoulomb3D,
    PowerCurve3D,
    PowerLaw3D,
    Shansep3D,
    Undrained3D,
    UndrainedDepth3D,
    material_from_spec_dict,
)

__all__ = [
    "MODEL_LABELS",
    "AnisotropicBedding3D",
    "BartonBandis3D",
    "HoekBrown3D",
    "HoekBrownDirect3D",
    "Material3D",
    "MohrCoulomb3D",
    "PowerCurve3D",
    "PowerLaw3D",
    "Shansep3D",
    "Undrained3D",
    "UndrainedDepth3D",
    "material_from_spec_dict",
]
