"""Design standards: partial factors on actions, material strengths and resistances.

With a design standard active the reported value is no longer a global factor of safety
but an over-design factor (ODF): the ratio of design resistance to design effect of
actions. ``ODF >= 1.0`` satisfies the limit state.

Factors are applied to the inputs before the limit-equilibrium solution:

* ``permanent_action``   multiplies soil unit weights (``gamma`` and ``gamma_sat``).
  Self-weight both drives and resists sliding, so it is factored as a single source.
  Pore-water pressures are not factored.
* ``variable_action``    multiplies patch and point load magnitudes.
* ``cohesion``           divides the cohesive strength term of effective-stress materials.
* ``friction``           divides ``tan(phi')``.
* ``undrained``          divides the strength of total-stress (undrained) materials.
* ``support_resistance`` divides bolt / anchor / nail / pile capacities.
* ``earth_resistance``   divides the whole mobilised shear resistance (EC7 ``gamma_R;e``).

For stress-dependent envelopes (Hoek-Brown, Barton-Bandis, power curves, tabulated
shear-normal functions) ``cohesion`` and ``friction`` act on the instantaneous tangent
``c`` and ``tan(phi)``, which is equivalent to dividing the shear strength when the two
factors are equal, as they are in every Eurocode 7 preset.

The Eurocode 7 presets follow the recommended values of EN 1997-1:2004 Annex A. National
annexes may prescribe different values; check them and use the user-defined set if so.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

from slipsurface3d.loads.supports import LoadSet3D, PatchLoad3D, PointLoad3D, SupportType3D
from slipsurface3d.materials.models import Material3D

_FACTOR_FIELDS = (
    "permanent_action",
    "variable_action",
    "cohesion",
    "friction",
    "undrained",
    "support_resistance",
    "earth_resistance",
)


@dataclass
class DesignFactors3D:
    """A set of partial factors. All factors are >= 1 divisors or multipliers (1 = inactive)."""

    key: str = "none"
    label: str = "None (characteristic values, global FoS)"
    permanent_action: float = 1.0
    variable_action: float = 1.0
    cohesion: float = 1.0
    friction: float = 1.0
    undrained: float = 1.0
    support_resistance: float = 1.0
    earth_resistance: float = 1.0

    def __post_init__(self) -> None:
        for name in _FACTOR_FIELDS:
            value = float(getattr(self, name))
            if not 0.1 <= value <= 10.0:
                raise ValueError(f"Design factor '{name}' must be within [0.1, 10] (got {value}).")
            setattr(self, name, value)

    @property
    def active(self) -> bool:
        return any(abs(float(getattr(self, name)) - 1.0) > 1e-12 for name in _FACTOR_FIELDS)

    @property
    def result_label(self) -> str:
        """Name of the reported ratio: over-design factor when factors are active."""
        return "ODF" if self.active else "FoS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> DesignFactors3D:
        if not d:
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})

    def summary(self) -> str:
        parts = [
            f"γG={self.permanent_action:g}",
            f"γQ={self.variable_action:g}",
            f"γc'={self.cohesion:g}",
            f"γφ'={self.friction:g}",
            f"γcu={self.undrained:g}",
            f"γa={self.support_resistance:g}",
            f"γR;e={self.earth_resistance:g}",
        ]
        return f"{self.label}: " + ", ".join(parts)


DESIGN_PRESETS: dict[str, DesignFactors3D] = {
    "none": DesignFactors3D(),
    "ec7_da1_c1": DesignFactors3D(
        key="ec7_da1_c1",
        label="Eurocode 7 — DA1 Combination 1 (A1 + M1 + R1)",
        permanent_action=1.35,
        variable_action=1.5,
        support_resistance=1.1,
    ),
    "ec7_da1_c2": DesignFactors3D(
        key="ec7_da1_c2",
        label="Eurocode 7 — DA1 Combination 2 (A2 + M2 + R1)",
        variable_action=1.3,
        cohesion=1.25,
        friction=1.25,
        undrained=1.4,
        support_resistance=1.1,
    ),
    "ec7_da2": DesignFactors3D(
        key="ec7_da2",
        label="Eurocode 7 — DA2 (A1 + M1 + R2)",
        permanent_action=1.35,
        variable_action=1.5,
        support_resistance=1.1,
        earth_resistance=1.1,
    ),
    "ec7_da3": DesignFactors3D(
        key="ec7_da3",
        label="Eurocode 7 — DA3 (A2 + M2 + R3)",
        variable_action=1.3,
        cohesion=1.25,
        friction=1.25,
        undrained=1.4,
    ),
}

USER_DESIGN_KEY = "user"
USER_DESIGN_LABEL = "User-defined partial factors"


def design_preset(key: str) -> DesignFactors3D:
    """Return a fresh copy of a named preset (``user`` starts from characteristic values)."""
    if key == USER_DESIGN_KEY:
        return DesignFactors3D(key=USER_DESIGN_KEY, label=USER_DESIGN_LABEL)
    if key not in DESIGN_PRESETS:
        raise ValueError(
            f"Unknown design standard {key!r} (expected one of {list(DESIGN_PRESETS)})."
        )
    return copy.deepcopy(DESIGN_PRESETS[key])


def apply_design_to_materials(
    materials: list[Material3D], design: DesignFactors3D
) -> list[Material3D]:
    """Factor unit weights and strengths in place and return the same list."""
    if not design.active:
        return materials
    for mat in materials:
        mat.gamma *= design.permanent_action
        mat.gamma_sat *= design.permanent_action
        strength_factor = design.undrained if mat.total_stress else design.cohesion
        mat.pf_c = strength_factor * design.earth_resistance
        mat.pf_tanphi = design.friction * design.earth_resistance
    return materials


def apply_design_to_loads(loads: LoadSet3D, design: DesignFactors3D) -> LoadSet3D:
    """Return a load set with factored variable actions and support resistances."""
    if not design.active:
        return loads
    gq = design.variable_action
    ga = design.support_resistance
    patches: list[PatchLoad3D] = []
    for pl in loads.patch_loads:
        new_pl = copy.copy(pl)
        new_pl.q1 = pl.q1 * gq
        new_pl.q2 = None if pl.q2 is None else pl.q2 * gq
        patches.append(new_pl)
    points: list[PointLoad3D] = []
    for pt in loads.point_loads:
        new_pt = copy.copy(pt)
        new_pt.magnitude_kn = pt.magnitude_kn * gq
        points.append(new_pt)
    factored_types: dict[int, SupportType3D] = {}
    bolts = []
    for bolt in loads.bolts:
        st = bolt.support_type
        if id(st) not in factored_types:
            new_st = copy.copy(st)
            new_st.tensile_kn = st.tensile_kn / ga
            new_st.plate_kn = st.plate_kn / ga
            new_st.bond_strength_kn_m = st.bond_strength_kn_m / ga
            new_st.pile_shear_kn = st.pile_shear_kn / ga
            factored_types[id(st)] = new_st
        new_bolt = copy.copy(bolt)
        new_bolt.support_type = factored_types[id(st)]
        bolts.append(new_bolt)
    return LoadSet3D(
        patch_loads=patches,
        point_loads=points,
        bolts=bolts,
        tension_crack=loads.tension_crack,
    )
