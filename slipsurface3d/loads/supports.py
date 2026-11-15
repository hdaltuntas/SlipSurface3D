"""3D surface loads, point loads, reinforcement bolts/sheets, and tension crack water forces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from slipsurface3d.geometry.surfaces import SlipSurface3D

SUPPORT_KINDS_3D = (
    "end_anchored",
    "grouted_tieback",
    "soil_nail",
    "micropile",
    "geotextile",
)


@dataclass
class PatchLoad3D:
    """Rectangular plan surcharge patch load q(x, y) [kPa] on [x1, x2] x [y1, y2].

    Supports linear variation along X from ``q1`` at ``x1`` to ``q2`` at ``x2`` so extruded 2D
    trapezoidal distributed loads reproduce exact 2D surcharge magnitudes.
    """

    x1: float
    x2: float
    y1: float
    y2: float
    q1: float
    q2: float | None = None
    azimuth_deg: float = 0.0
    plunge_deg: float = 90.0  # 90 deg = vertical downward (+Z is up, so force is -Z)
    name: str = "Patch Load"

    def __post_init__(self) -> None:
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
            if self.q2 is not None:
                self.q1, self.q2 = self.q2, self.q1
        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1
        if self.q2 is None:
            self.q2 = self.q1

    def pressure_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate surcharge pressure [kPa] on column centres (x, y)."""
        q2 = self.q1 if self.q2 is None else self.q2
        inside = (x >= self.x1) & (x <= self.x2) & (y >= self.y1) & (y <= self.y2)
        out = np.zeros_like(x, dtype=float)
        if np.any(inside):
            span = max(self.x2 - self.x1, 1e-12)
            t = np.clip((x[inside] - self.x1) / span, 0.0, 1.0)
            out[inside] = self.q1 + t * (q2 - self.q1)
        return out

    def unit_direction_xyz(self) -> np.ndarray:
        """Unit force vector in world (X, Y, Z). Default plunge=90° gives (0, 0, -1)."""
        pl = float(np.radians(self.plunge_deg))
        az = float(np.radians(self.azimuth_deg))
        ch = float(np.cos(pl))
        return np.array([ch * np.cos(az), ch * np.sin(az), -np.sin(pl)], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x1": float(self.x1),
            "x2": float(self.x2),
            "y1": float(self.y1),
            "y2": float(self.y2),
            "q1": float(self.q1),
            "q2": float(self.q1 if self.q2 is None else self.q2),
            "azimuth_deg": float(self.azimuth_deg),
            "plunge_deg": float(self.plunge_deg),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PatchLoad3D:
        return cls(
            x1=float(d["x1"]),
            x2=float(d["x2"]),
            y1=float(d["y1"]),
            y2=float(d["y2"]),
            q1=float(d["q1"]),
            q2=float(d.get("q2", d["q1"])),
            azimuth_deg=float(d.get("azimuth_deg", 0.0)),
            plunge_deg=float(d.get("plunge_deg", 90.0)),
            name=str(d.get("name", "Patch Load")),
        )


@dataclass
class PointLoad3D:
    """Concentrated 3D force [kN] applied at surface/interior point (x, y, z)."""

    x: float
    y: float
    z: float
    magnitude_kn: float
    azimuth_deg: float = 180.0
    plunge_deg: float = 90.0  # 90 deg = vertical downward
    name: str = "Point Load"

    def force_xyz(self) -> np.ndarray:
        pl = float(np.radians(self.plunge_deg))
        az = float(np.radians(self.azimuth_deg))
        ch = float(np.cos(pl))
        return float(self.magnitude_kn) * np.array(
            [ch * np.cos(az), ch * np.sin(az), -np.sin(pl)], dtype=float
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "magnitude_kn": float(self.magnitude_kn),
            "azimuth_deg": float(self.azimuth_deg),
            "plunge_deg": float(self.plunge_deg),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PointLoad3D:
        return cls(
            x=float(d["x"]),
            y=float(d["y"]),
            z=float(d["z"]),
            magnitude_kn=float(d["magnitude_kn"]),
            azimuth_deg=float(d.get("azimuth_deg", 180.0)),
            plunge_deg=float(d.get("plunge_deg", 90.0)),
            name=str(d.get("name", "Point Load")),
        )


@dataclass
class SupportType3D:
    """Strength and application properties of a 3D discrete reinforcement element."""

    name: str = "Tieback 1"
    kind: str = "grouted_tieback"
    tensile_kn: float = 200.0
    plate_kn: float = 150.0
    bond_strength_kn_m: float = 80.0  # pullout resistance per metre of bonded length [kN/m]
    bond_length_m: float = 5.0  # bonded tail length (0 = full length for soil nails)
    pile_shear_kn: float = 100.0  # transverse shear capacity for micropiles [kN]
    application: str = "active"  # active (unfactored) | passive (divided by F)
    color: str = "#d9534f"

    def __post_init__(self) -> None:
        if self.kind not in SUPPORT_KINDS_3D:
            raise ValueError(f"{self.name}: unknown support kind {self.kind!r}.")
        if self.application not in ("active", "passive"):
            raise ValueError(f"{self.name}: application must be 'active' or 'passive'.")
        if self.tensile_kn < 0 or self.plate_kn < 0 or self.bond_strength_kn_m < 0:
            raise ValueError(f"{self.name}: capacities must be non-negative.")

    def capacity_kn(self, l_above: float, l_below: float) -> float:
        """Governing axial (or shear) mobilized force [kN] at slip intersection."""
        if l_below <= 1e-6 or l_above < -1e-6:
            return 0.0
        if self.kind == "end_anchored":
            return float(
                min(self.tensile_kn, self.plate_kn if self.plate_kn > 0 else self.tensile_kn)
            )
        if self.kind == "grouted_tieback":
            total = l_above + l_below
            bonded_in_anchor = max(
                0.0, min(l_below, self.bond_length_m if self.bond_length_m > 0 else total)
            )
            pullout = self.bond_strength_kn_m * bonded_in_anchor
            head_cap = self.plate_kn + self.bond_strength_kn_m * max(
                0.0, l_above - (total - self.bond_length_m)
            )
            if self.plate_kn <= 0:
                head_cap = self.tensile_kn
            return float(min(self.tensile_kn, pullout, head_cap))
        if self.kind in ("soil_nail", "geotextile"):
            pullout_below = self.bond_strength_kn_m * l_below
            pullout_above = self.plate_kn + self.bond_strength_kn_m * max(0.0, l_above)
            return float(min(self.tensile_kn, pullout_below, pullout_above))
        if self.kind == "micropile":
            return float(self.pile_shear_kn)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "tensile_kn": float(self.tensile_kn),
            "plate_kn": float(self.plate_kn),
            "bond_strength_kn_m": float(self.bond_strength_kn_m),
            "bond_length_m": float(self.bond_length_m),
            "pile_shear_kn": float(self.pile_shear_kn),
            "application": self.application,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SupportType3D:
        return cls(
            name=str(d.get("name", "Support 1")),
            kind=str(d.get("kind", "grouted_tieback")),
            tensile_kn=float(d.get("tensile_kn", 200.0)),
            plate_kn=float(d.get("plate_kn", 150.0)),
            bond_strength_kn_m=float(d.get("bond_strength_kn_m", 80.0)),
            bond_length_m=float(d.get("bond_length_m", 5.0)),
            pile_shear_kn=float(d.get("pile_shear_kn", 100.0)),
            application=str(d.get("application", "active")),
            color=str(d.get("color", "#d9534f")),
        )


@dataclass
class Bolt3D:
    """Discrete 3D reinforcement element from head ``head_xyz`` to anchor tip ``tip_xyz``."""

    head_xyz: tuple[float, float, float]
    tip_xyz: tuple[float, float, float]
    support_type: SupportType3D
    label: str = "Bolt"

    def intersect_surface(
        self, surface: SlipSurface3D, n_samples: int = 65
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Find intersection with ``surface``.

        Returns ``(hit_xyz, force_vec_kn_on_mass, capacity_kn)`` if the bolt crosses from above
        the slip surface (inside sliding mass) to below it (stable ground), else ``None``.
        """
        a = np.asarray(self.head_xyz, dtype=float)
        b = np.asarray(self.tip_xyz, dtype=float)
        ab = b - a
        total_len = float(np.linalg.norm(ab))
        if total_len <= 1e-6:
            return None

        hit = surface.intersect_segment(a, b, n_samples=n_samples)
        if hit is None:
            return None
        hit_xyz, _normal = hit
        hit_t = float(np.dot(hit_xyz - a, ab) / (total_len * total_len))
        if not 0.0 <= hit_t <= 1.0:
            return None

        l_above = hit_t * total_len
        l_below = (1.0 - hit_t) * total_len
        cap = self.support_type.capacity_kn(l_above, l_below)
        if cap <= 0.0:
            return None

        if self.support_type.kind == "micropile":
            # Micropile resists along local basal shear plane (handled by caller using sliding azimuth)
            unit_vec = ab / total_len
        else:
            # Axial tension pulls the sliding mass toward the embedded anchor tip ``b``
            unit_vec = ab / total_len
        return hit_xyz, cap * unit_vec, cap


@dataclass
class TensionCrack3D:
    """Vertical tension crack cutting the rear of the 3D sliding mass up to depth ``depth_m``."""

    depth_m: float = 0.0
    water_fill_fraction: float = 0.0  # 0.0 = dry, 1.0 = water-filled to top of crack

    def __post_init__(self) -> None:
        if self.depth_m < 0 or not 0.0 <= self.water_fill_fraction <= 1.0:
            raise ValueError(
                "TensionCrack3D requires depth_m >= 0 and water_fill_fraction in [0, 1]."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_m": float(self.depth_m),
            "water_fill_fraction": float(self.water_fill_fraction),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TensionCrack3D:
        return cls(
            depth_m=float(d.get("depth_m", 0.0)),
            water_fill_fraction=float(d.get("water_fill_fraction", 0.0)),
        )


@dataclass
class LoadSet3D:
    """Collection of external 3D loads, bolts, and tension crack settings."""

    patch_loads: list[PatchLoad3D] = field(default_factory=list)
    point_loads: list[PointLoad3D] = field(default_factory=list)
    bolts: list[Bolt3D] = field(default_factory=list)
    tension_crack: TensionCrack3D | None = None


@dataclass
class BoltPatternSpec3D:
    """Specification for an automatic 3D slope-face reinforcement bolt array."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    spacing_x: float = 3.0
    spacing_y: float = 3.0
    length_m: float = 12.0
    plunge_deg: float = 15.0  # downward angle below horizontal [deg]
    trend_azimuth_deg: float = 0.0  # horizontal bearing from collar toward anchor tip [deg]
    support_type_name: str = "Tieback 1"
    label_prefix: str = "Pattern Bolt"
    stagger_rows: bool = False


def generate_bolt_pattern_3d(
    ground_z_fn: Callable[[Any, Any], np.ndarray],
    spec: BoltPatternSpec3D,
) -> list[dict[str, Any]]:
    """Generate collar-on-terrain 3D bolt dicts ``[{head, tip, type, label}, ...]`` over a patch."""
    if spec.length_m <= 0.0 or spec.spacing_x <= 0.0 or spec.spacing_y <= 0.0:
        raise ValueError("BoltPatternSpec3D requires positive length_m, spacing_x, and spacing_y.")
    x0, x1 = (spec.x_min, spec.x_max) if spec.x_min <= spec.x_max else (spec.x_max, spec.x_min)
    y0, y1 = (spec.y_min, spec.y_max) if spec.y_min <= spec.y_max else (spec.y_max, spec.y_min)

    nx = max(1, int(np.floor((x1 - x0) / spec.spacing_x + 1e-9)) + 1)
    ny = max(1, int(np.floor((y1 - y0) / spec.spacing_y + 1e-9)) + 1)
    xs = x0 + np.arange(nx, dtype=float) * spec.spacing_x if nx > 1 else np.array([0.5 * (x0 + x1)])
    ys_base = (
        y0 + np.arange(ny, dtype=float) * spec.spacing_y if ny > 1 else np.array([0.5 * (y0 + y1)])
    )

    pl_rad = float(np.radians(spec.plunge_deg))
    az_rad = float(np.radians(spec.trend_azimuth_deg))
    dir_vec = float(spec.length_m) * np.array(
        [np.cos(pl_rad) * np.cos(az_rad), np.cos(pl_rad) * np.sin(az_rad), -np.sin(pl_rad)],
        dtype=float,
    )

    bolts_out: list[dict[str, Any]] = []
    idx = 1
    for r_i, xv in enumerate(xs):
        y_shift = (0.5 * spec.spacing_y) if (spec.stagger_rows and (r_i % 2 == 1)) else 0.0
        for yv_raw in ys_base:
            yv = float(yv_raw + y_shift)
            if yv > y1 + 1e-6:
                continue
            zv = float(np.asarray(ground_z_fn(float(xv), yv), dtype=float).ravel()[0])
            head = np.array([float(xv), yv, zv], dtype=float)
            tip = head + dir_vec
            bolts_out.append(
                {
                    "head": [float(head[0]), float(head[1]), float(head[2])],
                    "tip": [float(tip[0]), float(tip[1]), float(tip[2])],
                    "type": spec.support_type_name,
                    "label": f"{spec.label_prefix} {idx}",
                }
            )
            idx += 1
    return bolts_out


@dataclass
class GeotextileSheet3D:
    """Horizontal sheet of planar reinforcement (geotextile / geogrid) at elevation ``z``.

    The sheet covers the plan box ``[x1, x2] x [y1, y2]`` and carries tension along
    ``direction`` ("x" or "y"). It is analysed as parallel strips ``strip_spacing`` wide, so
    each strip is cut by the slip surface where it actually crosses and takes

        T = min(tensile, pullout * L_anchored, anchorage + pullout * L_in_sliding_mass)

    per metre width, where ``tensile_kn_m`` is the long-term design strength [kN/m],
    ``pullout_kpa`` the pullout resistance per unit plan area of sheet (both faces together)
    and ``anchorage_kn_m`` the capacity of the connection at the slope face (0 = free end).
    """

    name: str = "Geotextile 1"
    x1: float = 0.0
    x2: float = 10.0
    y1: float = 0.0
    y2: float = 10.0
    z: float = 0.0
    tensile_kn_m: float = 50.0
    pullout_kpa: float = 20.0
    anchorage_kn_m: float = 0.0
    direction: str = "x"
    strip_spacing: float = 1.0
    application: str = "passive"

    def __post_init__(self) -> None:
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1
        if self.direction not in ("x", "y"):
            raise ValueError(f"{self.name}: direction must be 'x' or 'y'.")
        if self.application not in ("active", "passive"):
            raise ValueError(f"{self.name}: application must be 'active' or 'passive'.")
        if self.x2 - self.x1 <= 1e-9 or self.y2 - self.y1 <= 1e-9:
            raise ValueError(f"{self.name}: the sheet needs a non-zero plan area.")
        if self.strip_spacing <= 0 or min(self.tensile_kn_m, self.pullout_kpa) < 0:
            raise ValueError(f"{self.name}: spacing must be > 0 and strengths >= 0.")
        if self.anchorage_kn_m < 0:
            raise ValueError(f"{self.name}: anchorage capacity must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeotextileSheet3D:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_bolts(self, ground_z_fn: Callable[[Any, Any], np.ndarray]) -> list[Bolt3D]:
        """Expand into strip elements; each strip's head is its end nearest the slope face."""
        along_x = self.direction == "x"
        width = (self.y2 - self.y1) if along_x else (self.x2 - self.x1)
        n_strips = max(1, int(round(width / self.strip_spacing)))
        w_strip = width / n_strips
        strip_type = SupportType3D(
            name=f"{self.name} strip",
            kind="geotextile",
            tensile_kn=self.tensile_kn_m * w_strip,
            plate_kn=self.anchorage_kn_m * w_strip,
            bond_strength_kn_m=self.pullout_kpa * w_strip,
            bond_length_m=0.0,
            application=self.application,
            color="#0ea5e9",
        )
        bolts: list[Bolt3D] = []
        for i in range(n_strips):
            offset = (i + 0.5) * w_strip
            if along_x:
                a = (self.x1, self.y1 + offset, self.z)
                b = (self.x2, self.y1 + offset, self.z)
            else:
                a = (self.x1 + offset, self.y1, self.z)
                b = (self.x1 + offset, self.y2, self.z)
            # The face end is the one with less soil cover above the sheet.
            cover_a = float(np.asarray(ground_z_fn(a[0], a[1])).ravel()[0]) - self.z
            cover_b = float(np.asarray(ground_z_fn(b[0], b[1])).ravel()[0]) - self.z
            head, tip = (a, b) if cover_a <= cover_b else (b, a)
            bolts.append(
                Bolt3D(
                    head_xyz=head,
                    tip_xyz=tip,
                    support_type=strip_type,
                    label=f"{self.name} strip {i + 1}",
                )
            )
        return bolts


@dataclass
class PileRow3D:
    """Row of vertical stabilising piles between ``(x1, y1)`` and ``(x2, y2)``.

    Piles start at the ground surface and are ``length_m`` long. A pile cut by the slip
    surface resists sliding with ``shear_kn`` along the base of the column it crosses; piles
    that do not reach below the slip surface contribute nothing.
    """

    name: str = "Pile Row 1"
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 10.0
    spacing: float = 2.0
    length_m: float = 12.0
    shear_kn: float = 150.0
    application: str = "passive"

    def __post_init__(self) -> None:
        if self.spacing <= 0 or self.length_m <= 0 or self.shear_kn < 0:
            raise ValueError(f"{self.name}: spacing and length must be > 0, shear >= 0.")
        if self.application not in ("active", "passive"):
            raise ValueError(f"{self.name}: application must be 'active' or 'passive'.")

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PileRow3D:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def pile_positions(self) -> np.ndarray:
        """Plan (x, y) of every pile, evenly spaced and centred on the row."""
        row_len = float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))
        n_piles = int(np.floor(row_len / self.spacing + 1e-9)) + 1
        if n_piles == 1 or row_len < 1e-9:
            return np.array([[0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2)]])
        margin = 0.5 * (row_len - (n_piles - 1) * self.spacing)
        t = (margin + np.arange(n_piles) * self.spacing) / row_len
        return np.column_stack(
            [self.x1 + t * (self.x2 - self.x1), self.y1 + t * (self.y2 - self.y1)]
        )

    def to_bolts(self, ground_z_fn: Callable[[Any, Any], np.ndarray]) -> list[Bolt3D]:
        pile_type = SupportType3D(
            name=f"{self.name} pile",
            kind="micropile",
            tensile_kn=0.0,
            plate_kn=0.0,
            bond_strength_kn_m=0.0,
            pile_shear_kn=self.shear_kn,
            application=self.application,
            color="#64748b",
        )
        bolts: list[Bolt3D] = []
        for i, (px, py) in enumerate(self.pile_positions(), 1):
            z_head = float(np.asarray(ground_z_fn(float(px), float(py))).ravel()[0])
            bolts.append(
                Bolt3D(
                    head_xyz=(float(px), float(py), z_head),
                    tip_xyz=(float(px), float(py), z_head - self.length_m),
                    support_type=pile_type,
                    label=f"{self.name} pile {i}",
                )
            )
        return bolts
