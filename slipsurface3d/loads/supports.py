"""3D surface loads, point loads, reinforcement bolts/sheets, and tension crack water forces."""

from __future__ import annotations

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

        ts = np.linspace(0.0, 1.0, n_samples)
        pts = a[None, :] + ts[:, None] * ab[None, :]
        zb = surface.z(pts[:, 0], pts[:, 1])
        diff = pts[:, 2] - zb  # > 0 inside/above slip surface, < 0 below slip surface
        valid = np.isfinite(diff)
        if np.count_nonzero(valid) < 2:
            return None

        # Locate crossing from diff >= 0 (head side) to diff < 0 (anchor side)
        hit_t: float | None = None
        for k in range(n_samples - 1):
            if valid[k] and valid[k + 1] and diff[k] >= 0.0 and diff[k + 1] < 0.0:
                denom = diff[k] - diff[k + 1]
                frac = float(diff[k] / denom) if abs(denom) > 1e-12 else 0.5
                hit_t = float(ts[k] + frac * (ts[k + 1] - ts[k]))
                break
        if hit_t is None:
            return None

        l_above = hit_t * total_len
        l_below = (1.0 - hit_t) * total_len
        cap = self.support_type.capacity_kn(l_above, l_below)
        if cap <= 0.0:
            return None

        hit_xyz = a + hit_t * ab
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
