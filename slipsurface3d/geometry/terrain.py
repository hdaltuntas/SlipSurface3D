"""3D terrain, stratigraphy, borehole interpolation, and 2D profile extrusion.

World coordinate system
-----------------------
* X : primary horizontal cross-section axis [m] (matches 2D `x`)
* Y : transverse horizontal axis [m]
* Z : vertical elevation axis, positive upward [m] (matches 2D `y`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyvista as pv
import shapely
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union


def _as_2d_polyline(pts: Any, label: str = "polyline") -> np.ndarray:
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{label}: expected finite (n>=2, 2) array of [x, z] points.")
    order = np.argsort(arr[:, 0], kind="mergesort")
    arr = arr[order]
    if np.any(np.diff(arr[:, 0]) <= 0):
        # Collapse duplicate x values only when z is virtually identical; otherwise reject
        unique_x, idx = np.unique(np.round(arr[:, 0], 10), return_index=True)
        if len(unique_x) < 2 or len(unique_x) != len(arr):
            raise ValueError(f"{label}: x-coordinates must be strictly increasing.")
        arr = arr[idx]
    return arr


@dataclass
class HorizonSurface:
    """Single-valued elevation surface z = f(x, y).

    Supported representations (`kind`):
    * ``"extruded"``  : 2D profile ``profile`` ``(n, 2)`` invariant along Y.
    * ``"lofted"``    : 2D profiles at sorted ``y_sections`` linearly lofted along Y.
    * ``"grid"``      : regular elevation grid ``z_grid`` of shape ``(len(x_coords), len(y_coords))``.
    * ``"scattered"`` : scattered 3D points ``points`` ``(m, 3)`` interpolated via thin-plate RBF.
    """

    kind: str = "extruded"
    name: str = "Horizon"
    profile: np.ndarray | None = None
    y_sections: np.ndarray | None = None
    section_profiles: list[np.ndarray] | None = None
    x_coords: np.ndarray | None = None
    y_coords: np.ndarray | None = None
    z_grid: np.ndarray | None = None
    points: np.ndarray | None = None

    _grid_interp: RegularGridInterpolator | None = field(default=None, init=False, repr=False)
    _rbf_interp: RBFInterpolator | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind == "extruded":
            if self.profile is None:
                raise ValueError(f"{self.name}: extruded horizon requires 'profile'.")
            self.profile = _as_2d_polyline(self.profile, self.name)
        elif self.kind == "lofted":
            if self.y_sections is None or not self.section_profiles:
                raise ValueError(f"{self.name}: lofted horizon requires y_sections and profiles.")
            ys = np.asarray(self.y_sections, dtype=float)
            if ys.ndim != 1 or len(ys) < 2 or np.any(np.diff(ys) <= 0):
                raise ValueError(f"{self.name}: y_sections must be strictly increasing (n>=2).")
            if len(ys) != len(self.section_profiles):
                raise ValueError(f"{self.name}: y_sections and section_profiles length mismatch.")
            self.y_sections = ys
            self.section_profiles = [
                _as_2d_polyline(p, f"{self.name}[{i}]") for i, p in enumerate(self.section_profiles)
            ]
        elif self.kind == "grid":
            if self.x_coords is None or self.y_coords is None or self.z_grid is None:
                raise ValueError(f"{self.name}: grid horizon requires x_coords, y_coords, z_grid.")
            xc = np.asarray(self.x_coords, dtype=float)
            yc = np.asarray(self.y_coords, dtype=float)
            zg = np.asarray(self.z_grid, dtype=float)
            if xc.ndim != 1 or yc.ndim != 1 or len(xc) < 2 or len(yc) < 2:
                raise ValueError(f"{self.name}: x_coords and y_coords must be 1D arrays (len>=2).")
            if np.any(np.diff(xc) <= 0) or np.any(np.diff(yc) <= 0):
                raise ValueError(f"{self.name}: x_coords and y_coords must be strictly increasing.")
            if zg.shape != (len(xc), len(yc)) or not np.all(np.isfinite(zg)):
                raise ValueError(
                    f"{self.name}: z_grid must be finite with shape ({len(xc)}, {len(yc)})."
                )
            self.x_coords, self.y_coords, self.z_grid = xc, yc, zg
            self._grid_interp = RegularGridInterpolator(
                (xc, yc), zg, method="linear", bounds_error=False, fill_value=None
            )
        elif self.kind == "scattered":
            if self.points is None:
                raise ValueError(f"{self.name}: scattered horizon requires 'points'.")
            pts = np.asarray(self.points, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 1 or not np.all(np.isfinite(pts)):
                raise ValueError(f"{self.name}: points must be a finite (m>=1, 3) array.")
            self.points = pts
            if len(pts) >= 4:
                # Check non-collinearity of (x, y) before degree=1 thin-plate spline
                xy = pts[:, :2]
                centered = xy - xy.mean(axis=0)
                rank = np.linalg.matrix_rank(centered, tol=1e-8)
                if rank >= 2:
                    self._rbf_interp = RBFInterpolator(
                        xy, pts[:, 2], kernel="thin_plate_spline", smoothing=1e-6
                    )
        else:
            raise ValueError(f"Unknown HorizonSurface kind: {self.kind!r}")

    @classmethod
    def from_extruded(cls, profile: Any, name: str = "Horizon") -> HorizonSurface:
        return cls(kind="extruded", name=name, profile=_as_2d_polyline(profile, name))

    @classmethod
    def from_lofted(
        cls, y_sections: Any, profiles: list[Any], name: str = "Lofted Horizon"
    ) -> HorizonSurface:
        return cls(
            kind="lofted",
            name=name,
            y_sections=np.asarray(y_sections, dtype=float),
            section_profiles=[_as_2d_polyline(p, name) for p in profiles],
        )

    @classmethod
    def from_grid(
        cls, x_coords: Any, y_coords: Any, z_grid: Any, name: str = "Grid Horizon"
    ) -> HorizonSurface:
        return cls(
            kind="grid",
            name=name,
            x_coords=np.asarray(x_coords, dtype=float),
            y_coords=np.asarray(y_coords, dtype=float),
            z_grid=np.asarray(z_grid, dtype=float),
        )

    @classmethod
    def from_scattered(cls, points: Any, name: str = "Borehole Horizon") -> HorizonSurface:
        return cls(kind="scattered", name=name, points=np.asarray(points, dtype=float))

    @classmethod
    def from_plane(
        cls,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z0: float,
        dip_deg: float = 0.0,
        dip_dir_deg: float = 180.0,
        name: str = "Planar Horizon",
    ) -> HorizonSurface:
        """Planar horizon passing through midpoint (xm, ym, z0) with geological dip & dip-direction.

        Here ``dip_dir_deg`` is measured in the mathematical XY plane (0° = +X, 90° = +Y, 180° = -X)
        so ``dip_dir_deg = 180°`` dips toward -X (matching right-to-left slope faces).
        """
        xm = 0.5 * (x_bounds[0] + x_bounds[1])
        ym = 0.5 * (y_bounds[0] + y_bounds[1])
        xs = np.linspace(x_bounds[0], x_bounds[1], 5)
        ys = np.linspace(y_bounds[0], y_bounds[1], 5)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        az = np.radians(dip_dir_deg)
        slope = np.tan(np.radians(dip_deg))
        zz = z0 - slope * ((xx - xm) * np.cos(az) + (yy - ym) * np.sin(az))
        return cls.from_grid(xs, ys, zz, name=name)

    def z(self, x: Any, y: Any) -> np.ndarray:
        """Evaluate surface elevation z(x, y) on arbitrary broadcastable arrays."""
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        shape = xa.shape
        xf = xa.ravel()
        yf = ya.ravel()

        if self.kind == "extruded":
            assert self.profile is not None
            out = np.interp(xf, self.profile[:, 0], self.profile[:, 1])
            return out.reshape(shape)

        if self.kind == "lofted":
            assert self.y_sections is not None and self.section_profiles is not None
            ys = self.y_sections
            profs = self.section_profiles
            sec_vals = np.vstack([np.interp(xf, p[:, 0], p[:, 1]) for p in profs])
            yc = np.clip(yf, ys[0], ys[-1])
            idx = np.searchsorted(ys, yc, side="right") - 1
            idx = np.clip(idx, 0, len(ys) - 2)
            y0 = ys[idx]
            y1 = ys[idx + 1]
            w = np.where(y1 > y0, (yc - y0) / (y1 - y0), 0.0)
            cols = np.arange(len(xf))
            out = (1.0 - w) * sec_vals[idx, cols] + w * sec_vals[idx + 1, cols]
            return out.reshape(shape)

        if self.kind == "grid":
            assert (
                self._grid_interp is not None
                and self.x_coords is not None
                and self.y_coords is not None
            )
            xc = np.clip(xf, self.x_coords[0], self.x_coords[-1])
            yc = np.clip(yf, self.y_coords[0], self.y_coords[-1])
            out = np.asarray(self._grid_interp(np.column_stack([xc, yc])), dtype=float)
            return out.reshape(shape)

        assert self.points is not None
        pts = self.points
        if self._rbf_interp is not None:
            out = np.asarray(self._rbf_interp(np.column_stack([xf, yf])), dtype=float)
            return out.reshape(shape)
        if len(pts) == 1:
            return np.full(shape, float(pts[0, 2]), dtype=float)
        # Inverse-distance weighting fallback for 2-3 points or collinear boreholes
        dx = xf[:, None] - pts[None, :, 0]
        dy = yf[:, None] - pts[None, :, 1]
        dist2 = dx * dx + dy * dy
        weights = 1.0 / np.maximum(dist2, 1e-12)
        out = (weights @ pts[:, 2]) / np.sum(weights, axis=1)
        return out.reshape(shape)

    def to_mesh(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        nx: int = 60,
        ny: int = 40,
    ) -> pv.StructuredGrid:
        """Build a PyVista StructuredGrid mesh of this horizon over (x_bounds, y_bounds)."""
        xs = np.linspace(x_bounds[0], x_bounds[1], max(2, nx))
        ys = np.linspace(y_bounds[0], y_bounds[1], max(2, ny))
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        zz = self.z(xx, yy)
        grid = pv.StructuredGrid(xx, yy, zz)
        grid["Elevation"] = zz.ravel(order="F")
        return grid

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "name": self.name}
        if self.kind == "extruded" and self.profile is not None:
            d["profile"] = self.profile.tolist()
        elif (
            self.kind == "lofted"
            and self.y_sections is not None
            and self.section_profiles is not None
        ):
            d["y_sections"] = self.y_sections.tolist()
            d["section_profiles"] = [p.tolist() for p in self.section_profiles]
        elif (
            self.kind == "grid"
            and self.x_coords is not None
            and self.y_coords is not None
            and self.z_grid is not None
        ):
            d["x_coords"] = self.x_coords.tolist()
            d["y_coords"] = self.y_coords.tolist()
            d["z_grid"] = self.z_grid.tolist()
        elif self.kind == "scattered" and self.points is not None:
            d["points"] = self.points.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HorizonSurface:
        kind = str(d.get("kind", "extruded"))
        name = str(d.get("name", "Horizon"))
        if kind == "extruded":
            return cls.from_extruded(d["profile"], name=name)
        if kind == "lofted":
            return cls.from_lofted(d["y_sections"], d["section_profiles"], name=name)
        if kind == "grid":
            return cls.from_grid(d["x_coords"], d["y_coords"], d["z_grid"], name=name)
        if kind == "scattered":
            return cls.from_scattered(d["points"], name=name)
        raise ValueError(f"Unknown HorizonSurface kind: {kind!r}")


@dataclass
class BoreholeLayer:
    """Single stratigraphic interval inside a borehole log (top elevation to bottom elevation)."""

    material: str
    z_top: float
    z_bottom: float


@dataclass
class BoreholeLog:
    """Geotechnical borehole at plan coordinates (x, y) with collar elevation and intervals."""

    name: str
    x: float
    y: float
    collar_z: float
    layers: list[BoreholeLayer] = field(default_factory=list)
    water_z: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x": float(self.x),
            "y": float(self.y),
            "collar_z": float(self.collar_z),
            "water_z": None if self.water_z is None else float(self.water_z),
            "layers": [
                {
                    "material": lay.material,
                    "z_top": float(lay.z_top),
                    "z_bottom": float(lay.z_bottom),
                }
                for lay in self.layers
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BoreholeLog:
        return cls(
            name=str(d.get("name", "BH-1")),
            x=float(d["x"]),
            y=float(d["y"]),
            collar_z=float(d["collar_z"]),
            water_z=None if d.get("water_z") is None else float(d["water_z"]),
            layers=[
                BoreholeLayer(
                    material=str(it["material"]),
                    z_top=float(it["z_top"]),
                    z_bottom=float(it["z_bottom"]),
                )
                for it in d.get("layers", [])
            ],
        )


def horizons_from_boreholes(
    boreholes: list[BoreholeLog],
) -> tuple[HorizonSurface, list[tuple[str, HorizonSurface]], HorizonSurface | None, float]:
    """Interpolate ground surface, internal layer top horizons, water table, and z_min from boreholes."""
    if not boreholes:
        raise ValueError("At least one borehole is required to interpolate horizons.")

    collar_pts = np.array([[bh.x, bh.y, bh.collar_z] for bh in boreholes], dtype=float)
    ground = HorizonSurface.from_scattered(collar_pts, name="Ground Surface")

    # Preserve stratigraphic order from the first borehole that contains layers
    ordered_materials: list[str] = []
    for bh in boreholes:
        for lay in bh.layers:
            if lay.material not in ordered_materials:
                ordered_materials.append(lay.material)
    if not ordered_materials:
        raise ValueError("Boreholes must contain at least one stratigraphic layer.")

    layer_horizons: list[tuple[str, HorizonSurface]] = []
    z_min = float(np.min(collar_pts[:, 2])) - 10.0
    for idx, mat in enumerate(ordered_materials):
        if idx == 0:
            layer_horizons.append((mat, ground))
            for bh in boreholes:
                for lay in bh.layers:
                    z_min = min(z_min, float(lay.z_bottom))
            continue
        pts = []
        for bh in boreholes:
            for lay in bh.layers:
                z_min = min(z_min, float(lay.z_bottom))
                if lay.material == mat:
                    pts.append([bh.x, bh.y, lay.z_top])
                    break
        if pts:
            layer_horizons.append(
                (
                    mat,
                    HorizonSurface.from_scattered(
                        np.asarray(pts, dtype=float), name=f"Top of {mat}"
                    ),
                )
            )

    wt_pts = [[bh.x, bh.y, bh.water_z] for bh in boreholes if bh.water_z is not None]
    water = (
        HorizonSurface.from_scattered(np.asarray(wt_pts, dtype=float), name="Water Table")
        if wt_pts
        else None
    )
    return ground, layer_horizons, water, z_min


@dataclass
class ExtrudedProfileRegion:
    """Closed 2D (X, Z) polygon extruded along Y in [y_min, y_max]."""

    polygon: Polygon
    material: str

    def to_dict(self) -> dict[str, Any]:
        coords = np.asarray(self.polygon.exterior.coords, dtype=float).tolist()
        holes = [np.asarray(r.coords, dtype=float).tolist() for r in self.polygon.interiors]
        return {"material": self.material, "exterior": coords, "holes": holes}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtrudedProfileRegion:
        poly = Polygon(d["exterior"], d.get("holes") or None)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return cls(polygon=poly, material=str(d["material"]))


def _upper_profile_xz(poly: Polygon) -> np.ndarray:
    """Upper envelope z(x) of a 2D (x, z) polygon as an x-increasing polyline."""
    xs = np.unique(np.round(np.asarray(poly.exterior.coords)[:, 0], 10))
    zmin, zmax = poly.bounds[1] - 1.0, poly.bounds[3] + 1.0
    pts: list[tuple[float, float]] = []
    for x in xs:
        cut = LineString([(float(x), zmin), (float(x), zmax)]).intersection(poly)
        if cut.is_empty:
            continue
        pts.append((float(x), float(cut.bounds[3])))
    if len(pts) < 2:
        raise ValueError("Could not extract a valid upper ground profile from external polygon.")
    return np.asarray(pts, dtype=float)


def _attach_ends_2d(ml: np.ndarray, boundary: Any, domain: Polygon, tol: float) -> np.ndarray:
    ml = ml.copy()
    for end, prev in ((0, 1), (-1, -2)):
        p = Point(ml[end])
        if boundary.distance(p) <= tol:
            q = boundary.interpolate(boundary.project(p))
            d = ml[end] - ml[prev]
            n = float(np.hypot(*d))
            d = d / n if n > 0 else np.zeros(2)
            cand = np.array(q.coords[0]) + d * tol * 0.5
            if domain.contains(Point(cand)):
                cand = np.array(q.coords[0]) - d * tol * 0.5
            if domain.contains(Point(cand)):
                verts = np.asarray(boundary.coords)
                k = int(np.argmin(np.hypot(*(verts - q.coords[0]).T)))
                near = np.hypot(*(verts[k] - q.coords[0])) <= tol
                cand = verts[k] if near else np.array(q.coords[0])
            ml[end] = cand
    return ml


def build_2d_profile_regions(
    external: Any,
    material_boundaries: list[Any] | tuple[Any, ...] = (),
    assignments: list[tuple[float, float, str]] | tuple[tuple[float, float, str], ...] = (),
    default_material: str = "Soil 1",
) -> tuple[np.ndarray, list[ExtrudedProfileRegion], float]:
    """Polygonize Slide2 / SlipSurfaceIQ-LE 2D boundaries into (ground_profile, regions, z_min)."""
    ext = np.asarray(external, dtype=float)
    if ext.ndim != 2 or ext.shape[1] != 2 or len(ext) < 3 or not np.all(np.isfinite(ext)):
        raise ValueError("External 2D boundary must contain at least 3 finite [x, z] vertices.")
    domain = Polygon(ext)
    if not domain.is_valid:
        domain = domain.buffer(0)
    if not isinstance(domain, Polygon) or domain.area <= 0:
        raise ValueError("External 2D boundary must form a simple non-zero-area polygon.")
    if hasattr(shapely, "orient_polygons"):
        domain = shapely.orient_polygons(domain)

    size = max(domain.bounds[2] - domain.bounds[0], domain.bounds[3] - domain.bounds[1])
    tol = 2e-3 * size
    boundary = domain.exterior
    lines: list[Any] = [boundary]
    for i, raw in enumerate(material_boundaries or []):
        ml = np.asarray(raw, dtype=float)
        if ml.ndim != 2 or ml.shape[1] != 2 or len(ml) < 2 or not np.all(np.isfinite(ml)):
            raise ValueError(
                f"Material boundary {i + 1}: expected at least 2 finite [x, z] points."
            )
        ml = _attach_ends_2d(ml, boundary, domain, tol)
        lines.append(LineString(ml))

    noded = unary_union(lines)
    polys = [
        p
        for p in polygonize(noded)
        if p.area > 1e-6 and domain.buffer(tol).contains(p.representative_point())
    ]
    if not polys:
        polys = [domain]

    regions: list[ExtrudedProfileRegion] = []
    for p in polys:
        mat = default_material
        for x, z, name in assignments or []:
            if p.contains(Point(float(x), float(z))):
                mat = name
        regions.append(ExtrudedProfileRegion(polygon=p, material=mat))

    ground = _upper_profile_xz(domain)
    z_min = float(domain.bounds[1])
    return ground, regions, z_min


@dataclass
class WeakLayer3D:
    """A weak surface (bedding plane, shear zone, liner) with its own strength material.

    Any slip surface that would dip below ``horizon`` is clipped to run along it, and the
    column bases lying on it take the strength of ``material``.

    ``thickness`` models a seam of finite thickness lying on top of ``horizon``: a slip
    surface passing through the band ``[horizon, horizon + thickness]`` without reaching the
    bottom also takes the weak strength there. With ``thickness = 0`` only bases that run
    along the layer itself are weak. Unit weights are never changed by a weak layer; give
    the seam its own stratum if its density matters.
    """

    name: str
    material: str
    horizon: HorizonSurface
    thickness: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.thickness) or self.thickness < 0.0:
            raise ValueError(f"Weak layer '{self.name}': thickness must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "material": self.material,
            "horizon": self.horizon.to_dict(),
            "thickness": float(self.thickness),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WeakLayer3D:
        return cls(
            name=str(d.get("name", "Weak Layer")),
            material=str(d["material"]),
            horizon=HorizonSurface.from_dict(d["horizon"]),
            thickness=float(d.get("thickness", 0.0)),
        )


@dataclass
class SlopeModel3D:
    """Three-dimensional geotechnical slope geometry and stratigraphy.

    Supports both:
    * Stratified 3D ``layers``: ``[(material_name, top_horizon), ...]`` ordered from top ground
      surface downward; deeper layers take precedence where an internal horizon outcrops, matching
      standard geological/geotechnical horizon stacking.
    * ``profile_regions``: exact 2D ``(X, Z)`` polygons extruded along ``Y`` (used when importing
      arbitrary 2D ``.slp.json`` models with lenses, embankments, or non-layered zones).
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    ground: HorizonSurface
    layers: list[tuple[str, HorizonSurface]] = field(default_factory=list)
    profile_regions: list[ExtrudedProfileRegion] = field(default_factory=list)
    water_table: HorizonSurface | None = None
    name: str = "3D Slope Model"
    weak_layers: list[WeakLayer3D] = field(default_factory=list)
    # Named piezometric surfaces that individual materials can use instead of the water table
    piezo_surfaces: dict[str, HorizonSurface] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must exceed x_min ({self.x_min}).")
        if self.y_max <= self.y_min:
            raise ValueError(f"y_max ({self.y_max}) must exceed y_min ({self.y_min}).")
        if not self.layers and not self.profile_regions:
            raise ValueError("SlopeModel3D requires at least one layer or extruded profile region.")

    @property
    def x_bounds(self) -> tuple[float, float]:
        return (self.x_min, self.x_max)

    @property
    def y_bounds(self) -> tuple[float, float]:
        return (self.y_min, self.y_max)

    @property
    def span_x(self) -> float:
        return self.x_max - self.x_min

    @property
    def span_y(self) -> float:
        return self.y_max - self.y_min

    def ground_z(self, x: Any, y: Any) -> np.ndarray:
        return self.ground.z(x, y)

    def water_z(self, x: Any, y: Any) -> np.ndarray | None:
        if self.water_table is None:
            return None
        return self.water_table.z(x, y)

    def material_indices_at(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        material_names: list[str],
    ) -> np.ndarray:
        """Vectorized lookup of material index in ``material_names`` for 3D sample points (x, y, z)."""
        xa, ya, za = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        shape = xa.shape
        xf, yf, zf = xa.ravel(), ya.ravel(), za.ravel()
        name_to_idx = {name: idx for idx, name in enumerate(material_names)}
        out = np.zeros(len(xf), dtype=int)

        if self.profile_regions:
            for reg in self.profile_regions:
                idx = name_to_idx.get(reg.material, 0)
                mask = shapely.contains_xy(reg.polygon, xf, zf)
                out[mask] = idx
            return out.reshape(shape)

        # Horizon stack: layer 0 is default (below ground); deeper layers override when z <= top_z(x, y)
        if self.layers:
            out[:] = name_to_idx.get(self.layers[0][0], 0)
            for mat_name, horizon in self.layers[1:]:
                idx = name_to_idx.get(mat_name, 0)
                z_top = horizon.z(xf, yf)
                out[zf <= z_top + 1e-9] = idx
        return out.reshape(shape)

    def material_indices_columns(
        self,
        x_col: np.ndarray,
        y_col: np.ndarray,
        z_samples: np.ndarray,
        material_names: list[str],
    ) -> np.ndarray:
        """Fast column-wise material lookup where ``z_samples`` has shape ``(n_cols, n_sub)``.

        Evaluates 3D horizons once per ``(x_col, y_col)`` column instead of ``n_sub`` times.
        """
        if self.profile_regions or z_samples.ndim != 2:
            x_b = np.broadcast_to(x_col[:, None], z_samples.shape)
            y_b = np.broadcast_to(y_col[:, None], z_samples.shape)
            return self.material_indices_at(x_b, y_b, z_samples, material_names)

        n_cols, n_sub = z_samples.shape
        name_to_idx = {name: idx for idx, name in enumerate(material_names)}
        out = np.zeros((n_cols, n_sub), dtype=int)
        if self.layers:
            out[:] = name_to_idx.get(self.layers[0][0], 0)
            for mat_name, horizon in self.layers[1:]:
                idx = name_to_idx.get(mat_name, 0)
                z_top = horizon.z(x_col, y_col)[:, None]
                out[z_samples <= z_top + 1e-9] = idx
        return out

    def bake_horizon_grids(self, nx: int = 141, ny: int = 141) -> SlopeModel3D:
        """Return a fast-evaluating copy with scattered RBF horizons pre-baked onto regular grids."""
        has_scattered = (
            self.ground.kind == "scattered"
            or any(h.kind == "scattered" for _, h in self.layers)
            or (self.water_table is not None and self.water_table.kind == "scattered")
            or any(wl.horizon.kind == "scattered" for wl in self.weak_layers)
            or any(h.kind == "scattered" for h in self.piezo_surfaces.values())
        )
        if not has_scattered:
            return self

        xs = np.linspace(self.x_min, self.x_max, max(8, nx))
        ys = np.linspace(self.y_min, self.y_max, max(8, ny))
        xx, yy = np.meshgrid(xs, ys, indexing="ij")

        def _bake(h: HorizonSurface) -> HorizonSurface:
            if h.kind != "scattered":
                return h
            return HorizonSurface.from_grid(xs, ys, h.z(xx, yy), name=h.name)

        g_fast = _bake(self.ground)
        layers_fast = [
            (m_name, g_fast if h is self.ground else _bake(h)) for m_name, h in self.layers
        ]
        wt_fast = _bake(self.water_table) if self.water_table is not None else None
        return SlopeModel3D(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            z_min=self.z_min,
            ground=g_fast,
            layers=layers_fast,
            profile_regions=list(self.profile_regions),
            water_table=wt_fast,
            name=self.name,
            weak_layers=[
                WeakLayer3D(
                    name=wl.name,
                    material=wl.material,
                    horizon=_bake(wl.horizon),
                    thickness=wl.thickness,
                )
                for wl in self.weak_layers
            ],
            piezo_surfaces={k: _bake(h) for k, h in self.piezo_surfaces.items()},
        )

    def ground_mesh(self, nx: int = 64, ny: int = 44) -> pv.StructuredGrid:
        return self.ground.to_mesh(self.x_bounds, self.y_bounds, nx=nx, ny=ny)

    def water_mesh(self, nx: int = 56, ny: int = 38) -> pv.StructuredGrid | None:
        if self.water_table is None:
            return None
        return self.water_table.to_mesh(self.x_bounds, self.y_bounds, nx=nx, ny=ny)

    def layer_solid_meshes(self, nx: int = 52, ny: int = 36) -> list[tuple[str, pv.StructuredGrid]]:
        """Build 3D StructuredGrid hexahedral layer volumes between top and bottom surfaces."""
        xs = np.linspace(self.x_min, self.x_max, max(2, nx))
        ys = np.linspace(self.y_min, self.y_max, max(2, ny))
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        zg = self.ground.z(xx, yy)

        if self.layers and not self.profile_regions:
            # Effective upper/lower surfaces of each layer (deeper layers clip upper layers)
            n_lay = len(self.layers)
            raw_tops = [zg] + [h.z(xx, yy) for _, h in self.layers[1:]]
            solids: list[tuple[str, pv.StructuredGrid]] = []
            for k in range(n_lay):
                mat_name = self.layers[k][0]
                # Top of layer k is min(zg, raw_tops[k]), clipped above deeper layers
                z_top_k = np.minimum(zg, raw_tops[k])
                if k + 1 < n_lay:
                    deeper_max = np.max(np.stack(raw_tops[k + 1 :], axis=0), axis=0)
                    z_bot_k = np.clip(deeper_max, self.z_min, z_top_k)
                else:
                    z_bot_k = np.minimum(np.full_like(zg, self.z_min), z_top_k)
                if float(np.max(z_top_k - z_bot_k)) <= 1e-6:
                    continue
                x3 = np.stack([xx, xx], axis=2)
                y3 = np.stack([yy, yy], axis=2)
                z3 = np.stack([z_bot_k, z_top_k], axis=2)
                solids.append((mat_name, pv.StructuredGrid(x3, y3, z3)))
            return solids

        # Fallback single-block or extruded profile solid mesh
        z_bot = np.full_like(zg, self.z_min)
        x3 = np.stack([xx, xx], axis=2)
        y3 = np.stack([yy, yy], axis=2)
        z3 = np.stack([z_bot, zg], axis=2)
        mat0 = self.profile_regions[0].material if self.profile_regions else "Soil 1"
        return [(mat0, pv.StructuredGrid(x3, y3, z3))]


def simple_3d_slope(
    height: float = 10.0,
    ratio_h_to_v: float = 2.0,
    toe_x: float = 15.0,
    left_bench: float = 15.0,
    right_bench: float = 20.0,
    depth_below_toe: float = 10.0,
    width_y: float = 40.0,
    material: str = "Soil 1",
    water_table_profile: Any | None = None,
    name: str = "Simple 3D Slope",
) -> SlopeModel3D:
    """Create a homogeneous 3D slope extruded along Y in [0, width_y].

    Matches 2D convention: toe at ``(toe_x, 0)``, crest at ``(toe_x + ratio*H, H)``,
    slope face dipping toward ``-X`` (``azimuth_deg = 180°``).
    """
    if height <= 0 or ratio_h_to_v <= 0 or width_y <= 0:
        raise ValueError("height, ratio_h_to_v, and width_y must be positive.")
    x0 = toe_x - left_bench
    crest_x = toe_x + ratio_h_to_v * height
    x1 = crest_x + right_bench
    z_min = -depth_below_toe
    profile = np.array([[x0, 0.0], [toe_x, 0.0], [crest_x, height], [x1, height]], dtype=float)
    ground = HorizonSurface.from_extruded(profile, name="Ground Surface")
    wt = (
        HorizonSurface.from_extruded(water_table_profile, name="Water Table")
        if water_table_profile is not None
        else None
    )
    return SlopeModel3D(
        x_min=x0,
        x_max=x1,
        y_min=0.0,
        y_max=width_y,
        z_min=z_min,
        ground=ground,
        layers=[(material, ground)],
        water_table=wt,
        name=name,
    )
