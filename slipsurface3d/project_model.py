"""3D Project model (.slp3d.json), validation, 2D .slp.json importer, and 3D benchmarks."""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from slipsurface3d.design import (
    DesignFactors3D,
    apply_design_to_loads,
    apply_design_to_materials,
)
from slipsurface3d.geometry.surfaces import (
    SlipSurface3D,
    surface_from_dict,
)
from slipsurface3d.geometry.terrain import (
    BoreholeLog,
    ExtrudedProfileRegion,
    HorizonSurface,
    SlopeModel3D,
    WeakLayer3D,
    horizons_from_boreholes,
)
from slipsurface3d.lem.columns import GAMMA_W, ColumnOptions
from slipsurface3d.lem.solvers import INTERCOLUMN_FUNCTIONS, M_ALPHA_MIN_DEFAULT, METHODS_3D
from slipsurface3d.loads.supports import (
    Bolt3D,
    GeotextileSheet3D,
    LoadSet3D,
    PatchLoad3D,
    PileRow3D,
    PointLoad3D,
    SupportType3D,
    TensionCrack3D,
)
from slipsurface3d.materials.models import MODEL_LABELS, Material3D, material_from_spec_dict
from slipsurface3d.optimizer.search import EFFORT_PRESETS, SEARCH_ALGORITHMS, SearchSettings3D
from slipsurface3d.porewater import PorePressureGrid3D
from slipsurface3d.seepage import SeepageBC3D, solve_seepage_3d

FORMAT_3D = "slipsurface-3d"
VERSION_3D = 1

PALETTE_3D = [
    "#e0b438",
    "#4ea86f",
    "#e57c44",
    "#6c82d6",
    "#9b7ec8",
    "#57a8bd",
    "#c99b72",
    "#8cb84b",
]


def _now_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _assert_finite_tree(value: Any, path: str = "project") -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _assert_finite_tree(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _assert_finite_tree(v, f"{path}[{i}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"{path}: finite numeric value required (got {value!r}).")


@dataclass
class ProjectSummary3D:
    title: str = "New 3D Slope Project"
    author: str = ""
    company: str = ""
    date_created: str = field(default_factory=_now_str)
    notes: str = ""


@dataclass
class MaterialSpec3D:
    name: str = "Soil 1"
    model: str = "MC"
    gamma: float = 19.0
    gamma_sat: float = 20.0
    use_sat: bool = True
    use_water: bool = True
    hu: float = 1.0
    bbar: float = 0.0
    excess_load: bool = False
    # Piezometric surface used for this material's pore pressure ("" = global water table)
    piezo: str = ""
    color: str = PALETTE_3D[0]

    # Mohr-Coulomb
    c: float = 10.0
    phi: float = 30.0
    # Undrained / UndrainedDepth
    su: float = 40.0
    su_top: float = 25.0
    su_bottom: float = 55.0
    z_top: float = 10.0
    z_bottom: float = 0.0
    # Hoek-Brown
    sigma_ci: float = 25_000.0
    gsi: float = 50.0
    mi: float = 10.0
    d: float = 0.0
    hb_mb: float = 1.5
    hb_s: float = 0.004
    hb_a: float = 0.506
    # Barton-Bandis
    bb_phir: float = 25.0
    bb_jrc: float = 10.0
    bb_jcs: float = 50_000.0
    bb_l0: float = 0.1
    bb_ln: float = 0.1
    # Power Curve / Power Law
    pc_a: float = 1.5
    pc_b: float = 0.8
    pc_c: float = 0.0
    pc_d: float = 0.0
    coefficient: float = 0.5
    exponent: float = 0.8
    reference_pressure: float = 100.0
    # SHANSEP
    shansep_s: float = 0.22
    shansep_m: float = 0.8
    ocr: float = 1.0
    shansep_su_min: float = 0.0
    ocr_bottom: float = 1.0
    # 3D Anisotropic Bedding
    c_mass: float = 40.0
    phi_mass: float = 35.0
    c_bedding: float = 5.0
    phi_bedding: float = 18.0
    dip_deg: float = 20.0
    dip_dir_deg: float = 180.0
    transition_deg: float = 20.0
    # Shear / Normal function: [[sigma_n, tau], ...] in kPa
    sn_table: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0], [100.0, 60.0], [300.0, 150.0]]
    )

    def to_material(self) -> Material3D:
        return material_from_spec_dict(asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MaterialSpec3D:
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered: dict[str, Any] = {}
        for k, v in d.items():
            if k in valid_keys and v is not None:
                filtered[k] = v
        # Map 2D elevation keys if importing from 2D MaterialSpec
        if "su_y_top" in d and "z_top" not in filtered:
            filtered["z_top"] = float(d["su_y_top"])
        if "su_y_bottom" in d and "z_bottom" not in filtered:
            filtered["z_bottom"] = float(d["su_y_bottom"])
        return cls(**filtered)


@dataclass
class AnalysisSettings3D:
    methods: list[str] = field(default_factory=lambda: ["bishop", "janbu_simplified", "spencer"])
    nx: int = 28
    ny: int = 24
    n_sub: int = 20
    tolerance: float = 1e-4
    max_iter: int = 75
    sliding_mode: str = "auto"  # auto | right_to_left | left_to_right | custom
    azimuth_deg: float = 180.0
    pore_mode: str = "water_table"  # water_table | ru | seepage | grid | none
    gamma_w: float = GAMMA_W
    ru: float = 0.0
    kh: float = 0.0
    kv: float = 0.0
    both_kv_signs: bool = False
    min_depth: float = 0.1
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT
    check_admissibility: bool = True
    gle_interslice: str = "half-sine"
    n_workers: int = 1


@dataclass
class SurfaceConfig3D:
    mode: str = "search"  # search | user
    surface_kind: str = "ellipsoid"  # sphere | ellipsoid | spline_patch | wedge
    effort: str = "normal"  # fast | normal | fine
    primary_method: str = "bishop"
    algorithm: str = "cuckoo"  # cuckoo | pso
    optimize_azimuth: bool = False
    sao_enabled: bool = False
    seed: int = 42
    user_surface: dict[str, Any] | None = None
    y_center_range: list[float] | None = None
    ry_bounds: list[float] | None = None
    # Number of distinct critical surfaces reported by the search (1 = global minimum only)
    n_local_minima: int = 1
    # Plan search limits [x_min, x_max, y_min, y_max] the sliding mass must stay inside
    search_limits: list[float] | None = None


@dataclass
class Stage3D:
    """A construction stage: a box excavation or fill battered back to the existing ground.

    Inside the plan box the ground is cut down to (``excavation``) or built up to (``fill``)
    ``level``. Outside it the new surface rises (or falls) at ``batter`` horizontal to
    1 vertical until it meets the existing ground.
    """

    name: str = "Stage 1"
    kind: str = "excavation"  # excavation | fill
    x1: float = 0.0
    x2: float = 10.0
    y1: float = 0.0
    y2: float = 10.0
    level: float = 0.0
    batter: float = 1.5
    fill_material: str = ""
    # Flat water-table elevation from this stage on (None = unchanged from the stage before)
    water_level: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.water_level, str):
            text = self.water_level.strip().replace(",", ".")
            self.water_level = float(text) if text else None
        if self.kind not in ("excavation", "fill"):
            raise ValueError(f"{self.name}: stage kind must be 'excavation' or 'fill'.")
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1
        if self.batter < 0.0:
            raise ValueError(f"{self.name}: batter (H:V) must be >= 0.")

    def target_z(self, x: Any, y: Any) -> np.ndarray:
        """Elevation of the stage's cut / fill surface before merging with the ground."""
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        dx = np.maximum(np.maximum(self.x1 - xa, xa - self.x2), 0.0)
        dy = np.maximum(np.maximum(self.y1 - ya, ya - self.y2), 0.0)
        dist = np.hypot(dx, dy)
        rise = dist / self.batter if self.batter > 1e-9 else np.where(dist > 0.0, np.inf, 0.0)
        return self.level + rise if self.kind == "excavation" else self.level - rise

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Stage3D:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Project3D:
    summary: ProjectSummary3D = field(default_factory=ProjectSummary3D)
    materials: list[MaterialSpec3D] = field(default_factory=list)
    x_min: float = 0.0
    x_max: float = 55.0
    y_min: float = 0.0
    y_max: float = 40.0
    z_min: float = -10.0
    ground: HorizonSurface | None = None
    layers: list[tuple[str, HorizonSurface]] = field(default_factory=list)
    profile_regions: list[ExtrudedProfileRegion] = field(default_factory=list)
    boreholes: list[BoreholeLog] = field(default_factory=list)
    water_table: HorizonSurface | None = None
    patch_loads: list[PatchLoad3D] = field(default_factory=list)
    point_loads: list[PointLoad3D] = field(default_factory=list)
    support_types: list[SupportType3D] = field(default_factory=list)
    bolts: list[dict[str, Any]] = field(
        default_factory=list
    )  # [{head:[x,y,z], tip:[x,y,z], type:name, label}]
    tension_crack: TensionCrack3D | None = None
    seepage_bc: SeepageBC3D | None = None
    analysis: AnalysisSettings3D = field(default_factory=AnalysisSettings3D)
    surface_config: SurfaceConfig3D = field(default_factory=SurfaceConfig3D)
    design: DesignFactors3D = field(default_factory=DesignFactors3D)
    weak_layers: list[WeakLayer3D] = field(default_factory=list)
    piezo_surfaces: list[tuple[str, HorizonSurface]] = field(default_factory=list)
    # Scattered pore-pressure samples [[x, y, z, u_kPa], ...] used when pore_mode == "grid"
    pore_grid: list[list[float]] = field(default_factory=list)
    geotextiles: list[GeotextileSheet3D] = field(default_factory=list)
    pile_rows: list[PileRow3D] = field(default_factory=list)
    stages: list[Stage3D] = field(default_factory=list)
    # Set on the per-stage copies made by ``stages.stage_project``; a saved project is
    # always at stage 0 (initial ground), so these are not written to the project file.
    active_stage: int = 0
    stage_fills: list[tuple[str, HorizonSurface]] = field(default_factory=list)

    def copy(self) -> Project3D:
        return copy.deepcopy(self)

    @property
    def name(self) -> str:
        return self.summary.title

    def material_names(self) -> list[str]:
        return [m.name for m in self.materials]

    def material_colors(self) -> list[str]:
        return [m.color for m in self.materials]

    def rebuild_from_boreholes(self) -> None:
        """Regenerate ground, stratigraphic horizons, water table, and z_min from ``self.boreholes``."""
        ground, layer_horizons, water, z_min = horizons_from_boreholes(self.boreholes)
        self.ground = ground
        self.layers = layer_horizons
        if water is not None:
            self.water_table = water
        self.z_min = min(self.z_min, z_min)
        self.profile_regions = []

    @property
    def has_geometry(self) -> bool:
        """True when a ground surface or borehole logs define the 3D model."""
        return self.ground is not None or bool(self.boreholes)

    def validate(self, require_geometry: bool = True) -> None:
        """Check the project for consistency.

        ``require_geometry=False`` accepts a project that has no ground surface or
        boreholes yet (an empty project being set up), so it can still be saved and
        reopened. Analyses always validate with geometry required.
        """
        _assert_finite_tree(self.to_dict())
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Model domain bounds require x_max > x_min and y_max > y_min.")
        if not self.materials:
            raise ValueError("Define at least one material.")
        names = self.material_names()
        if len(set(names)) != len(names):
            raise ValueError("Material names must be unique.")
        for m in self.materials:
            if m.model not in MODEL_LABELS and m.model not in ("HS", "CN", "SS"):
                raise ValueError(f"{m.name}: unsupported material model {m.model!r}.")
            m.to_material()

        if require_geometry or self.has_geometry:
            if not self.has_geometry:
                raise ValueError("Project3D requires a ground surface or borehole logs.")
            if not self.layers and not self.profile_regions and not self.boreholes:
                raise ValueError(
                    "Project3D requires at least one stratigraphic layer or extruded region."
                )

        a = self.analysis
        if not a.methods:
            raise ValueError("Select at least one 3D LEM analysis method.")
        for meth in a.methods:
            if meth not in METHODS_3D:
                raise ValueError(f"Unknown 3D LEM method: {meth!r}.")
        if a.nx < 3 or a.ny < 1 or a.n_sub < 4:
            raise ValueError("Column grid requires nx >= 3, ny >= 1, n_sub >= 4.")
        if a.tolerance <= 0 or a.max_iter < 1:
            raise ValueError("Solver tolerance > 0 and max_iter >= 1 required.")
        if a.pore_mode not in ("water_table", "ru", "seepage", "grid", "none"):
            raise ValueError(f"Unknown pore_mode: {a.pore_mode!r}.")
        if a.pore_mode == "grid":
            if not self.pore_grid:
                raise ValueError("pore_mode='grid' requires imported pore-pressure samples.")
            PorePressureGrid3D(self.pore_grid, gamma_w=a.gamma_w)
        for stage in self.stages:
            if stage.kind == "fill":
                if stage.fill_material and stage.fill_material not in names:
                    raise ValueError(
                        f"Stage '{stage.name}': fill material {stage.fill_material!r} is not defined."
                    )
        for wl in self.weak_layers:
            if wl.material not in names:
                raise ValueError(
                    f"Weak layer '{wl.name}': material {wl.material!r} is not defined."
                )
        piezo_names = [n for n, _ in self.piezo_surfaces]
        if len(set(piezo_names)) != len(piezo_names) or "" in piezo_names:
            raise ValueError("Piezometric surface names must be unique and non-empty.")
        for m in self.materials:
            if m.piezo and m.piezo not in piezo_names:
                raise ValueError(f"{m.name}: piezometric surface {m.piezo!r} is not defined.")
        if a.gamma_w <= 0 or not 0.0 <= a.ru <= 1.0:
            raise ValueError("Require gamma_w > 0 and 0 <= ru <= 1.")
        if a.kh < 0 or not -1.0 < a.kv < 1.0:
            raise ValueError("Require kh >= 0 and -1 < kv < 1.")
        if a.sliding_mode not in ("auto", "right_to_left", "left_to_right", "custom"):
            raise ValueError(f"Unknown sliding_mode: {a.sliding_mode!r}.")
        if a.gle_interslice not in INTERCOLUMN_FUNCTIONS:
            raise ValueError(f"Unknown gle_interslice: {a.gle_interslice!r}.")
        if a.n_workers < 0:
            raise ValueError("Require n_workers >= 0.")

        sc = self.surface_config
        if sc.mode not in ("search", "user"):
            raise ValueError("surface_config.mode must be 'search' or 'user'.")
        if sc.surface_kind not in ("sphere", "ellipsoid", "spline_patch", "wedge"):
            raise ValueError(f"Unknown surface_kind: {sc.surface_kind!r}.")
        if sc.effort not in EFFORT_PRESETS:
            raise ValueError(f"Unknown search effort: {sc.effort!r}.")
        if sc.algorithm not in SEARCH_ALGORITHMS:
            raise ValueError(f"Unknown search algorithm: {sc.algorithm!r}.")
        if sc.primary_method not in METHODS_3D:
            raise ValueError(f"Unknown primary search method: {sc.primary_method!r}.")
        if not 1 <= sc.n_local_minima <= 12:
            raise ValueError("surface_config.n_local_minima must be between 1 and 12.")
        if sc.search_limits is not None:
            if len(sc.search_limits) != 4:
                raise ValueError("search_limits requires [x_min, x_max, y_min, y_max].")
            lx0, lx1, ly0, ly1 = (float(v) for v in sc.search_limits)
            if lx1 <= lx0 or ly1 <= ly0:
                raise ValueError("search_limits require x_max > x_min and y_max > y_min.")
            if lx1 <= self.x_min or lx0 >= self.x_max or ly1 <= self.y_min or ly0 >= self.y_max:
                raise ValueError("search_limits lie outside the model domain.")
        if sc.mode == "user":
            if sc.user_surface is None:
                raise ValueError("surface_config.mode='user' requires user_surface definition.")
            surface_from_dict(sc.user_surface)

        st_names = {st.name for st in self.support_types}
        for i, b in enumerate(self.bolts, 1):
            if b.get("type") not in st_names:
                raise ValueError(f"Bolt {i}: support type {b.get('type')!r} is not defined.")
            head = np.asarray(b.get("head", []), dtype=float)
            tip = np.asarray(b.get("tip", []), dtype=float)
            if head.shape != (3,) or tip.shape != (3,) or np.linalg.norm(tip - head) <= 1e-6:
                raise ValueError(f"Bolt {i}: distinct 3D head and tip coordinates are required.")

    def build_model(self) -> SlopeModel3D:
        if self.ground is None and self.boreholes:
            self.rebuild_from_boreholes()
        assert self.ground is not None
        return SlopeModel3D(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            z_min=self.z_min,
            ground=self.ground,
            layers=list(self.layers),
            profile_regions=list(self.profile_regions),
            water_table=self.water_table,
            name=self.summary.title,
            weak_layers=list(self.weak_layers),
            piezo_surfaces=dict(self.piezo_surfaces),
            fill_layers=list(self.stage_fills),
        )

    def build_materials(self) -> list[Material3D]:
        return apply_design_to_materials([m.to_material() for m in self.materials], self.design)

    def build_loads(self) -> LoadSet3D:
        # Loads and supports act from their ``from_stage`` onward; a plain analysis is stage 0.
        stage = self.active_stage
        st_map = {st.name: st for st in self.support_types}
        bolt_objs: list[Bolt3D] = []
        for i, b in enumerate(self.bolts, 1):
            st = st_map.get(b["type"])
            if st is None or int(b.get("from_stage", 0)) > stage:
                continue
            h = tuple(float(v) for v in b["head"])
            t = tuple(float(v) for v in b["tip"])
            bolt_objs.append(
                Bolt3D(
                    head_xyz=(h[0], h[1], h[2]),
                    tip_xyz=(t[0], t[1], t[2]),
                    support_type=st,
                    label=str(b.get("label", f"Bolt {i}")),
                )
            )
        if self.geotextiles or self.pile_rows:
            if self.ground is None and self.boreholes:
                self.rebuild_from_boreholes()
            assert self.ground is not None
            for sheet in self.geotextiles:
                if sheet.from_stage <= stage:
                    bolt_objs.extend(sheet.to_bolts(self.ground.z))
            for row in self.pile_rows:
                if row.from_stage <= stage:
                    bolt_objs.extend(row.to_bolts(self.ground.z))
        loads = LoadSet3D(
            patch_loads=[pl for pl in self.patch_loads if pl.from_stage <= stage],
            point_loads=[pt for pt in self.point_loads if pt.from_stage <= stage],
            bolts=bolt_objs,
            tension_crack=self.tension_crack,
        )
        return apply_design_to_loads(loads, self.design)

    def build_column_options(self, kv_override: float | None = None) -> ColumnOptions:
        a = self.analysis
        pore = a.pore_mode
        pore_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None
        if pore == "water_table" and self.water_table is None and not self.piezo_surfaces:
            pore = "none"
        elif pore == "grid":
            pore_fn = PorePressureGrid3D(self.pore_grid, gamma_w=a.gamma_w)
        elif pore == "seepage":
            seep_grid = solve_seepage_3d(self.build_model(), bc=self.seepage_bc, gamma_w=a.gamma_w)
            pore_fn = seep_grid.pore_pressure_at
            if self.water_table is None:
                self.water_table = seep_grid.phreatic_horizon()
        return ColumnOptions(
            nx=a.nx,
            ny=a.ny,
            n_sub=a.n_sub,
            gamma_w=a.gamma_w,
            pore_mode=pore,
            ru=a.ru,
            kh=a.kh,
            kv=a.kv if kv_override is None else kv_override,
            sliding_mode=a.sliding_mode,
            azimuth_deg=a.azimuth_deg,
            min_depth=a.min_depth,
            loads=self.build_loads(),
            pore_pressure_fn=pore_fn,
        )

    def build_search_settings(self) -> SearchSettings3D:
        sc = self.surface_config
        return SearchSettings3D(
            surface_kind=sc.surface_kind,
            effort=sc.effort,
            primary_method=sc.primary_method,
            algorithm=sc.algorithm,
            optimize_azimuth=sc.optimize_azimuth,
            sao_enabled=sc.sao_enabled,
            n_workers=self.analysis.n_workers,
            gle_interslice=self.analysis.gle_interslice,
            check_admissibility=self.analysis.check_admissibility,
            m_alpha_min=self.analysis.m_alpha_min,
            seed=sc.seed,
            n_local_minima=int(sc.n_local_minima),
            search_limits=(
                (
                    float(sc.search_limits[0]),
                    float(sc.search_limits[1]),
                    float(sc.search_limits[2]),
                    float(sc.search_limits[3]),
                )
                if sc.search_limits and len(sc.search_limits) == 4
                else None
            ),
            y_center_range=(
                (float(sc.y_center_range[0]), float(sc.y_center_range[1]))
                if sc.y_center_range and len(sc.y_center_range) == 2
                else None
            ),
            ry_bounds=(
                (float(sc.ry_bounds[0]), float(sc.ry_bounds[1]))
                if sc.ry_bounds and len(sc.ry_bounds) == 2
                else None
            ),
        )

    def user_slip_surface(self) -> SlipSurface3D | None:
        if self.surface_config.user_surface is None:
            return None
        return surface_from_dict(self.surface_config.user_surface)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT_3D,
            "version": VERSION_3D,
            "summary": asdict(self.summary),
            "materials": [asdict(m) for m in self.materials],
            "domain": {
                "x_min": float(self.x_min),
                "x_max": float(self.x_max),
                "y_min": float(self.y_min),
                "y_max": float(self.y_max),
                "z_min": float(self.z_min),
            },
            "ground": None if self.ground is None else self.ground.to_dict(),
            "layers": [{"material": mat, "horizon": h.to_dict()} for mat, h in self.layers],
            "profile_regions": [r.to_dict() for r in self.profile_regions],
            "boreholes": [bh.to_dict() for bh in self.boreholes],
            "water_table": None if self.water_table is None else self.water_table.to_dict(),
            "patch_loads": [pl.to_dict() for pl in self.patch_loads],
            "point_loads": [pt.to_dict() for pt in self.point_loads],
            "support_types": [st.to_dict() for st in self.support_types],
            "bolts": copy.deepcopy(self.bolts),
            "tension_crack": None if self.tension_crack is None else self.tension_crack.to_dict(),
            "seepage_bc": None if self.seepage_bc is None else self.seepage_bc.to_dict(),
            "analysis": asdict(self.analysis),
            "surface_config": asdict(self.surface_config),
            "design": self.design.to_dict(),
            "weak_layers": [wl.to_dict() for wl in self.weak_layers],
            "piezo_surfaces": [
                {"name": name, "horizon": hor.to_dict()} for name, hor in self.piezo_surfaces
            ],
            "pore_grid": [[float(v) for v in row] for row in self.pore_grid],
            "geotextiles": [g.to_dict() for g in self.geotextiles],
            "pile_rows": [r.to_dict() for r in self.pile_rows],
            "stages": [s.to_dict() for s in self.stages],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Project3D:
        if not isinstance(d, dict):
            raise ValueError("Project file must contain a JSON object.")
        fmt = d.get("format")
        if fmt == "slipsurface-le":
            from slipsurface3d.project_import import import_2d_project

            return import_2d_project(d)
        if fmt != FORMAT_3D:
            raise ValueError(f"Unsupported project format {fmt!r} (expected {FORMAT_3D!r}).")

        dom = d.get("domain", {})
        summary = ProjectSummary3D(
            **{
                k: v
                for k, v in d.get("summary", {}).items()
                if k in ProjectSummary3D.__dataclass_fields__
            }
        )
        mats = [MaterialSpec3D.from_dict(m) for m in d.get("materials", [])]
        ground = HorizonSurface.from_dict(d["ground"]) if d.get("ground") else None
        layers = [
            (str(item["material"]), HorizonSurface.from_dict(item["horizon"]))
            for item in d.get("layers", [])
        ]
        regions = [ExtrudedProfileRegion.from_dict(r) for r in d.get("profile_regions", [])]
        boreholes = [BoreholeLog.from_dict(b) for b in d.get("boreholes", [])]
        wt = HorizonSurface.from_dict(d["water_table"]) if d.get("water_table") else None
        patch_loads = [PatchLoad3D.from_dict(p) for p in d.get("patch_loads", [])]
        point_loads = [PointLoad3D.from_dict(p) for p in d.get("point_loads", [])]
        support_types = [SupportType3D.from_dict(s) for s in d.get("support_types", [])]
        tc = TensionCrack3D.from_dict(d["tension_crack"]) if d.get("tension_crack") else None
        seep_bc = SeepageBC3D.from_dict(d["seepage_bc"]) if d.get("seepage_bc") else None

        an_raw = {
            k: v
            for k, v in d.get("analysis", {}).items()
            if k in AnalysisSettings3D.__dataclass_fields__
        }
        sc_raw = {
            k: v
            for k, v in d.get("surface_config", {}).items()
            if k in SurfaceConfig3D.__dataclass_fields__
        }

        proj = cls(
            summary=summary,
            materials=mats,
            x_min=float(dom.get("x_min", 0.0)),
            x_max=float(dom.get("x_max", 55.0)),
            y_min=float(dom.get("y_min", 0.0)),
            y_max=float(dom.get("y_max", 40.0)),
            z_min=float(dom.get("z_min", -10.0)),
            ground=ground,
            layers=layers,
            profile_regions=regions,
            boreholes=boreholes,
            water_table=wt,
            patch_loads=patch_loads,
            point_loads=point_loads,
            support_types=support_types,
            bolts=list(d.get("bolts", [])),
            tension_crack=tc,
            seepage_bc=seep_bc,
            analysis=AnalysisSettings3D(**an_raw),
            surface_config=SurfaceConfig3D(**sc_raw),
            design=DesignFactors3D.from_dict(d.get("design")),
            weak_layers=[WeakLayer3D.from_dict(w) for w in d.get("weak_layers", [])],
            piezo_surfaces=[
                (str(item["name"]), HorizonSurface.from_dict(item["horizon"]))
                for item in d.get("piezo_surfaces", [])
            ],
            pore_grid=[[float(v) for v in row] for row in d.get("pore_grid", [])],
            geotextiles=[GeotextileSheet3D.from_dict(g) for g in d.get("geotextiles", [])],
            pile_rows=[PileRow3D.from_dict(r) for r in d.get("pile_rows", [])],
            stages=[Stage3D.from_dict(s) for s in d.get("stages", [])],
        )
        proj.validate(require_geometry=False)
        return proj

    def save(self, path: str | Path) -> Path:
        self.validate(require_geometry=False)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2)
        fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".slp3d_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_name, target)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
        return target

    @classmethod
    def load(cls, path: str | Path) -> Project3D:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)
