"""3D Project model (.slp3d.json), validation, 2D .slp.json importer, and 3D benchmarks."""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from slipsurface3d.geometry.surfaces import (
    CompositeEllipsoidPlaneSurface,
    EllipsoidSurface,
    SlipSurface3D,
    cylindrical_surface,
    sphere_surface,
    surface_from_dict,
)
from slipsurface3d.geometry.terrain import (
    BoreholeLog,
    ExtrudedProfileRegion,
    HorizonSurface,
    SlopeModel3D,
    build_2d_profile_regions,
    horizons_from_boreholes,
    simple_3d_slope,
)
from slipsurface3d.lem.columns import GAMMA_W, ColumnOptions
from slipsurface3d.lem.solvers import M_ALPHA_MIN_DEFAULT, METHODS_3D
from slipsurface3d.loads.supports import (
    Bolt3D,
    LoadSet3D,
    PatchLoad3D,
    PointLoad3D,
    SupportType3D,
    TensionCrack3D,
)
from slipsurface3d.materials.models import MODEL_LABELS, Material3D, material_from_spec_dict
from slipsurface3d.optimizer.search import EFFORT_PRESETS, SearchSettings3D

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
    pore_mode: str = "water_table"  # water_table | ru | none
    gamma_w: float = GAMMA_W
    ru: float = 0.0
    kh: float = 0.0
    kv: float = 0.0
    both_kv_signs: bool = False
    min_depth: float = 0.1
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT
    check_admissibility: bool = True


@dataclass
class SurfaceConfig3D:
    mode: str = "search"  # search | user
    surface_kind: str = "ellipsoid"  # sphere | ellipsoid | spline_patch
    effort: str = "normal"  # fast | normal | fine
    primary_method: str = "bishop"
    optimize_azimuth: bool = False
    seed: int = 42
    user_surface: dict[str, Any] | None = None
    y_center_range: list[float] | None = None
    ry_bounds: list[float] | None = None


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
    analysis: AnalysisSettings3D = field(default_factory=AnalysisSettings3D)
    surface_config: SurfaceConfig3D = field(default_factory=SurfaceConfig3D)

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

    def validate(self) -> None:
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

        if self.ground is None and not self.boreholes:
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
        if a.pore_mode not in ("water_table", "ru", "none"):
            raise ValueError(f"Unknown pore_mode: {a.pore_mode!r}.")
        if a.gamma_w <= 0 or not 0.0 <= a.ru <= 1.0:
            raise ValueError("Require gamma_w > 0 and 0 <= ru <= 1.")
        if a.kh < 0 or not -1.0 < a.kv < 1.0:
            raise ValueError("Require kh >= 0 and -1 < kv < 1.")
        if a.sliding_mode not in ("auto", "right_to_left", "left_to_right", "custom"):
            raise ValueError(f"Unknown sliding_mode: {a.sliding_mode!r}.")

        sc = self.surface_config
        if sc.mode not in ("search", "user"):
            raise ValueError("surface_config.mode must be 'search' or 'user'.")
        if sc.surface_kind not in ("sphere", "ellipsoid", "spline_patch"):
            raise ValueError(f"Unknown surface_kind: {sc.surface_kind!r}.")
        if sc.effort not in EFFORT_PRESETS:
            raise ValueError(f"Unknown search effort: {sc.effort!r}.")
        if sc.primary_method not in METHODS_3D:
            raise ValueError(f"Unknown primary search method: {sc.primary_method!r}.")
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
        )

    def build_materials(self) -> list[Material3D]:
        return [m.to_material() for m in self.materials]

    def build_loads(self) -> LoadSet3D:
        st_map = {st.name: st for st in self.support_types}
        bolt_objs: list[Bolt3D] = []
        for i, b in enumerate(self.bolts, 1):
            st = st_map.get(b["type"])
            if st is None:
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
        return LoadSet3D(
            patch_loads=list(self.patch_loads),
            point_loads=list(self.point_loads),
            bolts=bolt_objs,
            tension_crack=self.tension_crack,
        )

    def build_column_options(self, kv_override: float | None = None) -> ColumnOptions:
        a = self.analysis
        pore = a.pore_mode
        if pore == "water_table" and self.water_table is None:
            pore = "none"
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
        )

    def build_search_settings(self) -> SearchSettings3D:
        sc = self.surface_config
        return SearchSettings3D(
            surface_kind=sc.surface_kind,
            effort=sc.effort,
            primary_method=sc.primary_method,
            optimize_azimuth=sc.optimize_azimuth,
            check_admissibility=self.analysis.check_admissibility,
            m_alpha_min=self.analysis.m_alpha_min,
            seed=sc.seed,
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
            "analysis": asdict(self.analysis),
            "surface_config": asdict(self.surface_config),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Project3D:
        if not isinstance(d, dict):
            raise ValueError("Project file must contain a JSON object.")
        fmt = d.get("format")
        if fmt == "slipsurface-le":
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
            analysis=AnalysisSettings3D(**an_raw),
            surface_config=SurfaceConfig3D(**sc_raw),
        )
        proj.validate()
        return proj

    def save(self, path: str | Path) -> Path:
        self.validate()
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


# ------------------------------------------------------- 2D .slp.json Importer
def import_2d_project(
    source: str | Path | dict[str, Any],
    y_min: float = 0.0,
    y_max: float = 30.0,
    extrude_surface_as_cylinder: bool = False,
) -> Project3D:
    """Import a 2D SlipSurfaceIQ-LE (.slp.json) project and extrude it along Y in [y_min, y_max]."""
    if y_max <= y_min:
        raise ValueError("Extrusion span requires y_max > y_min.")
    if isinstance(source, (str, Path)):
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        data = copy.deepcopy(source)

    if data.get("format") == FORMAT_3D:
        return Project3D.from_dict(data)
    if data.get("format") not in ("slipsurface-le", None):
        raise ValueError(f"Unrecognized 2D project format: {data.get('format')!r}.")

    sum2 = data.get("summary", {})
    comments = [c for c in sum2.get("comments", []) if isinstance(c, str) and c.strip()]
    summary = ProjectSummary3D(
        title=f"{sum2.get('title', 'Imported 2D Project')} (3D Extruded)",
        author=str(sum2.get("author", "")),
        company=str(sum2.get("company", "")),
        notes="\n".join(comments),
    )

    mats = [MaterialSpec3D.from_dict(m) for m in data.get("materials", [])]
    if not mats:
        mats = [MaterialSpec3D(name="Soil 1")]

    geom2 = data.get("geometry", {})
    ext = geom2.get("external")
    if not ext:
        raise ValueError("2D project has no external boundary in geometry.external.")

    assign_tuples = [
        (float(a["point"][0]), float(a["point"][1]), str(a["material"]))
        for a in geom2.get("assignments", [])
    ]
    ground_prof, regions, z_min = build_2d_profile_regions(
        external=ext,
        material_boundaries=geom2.get("material_boundaries", []),
        assignments=assign_tuples,
        default_material=mats[0].name,
    )
    x_min = float(ground_prof[0, 0])
    x_max = float(ground_prof[-1, 0])
    ground = HorizonSurface.from_extruded(ground_prof, name="Extruded Ground")

    wt_raw = geom2.get("water_table")
    water = HorizonSurface.from_extruded(wt_raw, name="Extruded Water Table") if wt_raw else None

    # Settings mapping
    st2 = data.get("settings", {})
    gen2 = st2.get("general", {})
    meth2 = st2.get("methods", {})
    gw2 = st2.get("groundwater", {})
    seis2 = st2.get("seismic", {})
    adv2 = st2.get("advanced", {})

    fdir = str(gen2.get("failure_direction", "right_to_left"))
    sliding_mode = fdir if fdir in ("auto", "right_to_left", "left_to_right") else "auto"
    az_deg = 0.0 if sliding_mode == "left_to_right" else 180.0

    methods_3d = [m for m in meth2.get("methods", []) if m in METHODS_3D]
    if not methods_3d:
        methods_3d = ["bishop", "janbu_simplified", "spencer"]

    nx_3d = int(np.clip(meth2.get("n_slices", 30), 10, 80))
    ny_3d = max(12, int(round(nx_3d * min(1.0, (y_max - y_min) / max(1.0, x_max - x_min)))))

    # Loads mapping
    patch_loads: list[PatchLoad3D] = []
    point_loads: list[PointLoad3D] = []
    for idx_l, ld in enumerate(data.get("loads", []), 1):
        kind = ld.get("kind")
        if kind == "distributed":
            (lx1, lz1), (lx2, lz2) = ld["points"]
            q1, q2 = float(ld["q"][0]), float(ld["q"][1])
            orient = ld.get("orientation", "vertical")
            plunge = 90.0
            az_l = 180.0
            if orient == "normal" and abs(lx2 - lx1) > 1e-6:
                slope_dz_dx = (lz2 - lz1) / (lx2 - lx1)
                # Inward normal to segment (dx, dz) has components (+dz, -dx)
                nx_in = slope_dz_dx
                nz_in = -1.0
                norm = float(np.hypot(nx_in, nz_in))
                az_l = 0.0 if nx_in >= 0 else 180.0
                plunge = float(np.degrees(np.arcsin(abs(nz_in) / norm)))
            patch_loads.append(
                PatchLoad3D(
                    x1=float(lx1),
                    x2=float(lx2),
                    y1=y_min,
                    y2=y_max,
                    q1=q1,
                    q2=q2,
                    azimuth_deg=az_l,
                    plunge_deg=plunge,
                    name=f"2D Distributed Load {idx_l}",
                )
            )
        elif kind == "line":
            px, pz = float(ld["point"][0]), float(ld["point"][1])
            mag_per_m = float(ld["magnitude"])  # kN/m along Y
            ang2d = float(ld.get("angle", 270.0))  # 270° = straight down in 2D
            fx_m = float(np.cos(np.radians(ang2d)))
            fz_m = float(np.sin(np.radians(ang2d)))
            az_pt = 0.0 if fx_m >= 0 else 180.0
            plunge_pt = float(np.degrees(np.arctan2(-fz_m, abs(fx_m))))
            n_pts_y = max(4, int(round((y_max - y_min) / 2.5)))
            dy_pt = (y_max - y_min) / n_pts_y
            for j_y in range(n_pts_y):
                yj = y_min + (j_y + 0.5) * dy_pt
                point_loads.append(
                    PointLoad3D(
                        x=px,
                        y=yj,
                        z=pz,
                        magnitude_kn=mag_per_m * dy_pt,
                        azimuth_deg=az_pt,
                        plunge_deg=plunge_pt,
                        name=f"Line Load {idx_l}.{j_y + 1}",
                    )
                )

    # Supports mapping (expand 2D bolts along Y at out-of-plane spacing)
    support_types: list[SupportType3D] = []
    spacing_map: dict[str, float] = {}
    for st in data.get("support_types", []):
        st_name = str(st.get("name", "Support 1"))
        sp_y = max(0.5, float(st.get("spacing", 2.0)))
        spacing_map[st_name] = sp_y
        kind2 = str(st.get("kind", "grouted_tieback"))
        kind3 = (
            kind2
            if kind2 in ("end_anchored", "grouted_tieback", "soil_nail", "micropile", "geotextile")
            else "grouted_tieback"
        )
        support_types.append(
            SupportType3D(
                name=st_name,
                kind=kind3,
                tensile_kn=float(st.get("tensile", 200.0)),
                plate_kn=float(st.get("plate", 150.0)),
                bond_strength_kn_m=float(st.get("bond_strength", st.get("pullout", 60.0))),
                bond_length_m=float(st.get("bond_length", 5.0)),
                pile_shear_kn=float(st.get("pile_shear", 100.0)),
                application=str(st.get("application", "active")),
                color=str(st.get("color", "#d9534f")),
            )
        )

    bolts_3d: list[dict[str, Any]] = []
    for idx_s, sp in enumerate(data.get("supports", []), 1):
        st_name = str(sp["type"])
        sp_y = spacing_map.get(st_name, 2.0)
        n_rows = max(1, int(np.floor((y_max - y_min) / sp_y)))
        ys_b = (
            np.linspace(y_min + 0.5 * sp_y, y_max - 0.5 * sp_y, n_rows)
            if n_rows > 1
            else np.array([0.5 * (y_min + y_max)])
        )
        ax, az_head = float(sp["a"][0]), float(sp["a"][1])
        bx, bz_tip = float(sp["b"][0]), float(sp["b"][1])
        for r_i, yb_i in enumerate(ys_b, 1):
            bolts_3d.append(
                {
                    "head": [ax, float(yb_i), az_head],
                    "tip": [bx, float(yb_i), bz_tip],
                    "type": st_name,
                    "label": f"Bolt {idx_s}.{r_i}",
                }
            )

    # Tension crack mapping
    tc_line = geom2.get("tension_crack")
    tc_3d: TensionCrack3D | None = None
    if tc_line and len(tc_line) >= 2:
        tc_arr = np.asarray(tc_line, dtype=float)
        gz_tc = np.interp(tc_arr[:, 0], ground_prof[:, 0], ground_prof[:, 1])
        depth_est = float(np.max(np.maximum(gz_tc - tc_arr[:, 1], 0.0)))
        if depth_est > 0:
            tc_3d = TensionCrack3D(
                depth_m=depth_est,
                water_fill_fraction=float(np.clip(gw2.get("crack_water", 0.0), 0.0, 1.0)),
            )

    # Surface settings mapping
    surf2 = data.get("surfaces", {})
    sc_mode = "search"
    user_surf_dict: dict[str, Any] | None = None
    ucirc = surf2.get("user_circle")
    if ucirc and len(ucirc) == 3:
        sc_mode = "user"
        cx, cz, cr = float(ucirc[0]), float(ucirc[1]), float(ucirc[2])
        if extrude_surface_as_cylinder:
            user_surf_dict = cylindrical_surface(
                cx, cz, cr, y_center=0.5 * (y_min + y_max), azimuth_deg=az_deg
            ).to_dict()
        else:
            user_surf_dict = sphere_surface(cx, 0.5 * (y_min + y_max), cz, cr).to_dict()

    proj = Project3D(
        summary=summary,
        materials=mats,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        ground=ground,
        layers=[],
        profile_regions=regions,
        water_table=water,
        patch_loads=patch_loads,
        point_loads=point_loads,
        support_types=support_types,
        bolts=bolts_3d,
        tension_crack=tc_3d,
        analysis=AnalysisSettings3D(
            methods=methods_3d,
            nx=nx_3d,
            ny=ny_3d,
            tolerance=float(meth2.get("tolerance", 1e-4)),
            max_iter=int(meth2.get("max_iter", 75)),
            sliding_mode=sliding_mode,
            azimuth_deg=az_deg,
            pore_mode=str(gw2.get("method", "water_table")),
            gamma_w=float(gw2.get("gamma_w", GAMMA_W)),
            ru=float(gw2.get("ru", 0.0)),
            kh=float(seis2.get("kh", 0.0)),
            kv=float(seis2.get("kv", 0.0)),
            both_kv_signs=bool(seis2.get("both_vertical", False)),
            min_depth=float(adv2.get("min_depth", 0.05)),
            m_alpha_min=float(adv2.get("m_alpha_min", M_ALPHA_MIN_DEFAULT)),
            check_admissibility=bool(adv2.get("check_admissibility", True)),
        ),
        surface_config=SurfaceConfig3D(
            mode=sc_mode,
            surface_kind="ellipsoid",
            effort=str(surf2.get("effort", "normal"))
            if surf2.get("effort") in EFFORT_PRESETS
            else "normal",
            primary_method=methods_3d[0],
            seed=int(surf2.get("seed", 42)),
            user_surface=user_surf_dict,
        ),
    )
    proj.validate()
    return proj


# ----------------------------------------- Built-in 3D Benchmarks & Examples
def hungr_1987_hemisphere_project() -> Project3D:
    """Analytical phi=0 3D spherical cap on an inclined planar 2H:1V slope (Baligh & Azzouz 1975; Hungr 1987).

    Sphere radius R = 15 m at perpendicular offset d = 11.5 m above a uniform 2H:1V (beta = arctan(0.5))
    planar slope with su = 30 kPa, gamma = 20 kN/m^3.
    Exact closed-form 3D rotational integral FoS = 2.31611.
    """
    slope_ratio = 0.5
    beta = float(np.arctan(slope_ratio))
    x_min, x_max = 0.0, 50.0
    y_min, y_max = 0.0, 40.0
    radius = 15.0
    d_perp = 11.5
    xc = 25.0 - d_perp * float(np.sin(beta))
    yc = 20.0
    zc = 12.5 + d_perp * float(np.cos(beta))

    prof = np.array([[x_min, slope_ratio * x_min], [x_max, slope_ratio * x_max]], dtype=float)
    ground = HorizonSurface.from_extruded(prof, name="Planar 2H:1V Slope")
    mat = MaterialSpec3D(
        name="Undrained Clay",
        model="UD",
        gamma=20.0,
        gamma_sat=20.0,
        su=30.0,
        use_water=False,
        color="#57a8bd",
    )
    surf = sphere_surface(xc=xc, yc=yc, zc=zc, radius=radius)
    return Project3D(
        summary=ProjectSummary3D(
            title="Hungr (1987) / Baligh & Azzouz (1975) Analytical Spherical Cap (phi=0)",
            notes="Exact closed-form 3D rotational FoS = 2.3161 (R=15 m, d=11.5 m, su=30 kPa, gamma=20 kN/m3).",
        ),
        materials=[mat],
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=-5.0,
        ground=ground,
        layers=[(mat.name, ground)],
        analysis=AnalysisSettings3D(
            methods=["ordinary", "bishop", "janbu_simplified", "spencer"],
            nx=64,
            ny=64,
            sliding_mode="right_to_left",
            pore_mode="none",
        ),
        surface_config=SurfaceConfig3D(
            mode="user",
            surface_kind="sphere",
            primary_method="bishop",
            user_surface=surf.to_dict(),
        ),
    )


def zhang_1988_homog_ellipsoid_project() -> Project3D:
    """Zhang (1988) / Hungr et al. (1989) Example 1: Homogeneous 12.19 m (40 ft) 2H:1V slope.

    Fredlund & Krahn (1977) / Zhang (1988) geometry: H = 40 ft (12.192 m), 2H:1V,
    c = 600 psf (28.73 kPa), phi = 20 deg, gamma = 120 pcf (18.84 kN/m^3),
    xc = toe + 30 ft (9.144 m), zc = toe + 70 ft (21.336 m), rx = rz = 80 ft (24.384 m),
    ry = 200 ft (60.96 m). Published 3D Bishop FoS = 2.122 (Zhang 1988; Hungr et al. 1989).
    """
    ft = 0.3048
    h = 40.0 * ft  # 12.192 m
    toe_x = 15.0
    model = simple_3d_slope(
        height=h,
        ratio_h_to_v=2.0,
        toe_x=toe_x,
        left_bench=15.0,
        right_bench=20.0,
        depth_below_toe=12.0,
        width_y=140.0,
        material="Homogeneous Clayey Silt",
    )
    mat = MaterialSpec3D(
        name="Homogeneous Clayey Silt",
        model="MC",
        gamma=18.84,
        gamma_sat=18.84,
        c=28.73,
        phi=20.0,
        use_water=False,
        color="#e0b438",
    )
    ell = EllipsoidSurface(
        xc=toe_x + 30.0 * ft,
        yc=70.0,
        zc=70.0 * ft,
        rx=80.0 * ft,
        ry=200.0 * ft,
        rz=80.0 * ft,
        azimuth_deg=180.0,
    )
    return Project3D(
        summary=ProjectSummary3D(
            title="Zhang (1988) / Hungr et al. (1989) Ex. 1 — Homogeneous 3D Ellipsoid",
            notes="12.19 m (40 ft) 2H:1V slope, c=28.73 kPa, phi=20 deg, gamma=18.84 kN/m3 (published 3D FoS = 2.122).",
        ),
        materials=[mat],
        x_min=model.x_min,
        x_max=model.x_max,
        y_min=model.y_min,
        y_max=model.y_max,
        z_min=model.z_min,
        ground=model.ground,
        layers=model.layers,
        analysis=AnalysisSettings3D(
            methods=["bishop", "janbu_simplified", "janbu_corrected", "spencer"],
            nx=36,
            ny=36,
            sliding_mode="right_to_left",
            pore_mode="none",
        ),
        surface_config=SurfaceConfig3D(
            mode="user",
            surface_kind="ellipsoid",
            primary_method="bishop",
            user_surface=ell.to_dict(),
        ),
    )


def zhang_1988_weak_seam_project() -> Project3D:
    """Zhang (1988) / Hungr et al. (1989) Example 2: 3D Ellipsoid truncated by a weak horizontal seam.

    Same 12.19 m 2H:1V slope and ellipsoid as Ex. 1, truncated by a weak horizontal seam
    (c = 0 kPa, phi = 10 deg) at z = 0.0 m (published 3D Bishop FoS = 1.553, Zhang 1988).
    """
    ft = 0.3048
    base_proj = zhang_1988_homog_ellipsoid_project()
    weak_mat = MaterialSpec3D(
        name="Weak Clay Seam",
        model="MC",
        gamma=18.84,
        gamma_sat=18.84,
        c=0.0,
        phi=10.0,
        use_water=False,
        color="#d9534f",
    )
    base_proj.summary.title = "Zhang (1988) / Hungr et al. (1989) Ex. 2 — Ellipsoid + Weak Seam"
    base_proj.summary.notes = "12.19 m 2H:1V slope underlain by horizontal weak seam (c=0, phi=10 deg) at z=0.0 m (published 3D FoS = 1.553)."
    base_proj.materials.append(weak_mat)
    seam_top = HorizonSurface.from_extruded(
        [[base_proj.x_min, 0.15], [base_proj.x_max, 0.15]], name="Weak Seam Top"
    )
    assert base_proj.ground is not None
    base_proj.layers = [
        (base_proj.materials[0].name, base_proj.ground),
        (weak_mat.name, seam_top),
    ]
    ell = EllipsoidSurface(
        xc=15.0 + 30.0 * ft,
        yc=70.0,
        zc=70.0 * ft,
        rx=80.0 * ft,
        ry=200.0 * ft,
        rz=80.0 * ft,
        azimuth_deg=180.0,
    )
    comp = CompositeEllipsoidPlaneSurface(
        ellipsoid=ell,
        plane_x0=24.144,
        plane_y0=70.0,
        plane_z0=0.0,
        dip_deg=0.0,
        dip_dir_deg=180.0,
    )
    base_proj.surface_config.user_surface = comp.to_dict()
    return base_proj


def extruded_acads_1a_project() -> Project3D:
    """ACADS 1(a) homogeneous 2H:1V slope (H=10 m, c=3 kPa, phi=19.6 deg, gamma=20 kN/m3) in 3D."""
    model = simple_3d_slope(
        height=10.0,
        ratio_h_to_v=2.0,
        toe_x=15.0,
        left_bench=15.0,
        right_bench=20.0,
        depth_below_toe=10.0,
        width_y=40.0,
        material="ACADS Soil",
    )
    mat = MaterialSpec3D(
        name="ACADS Soil",
        model="MC",
        gamma=20.0,
        gamma_sat=20.0,
        c=3.0,
        phi=19.6,
        use_water=False,
        color="#4ea86f",
    )
    return Project3D(
        summary=ProjectSummary3D(
            title="ACADS 1(a) 3D Slope (Search Demo)",
            notes="Classic ACADS 1(a) slope (2D FoS = 1.00; 3D finite-width ellipsoid FoS ~ 1.06-1.12).",
        ),
        materials=[mat],
        x_min=model.x_min,
        x_max=model.x_max,
        y_min=model.y_min,
        y_max=model.y_max,
        z_min=model.z_min,
        ground=model.ground,
        layers=model.layers,
        analysis=AnalysisSettings3D(
            methods=["bishop", "janbu_simplified", "spencer"],
            nx=28,
            ny=24,
            sliding_mode="right_to_left",
            pore_mode="none",
        ),
        surface_config=SurfaceConfig3D(
            mode="search",
            surface_kind="ellipsoid",
            effort="normal",
            primary_method="bishop",
        ),
    )


def reinforced_3d_valley_project() -> Project3D:
    """3D lofted concave cut slope with two layers, water table, crest surcharge, and tieback bolts."""
    y_secs = [0.0, 20.0, 40.0]
    # Central section (y=20) is steeper (toe shifted left) creating a 3D amphitheatre / valley cut
    p_edge = [[0.0, 0.0], [15.0, 0.0], [33.0, 12.0], [52.0, 12.0]]
    p_mid = [[0.0, 0.0], [12.0, 0.0], [28.0, 12.0], [52.0, 12.0]]
    ground = HorizonSurface.from_lofted(y_secs, [p_edge, p_mid, p_edge], name="3D Cut Ground")
    weather_top = HorizonSurface.from_extruded(
        [[0.0, -1.5], [20.0, 1.0], [52.0, 4.5]], name="Dense Till Top"
    )
    wt = HorizonSurface.from_extruded(
        [[0.0, 0.0], [14.0, 0.2], [32.0, 6.5], [52.0, 8.0]], name="Water Table"
    )

    m1 = MaterialSpec3D(
        name="Upper Colluvium",
        model="MC",
        gamma=18.5,
        gamma_sat=19.8,
        c=8.0,
        phi=26.0,
        color="#e0b438",
    )
    m2 = MaterialSpec3D(
        name="Dense Glacial Till",
        model="MC",
        gamma=20.5,
        gamma_sat=21.5,
        c=18.0,
        phi=34.0,
        color="#6c82d6",
    )
    st = SupportType3D(
        name="Grouted Tieback 250 kN",
        kind="grouted_tieback",
        tensile_kn=250.0,
        plate_kn=220.0,
        bond_strength_kn_m=90.0,
        bond_length_m=6.0,
        application="active",
        color="#d9534f",
    )
    bolts = []
    for r_idx, yb in enumerate([12.0, 16.0, 20.0, 24.0, 28.0], 1):
        xg = 20.0
        zg = float(ground.z(xg, yb))
        ang = np.radians(15.0)
        length = 14.0
        bolts.append(
            {
                "head": [xg, yb, zg],
                "tip": [xg + length * float(np.cos(ang)), yb, zg - length * float(np.sin(ang))],
                "type": st.name,
                "label": f"Tieback {r_idx}",
            }
        )

    surcharge = PatchLoad3D(
        x1=31.0,
        x2=42.0,
        y1=10.0,
        y2=30.0,
        q1=25.0,
        q2=25.0,
        name="Highway Crest Surcharge (25 kPa)",
    )
    return Project3D(
        summary=ProjectSummary3D(
            title="3D Lofted Highway Cut with Tiebacks & Groundwater",
            notes="Concave 3D cut slope with two soil horizons, water table, 25 kPa patch surcharge, and 5 grouted tiebacks.",
        ),
        materials=[m1, m2],
        x_min=0.0,
        x_max=52.0,
        y_min=0.0,
        y_max=40.0,
        z_min=-10.0,
        ground=ground,
        layers=[(m1.name, ground), (m2.name, weather_top)],
        water_table=wt,
        patch_loads=[surcharge],
        support_types=[st],
        bolts=bolts,
        analysis=AnalysisSettings3D(
            methods=["bishop", "janbu_simplified", "spencer"],
            nx=26,
            ny=22,
            sliding_mode="right_to_left",
            pore_mode="water_table",
        ),
        surface_config=SurfaceConfig3D(
            mode="search",
            surface_kind="ellipsoid",
            effort="fast",
            primary_method="bishop",
        ),
    )


EXAMPLE_PROJECTS_3D: dict[str, tuple[str, Any]] = {
    "acads_1a_3d": ("ACADS 1(a) Homogeneous 3D Slope (Search)", extruded_acads_1a_project),
    "hungr_1987_hemisphere": (
        "Hungr (1987) Analytical Hemisphere (phi=0)",
        hungr_1987_hemisphere_project,
    ),
    "zhang_1988_homog": (
        "Zhang (1988) Ex. 1 — Homogeneous 3D Ellipsoid",
        zhang_1988_homog_ellipsoid_project,
    ),
    "zhang_1988_weak_seam": (
        "Zhang (1988) Ex. 2 — 3D Ellipsoid + Weak Seam",
        zhang_1988_weak_seam_project,
    ),
    "reinforced_valley_3d": (
        "3D Lofted Cut Slope with Tiebacks & Water",
        reinforced_3d_valley_project,
    ),
}
