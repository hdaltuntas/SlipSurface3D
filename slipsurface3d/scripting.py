"""Ergonomic Python Scripting & Batch Automation API for SlipSurfaceIQ-3D."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from slipsurface3d.analysis import AnalysisResults3D, report_text_3d, run_analysis_3d
from slipsurface3d.cross_section import (
    CrossSection2DResult,
    compare_2d_cross_section,
    export_2d_slp_json,
)
from slipsurface3d.design import DesignFactors3D, design_preset
from slipsurface3d.geometry.terrain import WeakLayer3D, simple_3d_slope
from slipsurface3d.loads.supports import GeotextileSheet3D, PileRow3D
from slipsurface3d.probabilistic import (
    ProbabilisticResult3D,
    RandomVarSpec3D,
    SensitivityCurve3D,
    SensitivityParamSpec3D,
    run_probabilistic_3d,
    run_sensitivity_3d,
)
from slipsurface3d.project import (
    EXAMPLE_PROJECTS_3D,
    AnalysisSettings3D,
    MaterialSpec3D,
    Project3D,
    ProjectSummary3D,
    Stage3D,
    SurfaceConfig3D,
)
from slipsurface3d.seismic import (
    NewmarkResult3D,
    YieldAccelerationResult3D,
    estimate_newmark_displacement_3d,
    find_yield_acceleration_3d,
)
from slipsurface3d.stages import StageResult3D, run_staged_analysis_3d, stage_project


def create_simple_slope_project(
    *,
    title: str = "Scripted 3D Slope",
    height: float = 12.0,
    ratio_h_to_v: float = 2.0,
    width_y: float = 40.0,
    c: float = 15.0,
    phi: float = 26.0,
    gamma: float = 19.0,
    methods: Sequence[str] = ("bishop", "janbu_simplified", "spencer", "gle_mp"),
    effort: str = "normal",
    sao_enabled: bool = False,
) -> Project3D:
    """Create a ready-to-solve homogeneous 3D slope project in one call."""
    model = simple_3d_slope(
        height=height,
        ratio_h_to_v=ratio_h_to_v,
        width_y=width_y,
        material="Slope Soil",
    )
    mat = MaterialSpec3D(
        name="Slope Soil",
        model="MC",
        gamma=gamma,
        gamma_sat=gamma + 1.0,
        c=c,
        phi=phi,
        use_water=False,
    )
    return Project3D(
        summary=ProjectSummary3D(title=title),
        materials=[mat],
        x_min=model.x_min,
        x_max=model.x_max,
        y_min=model.y_min,
        y_max=model.y_max,
        z_min=model.z_min,
        ground=model.ground,
        layers=model.layers,
        analysis=AnalysisSettings3D(methods=list(methods)),
        surface_config=SurfaceConfig3D(
            mode="search",
            surface_kind="ellipsoid",
            effort=effort,
            primary_method=methods[0] if methods else "bishop",
            sao_enabled=sao_enabled,
        ),
    )


def load_project(path: str | Path) -> Project3D:
    """Load either a 3D `.slp3d.json` project or a 2D `.slp.json` project (auto-extruded)."""
    return Project3D.load(path)


def save_project(project: Project3D, path: str | Path) -> Path:
    """Validate and save ``project`` to ``path``."""
    return project.save(path)


def load_example(key: str) -> Project3D:
    """Instantiate one of the built-in 3D verification benchmarks by key."""
    if key not in EXAMPLE_PROJECTS_3D:
        raise KeyError(f"Unknown example {key!r} (available: {list(EXAMPLE_PROJECTS_3D)}).")
    _, factory = EXAMPLE_PROJECTS_3D[key]
    proj: Project3D = factory()
    return proj


def solve(
    project: Project3D,
    *,
    methods: Sequence[str] | None = None,
    effort: str | None = None,
    algorithm: str | None = None,
    sao: bool | None = None,
    n_local_minima: int | None = None,
    design: str | DesignFactors3D | None = None,
    n_workers: int | None = None,
) -> AnalysisResults3D:
    """Run 3D limit-equilibrium analysis with optional non-destructive overrides."""
    proj = project.copy()
    if methods is not None:
        proj.analysis.methods = list(methods)
        if proj.surface_config.primary_method not in proj.analysis.methods:
            proj.surface_config.primary_method = proj.analysis.methods[0]
    if effort is not None:
        proj.surface_config.effort = effort
    if algorithm is not None:
        proj.surface_config.algorithm = algorithm
    if sao is not None:
        proj.surface_config.sao_enabled = sao
    if n_local_minima is not None:
        proj.surface_config.n_local_minima = int(n_local_minima)
    if design is not None:
        proj.design = design_preset(design) if isinstance(design, str) else design
    if n_workers is not None:
        proj.analysis.n_workers = n_workers
    return run_analysis_3d(proj)


def run_probabilistic(
    project: Project3D,
    var_specs: list[RandomVarSpec3D],
    *,
    n_samples: int = 200,
    sampling: str = "lhs",
    fixed_surface: bool = True,
    seed: int = 42,
) -> ProbabilisticResult3D:
    """Run 3D Latin Hypercube or Monte Carlo reliability analysis."""
    return run_probabilistic_3d(
        project,
        var_specs,
        n_samples=n_samples,
        sampling=sampling,
        fixed_surface=fixed_surface,
        seed=seed,
    )


def run_sensitivity(
    project: Project3D,
    specs: list[SensitivityParamSpec3D] | None = None,
    *,
    n_points: int = 9,
) -> list[SensitivityCurve3D]:
    """Run 3D one-at-a-time tornado sensitivity analysis."""
    return run_sensitivity_3d(project, specs=specs, n_points=n_points)


def find_ky(
    project: Project3D,
    *,
    method: str | None = None,
    pga_g: float | None = None,
    mw: float = 6.5,
) -> tuple[YieldAccelerationResult3D, NewmarkResult3D | None]:
    """Compute critical seismic yield acceleration ``k_y`` and optional Newmark displacement."""
    ky_res = find_yield_acceleration_3d(project, method=method)
    if pga_g is None or not ky_res.converged:
        return ky_res, None
    return ky_res, estimate_newmark_displacement_3d(ky_res.ky_g, pga_g=pga_g, mw=mw)


def compare_2d_section(
    project: Project3D,
    results: AnalysisResults3D | None = None,
    *,
    y0: float | None = None,
) -> CrossSection2DResult:
    """Extract a 2D cross-section at ``y = y0`` and compute 3D/2D FoS end-effect ratios."""
    return compare_2d_cross_section(project, results=results, y0=y0)


def run_stages(project: Project3D) -> list[StageResult3D]:
    """Analyse the initial ground and every construction stage defined on ``project``."""
    return run_staged_analysis_3d(project)


def batch_solve(
    items: Sequence[Project3D | str | Path],
    *,
    n_workers: int = 1,
) -> list[tuple[str, AnalysisResults3D]]:
    """Solve a sequence of projects or file paths (optionally in parallel across worker threads)."""

    def _run_one(item: Project3D | str | Path) -> tuple[str, AnalysisResults3D]:
        proj = item if isinstance(item, Project3D) else Project3D.load(item)
        return proj.name, run_analysis_3d(proj)

    if n_workers <= 1 or len(items) <= 1:
        return [_run_one(it) for it in items]
    with ThreadPoolExecutor(max_workers=int(n_workers)) as pool:
        return list(pool.map(_run_one, items))


__all__ = [
    "AnalysisResults3D",
    "CrossSection2DResult",
    "DesignFactors3D",
    "GeotextileSheet3D",
    "NewmarkResult3D",
    "PileRow3D",
    "ProbabilisticResult3D",
    "Project3D",
    "RandomVarSpec3D",
    "SensitivityCurve3D",
    "SensitivityParamSpec3D",
    "Stage3D",
    "StageResult3D",
    "WeakLayer3D",
    "YieldAccelerationResult3D",
    "batch_solve",
    "compare_2d_section",
    "create_simple_slope_project",
    "design_preset",
    "export_2d_slp_json",
    "find_ky",
    "load_example",
    "load_project",
    "report_text_3d",
    "run_probabilistic",
    "run_sensitivity",
    "run_stages",
    "save_project",
    "solve",
    "stage_project",
]
