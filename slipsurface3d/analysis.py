"""High-level 3D analysis runner and plain-text engineering summary."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from slipsurface3d.geometry.surfaces import SlipSurface3D
from slipsurface3d.geometry.terrain import SlopeModel3D
from slipsurface3d.lem.columns import ColumnGrid, discretize_columns
from slipsurface3d.lem.solvers import METHODS_3D, LEMResult3D, solve_surface_3d
from slipsurface3d.optimizer.search import search_critical_surface_3d
from slipsurface3d.project import Project3D


@dataclass
class AnalysisResults3D:
    model: SlopeModel3D
    critical_surface: SlipSurface3D
    columns: ColumnGrid
    method_results: dict[str, LEMResult3D]
    primary_method: str
    governing_kv: float = 0.0
    trials: list[tuple[SlipSurface3D, float]] = field(default_factory=list)
    evaluations: int = 0
    elapsed_s: float = 0.0

    @property
    def primary_result(self) -> LEMResult3D:
        if self.primary_method in self.method_results:
            return self.method_results[self.primary_method]
        return next(iter(self.method_results.values()))

    @property
    def critical_fos(self) -> float:
        return self.primary_result.fos


def _run_for_kv(
    project: Project3D,
    kv_val: float,
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> AnalysisResults3D:
    t0 = time.perf_counter()
    model = project.build_model()
    materials = project.build_materials()
    colors = project.material_colors()
    col_opts = project.build_column_options(kv_override=kv_val)
    an = project.analysis
    sc = project.surface_config
    primary = sc.primary_method if sc.primary_method in an.methods else an.methods[0]

    if sc.mode == "user":
        surf = project.user_slip_surface()
        assert surf is not None
        cols = discretize_columns(model, surf, materials, opts=col_opts, material_colors=colors)
        trials: list[tuple[SlipSurface3D, float]] = []
        evals = 1
    else:
        s_cfg = project.build_search_settings()
        s_cfg.primary_method = primary
        sr = search_critical_surface_3d(
            model=model,
            materials=materials,
            col_opts=col_opts,
            settings=s_cfg,
            solver_tolerance=an.tolerance,
            solver_max_iter=an.max_iter,
            material_colors=colors,
            progress_cb=progress_cb,
        )
        surf = sr.critical_surface
        cols = sr.critical_columns
        trials = sr.trials
        evals = sr.evaluations

    method_results: dict[str, LEMResult3D] = {}
    for meth in an.methods:
        method_results[meth] = solve_surface_3d(
            cols,
            method=meth,
            tolerance=an.tolerance,
            max_iter=an.max_iter,
            m_alpha_min=an.m_alpha_min,
        )

    elapsed = time.perf_counter() - t0
    return AnalysisResults3D(
        model=model,
        critical_surface=surf,
        columns=cols,
        method_results=method_results,
        primary_method=primary,
        governing_kv=kv_val,
        trials=trials,
        evaluations=evals,
        elapsed_s=elapsed,
    )


def run_analysis_3d(
    project: Project3D,
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> AnalysisResults3D:
    """Validate and execute 3D slope stability analysis for ``project``."""
    project.validate()
    kv = project.analysis.kv
    if project.analysis.both_kv_signs and abs(kv) > 1e-9:
        r_pos = _run_for_kv(project, +abs(kv), progress_cb=progress_cb)
        r_neg = _run_for_kv(project, -abs(kv), progress_cb=progress_cb)
        f_pos = r_pos.critical_fos if np.isfinite(r_pos.critical_fos) else np.inf
        f_neg = r_neg.critical_fos if np.isfinite(r_neg.critical_fos) else np.inf
        best = r_pos if f_pos <= f_neg else r_neg
        best.elapsed_s = r_pos.elapsed_s + r_neg.elapsed_s
        best.evaluations = r_pos.evaluations + r_neg.evaluations
        return best
    return _run_for_kv(project, kv, progress_cb=progress_cb)


def report_text_3d(project: Project3D, res: AnalysisResults3D) -> str:
    """Format a structured plain-text engineering summary of 3D analysis results."""
    cols = res.columns
    lines = [
        f"SlipSurfaceIQ-3D Engineering Summary — {project.summary.title}",
        "=" * 72,
        f"Domain X: [{project.x_min:.2f}, {project.x_max:.2f}] m | "
        f"Y: [{project.y_min:.2f}, {project.y_max:.2f}] m | z_min: {project.z_min:.2f} m",
        f"Active Columns: {cols.n} (grid {cols.nx_grid} x {cols.ny_grid}, "
        f"dx={cols.dx:.3f} m, dy={cols.dy:.3f} m)",
        f"Sliding Volume: {cols.total_volume:.2f} m^3 | Total Weight: {cols.total_weight:.1f} kN | "
        f"Max Depth: {cols.max_depth:.2f} m",
        f"Sliding Azimuth: {cols.azimuth_deg:.1f} deg | Seismic: kh={cols.kh:g}, kv={res.governing_kv:g}",
        f"Reinforcement Intersections: {len(cols.support_hits)} | "
        f"Evaluations: {res.evaluations} ({res.elapsed_s:.2f} s)",
        "-" * 72,
        f"{'Method':<26} {'FoS':>9} {'Conv':>6} {'Adm':>6} {'lambda':>9} {'m_a,min':>9}",
        "-" * 72,
    ]
    for key, r in res.method_results.items():
        label = METHODS_3D[key][0]
        fos_str = f"{r.fos:.4f}" if np.isfinite(r.fos) else "N/A"
        lam_str = f"{r.lam:.4f}" if np.isfinite(r.lam) else "-"
        ma_str = f"{r.m_alpha_min:.3f}" if np.isfinite(r.m_alpha_min) else "-"
        lines.append(
            f"{label:<26} {fos_str:>9} {str(r.converged):>6} {str(r.admissible):>6} {lam_str:>9} {ma_str:>9}"
        )
        if r.issue:
            lines.append(f"  * Admissibility note: {r.issue}")
        if r.message:
            lines.append(f"  * Solver message: {r.message}")
    lines.append("=" * 72)
    return "\n".join(lines)
