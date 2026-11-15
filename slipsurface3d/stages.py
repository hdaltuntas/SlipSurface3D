"""Construction stages: box excavations and fills applied cumulatively to the ground surface.

Each stage reshapes the ground inside a plan box and batters back to the existing ground
outside it. ``stage_project(project, n)`` returns the project as it stands after the first
``n`` stages, and ``run_staged_analysis_3d`` analyses the initial state and every stage.

Only the ground surface (and, for fills, the stratigraphy above the old ground) changes
between stages. Water table, loads, supports and weak layers are the same in every stage,
and each stage is an independent limit-equilibrium analysis: no stress history is carried.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from slipsurface3d.analysis import AnalysisResults3D, run_analysis_3d
from slipsurface3d.geometry.terrain import HorizonSurface
from slipsurface3d.project import Project3D

STAGE_GRID_MAX = 241


def _stage_grid(project: Project3D) -> tuple[np.ndarray, np.ndarray]:
    """Regular plan grid fine enough to carry stage edges (about 0.5 m, capped)."""
    nx = int(np.clip(round((project.x_max - project.x_min) / 0.5) + 1, 61, STAGE_GRID_MAX))
    ny = int(np.clip(round((project.y_max - project.y_min) / 0.5) + 1, 61, STAGE_GRID_MAX))
    return (
        np.linspace(project.x_min, project.x_max, nx),
        np.linspace(project.y_min, project.y_max, ny),
    )


def stage_project(project: Project3D, n_stages: int) -> Project3D:
    """Return a copy of ``project`` with its first ``n_stages`` stages built."""
    if not 0 <= n_stages <= len(project.stages):
        raise ValueError(f"Stage count must be between 0 and {len(project.stages)}.")
    staged = project.copy()
    staged.stages = []
    if n_stages == 0:
        return staged
    if staged.ground is None:
        staged.build_model()  # generates the ground from boreholes
    assert staged.ground is not None

    xs, ys = _stage_grid(project)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    ground = np.asarray(staged.ground.z(xx, yy), dtype=float)
    layers = list(staged.layers)

    for stage in project.stages[:n_stages]:
        target = stage.target_z(xx, yy)
        if stage.kind == "fill":
            if staged.profile_regions:
                raise ValueError(
                    f"Stage '{stage.name}': fills need a layered model, not extruded 2D regions."
                )
            if not layers:
                raise ValueError(f"Stage '{stage.name}': the model has no stratigraphic layers.")
            new_ground = np.maximum(ground, target)
            old_top = HorizonSurface.from_grid(xs, ys, ground, name=f"Ground before {stage.name}")
            new_top = HorizonSurface.from_grid(xs, ys, new_ground, name=f"{stage.name} surface")
            fill_material = stage.fill_material or layers[0][0]
            # Layer 0 is the default below ground; later layers apply at or below their top.
            layers = [(fill_material, new_top), (layers[0][0], old_top), *layers[1:]]
            ground = new_ground
        else:
            ground = np.minimum(ground, target)

    final = HorizonSurface.from_grid(xs, ys, ground, name=f"Ground after stage {n_stages}")
    staged.ground = final
    if layers:
        layers[0] = (layers[0][0], final)
    staged.layers = layers
    staged.summary.title = f"{project.summary.title} — {project.stages[n_stages - 1].name}"
    return staged


@dataclass
class StageResult3D:
    index: int  # 0 = initial ground, k = after stage k
    name: str
    project: Project3D
    results: AnalysisResults3D | None
    error: str = ""

    @property
    def fos(self) -> float:
        return self.results.critical_fos if self.results is not None else float("nan")


def run_staged_analysis_3d(
    project: Project3D,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    **run_kwargs: Any,
) -> list[StageResult3D]:
    """Analyse the initial ground and the ground after each construction stage.

    A stage whose analysis fails (for example no admissible surface) is reported with
    ``results=None`` and the error text instead of aborting the remaining stages.
    """
    project.validate()
    names = ["Initial ground"] + [s.name for s in project.stages]
    out: list[StageResult3D] = []
    for k, name in enumerate(names):
        if cancel_check is not None and cancel_check():
            break
        if progress_cb is not None:
            progress_cb(k, len(names), name)
        staged = stage_project(project, k)
        try:
            res = run_analysis_3d(staged, cancel_check=cancel_check, **run_kwargs)
            out.append(StageResult3D(index=k, name=name, project=staged, results=res))
        except ValueError as exc:  # includes ColumnDiscretizationError
            out.append(
                StageResult3D(index=k, name=name, project=staged, results=None, error=str(exc))
            )
    return out
