"""2D Cross-Section Extraction, F_3D / F_2D End-Effect Analyzer, and 2D .slp.json Exporter."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from slipsurface3d.analysis import AnalysisResults3D, run_analysis_3d
from slipsurface3d.geometry.surfaces import EllipsoidSurface, SlipSurface3D
from slipsurface3d.geometry.terrain import HorizonSurface, SlopeModel3D
from slipsurface3d.lem.columns import ColumnOptions, discretize_columns
from slipsurface3d.lem.solvers import solve_surface_3d
from slipsurface3d.loads.supports import LoadSet3D, PatchLoad3D
from slipsurface3d.project import Project3D


class _Extruded1DSlipSurface(SlipSurface3D):
    """Exact plane-strain extrusion (dz/dy == 0) of a 1D cross-sectional slip curve z_b(x)."""

    def __init__(
        self,
        x_pts: np.ndarray,
        z_pts: np.ndarray,
        y_bounds: tuple[float, float],
        parent_surface: SlipSurface3D,
    ) -> None:
        self.x_pts = np.asarray(x_pts, dtype=float)
        self.z_pts = np.asarray(z_pts, dtype=float)
        self.y_bounds = y_bounds
        self._parent = parent_surface
        self._dz_dx_pts = np.gradient(self.z_pts, self.x_pts)

    def z(self, x: Any, y: Any) -> np.ndarray:
        xa, _ = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        inside = (xa >= self.x_pts[0]) & (xa <= self.x_pts[-1])
        vals = np.interp(np.clip(xa, self.x_pts[0], self.x_pts[-1]), self.x_pts, self.z_pts)
        return np.where(inside, vals, np.nan)

    def gradients(self, x: Any, y: Any, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        del eps
        xa, _ = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        inside = (xa >= self.x_pts[0]) & (xa <= self.x_pts[-1])
        gx = np.interp(np.clip(xa, self.x_pts[0], self.x_pts[-1]), self.x_pts, self._dz_dx_pts)
        return np.where(inside, gx, np.nan), np.where(inside, 0.0, np.nan)

    def rotation_center_sz(self, azimuth_deg: float) -> tuple[float, float] | None:
        return self._parent.rotation_center_sz(azimuth_deg)

    def footprint_bounds(self) -> tuple[float, float, float, float]:
        return (
            float(self.x_pts[0]),
            float(self.x_pts[-1]),
            float(self.y_bounds[0]),
            float(self.y_bounds[1]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "mesh",
            "vertices": [
                [float(xv), float(yv), float(zv)]
                for yv in self.y_bounds
                for xv, zv in zip(self.x_pts, self.z_pts, strict=False)
            ],
        }


@dataclass
class CrossSection2DResult:
    """Comparison between 3D slope stability and its extracted 2D plane-strain cross-section."""

    y0: float
    x_profile: np.ndarray
    z_ground: np.ndarray
    z_slip: np.ndarray
    z_water: np.ndarray | None
    fos_3d_by_method: dict[str, float]
    fos_2d_by_method: dict[str, float]
    end_effect_ratios: dict[str, float]
    primary_method: str
    primary_fos_3d: float
    primary_fos_2d: float
    primary_ratio_3d_2d: float


def compare_2d_cross_section(
    project: Project3D,
    results: AnalysisResults3D | None = None,
    y0: float | None = None,
    n_slices_2d: int = 50,
    n_profile_pts: int = 121,
) -> CrossSection2DResult:
    """Slice ``project`` along transverse plane ``y = y0`` and compare exact 2D vs 3D FoS."""
    res3d = results if results is not None else run_analysis_3d(project)
    mdl3d = res3d.model
    surf3d = res3d.critical_surface
    an = project.analysis

    # Default y0 to weight-averaged transverse centroid of the 3D sliding mass
    y_sec = (
        float(y0)
        if y0 is not None
        else float(
            np.sum(res3d.columns.W * res3d.columns.y) / max(res3d.columns.total_weight, 1e-9)
        )
    )
    y_sec = float(np.clip(y_sec, mdl3d.y_min, mdl3d.y_max))

    x_breaks: list[np.ndarray] = [np.linspace(mdl3d.x_min, mdl3d.x_max, max(21, n_profile_pts))]
    for h_cand in [mdl3d.ground, *(h for _, h in mdl3d.layers), mdl3d.water_table]:
        if h_cand is None:
            continue
        if h_cand.profile is not None:
            x_breaks.append(h_cand.profile[:, 0])
        if h_cand.x_coords is not None:
            x_breaks.append(h_cand.x_coords)
    xs_raw = np.concatenate(x_breaks)
    xs_raw = xs_raw[(xs_raw >= mdl3d.x_min - 1e-9) & (xs_raw <= mdl3d.x_max + 1e-9)]
    xs_raw = np.clip(xs_raw, mdl3d.x_min, mdl3d.x_max)
    _, uniq_idx = np.unique(np.round(xs_raw, 8), return_index=True)
    xs_prof = np.sort(xs_raw[uniq_idx])
    ys_const = np.full_like(xs_prof, y_sec)
    zg_prof = mdl3d.ground_z(xs_prof, ys_const)
    zs_prof = surf3d.z(xs_prof, ys_const)
    zw_prof = mdl3d.water_z(xs_prof, ys_const)

    # Build 2D extruded plane-strain SlopeModel3D along y = y_sec (unit width [y_sec - 0.5, y_sec + 0.5])
    y_lo, y_hi = y_sec - 0.5, y_sec + 0.5
    ground_2d = HorizonSurface.from_extruded(
        np.column_stack([xs_prof, zg_prof]), name=f"2D Ground (y={y_sec:.2f})"
    )
    layers_2d: list[tuple[str, HorizonSurface]] = []
    for mat_name, h_surf in mdl3d.layers:
        zh = h_surf.z(xs_prof, ys_const)
        layers_2d.append(
            (
                mat_name,
                HorizonSurface.from_extruded(np.column_stack([xs_prof, zh]), name=h_surf.name),
            )
        )
    wt_2d = (
        HorizonSurface.from_extruded(np.column_stack([xs_prof, zw_prof]), name="2D Water Table")
        if zw_prof is not None
        else None
    )
    mdl_2d = SlopeModel3D(
        x_min=mdl3d.x_min,
        x_max=mdl3d.x_max,
        y_min=y_lo,
        y_max=y_hi,
        z_min=mdl3d.z_min,
        ground=ground_2d,
        layers=layers_2d,
        profile_regions=list(mdl3d.profile_regions),
        water_table=wt_2d,
        name=f"{mdl3d.name} [2D Section y={y_sec:.2f} m]",
    )

    # Build exact 2D extruded slip surface along y = y_sec
    surf_2d: SlipSurface3D
    if isinstance(surf3d, EllipsoidSurface) and abs(surf3d.azimuth_deg % 180.0) < 1e-6:
        eta = (y_sec - surf3d.yc) / max(surf3d.ry, 1e-9)
        if abs(eta) < 0.995:
            scale = float(np.sqrt(max(1.0 - eta * eta, 1e-6)))
            surf_2d = EllipsoidSurface(
                xc=surf3d.xc,
                yc=y_sec,
                zc=surf3d.zc,
                rx=surf3d.rx * scale,
                ry=1.0e9,
                rz=surf3d.rz * scale,
                azimuth_deg=surf3d.azimuth_deg,
            )
        else:
            raise ValueError(f"Section y={y_sec:.3f} m lies outside the active 3D ellipsoid trace.")
    else:
        fx0, fx1, _, _ = surf3d.footprint_bounds()
        xs_dense = np.linspace(max(mdl3d.x_min, fx0), min(mdl3d.x_max, fx1), 401)
        zs_dense = surf3d.z(xs_dense, np.full_like(xs_dense, y_sec))
        valid_s = np.isfinite(zs_dense)
        if int(np.count_nonzero(valid_s)) < 5:
            raise ValueError(
                f"3D slip surface has insufficient active trace at section y={y_sec:.3f} m."
            )
        surf_2d = _Extruded1DSlipSurface(xs_dense[valid_s], zs_dense[valid_s], (y_lo, y_hi), surf3d)

    # Filter patch loads intersecting y = y_sec
    patch_2d = [
        PatchLoad3D(
            x1=pl.x1,
            x2=pl.x2,
            y1=y_lo,
            y2=y_hi,
            q1=pl.q1,
            q2=pl.q2,
            azimuth_deg=pl.azimuth_deg,
            plunge_deg=pl.plunge_deg,
            name=pl.name,
        )
        for pl in project.patch_loads
        if pl.y1 <= y_sec <= pl.y2
    ]

    col_opts_3d = project.build_column_options(kv_override=res3d.governing_kv)
    az_2d = (
        0.0 if (res3d.columns.azimuth_deg < 90.0 or res3d.columns.azimuth_deg > 270.0) else 180.0
    )
    opts_2d = ColumnOptions(
        nx=max(12, int(n_slices_2d)),
        ny=1,
        n_sub=col_opts_3d.n_sub,
        gamma_w=col_opts_3d.gamma_w,
        pore_mode=col_opts_3d.pore_mode,
        ru=col_opts_3d.ru,
        kh=col_opts_3d.kh,
        kv=col_opts_3d.kv,
        sliding_mode="custom",
        azimuth_deg=az_2d,
        min_depth=col_opts_3d.min_depth,
        min_columns=4,
        loads=LoadSet3D(patch_loads=patch_2d, tension_crack=project.tension_crack),
        pore_pressure_fn=col_opts_3d.pore_pressure_fn,
    )

    cols_2d = discretize_columns(
        mdl_2d,
        surf_2d,
        project.build_materials(),
        opts=opts_2d,
        material_colors=project.material_colors(),
    )

    fos_3d_map: dict[str, float] = {}
    fos_2d_map: dict[str, float] = {}
    ratios: dict[str, float] = {}

    for meth, mr3 in res3d.method_results.items():
        fos_3d_map[meth] = float(mr3.fos)
        mr2 = solve_surface_3d(
            cols_2d,
            method=meth,
            tolerance=an.tolerance,
            max_iter=an.max_iter,
            m_alpha_min=an.m_alpha_min,
            interslice=an.gle_interslice,
        )
        f2 = float(mr2.fos) if mr2.ok else np.nan
        fos_2d_map[meth] = f2
        ratios[meth] = (
            float(mr3.fos / f2)
            if (np.isfinite(mr3.fos) and np.isfinite(f2) and f2 > 0.0)
            else np.nan
        )

    prim = res3d.primary_method
    f3_p = fos_3d_map.get(prim, res3d.critical_fos)
    f2_p = fos_2d_map.get(prim, np.nan)
    rat_p = ratios.get(prim, np.nan)

    return CrossSection2DResult(
        y0=y_sec,
        x_profile=xs_prof,
        z_ground=zg_prof,
        z_slip=zs_prof,
        z_water=zw_prof,
        fos_3d_by_method=fos_3d_map,
        fos_2d_by_method=fos_2d_map,
        end_effect_ratios=ratios,
        primary_method=prim,
        primary_fos_3d=f3_p,
        primary_fos_2d=f2_p,
        primary_ratio_3d_2d=rat_p,
    )


def build_2d_slp_dict(
    project: Project3D,
    y0: float | None = None,
    n_pts: int = 41,
) -> dict[str, Any]:
    """Export a 2D cross-section at ``y = y0`` into standalone 2D SlipSurfaceIQ-LE (`.slp.json`) dict."""
    mdl = project.build_model()
    y_sec = float(y0) if y0 is not None else 0.5 * (mdl.y_min + mdl.y_max)
    xs = np.linspace(mdl.x_min, mdl.x_max, max(11, n_pts))
    ys = np.full_like(xs, y_sec)
    zg = mdl.ground_z(xs, ys)

    # Closed counter-clockwise external boundary polygon
    top_chain = [[float(xv), float(zv)] for xv, zv in zip(xs, zg, strict=False)]
    external = [[float(mdl.x_min), float(mdl.z_min)], [float(mdl.x_max), float(mdl.z_min)]] + list(
        reversed(top_chain)
    )

    mat_bounds: list[list[list[float]]] = []
    assignments: list[dict[str, Any]] = []
    xm = 0.5 * (mdl.x_min + mdl.x_max)
    if mdl.layers:
        for idx_l, (mname, h_surf) in enumerate(mdl.layers):
            zh = h_surf.z(xs, ys)
            if idx_l > 0:
                mat_bounds.append([[float(xv), float(zv)] for xv, zv in zip(xs, zh, strict=False)])
            z_top_m = float(h_surf.z(xm, y_sec))
            z_bot_m = (
                float(mdl.layers[idx_l + 1][1].z(xm, y_sec))
                if idx_l + 1 < len(mdl.layers)
                else float(mdl.z_min)
            )
            if z_top_m > z_bot_m + 1e-3:
                assignments.append(
                    {"point": [float(xm), 0.5 * (z_top_m + z_bot_m)], "material": mname}
                )

    zw = mdl.water_z(xs, ys)
    wt_2d = (
        [[float(xv), float(zv)] for xv, zv in zip(xs, zw, strict=False)] if zw is not None else None
    )

    an = project.analysis
    fdir = (
        an.sliding_mode
        if an.sliding_mode in ("auto", "right_to_left", "left_to_right")
        else (
            "left_to_right"
            if (an.azimuth_deg < 90.0 or an.azimuth_deg > 270.0)
            else "right_to_left"
        )
    )
    allowed_2d = {"ordinary", "bishop", "janbu_simplified", "janbu_corrected", "spencer", "gle_mp"}
    meths_2d = [m for m in an.methods if m in allowed_2d] or ["bishop", "spencer"]

    return {
        "format": "slipsurface-le",
        "version": 2,
        "summary": {
            "title": f"{project.summary.title} — 2D Section y={y_sec:.2f} m",
            "author": project.summary.author,
            "company": project.summary.company,
            "comments": [project.summary.notes] if project.summary.notes else [],
        },
        "materials": [asdict(m) for m in project.materials],
        "geometry": {
            "external": external,
            "material_boundaries": mat_bounds,
            "assignments": assignments,
            "water_table": wt_2d,
        },
        "loads": [],
        "supports": [],
        "settings": {
            "general": {"failure_direction": fdir, "units": "metric"},
            "methods": {
                "methods": meths_2d,
                "n_slices": an.nx,
                "tolerance": an.tolerance,
                "max_iter": an.max_iter,
                "interslice_function": an.gle_interslice,
            },
            "groundwater": {
                "mode": "water_table"
                if wt_2d is not None
                else ("ru" if an.pore_mode == "ru" else "none"),
                "gamma_w": an.gamma_w,
                "ru": an.ru,
            },
            "seismic": {"kh": an.kh, "kv": an.kv},
            "advanced": {"m_alpha_min": an.m_alpha_min},
        },
    }


def export_2d_slp_json(
    path: str | Path,
    project: Project3D,
    y0: float | None = None,
) -> Path:
    """Write extracted 2D cross-section at ``y = y0`` to a `.slp.json` file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_2d_slp_dict(project, y0=y0)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
