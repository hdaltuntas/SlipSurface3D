"""Import a 2D SlipSurfaceIQ-LE (.slp.json) project and extrude it along Y."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from slipsurface3d.geometry.surfaces import cylindrical_surface, sphere_surface
from slipsurface3d.geometry.terrain import (
    HorizonSurface,
    build_2d_profile_regions,
)
from slipsurface3d.lem.columns import GAMMA_W
from slipsurface3d.lem.solvers import M_ALPHA_MIN_DEFAULT, METHODS_3D
from slipsurface3d.loads.supports import (
    PatchLoad3D,
    PointLoad3D,
    SupportType3D,
    TensionCrack3D,
)
from slipsurface3d.optimizer.search import EFFORT_PRESETS
from slipsurface3d.project_model import (
    FORMAT_3D,
    AnalysisSettings3D,
    MaterialSpec3D,
    Project3D,
    ProjectSummary3D,
    SurfaceConfig3D,
)


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
