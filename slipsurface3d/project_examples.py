"""Built-in 3D benchmark and example projects."""

from __future__ import annotations

from typing import Any

import numpy as np

from slipsurface3d.geometry.surfaces import (
    CompositeEllipsoidPlaneSurface,
    EllipsoidSurface,
    MultiPlaneWedgeSurface,
    WedgePlane3D,
    sphere_surface,
)
from slipsurface3d.geometry.terrain import (
    HorizonSurface,
    simple_3d_slope,
)
from slipsurface3d.loads.supports import PatchLoad3D, SupportType3D
from slipsurface3d.project_model import (
    AnalysisSettings3D,
    MaterialSpec3D,
    Project3D,
    ProjectSummary3D,
    SurfaceConfig3D,
)


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


def rock_wedge_3d_project() -> Project3D:
    """Hoek & Bray (1981) Symmetric 3D Rock Wedge Analytical Benchmark.

    Intersection line plunges psi_i = 30 deg toward -X (180 deg) from toe (x=10, y=25, z=0);
    joint planes tilt laterally at beta = 35 deg (half-wedge angle xi = 55 deg).
    For c = 0, phi = 38 deg, exact Hoek & Bray (1981) 3D FoS = tan(38)/(tan(30)*cos(35)) = 1.651975.
    """
    model = simple_3d_slope(
        height=20.0,
        ratio_h_to_v=0.5,
        toe_x=10.0,
        left_bench=10.0,
        right_bench=35.0,
        depth_below_toe=10.0,
        width_y=50.0,
        material="Jointed Granite",
    )
    mat = MaterialSpec3D(
        name="Jointed Granite",
        model="MC",
        gamma=26.0,
        gamma_sat=26.5,
        c=0.0,
        phi=38.0,
        use_water=False,
        color="#9b7ec8",
    )
    tan_psi = float(np.tan(np.radians(30.0)))
    # Dihedral tilt beta = 35 deg perpendicular to plunging intersection line -> dz/dy = tan(beta)/cos(psi_i)
    tan_beta = float(np.tan(np.radians(35.0)) / np.cos(np.radians(30.0)))
    dip_deg = float(np.degrees(np.arctan(np.hypot(tan_psi, tan_beta))))
    dd1 = float(np.degrees(np.arctan2(-tan_beta, -tan_psi)) % 360.0)
    dd2 = float(np.degrees(np.arctan2(tan_beta, -tan_psi)) % 360.0)
    wedge = MultiPlaneWedgeSurface(
        planes=[
            WedgePlane3D(
                dip_deg=dip_deg, dip_dir_deg=dd1, x0=10.0, y0=25.0, z0=0.0, name="Joint A"
            ),
            WedgePlane3D(
                dip_deg=dip_deg, dip_dir_deg=dd2, x0=10.0, y0=25.0, z0=0.0, name="Joint B"
            ),
        ],
        bounds=(10.0, 50.0, 2.0, 48.0),
    )
    return Project3D(
        summary=ProjectSummary3D(
            title="Hoek & Bray (1981) 3D Rock Wedge Benchmark",
            notes="Symmetric 2-plane rock wedge (plunge=30 deg, half-wedge=55 deg, phi=38 deg; exact 3D FoS = 1.6520).",
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
            methods=["janbu_simplified", "spencer", "gle_mp", "cheng_yip"],
            nx=32,
            ny=32,
            sliding_mode="auto",
            pore_mode="none",
        ),
        surface_config=SurfaceConfig3D(
            mode="user",
            surface_kind="wedge",
            primary_method="spencer",
            user_surface=wedge.to_dict(),
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
    "rock_wedge_3d": (
        "Hoek & Bray (1981) 3D Rock Wedge Benchmark",
        rock_wedge_3d_project,
    ),
    "reinforced_valley_3d": (
        "3D Lofted Cut Slope with Tiebacks & Water",
        reinforced_3d_valley_project,
    ),
}
