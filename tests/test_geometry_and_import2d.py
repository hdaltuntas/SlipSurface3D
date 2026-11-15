"""Tests for 3D terrain/horizon geometry, slip surfaces, borehole RBF, and 2D .slp.json import."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from slipsurface3d.analysis import run_analysis_3d
from slipsurface3d.geometry import (
    BoreholeLayer,
    BoreholeLog,
    CompositeEllipsoidPlaneSurface,
    EllipsoidSurface,
    HorizonSurface,
    ImplicitSlipSurface,
    MeshSlipSurface,
    SplinePatchSurface,
    cylindrical_surface,
    horizons_from_boreholes,
    simple_3d_slope,
    sphere_surface,
    surface_from_dict,
)
from slipsurface3d.project import (
    Project3D,
    extruded_acads_1a_project,
    hungr_1987_hemisphere_project,
    import_2d_project,
)


def test_horizon_representations_and_serialization():
    # Extruded
    h_ext = HorizonSurface.from_extruded([[0.0, 0.0], [10.0, 5.0], [20.0, 5.0]], name="Ext")
    assert np.isclose(float(h_ext.z(5.0, 12.0)), 2.5)
    assert np.allclose(h_ext.z([5.0, 5.0], [0.0, 100.0]), [2.5, 2.5])
    h_ext_rt = HorizonSurface.from_dict(h_ext.to_dict())
    assert np.isclose(float(h_ext_rt.z(5.0, 7.0)), 2.5)

    # Lofted
    h_loft = HorizonSurface.from_lofted(
        [0.0, 20.0],
        [[[0.0, 0.0], [10.0, 4.0]], [[0.0, 0.0], [10.0, 8.0]]],
        name="Loft",
    )
    assert np.isclose(float(h_loft.z(10.0, 10.0)), 6.0)
    assert HorizonSurface.from_dict(h_loft.to_dict()).kind == "lofted"

    # Regular Grid
    xs = np.linspace(0.0, 10.0, 6)
    ys = np.linspace(0.0, 8.0, 5)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = 2.0 + 0.5 * xx - 0.25 * yy
    h_grid = HorizonSurface.from_grid(xs, ys, zz, name="DEM")
    assert np.isclose(float(h_grid.z(4.0, 4.0)), 3.0)
    assert h_grid.to_mesh((0.0, 10.0), (0.0, 8.0), nx=10, ny=8).n_points == 80

    # Planar dipping horizon
    h_plane = HorizonSurface.from_plane(
        (0.0, 20.0), (0.0, 20.0), z0=10.0, dip_deg=45.0, dip_dir_deg=180.0
    )
    assert float(h_plane.z(15.0, 10.0)) > float(h_plane.z(5.0, 10.0))


def test_borehole_rbf_horizon_interpolation():
    bhs = [
        BoreholeLog(
            "BH1",
            0.0,
            0.0,
            2.0,
            [BoreholeLayer("Fill", 2.0, 0.0), BoreholeLayer("Bedrock", 0.0, -10.0)],
            water_z=1.0,
        ),
        BoreholeLog(
            "BH2",
            0.0,
            30.0,
            2.0,
            [BoreholeLayer("Fill", 2.0, 0.0), BoreholeLayer("Bedrock", 0.0, -10.0)],
            water_z=1.0,
        ),
        BoreholeLog(
            "BH3",
            40.0,
            0.0,
            12.0,
            [BoreholeLayer("Fill", 12.0, 4.0), BoreholeLayer("Bedrock", 4.0, -10.0)],
            water_z=8.0,
        ),
        BoreholeLog(
            "BH4",
            40.0,
            30.0,
            12.0,
            [BoreholeLayer("Fill", 12.0, 4.0), BoreholeLayer("Bedrock", 4.0, -10.0)],
            water_z=8.0,
        ),
    ]
    ground, layers, wt, z_min = horizons_from_boreholes(bhs)
    assert len(layers) == 2
    assert wt is not None
    assert z_min <= -10.0
    assert np.isclose(float(ground.z(20.0, 15.0)), 7.0, atol=0.2)
    assert np.isclose(float(wt.z(20.0, 15.0)), 4.5, atol=0.2)


def test_3d_slip_surface_gradients_and_roundtrip():
    ell = EllipsoidSurface(xc=20.0, yc=15.0, zc=18.0, rx=14.0, ry=22.0, rz=12.0, azimuth_deg=165.0)
    gx_an, gy_an = ell.gradients(np.array([17.0, 22.0]), np.array([13.0, 18.0]))
    gx_fd, gy_fd = super(EllipsoidSurface, ell).gradients(
        np.array([17.0, 22.0]), np.array([13.0, 18.0])
    )
    assert np.allclose(gx_an, gx_fd, atol=1e-5)
    assert np.allclose(gy_an, gy_fd, atol=1e-5)

    comp = CompositeEllipsoidPlaneSurface(
        ellipsoid=ell, plane_z0=8.5, dip_deg=5.0, dip_dir_deg=180.0
    )
    assert isinstance(surface_from_dict(comp.to_dict()), CompositeEllipsoidPlaneSurface)

    sp = SplinePatchSurface(base=ell, control_dz=np.full((3, 3), -0.5))
    assert float(sp.z(20.0, 15.0)) < float(ell.z(20.0, 15.0))
    assert isinstance(surface_from_dict(sp.to_dict()), SplinePatchSurface)

    pts = np.array([[0.0, 0.0, 5.0], [20.0, 0.0, 5.0], [0.0, 20.0, 5.0], [20.0, 20.0, 5.0]])
    ms = MeshSlipSurface(pts)
    assert np.isclose(float(ms.z(10.0, 10.0)), 5.0)
    assert isinstance(surface_from_dict(ms.to_dict()), MeshSlipSurface)

    s_sph = sphere_surface(10.0, 10.0, 15.0, 12.0)
    assert s_sph.is_spherical
    s_cyl = cylindrical_surface(10.0, 15.0, 12.0)
    assert not s_cyl.is_spherical


def test_closed_ellipsoid_returns_both_sheets_and_picks_the_floor():
    ell = EllipsoidSurface(xc=20.0, yc=15.0, zc=18.0, rx=14.0, ry=22.0, rz=12.0, sheet="both")
    z_hits, nrm = ell.vertical_hits(20.0, 15.0)
    finite = np.isfinite(z_hits)
    assert int(np.count_nonzero(finite)) == 2
    assert nrm[0, 2] < 0.0
    assert nrm[1, 2] > 0.0
    samp = ell.sample_base(20.0, 15.0, 40.0, 0.0)
    assert int(samp.n_hits) == 2
    assert np.isclose(float(samp.z), float(ell.z(20.0, 15.0)), atol=1e-9)
    assert float(samp.normal_xyz[2]) > 0.0
    rt = surface_from_dict(ell.to_dict())
    assert isinstance(rt, EllipsoidSurface) and rt.sheet == "both"


def test_overhanging_triangle_mesh_skips_roof_and_keeps_floor():
    verts = np.array(
        [
            [0.0, 0.0, 1.0],
            [10.0, 0.0, 1.0],
            [0.0, 10.0, 1.0],
            [10.0, 10.0, 1.0],
            [0.0, 0.0, 6.0],
            [10.0, 0.0, 6.0],
            [0.0, 10.0, 6.0],
            [10.0, 10.0, 6.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [4, 6, 5],
            [5, 6, 7],
        ],
        dtype=np.int64,
    )
    ms = MeshSlipSurface(verts, faces=faces)
    assert not ms.graph_valued
    z_hits, nrm = ms.vertical_hits(5.0, 5.0)
    assert int(np.count_nonzero(np.isfinite(z_hits))) >= 2
    assert nrm[0, 2] < 0.0
    samp = ms.sample_base(5.0, 5.0, 12.0, -1.0)
    assert int(samp.n_hits) == 2
    assert np.isclose(float(samp.z), 1.0, atol=1e-6)
    assert float(samp.normal_xyz[2]) > 0.0
    rt = surface_from_dict(ms.to_dict())
    assert isinstance(rt, MeshSlipSurface) and rt.faces is not None


def test_implicit_sphere_matches_closed_ellipsoid_floor():
    ell = EllipsoidSurface(xc=20.0, yc=15.0, zc=18.0, rx=12.0, ry=12.0, rz=12.0, sheet="both")

    def fn(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return ((x - 20.0) / 12.0) ** 2 + ((y - 15.0) / 12.0) ** 2 + ((z - 18.0) / 12.0) ** 2 - 1.0

    def grad_fn(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                2.0 * (x - 20.0) / 144.0,
                2.0 * (y - 15.0) / 144.0,
                2.0 * (z - 18.0) / 144.0,
            ],
            axis=-1,
        )

    imp = ImplicitSlipSurface(
        fn=fn, grad_fn=grad_fn, bounds=(5.0, 35.0, 0.0, 30.0, 0.0, 40.0), n_z_scan=64
    )
    assert not imp.graph_valued
    assert np.isclose(float(imp.z(20.0, 15.0)), float(ell.z(20.0, 15.0)), atol=0.05)
    samp = imp.sample_base(20.0, 15.0, 40.0, 0.0)
    assert int(samp.n_hits) >= 2
    assert float(samp.normal_xyz[2]) > 0.0


def test_closed_ellipsoid_columns_match_lower_sheet_fos():
    p_lo = hungr_1987_hemisphere_project()
    p_lo.analysis.nx = p_lo.analysis.ny = 24
    r_lo = run_analysis_3d(p_lo)

    p_both = p_lo.copy()
    surf = p_both.user_slip_surface()
    assert isinstance(surf, EllipsoidSurface)
    surf.sheet = "both"
    p_both.surface_config.user_surface = surf.to_dict()
    r_both = run_analysis_3d(p_both)
    assert np.isclose(r_lo.primary_result.fos, r_both.primary_result.fos, rtol=1e-6, atol=1e-6)


def test_project3d_save_load_validation(tmp_path: Path):
    proj = extruded_acads_1a_project()
    out = proj.save(tmp_path / "acads3d.slp3d.json")
    loaded = Project3D.load(out)
    assert loaded.summary.title == proj.summary.title
    assert len(loaded.materials) == 1

    bad = proj.to_dict()
    bad["domain"]["x_max"] = -100.0
    with pytest.raises(ValueError, match="x_max"):
        Project3D.from_dict(bad)

    m3d = simple_3d_slope(height=8.0, ratio_h_to_v=1.5, width_y=25.0)
    assert m3d.ground_mesh(nx=12, ny=10).n_points == 120


def test_import_2d_slp_json_project_with_loads_supports_and_water(tmp_path: Path):
    sample_2d = {
        "format": "slipsurface-le",
        "version": 3,
        "summary": {"title": "2D Embankment", "author": "GeoTest", "comments": ["Imported test"]},
        "materials": [
            {
                "name": "Fill",
                "model": "MC",
                "gamma": 19.0,
                "gamma_sat": 20.5,
                "use_sat": True,
                "c": 12.0,
                "phi": 28.0,
                "color": "#e0b438",
            },
            {
                "name": "Clay Base",
                "model": "UD",
                "gamma": 18.0,
                "gamma_sat": 18.0,
                "su": 45.0,
                "color": "#4ea86f",
            },
        ],
        "geometry": {
            "external": [
                [0.0, -8.0],
                [50.0, -8.0],
                [50.0, 10.0],
                [30.0, 10.0],
                [12.0, 0.0],
                [0.0, 0.0],
            ],
            "material_boundaries": [[[0.0, 0.0], [50.0, 0.0]]],
            "water_table": [[0.0, 0.0], [12.0, 0.0], [35.0, 6.0], [50.0, 6.0]],
            "assignments": [
                {"point": [35.0, 5.0], "material": "Fill"},
                {"point": [25.0, -4.0], "material": "Clay Base"},
            ],
        },
        "settings": {
            "general": {"failure_direction": "right_to_left"},
            "methods": {
                "methods": ["bishop", "spencer"],
                "n_slices": 30,
                "tolerance": 0.001,
                "max_iter": 60,
            },
            "groundwater": {
                "method": "water_table",
                "gamma_w": 9.81,
                "ru": 0.0,
                "crack_water": 0.5,
            },
            "seismic": {"kh": 0.08, "kv": 0.04, "both_vertical": True},
            "advanced": {"check_admissibility": True, "min_depth": 0.2, "m_alpha_min": 0.2},
        },
        "surfaces": {"surface_type": "user", "effort": "fast", "user_circle": [18.0, 15.0, 17.0]},
        "loads": [
            {
                "kind": "distributed",
                "points": [[32.0, 10.0], [42.0, 10.0]],
                "q": [20.0, 30.0],
                "orientation": "vertical",
            },
            {"kind": "line", "point": [35.0, 10.0], "magnitude": 15.0, "angle": 270.0},
        ],
        "support_types": [
            {
                "name": "Anchor",
                "kind": "grouted_tieback",
                "spacing": 5.0,
                "tensile": 220.0,
                "plate": 200.0,
                "bond_strength": 80.0,
                "bond_length": 5.0,
                "application": "active",
            }
        ],
        "supports": [{"a": [20.0, 4.4], "b": [32.0, 1.0], "type": "Anchor"}],
    }
    p2d_file = tmp_path / "embankment.slp.json"
    p2d_file.write_text(json.dumps(sample_2d), encoding="utf-8")

    # Direct Project3D.load auto-detects slipsurface-le format
    p3d = Project3D.load(p2d_file)
    assert len(p3d.materials) == 2
    assert len(p3d.profile_regions) >= 2
    assert p3d.water_table is not None
    assert len(p3d.patch_loads) == 1
    assert len(p3d.point_loads) >= 4
    assert len(p3d.bolts) == 6  # 30 m span / 5 m spacing = 6 bolts

    # Extrude as cylinder option
    p3d_cyl = import_2d_project(p2d_file, y_min=0.0, y_max=20.0, extrude_surface_as_cylinder=True)
    assert p3d_cyl.surface_config.user_surface["ry"] >= 1e8
