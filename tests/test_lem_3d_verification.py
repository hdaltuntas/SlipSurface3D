"""Verification tests for 3D LEM solvers, constitutive models, loads, bolts, and benchmarks."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from slipsurface3d.analysis import run_analysis_3d
from slipsurface3d.geometry import cylindrical_surface, sphere_surface
from slipsurface3d.loads import PatchLoad3D, SupportType3D, TensionCrack3D
from slipsurface3d.materials import (
    AnisotropicBedding3D,
    BartonBandis3D,
    HoekBrown3D,
    HoekBrownDirect3D,
    PowerCurve3D,
    PowerLaw3D,
    Shansep3D,
    UndrainedDepth3D,
)
from slipsurface3d.project import (
    extruded_acads_1a_project,
    hungr_1987_hemisphere_project,
    zhang_1988_homog_ellipsoid_project,
    zhang_1988_weak_seam_project,
)


def test_analytical_phi0_3d_spherical_cap_benchmark():
    beta = math.atan(0.5)
    r, d = 15.0, 11.5
    su, gamma = 30.0, 20.0
    a = math.sqrt(r * r - d * d)
    num, _ = quad(
        lambda t: 2.0 * su * r * math.sqrt(r * r - t * t) * math.acos(d / math.sqrt(r * r - t * t)),
        -a,
        a,
    )
    exact_fos = num / ((math.pi / 4.0) * gamma * ((r * r - d * d) ** 2) * math.sin(beta))

    proj = hungr_1987_hemisphere_project()
    proj.analysis.nx = 64
    proj.analysis.ny = 64
    res = run_analysis_3d(proj)
    for m in ("ordinary", "bishop", "spencer"):
        mr = res.method_results[m]
        assert mr.ok and mr.admissible
        assert abs(mr.fos - exact_fos) / exact_fos < 0.005


def test_2d_extrusion_width_invariance_all_methods():
    p1 = extruded_acads_1a_project()
    p1.surface_config.mode = "user"
    p1.surface_config.user_surface = cylindrical_surface(20.0, 16.0, 17.0, y_center=5.0).to_dict()
    p1.y_min, p1.y_max = 0.0, 10.0
    p1.analysis.methods = ["ordinary", "bishop", "janbu_simplified", "janbu_corrected", "spencer"]
    p1.analysis.nx, p1.analysis.ny = 70, 1

    p2 = p1.copy()
    p2.y_min, p2.y_max = 0.0, 60.0
    p2.surface_config.user_surface = cylindrical_surface(20.0, 16.0, 17.0, y_center=30.0).to_dict()
    p2.analysis.nx, p2.analysis.ny = 70, 12

    r1 = run_analysis_3d(p1)
    r2 = run_analysis_3d(p2)
    for meth in p1.analysis.methods:
        assert r1.method_results[meth].ok
        assert np.isclose(r1.method_results[meth].fos, r2.method_results[meth].fos, atol=1e-9)
    assert np.isclose(r1.method_results["spencer"].lam, r2.method_results["spencer"].lam, atol=1e-6)


def test_zhang_1988_homogeneous_and_weak_seam_benchmarks():
    r1 = run_analysis_3d(zhang_1988_homog_ellipsoid_project())
    assert abs(r1.method_results["bishop"].fos - 2.122) / 2.122 < 0.02
    assert r1.method_results["spencer"].ok

    r2 = run_analysis_3d(zhang_1988_weak_seam_project())
    # Weak horizontal seam drops FoS substantially from ~2.13 down to ~1.55-1.60
    assert 1.48 <= r2.method_results["bishop"].fos <= 1.65
    assert r2.method_results["bishop"].fos < r1.method_results["bishop"].fos - 0.4


def test_nonlinear_and_anisotropic_3d_constitutive_models():
    sn = np.array([25.0, 100.0, 300.0])
    for mat in (
        HoekBrown3D("HB", sigma_ci=30_000.0, gsi=55.0, mi=12.0, d=0.2),
        HoekBrownDirect3D("HBD", sigma_ci=30_000.0, mb=1.8, s=0.005, a=0.505),
        BartonBandis3D("BB", phi_r=26.0, jrc=12.0, jcs=40_000.0),
        PowerCurve3D("PC", pc_a=1.2, pc_b=0.75, pc_c=5.0, pc_d=2.0),
        PowerLaw3D("PL", coefficient=0.6, exponent=0.8),
    ):
        c, tp = mat.strength(sn)
        assert np.all(c >= 0.0) and np.all(tp > 0.0)

    udg = UndrainedDepth3D("UDG", su_top=20.0, su_bottom=60.0, z_top=10.0, z_bottom=0.0)
    c_u, tp_u = udg.strength(sn, z=np.array([10.0, 5.0, 0.0]))
    assert np.allclose(c_u, [20.0, 40.0, 60.0]) and np.all(tp_u == 0.0)

    sh = Shansep3D("SH", s_ratio=0.25, m_exp=0.8, ocr=2.0, su_min=5.0)
    c_sh, _ = sh.strength(sn, sigma_v=np.array([100.0, 100.0, 100.0]))
    assert np.all(c_sh > 40.0)

    # 3D Bedding Anisotropy: weak along dip=25 deg / dip_dir=180 deg, strong perpendicular
    aniso = AnisotropicBedding3D(
        "Shale",
        c_mass=50.0,
        phi_mass=38.0,
        c_bedding=4.0,
        phi_bedding=15.0,
        dip_deg=25.0,
        dip_dir_deg=180.0,
        transition_deg=20.0,
    )
    pole = aniso.bedding_pole()
    perp = np.array([0.0, 1.0, 0.0])
    normals = np.vstack([pole, perp])
    c_an, tp_an = aniso.strength(np.array([100.0, 100.0]), normal_xyz=normals)
    assert np.isclose(c_an[0], 4.0) and np.isclose(c_an[1], 50.0)
    assert tp_an[0] < tp_an[1]

    # Solve nonlinear Hoek-Brown 3D slope end-to-end
    p_hb = extruded_acads_1a_project()
    p_hb.materials[0].model = "HB"
    p_hb.materials[0].sigma_ci = 5_000.0
    p_hb.materials[0].gsi = 35.0
    p_hb.materials[0].mi = 8.0
    p_hb.surface_config.mode = "user"
    p_hb.surface_config.user_surface = sphere_surface(20.0, 20.0, 16.0, 17.0).to_dict()
    r_hb = run_analysis_3d(p_hb)
    assert r_hb.method_results["bishop"].ok and r_hb.method_results["spencer"].ok


def test_seismic_patch_load_bolts_and_tension_crack_physics():
    base = extruded_acads_1a_project()
    base.surface_config.mode = "user"
    base.surface_config.user_surface = sphere_surface(22.0, 20.0, 16.0, 17.0).to_dict()
    base.analysis.methods = ["bishop", "spencer"]
    r0 = run_analysis_3d(base)
    fos0 = r0.method_results["bishop"].fos

    # Horizontal seismic kh lowers FoS
    p_seis = base.copy()
    p_seis.analysis.kh = 0.12
    p_seis.analysis.kv = 0.05
    p_seis.analysis.both_kv_signs = True
    r_seis = run_analysis_3d(p_seis)
    assert r_seis.method_results["bishop"].fos < fos0

    # Crest surcharge lowers FoS
    p_load = base.copy()
    p_load.patch_loads = [PatchLoad3D(x1=34.0, x2=42.0, y1=12.0, y2=28.0, q1=50.0)]
    r_load = run_analysis_3d(p_load)
    assert r_load.method_results["bishop"].fos < fos0

    # Grouted tieback bolts intersecting the slip bowl increase FoS
    p_bolt = base.copy()
    st = SupportType3D(
        name="Tieback",
        kind="grouted_tieback",
        tensile_kn=400.0,
        plate_kn=400.0,
        bond_strength_kn_m=150.0,
        bond_length_m=8.0,
    )
    p_bolt.support_types = [st]
    p_bolt.bolts = [
        {"head": [22.0, y_b, 3.5], "tip": [38.0, y_b, -2.0], "type": "Tieback", "label": f"B{i}"}
        for i, y_b in enumerate([16.0, 20.0, 24.0], 1)
    ]
    r_bolt = run_analysis_3d(p_bolt)
    assert len(r_bolt.columns.support_hits) == 3
    assert r_bolt.method_results["bishop"].fos > fos0

    # Water-filled tension crack
    p_tc = base.copy()
    p_tc.tension_crack = TensionCrack3D(depth_m=1.5, water_fill_fraction=1.0)
    r_tc = run_analysis_3d(p_tc)
    assert r_tc.method_results["bishop"].ok
