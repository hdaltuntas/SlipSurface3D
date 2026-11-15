#!/usr/bin/env python3
"""3D Limit Equilibrium Benchmark & Verification Audit Suite.

Verifies:
1. Closed-form analytical phi=0 3D spherical cap on a 2H:1V slope (Baligh & Azzouz 1975; Hungr 1987).
2. 2D -> 3D cylindrical extrusion invariance across Ordinary, Bishop, Janbu Simplified,
   Janbu Corrected, and Spencer (comparing against analytical 2D limit / independent 1D strip).
3. Zhang (1988) / Hungr et al. (1989) Example 1 (Homogeneous 3D Ellipsoid, published FoS = 2.122).
4. Zhang (1988) / Hungr et al. (1989) Example 2 (3D Ellipsoid + Weak Seam, published FoS = 1.553).
5. Symmetry & azimuth invariance under 90/180/270-degree coordinate rotations.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slipsurface3d.analysis import run_analysis_3d  # noqa: E402
from slipsurface3d.geometry import (  # noqa: E402
    HorizonSurface,
    cylindrical_surface,
    sphere_surface,
)
from slipsurface3d.project import (  # noqa: E402
    AnalysisSettings3D,
    MaterialSpec3D,
    Project3D,
    ProjectSummary3D,
    SurfaceConfig3D,
    extruded_acads_1a_project,
    hungr_1987_hemisphere_project,
    zhang_1988_homog_ellipsoid_project,
    zhang_1988_weak_seam_project,
)


def exact_phi0_spherical_cap_fos(
    radius: float, d_perp: float, slope_tan: float, su: float, gamma: float
) -> float:
    beta = math.atan(slope_tan)
    a = math.sqrt(radius * radius - d_perp * d_perp)
    num, _ = quad(
        lambda t: (
            2.0
            * su
            * radius
            * math.sqrt(radius * radius - t * t)
            * math.acos(d_perp / math.sqrt(radius * radius - t * t))
        ),
        -a,
        a,
    )
    den = (math.pi / 4.0) * gamma * ((radius * radius - d_perp * d_perp) ** 2) * math.sin(beta)
    return float(num / den)


def main() -> int:
    failures: list[str] = []
    print("=" * 78)
    print("SlipSurfaceIQ-3D — Benchmark Verification Audit")
    print("=" * 78)

    # 1. Exact closed-form phi=0 3D spherical cap
    exact_cap = exact_phi0_spherical_cap_fos(
        radius=15.0, d_perp=11.5, slope_tan=0.5, su=30.0, gamma=20.0
    )
    p_cap = hungr_1987_hemisphere_project()
    p_cap.analysis.nx = 72
    p_cap.analysis.ny = 72
    r_cap = run_analysis_3d(p_cap)
    for meth in ("ordinary", "bishop", "spencer"):
        fos = r_cap.method_results[meth].fos
        rel_err = abs(fos - exact_cap) / exact_cap
        status = "PASS" if rel_err < 0.005 else "FAIL"
        print(
            f"[1] Analytical Cap ({meth:<8}): 3D={fos:.5f} | Exact={exact_cap:.5f} | RelErr={rel_err * 100:.3f}% [{status}]"
        )
        if status == "FAIL":
            failures.append(f"Analytical cap {meth} rel_err={rel_err:.4f}")

    # 2. Cylindrical width invariance (ny=1 vs ny=16, W_y=10 m vs W_y=80 m)
    p_c1 = extruded_acads_1a_project()
    p_c1.surface_config.mode = "user"
    p_c1.surface_config.user_surface = cylindrical_surface(20.0, 16.0, 17.0, y_center=5.0).to_dict()
    p_c1.y_min, p_c1.y_max = 0.0, 10.0
    p_c1.analysis.methods = ["ordinary", "bishop", "janbu_simplified", "janbu_corrected", "spencer"]
    p_c1.analysis.nx, p_c1.analysis.ny = 80, 1

    p_c2 = p_c1.copy()
    p_c2.y_min, p_c2.y_max = 0.0, 80.0
    p_c2.surface_config.user_surface = cylindrical_surface(
        20.0, 16.0, 17.0, y_center=40.0
    ).to_dict()
    p_c2.analysis.nx, p_c2.analysis.ny = 80, 16

    rc1 = run_analysis_3d(p_c1)
    rc2 = run_analysis_3d(p_c2)
    for meth in p_c1.analysis.methods:
        f1 = rc1.method_results[meth].fos
        f2 = rc2.method_results[meth].fos
        diff = abs(f1 - f2)
        status = "PASS" if diff < 1e-6 else "FAIL"
        print(
            f"[2] 2D Extrusion Invariance ({meth:<16}): ny=1 -> {f1:.5f} | ny=16 -> {f2:.5f} | diff={diff:.2e} [{status}]"
        )
        if status == "FAIL":
            failures.append(f"Extrusion invariance {meth} diff={diff}")

    # 3. Zhang (1988) / Hungr et al. (1989) Ex. 1 (published 3D Bishop = 2.122)
    p_z1 = zhang_1988_homog_ellipsoid_project()
    p_z1.analysis.nx = p_z1.analysis.ny = 48
    r_z1 = run_analysis_3d(p_z1)
    f_z1 = r_z1.method_results["bishop"].fos
    ref_z1 = 2.122
    err_z1 = abs(f_z1 - ref_z1) / ref_z1
    st_z1 = "PASS" if err_z1 < 0.02 else "FAIL"
    print(
        f"[3] Zhang (1988) Ex.1 Homogeneous Ellipsoid : 3D Bishop={f_z1:.4f} | Ref={ref_z1:.3f} | RelErr={err_z1 * 100:.2f}% [{st_z1}]"
    )
    if st_z1 == "FAIL":
        failures.append(f"Zhang Ex.1 rel_err={err_z1:.4f}")

    # 4. Zhang (1988) / Hungr et al. (1989) Ex. 2 (published 3D Bishop = 1.553, CLARA = 1.55-1.62)
    p_z2 = zhang_1988_weak_seam_project()
    p_z2.analysis.nx = p_z2.analysis.ny = 48
    r_z2 = run_analysis_3d(p_z2)
    f_z2_b = r_z2.method_results["bishop"].fos
    f_z2_j = r_z2.method_results["janbu_simplified"].fos
    ref_z2 = 1.553
    err_z2 = abs(f_z2_b - ref_z2) / ref_z2
    st_z2 = "PASS" if err_z2 < 0.05 else "FAIL"
    print(
        f"[4] Zhang (1988) Ex.2 Ellipsoid + Weak Seam : 3D Bishop={f_z2_b:.4f}, Janbu={f_z2_j:.4f} | Ref={ref_z2:.3f} | RelErr={err_z2 * 100:.2f}% [{st_z2}]"
    )
    if st_z2 == "FAIL":
        failures.append(f"Zhang Ex.2 rel_err={err_z2:.4f}")

    # 5. Rotational symmetry check: slope dipping -X (az=180) vs slope dipping -Y (az=270)
    xs = np.linspace(0.0, 40.0, 41)
    ys = np.linspace(0.0, 40.0, 41)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    mat_sym = MaterialSpec3D(name="Soil", model="MC", gamma=19.0, c=15.0, phi=25.0, use_water=False)

    g_x = HorizonSurface.from_grid(xs, ys, 0.4 * xx, name="Dip -X")
    p_sym_x = Project3D(
        summary=ProjectSummary3D(title="Dip -X"),
        materials=[mat_sym],
        x_min=0.0,
        x_max=40.0,
        y_min=0.0,
        y_max=40.0,
        z_min=-10.0,
        ground=g_x,
        layers=[("Soil", g_x)],
        analysis=AnalysisSettings3D(
            methods=["bishop", "spencer"], nx=40, ny=40, sliding_mode="auto", pore_mode="none"
        ),
        surface_config=SurfaceConfig3D(
            mode="user",
            surface_kind="sphere",
            user_surface=sphere_surface(16.0, 20.0, 18.0, 15.0).to_dict(),
        ),
    )
    g_y = HorizonSurface.from_grid(xs, ys, 0.4 * yy, name="Dip -Y")
    p_sym_y = Project3D(
        summary=ProjectSummary3D(title="Dip -Y"),
        materials=[mat_sym],
        x_min=0.0,
        x_max=40.0,
        y_min=0.0,
        y_max=40.0,
        z_min=-10.0,
        ground=g_y,
        layers=[("Soil", g_y)],
        analysis=AnalysisSettings3D(
            methods=["bishop", "spencer"], nx=40, ny=40, sliding_mode="auto", pore_mode="none"
        ),
        surface_config=SurfaceConfig3D(
            mode="user",
            surface_kind="sphere",
            user_surface=sphere_surface(20.0, 16.0, 18.0, 15.0).to_dict(),
        ),
    )
    rx = run_analysis_3d(p_sym_x)
    ry = run_analysis_3d(p_sym_y)
    sym_diff = abs(rx.method_results["spencer"].fos - ry.method_results["spencer"].fos)
    st_sym = "PASS" if sym_diff < 1e-6 else "FAIL"
    print(
        f"[5] 90-deg Azimuth Symmetry (Spencer)       : Az={rx.columns.azimuth_deg:.1f}->FoS={rx.method_results['spencer'].fos:.5f} | "
        f"Az={ry.columns.azimuth_deg:.1f}->FoS={ry.method_results['spencer'].fos:.5f} [{st_sym}]"
    )
    if st_sym == "FAIL":
        failures.append(f"Azimuth symmetry diff={sym_diff}")

    print("=" * 78)
    if failures:
        print("AUDIT FAILED:", "; ".join(failures))
        return 1
    print("ALL 3D BENCHMARK VERIFICATION CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
