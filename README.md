# SlipSurfaceIQ-3D

**Three-Dimensional Limit-Equilibrium Slope Stability Analysis (`Method of Columns` + `PyVista/VTK` + `PySide6`)**

`SlipSurfaceIQ-3D` is the standalone 3D companion to `SlipSurfaceIQ-LE`, engineered for 3D rotational, ellipsoidal, composite weak-seam, and spline-bowl slope stability analysis.

## Key Capabilities

- **3D Method of Columns Solvers (`slipsurface3d/lem/`)**:
  - **3D Ordinary / Fellenius** (Hovland 1977; Hungr 1987)
  - **3D Simplified Bishop** (Hungr 1987; Hungr et al. 1989)
  - **3D Simplified & Corrected Janbu** (Hungr et al. 1989)
  - **3D Rigorous Spencer** (Lam & Fredlund 1993; Cheng & Yip 2007) — coupled global horizontal/vertical force + moment equilibrium with intercolumn inclination $\lambda$ and Whitman & Bailey ($m_\alpha$) admissibility screening.
- **3D Geometry & Stratigraphy (`slipsurface3d/geometry/`)**:
  - Extruded 2D profiles, multi-section lofted 3D cuts/valleys, regular DEM elevation grids, and 3D borehole thin-plate RBF horizon interpolation.
  - Direct lossless import & extrusion of 2D `SlipSurfaceIQ-LE` (`.slp.json`) projects.
  - Tri-axial 3D ellipsoids, spheres, extruded 2D circular cylinders, weak-plane truncated composite ellipsoids, non-ellipsoidal 3D spline patches, and imported OBJ/DXF slip meshes.
- **Constitutive Strength Models (`slipsurface3d/materials/`)**:
  - Mohr-Coulomb, Undrained ($\phi = 0$), Depth-Linear $s_u(z)$, Generalized Hoek-Brown (`GSI, mi, D` & direct `mb, s, a`), Barton-Bandis, Power Curve, Power Law, SHANSEP, and **3D Dip / Dip-Direction Anisotropic Bedding**.
- **Loads, Groundwater & 3D Reinforcement (`slipsurface3d/loads/`)**:
  - Piezometric 3D water surfaces, $R_u$, $\bar{B}$ excess pore pressure, ponded water, water-filled 3D tension cracks, pseudo-static seismic ($k_h, \pm k_v$), rectangular patch surcharges, 3D point loads, and discrete 3D bolts (end-anchored, grouted tiebacks, soil nails, micropiles, geotextiles).
- **Interactive PyVista / VTK Desktop UI & Engineering Reports**:
  - Hardware-accelerated `QVTKRenderWindowInteractor` 3D viewport with automatic headless offscreen fallback (`QT_QPA_PLATFORM=offscreen`).
  - Interactive 3D column scalar maps ($\sigma'_n$, $\tau_{\text{mob}}$, $u$, $h$, $W$, $\alpha_s$, Material ID), mid-plane section clipping, cell picking, Dark/Light themes, 3D DXF (`3DFACE`) / Wavefront `.obj` exchange, and ReportLab A4 engineering PDF reports.

## Quick Start

```bash
# Headless 3D verification check
python3 main.py --check

# Full benchmark verification audit
python3 scripts/audit_verification_3d.py

# Launch interactive 3D desktop UI
python3 main.py
```
