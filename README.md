# SlipSurfaceIQ-3D

**Three-Dimensional Limit-Equilibrium Slope Stability Analysis (`Method of Columns` + `PyVista/VTK` + `PySide6`)**

`SlipSurfaceIQ-3D` is the standalone 3D companion to `SlipSurfaceIQ-LE`, engineered for 3D rotational, ellipsoidal, multi-plane rock wedge, composite weak-seam, and SAO spline-bowl slope stability analysis.

## Key Capabilities

- **7 Rigorous & Simplified 3D Method of Columns Solvers (`slipsurface3d/lem/`)**:
  - **3D Ordinary / Fellenius** (Hovland 1977; Hungr 1987)
  - **3D Simplified Bishop** (Hungr 1987; Hungr et al. 1989)
  - **3D Simplified & Corrected Janbu** (Hungr et al. 1989)
  - **3D Rigorous Spencer** (Lam & Fredlund 1993) — coupled global horizontal/vertical force + moment equilibrium
  - **3D General Limit Equilibrium / Morgenstern-Price (`gle_mp`)** — strip-wise intercolumn functions (`constant`, `half-sine`, `clipped-sine`, `trapezoidal`)
  - **3D Cheng & Yip (2007) Asymmetric Equilibrium (`cheng_yip`)** — solves $\sum F_s = 0$, $\sum F_t = 0$, and $\sum M = 0$ simultaneously for oblique 3D sliding azimuth $\psi^*$
- **Global Search, Multi-Scale SAO &Slope-Wide Safety Maps (`slipsurface3d/optimizer/`)**:
  - Multi-core parallel Cuckoo Search + Nelder-Mead polish + **Multi-Scale Surface Altering Optimization (`SAO: 3×3 → 5×5` bicubic spline refinement)**
  - **3D Slope-Wide Safety Map (`SafetyMap3D`)** projecting all trial slip bowls onto ground topography
- **Probabilistic Reliability, Seismic $k_y$/Newmark & 3D Seepage**:
  - **Latin Hypercube & Monte Carlo 3D Reliability** ($P_f$, $\beta_N$, $\beta_{LN}$) & **Tornado Sensitivity** (`slipsurface3d/probabilistic.py`)
  - **Critical Yield Acceleration $k_y$** ($F_s(k_y) = 1.0$) & **Newmark Permanent Displacement** (Jibson 2007, Ambraseys & Menu 1988, and exact accelerogram double integration in `slipsurface3d/seismic.py`)
  - **3D Steady-State Anisotropic Seepage Laplace Solver** $\nabla \cdot (\mathbf{K}\nabla H) = 0$ with atmospheric seepage-face breakout (`slipsurface3d/seepage.py`)
- **3D Geometry, Rock Wedges, Bolt Patterns & GIS/CAD Exchange**:
  - Extruded 2D profiles, multi-section lofted cuts, ESRI ASCII DEM grids (`.asc`), XYZ/CSV point clouds, Binary/ASCII `.stl`, Wavefront `.obj`, 3D `.dxf` (`3DFACE`), and 3D borehole RBF interpolation
  - **3D Multi-Plane Rock Wedge (`MultiPlaneWedgeSurface`)** matching closed-form Hoek & Bray (1981) 3D wedge mechanics
  - **Automatic 3D Bolt Pattern Generator** (`BoltPatternSpec3D`), 2D↔3D Cross-Section Slicer & $F_{3D}/F_{2D}$ End-Effect Analyzer (`slipsurface3d/cross_section.py`), and ergonomic Python Scripting API (`slipsurface3d/scripting.py`)
- **Interactive PyVista / VTK Desktop UI & Engineering Reports**:
  - Hardware-accelerated `QVTKRenderWindowInteractor` 3D viewport (Wayland XWayland-safe + headless offscreen fallback) with live search preview, stop/cancel button, picked-column 3D free-body force arrows ($W, N', S_{\text{mob}}, U$), terrain Safety Map overlay, and ReportLab A4 PDF reports.

## Quick Start

```bash
# Headless 3D verification check
python3 main.py --check

# Full 6-benchmark verification audit
python3 scripts/audit_verification_3d.py

# Launch interactive 3D desktop UI
python3 main.py
```
