# SlipSurfaceIQ-3D

**Three-Dimensional Limit-Equilibrium Slope Stability Analysis (`Method of Columns` + `PyVista/VTK` + `PySide6`)**

`SlipSurfaceIQ-3D` is the standalone 3D companion to `SlipSurfaceIQ-LE`, engineered for 3D rotational, ellipsoidal, multi-plane rock wedge, composite weak-seam, and SAO spline-bowl slope stability analysis.

## Key Capabilities

- **7 Rigorous & Simplified 3D Method of Columns Solvers (`slipsurface3d/lem/`)**:
  - **3D Ordinary / Fellenius** (Hovland 1977; Hungr 1987)
  - **3D Simplified Bishop** (Hungr 1987; Hungr et al. 1989)
  - **3D Simplified Janbu** (Hungr et al. 1989)
  - **3D Corrected Janbu** ($f_0 \cdot F_{\text{Janbu}}$)
  - **3D Rigorous Spencer** (Lam & Fredlund 1993) — coupled global horizontal/vertical force + moment equilibrium
  - **3D General Limit Equilibrium / Morgenstern-Price (`gle_mp`)** — strip-wise intercolumn functions (`constant`, `half-sine`, `clipped-sine`, `trapezoidal`)
  - **3D Cheng & Yip (2007) Asymmetric Equilibrium (`cheng_yip`)** — solves $\sum F_s = 0$, $\sum F_t = 0$, and $\sum M = 0$ simultaneously for oblique 3D sliding azimuth $\psi^*$
- **Global Search, Multi-Scale SAO & Slope-Wide Safety Maps (`slipsurface3d/optimizer/`)**:
  - Multi-core parallel Cuckoo Search or Particle Swarm + Nelder-Mead polish + **Multi-Scale Surface Altering Optimization (`SAO: 3×3 → 5×5` bicubic spline refinement)**
  - **3D Slope-Wide Safety Map (`SafetyMap3D`)** projecting all trial slip bowls onto ground topography
  - **Multi-modal search** (`n_local_minima`): reports several distinct failure modes ranked by FoS, each refined inside its own plan footprint and drawn in the viewport
  - **Plan search limits** (`search_limits`) confining the sliding mass to an `[x_min, x_max, y_min, y_max]` box
- **Design Standards & Strength Models (`slipsurface3d/design.py`, `slipsurface3d/materials/`)**:
  - **Partial-factor design** with Eurocode 7 presets (DA1-C1, DA1-C2, DA2, DA3) or user-defined factors on unit weight, variable loads, $c'$, $\tan\varphi'$, $c_u$, support resistance and earth resistance; results are then reported as over-design factors (ODF)
  - 12 strength models, including a **tabulated shear/normal function** and an **infinite-strength** material that slip surfaces may not cross
- **Weak Layers & Pore-Water Options**:
  - **Weak layers along any surface** (`WeakLayer3D`: plane, DEM, STL/OBJ, XYZ), with optional seam thickness: slip surfaces that would pass below a layer are clipped to run along it and take its strength material there, in both user-surface and search modes
  - **Per-material piezometric surfaces** (`piezo_surfaces`, `MaterialSpec3D.piezo`) and **scattered pore-pressure grids** (`pore_mode = "grid"`, `x, y, z, u` CSV import)
- **Areal Supports & Construction Stages**:
  - **Geotextile / geogrid sheets** (horizontal or dipping) and **pile rows** (`GeotextileSheet3D`, `PileRow3D`) cut individually by the 3D slip surface
  - **Staged excavations and fills** (`Stage3D`, `run_staged_analysis_3d`) with a factor of safety per stage; stages can change the water level, and loads and supports act from their own `from_stage`
- **Probabilistic Reliability, Seismic $k_y$/Newmark & 3D Seepage**:
  - **Latin Hypercube & Monte Carlo 3D Reliability** ($P_f$, $\beta_N$, $\beta_{LN}$) & **Tornado Sensitivity** (`slipsurface3d/probabilistic.py`)
  - **Critical Yield Acceleration $k_y$** ($F_s(k_y) = 1.0$) & **Newmark Permanent Displacement** (Jibson 2007, Ambraseys & Menu 1988, and exact accelerogram double integration in `slipsurface3d/seismic.py`)
  - **3D Steady-State Anisotropic Seepage Laplace Solver** $\nabla \cdot (\mathbf{K}\nabla H) = 0$ with atmospheric seepage-face breakout (`slipsurface3d/seepage.py`)
- **3D Geometry, Rock Wedges, Bolt Patterns & GIS/CAD Exchange**:
  - Extruded 2D profiles, multi-section lofted cuts, ESRI ASCII DEM grids (`.asc`), XYZ/CSV point clouds, Binary/ASCII `.stl`, Wavefront `.obj`, 3D `.dxf` (`3DFACE`), and 3D borehole RBF interpolation
  - Multi-valued 3D slip surfaces (closed ellipsoids, overhanging triangle meshes, implicit $F(x,y,z)=0$, scripting-only and not saved in project files) sampled by vertical-ray intersection; Method of Columns uses the first floor-like hit (`sample_base`). Imported triangle meshes are rewound automatically, so STL/OBJ winding order does not matter
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

# Launch interactive 3D desktop UI (opens with an empty project)
python3 main.py
```

The desktop application starts with an empty project. Build a model with **File → New Parametric 3D Slope**, import a 2D `.slp.json`, STL/DEM/CSV terrain or boreholes, or load one of the verification cases under **File → Built-in 3D Examples & Benchmarks**.

Interactive analyses default to a `28 × 24` column grid (`nx × ny`) so searches stay responsive. Published verification cases in [`docs/VERIFICATION_3D.md`](docs/VERIFICATION_3D.md) use `64–72` columns per axis. Raise `nx` / `ny` under **Analysis → Solvers, Columns & Slip Surface Settings** when comparing against published Factors of Safety.

> **Engineering disclaimer.** SlipSurfaceIQ-3D is provided without any warranty. Results depend entirely on the input geometry, strength parameters, groundwater and loads. Always review and verify them with engineering judgement and, where it matters, with independent calculations or software.

## License

Copyright © 2026 Hasan Deniz Altuntaş.

SlipSurfaceIQ-3D is free software: you can redistribute it and/or modify it under the terms of the [GNU Affero General Public License](LICENSE) as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. If you modify the program and make it available to users over a network (for example as a server or web service), the AGPL requires you to offer them the corresponding source code.

The program is distributed in the hope that it will be useful, but **without any warranty**; without even the implied warranty of merchantability or fitness for a particular purpose.

Third-party components and their licenses are listed under *Help → About*: Qt / PySide6 (LGPL-3.0), NumPy, SciPy, PyVista and VTK (BSD), Shapely (BSD), ReportLab (BSD), ezdxf and QtAwesome (MIT).
