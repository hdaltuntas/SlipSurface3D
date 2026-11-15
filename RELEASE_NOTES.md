# Release Notes — SlipSurfaceIQ-3D `2026.7.1`

Standalone 3D companion to SlipSurfaceIQ-LE. Licensed under GNU AGPL v3 or later.

## Solvers
- 3D Ordinary / Fellenius, Simplified Bishop, Simplified and Corrected Janbu, and Rigorous Spencer (Lam & Fredlund 1993).
- 3D General Limit Equilibrium / Morgenstern-Price (`gle_mp`) with strip-wise intercolumn functions: constant, half-sine, clipped-sine, trapezoidal.
- 3D Cheng & Yip (2007) asymmetric equilibrium (`cheng_yip`): simultaneous $\sum F_s = 0$, $\sum F_t = 0$, and $\sum M = 0$ for the oblique sliding azimuth $\psi^*$.

## Search and surfaces
- Cuckoo Search + Nelder-Mead global optimizer for spheres, tri-axial ellipsoids, and non-ellipsoidal 3D spline bowls.
- Multi-scale Surface Altering Optimization (SAO: 3×3 → 5×5 bicubic spline refinement) and multi-core parallel search.
- 3D slope-wide Safety Map projecting trial slip bowls onto ground topography.
- Multi-plane rock wedge (`MultiPlaneWedgeSurface`) matching Hoek & Bray (1981) closed-form 3D wedge mechanics.

## Design standards and strength models
- Partial-factor design (`slipsurface3d.design`): Eurocode 7 DA1-C1, DA1-C2, DA2 and DA3 presets using the recommended values of EN 1997-1 Annex A, plus user-defined factors. Results are reported as over-design factors (ODF). Pore pressures are not factored; check the National Annex before relying on a preset.
- Tabulated shear/normal strength function (`SN`) and infinite-strength material (`INF`) that rejects any slip surface passing through it.
- The Hoek-Brown (mb, s, a), Barton-Bandis, Power Curve, Power Law and SHANSEP parameters are now editable in the Materials dialog.

## Multi-modal search and search limits
- `surface_config.n_local_minima` reports up to 12 distinct critical surfaces (plan footprints overlapping by less than half), ranked by FoS and drawn in the viewport, the text summary and the PDF report. Secondary modes are sampled less densely than the global minimum, so their FoS is an upper bound on the local minimum.
- `surface_config.search_limits` restricts the sliding mass to a plan box.

## Groundwater, seismic, reliability
- 3D steady-state anisotropic seepage Laplace solver $\nabla \cdot (\mathbf{K}\nabla H) = 0$ with atmospheric seepage-face breakout.
- Critical yield acceleration $k_y$ ($F_s(k_y) = 1.0$) and Newmark permanent displacement (Jibson 2007, Ambraseys & Menu 1988, accelerogram double integration).
- Latin Hypercube and Monte Carlo 3D reliability ($P_f$, $\beta_N$, $\beta_{LN}$) with tornado sensitivity.

## Geometry, loads, and exchange
- Lossless 2D `.slp.json` importer and extruder (`import_2d_project`) compatible with SlipSurfaceIQ-LE.
- Ten constitutive models including 3D dip / dip-direction anisotropic bedding.
- 3D patch/point loads, discrete reinforcement bolts, automatic bolt-pattern generator, and tension cracks.
- 2D↔3D cross-section slicer and $F_{3D}/F_{2D}$ end-effect analyzer.
- ESRI ASCII DEM, XYZ/CSV, binary/ASCII STL, Wavefront OBJ, and 3D DXF (`3DFACE`) exchange.

## Desktop UI, scripting, and verification
- PySide6 + PyVista/VTK workstation with Wayland XWayland (`xcb`) fallback, offscreen rendering, live search preview, stop/cancel, picked-column free-body arrows, and ReportLab A4 PDF reports.
- Python scripting and batch API (`slipsurface3d.scripting`).
- Verified against the closed-form $\phi = 0$ 3D spherical cap (`< 0.01%` error), 2D cylindrical extrusion invariance (`|ΔFoS| < 1e-12` across `ny`, including GLE and Cheng & Yip), Zhang (1988) / Hungr et al. (1989) ellipsoid cases, 90° azimuth invariance, and Hoek & Bray (1981) 3D rock wedge (`|ΔFoS| < 1e-10`).
- Interactive analyses default to a `28 × 24` column grid; published verification cases use `64–72` columns per axis.
