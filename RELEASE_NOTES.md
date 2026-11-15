# Release Notes — SlipSurfaceIQ-3D `2026.7.1`

## Highlights
- Initial standalone release of **SlipSurfaceIQ-3D** (`2026.7.1`).
- Implemented 3D Method of Columns solvers: **3D Ordinary**, **3D Simplified Bishop**, **3D Simplified & Corrected Janbu**, and **3D Rigorous Spencer**.
- Added Cuckoo Search + Nelder-Mead global 3D optimizer for spheres, tri-axial ellipsoids, and non-ellipsoidal 3D spline bowls.
- Added lossless 2D `.slp.json` project importer & extruder (`import_2d_project`) compatible with `SlipSurfaceIQ-LE`.
- Added 10 constitutive models including **3D Dip / Dip-Direction Anisotropic Bedding**, plus 3D patch/point loads, discrete 3D reinforcement bolts, and tension cracks.
- Built PySide6 + PyVista/VTK interactive 3D desktop workstation with offscreen rendering support, 3D DXF/OBJ CAD exchange, and ReportLab PDF engineering reports.
- Verified against closed-form analytical $\phi = 0$ 3D spherical cap (`< 0.01%` error), 2D cylindrical extrusion invariance (`< 1e-12` across `ny`), and Zhang (1988) / Hungr et al. (1989) 3D benchmark cases.
