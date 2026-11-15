# SlipSurfaceIQ-3D — Benchmark Verification Documentation

This document records the analytical and published three-dimensional limit-equilibrium verification suite executed by `scripts/audit_verification_3d.py`.

## 1. Closed-Form $\phi = 0$ 3D Spherical Cap (Baligh & Azzouz 1975; Hungr 1987)

For a sphere of radius $R$ whose centre lies at perpendicular distance $d$ above a uniform planar slope of inclination $\beta$ in an undrained ($\phi = 0$) clay with shear strength $s_u$ and unit weight $\gamma$, the exact 3D rotational equilibrium Factor of Safety is given in closed integral form by:

$$F_{\text{3D, exact}} = \frac{2 s_u R \int_{-a}^{+a} \sqrt{R^2 - t^2} \arccos\!\left(\frac{d}{\sqrt{R^2 - t^2}}\right) dt}{\frac{\pi}{4} \gamma (R^2 - d^2)^2 \sin\beta}, \qquad a = \sqrt{R^2 - d^2}$$

For $R = 15.0\text{ m}$, $d = 11.5\text{ m}$, $\tan\beta = 0.5$ ($2\text{H}:1\text{V}$), $s_u = 30\text{ kPa}$, and $\gamma = 20\text{ kN/m}^3$:
- **Exact Analytical FoS**: `2.31611`
- **SlipSurfaceIQ-3D (`nx = ny = 72`)**: Ordinary = `2.3163`, Bishop = `2.3163`, Spencer = `2.3163` (**< 0.01% relative error**).

## 2. 2D → 3D Cylindrical Extrusion Invariance

When any 2D slope profile and circular slip surface (`ry -> infinity`) are extruded along $Y \in [y_{\min}, y_{\max}]$, every column has transverse basal inclination $\alpha_t = 0$, so cell width $\Delta y$ cancels identically from force and moment equilibrium.
- Verified across `ordinary`, `bishop`, `janbu_simplified`, `janbu_corrected`, `spencer`, `gle_mp`, and `cheng_yip`:
  - `ny = 1` ($W_y = 10\text{ m}$) vs `ny = 16` ($W_y = 80\text{ m}$): **`|ΔFoS| < 1e-12`**, matching 2D `SlipSurfaceIQ-LE` to 5 significant digits (including Spencer's intercolumn inclination $\lambda$).
- Interactive analyses default to `nx = 28`, `ny = 24`. The spherical-cap and Zhang cases in this document use `nx = ny = 48–72`.

## 3. Zhang (1988) / Hungr et al. (1989) Benchmark Cases

| Case | Description | Published Reference FoS | SlipSurfaceIQ-3D Bishop | Relative Difference |
|---|---|---:|---:|---:|
| **Ex. 1** | Homogeneous $12.19\text{ m}$ ($40\text{ ft}$) $2\text{H}:1\text{V}$ slope ($c'=28.73\text{ kPa}$, $\phi'=20^\circ$, $\gamma=18.84\text{ kN/m}^3$), ellipsoid $r_x=r_z=24.384\text{ m}$, $r_y=60.96\text{ m}$ | `2.122` (Zhang 1988; Hungr et al. 1989) | `2.135` | `0.6%` |
| **Ex. 2** | Same slope & ellipsoid truncated by horizontal weak seam ($c'_w=0$, $\phi'_w=10^\circ$) at $z = 0\text{ m}$ | `1.553` (Zhang 1988) / `1.55–1.62` (Hungr et al. 1989) | `1.59` (Janbu `1.56`) | `< 2.5%` |

## 4. Coordinate-Frame & Azimuth Invariance

Rotating a 3D slope and spherical slip surface by $90^\circ$ in the horizontal $(X, Y)$ plane yields automatic sliding-direction detection (`180.0°` vs `270.0°`) and identical Factors of Safety (`|ΔFoS| < 1e-12`) for both 3D Bishop and 3D Spencer.


## 5. Hoek & Bray (1981) Closed-Form 3D Rock Wedge Benchmark

For a symmetric two-plane rock wedge with line-of-intersection plunge $\psi_i = 30^\circ$, dihedral plane tilt $\beta = 35^\circ$ (half-wedge angle $\xi = 55^\circ$), cohesion $c' = 0$, and joint friction angle $\phi' = 38^\circ$:

$$F_{\text{3D, Hoek-Bray}} = \frac{\tan\phi'}{\tan\psi_i \sin\xi} = \frac{\tan 38^\circ}{\tan 30^\circ \cos 35^\circ} = 1.651984$$

- **SlipSurfaceIQ-3D (`MultiPlaneWedgeSurface`)**: Janbu Simplified, Spencer, GLE/M-P, and Cheng & Yip all yield **`1.651984`** (`|ΔFoS| < 1e-10`), reproducing exact 3D lateral wedging amplification ($1/\cos 35^\circ = 1.2208\times$ over 2D planar sliding).
