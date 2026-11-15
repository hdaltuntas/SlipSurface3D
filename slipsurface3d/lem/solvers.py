"""Three-dimensional Limit Equilibrium solvers (Method of Columns).

Implemented methods:
* ``ordinary_3d``         : 3D Ordinary / Fellenius moment equilibrium (Hovland 1977; Hungr 1987)
* ``bishop_3d``           : 3D Simplified Bishop moment equilibrium (Hungr 1987; Hungr et al. 1989)
* ``janbu_simplified_3d`` : 3D Simplified Janbu horizontal force equilibrium (Hungr et al. 1989)
* ``janbu_corrected_3d``  : 3D Corrected Janbu (f0 * F_Janbu)
* ``spencer_3d``          : 3D Rigorous Spencer coupled force + moment equilibrium
                            (Lam & Fredlund 1993; Cheng & Yip 2007)
* ``gle_mp_3d``           : 3D General Limit Equilibrium / Morgenstern-Price with selectable
                            intercolumn inclination functions (constant, half-sine, clipped-sine, trapezoidal)
* ``cheng_yip_3d``        : 3D Asymmetric Cheng & Yip (2007) simultaneous longitudinal force,
                            transverse force, and moment equilibrium solving true sliding azimuth
"""

from __future__ import annotations

from collections.abc import Callable

from slipsurface3d.lem.columns import ColumnGrid
from slipsurface3d.lem.common import M_ALPHA_MIN_DEFAULT, LEMResult3D
from slipsurface3d.lem.rigorous import (
    INTERCOLUMN_FUNCTIONS,
    cheng_yip_3d,
    gle_mp_3d,
    intercolumn_function_3d,
    spencer_3d,
)
from slipsurface3d.lem.simplified import (
    bishop_3d,
    janbu_corrected_3d,
    janbu_simplified_3d,
    ordinary_3d,
)

METHODS_3D: dict[str, tuple[str, Callable[..., LEMResult3D]]] = {
    "ordinary": ("3D Ordinary (Fellenius)", ordinary_3d),
    "bishop": ("3D Bishop Simplified", bishop_3d),
    "janbu_simplified": ("3D Janbu Simplified", janbu_simplified_3d),
    "janbu_corrected": ("3D Janbu Corrected", janbu_corrected_3d),
    "spencer": ("3D Spencer", spencer_3d),
    "gle_mp": ("3D GLE / Morgenstern-Price", gle_mp_3d),
    "cheng_yip": ("3D Cheng & Yip Asymmetric", cheng_yip_3d),
}


def solve_surface_3d(
    cols: ColumnGrid,
    method: str = "bishop",
    tolerance: float = 1e-4,
    max_iter: int = 75,
    m_alpha_min: float = M_ALPHA_MIN_DEFAULT,
    interslice: str = "half-sine",
) -> LEMResult3D:
    if method not in METHODS_3D:
        raise ValueError(f"Unknown 3D LEM method {method!r} (expected one of {list(METHODS_3D)}).")
    _, fn = METHODS_3D[method]
    if method in ("gle_mp", "cheng_yip"):
        use_fn = (
            interslice
            if method == "gle_mp"
            else (interslice if interslice in INTERCOLUMN_FUNCTIONS else "constant")
        )
        return fn(
            cols,
            tolerance=tolerance,
            max_iter=max_iter,
            m_alpha_min=m_alpha_min,
            interslice=use_fn,
        )
    return fn(cols, tolerance=tolerance, max_iter=max_iter, m_alpha_min=m_alpha_min)


__all__ = [
    "INTERCOLUMN_FUNCTIONS",
    "METHODS_3D",
    "M_ALPHA_MIN_DEFAULT",
    "LEMResult3D",
    "bishop_3d",
    "cheng_yip_3d",
    "gle_mp_3d",
    "intercolumn_function_3d",
    "janbu_corrected_3d",
    "janbu_simplified_3d",
    "ordinary_3d",
    "solve_surface_3d",
    "spencer_3d",
]
