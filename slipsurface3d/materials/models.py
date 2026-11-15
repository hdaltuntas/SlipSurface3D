"""Constitutive strength models for 3D Limit Equilibrium Analysis.

Every model evaluates instantaneous tangent parameters ``(c_eq, tan_phi_eq)``
at column base effective normal stress ``sigma_n >= 0`` so that shear strength is:
    tau_f(sigma_n) = c_eq + sigma_n * tan_phi_eq
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

MODEL_LABELS: dict[str, str] = {
    "MC": "Mohr-Coulomb",
    "UD": "Undrained (phi = 0)",
    "UDG": "Undrained Depth-Dependent su(z)",
    "HB": "Generalized Hoek-Brown (GSI, mi, D)",
    "HBD": "Generalized Hoek-Brown (mb, s, a)",
    "BB": "Barton-Bandis Joint",
    "PC": "Power Curve",
    "PL": "Power Law",
    "SH": "SHANSEP",
    "ANISO3D": "3D Anisotropic Bedding (Dip / Dip-Direction)",
}


@dataclass
class Material3D(ABC):
    """Abstract 3D geotechnical material."""

    name: str
    gamma: float = 19.0
    gamma_sat: float = 20.0
    use_water: bool = True
    hu: float = 1.0
    bbar: float = 0.0
    excess_load: bool = False

    uses_elevation: bool = False
    uses_vertical_stress: bool = False
    uses_normal_vector: bool = False
    total_stress: bool = False

    def __post_init__(self) -> None:
        if self.gamma <= 0 or self.gamma_sat <= 0:
            raise ValueError(f"{self.name}: unit weights gamma and gamma_sat must be positive.")
        if not 0.0 <= self.hu <= 1.5:
            raise ValueError(f"{self.name}: water pressure factor hu must be in [0, 1.5].")
        if not 0.0 <= self.bbar <= 1.0:
            raise ValueError(f"{self.name}: B-bar must be in [0, 1].")

    @abstractmethod
    def strength(
        self,
        sigma_n: np.ndarray,
        *,
        z: np.ndarray | None = None,
        sigma_v: np.ndarray | None = None,
        normal_xyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(c_eq, tan_phi_eq)`` arrays matching ``sigma_n`` shape."""


@dataclass
class MohrCoulomb3D(Material3D):
    c: float = 10.0
    phi: float = 30.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.c < 0 or not 0.0 <= self.phi < 89.0:
            raise ValueError(f"{self.name}: require c >= 0 and 0 <= phi < 89 deg.")

    def strength(
        self,
        sigma_n: np.ndarray,
        *,
        z: np.ndarray | None = None,
        sigma_v: np.ndarray | None = None,
        normal_xyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sn = np.asarray(sigma_n, dtype=float)
        return np.full_like(sn, self.c), np.full_like(sn, float(np.tan(np.radians(self.phi))))


@dataclass
class Undrained3D(Material3D):
    su: float = 40.0

    def __post_init__(self) -> None:
        self.total_stress = True
        super().__post_init__()
        if self.su < 0:
            raise ValueError(f"{self.name}: undrained shear strength su must be >= 0.")

    def strength(
        self,
        sigma_n: np.ndarray,
        *,
        z: np.ndarray | None = None,
        sigma_v: np.ndarray | None = None,
        normal_xyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sn = np.asarray(sigma_n, dtype=float)
        return np.full_like(sn, self.su), np.zeros_like(sn)


@dataclass
class UndrainedDepth3D(Material3D):
    su_top: float = 25.0
    su_bottom: float = 60.0
    z_top: float = 10.0
    z_bottom: float = 0.0

    def __post_init__(self) -> None:
        self.uses_elevation = True
        self.total_stress = True
        super().__post_init__()
        if self.su_top < 0 or self.su_bottom < 0:
            raise ValueError(f"{self.name}: su_top and su_bottom must be >= 0.")

    def strength(
        self,
        sigma_n: np.ndarray,
        *,
        z: np.ndarray | None = None,
        sigma_v: np.ndarray | None = None,
        normal_xyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sn = np.asarray(sigma_n, dtype=float)
        if z is None or abs(self.z_top - self.z_bottom) < 1e-9:
            return np.full_like(sn, self.su_top), np.zeros_like(sn)
        za = np.asarray(z, dtype=float)
        t = np.clip((self.z_top - za) / (self.z_top - self.z_bottom), 0.0, 1.0)
        su = self.su_top + t * (self.su_bottom - self.su_top)
        return su, np.zeros_like(sn)


def _hoek_brown_tangent(
    sigma_n: np.ndarray, sigma_ci: float, mb: float, s: float, a: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Hoek-Brown (2002) instantaneous tangent (c_t, tan_phi_t) at effective normal stress sigma_n."""
    sn = np.maximum(np.asarray(sigma_n, dtype=float), 1e-6)
    # Solve parametric sigma_n(sigma_3) = sigma_3 + sigma_ci * u^a / (2 + a * mb * u^(a-1))
    # via 12 vectorized Newton steps in normalized variable x = mb * sigma_3 / sigma_ci + s
    x = np.maximum(mb * sn / sigma_ci + s, max(s, 1e-9) + 1e-9)
    for _ in range(14):
        xa = np.power(x, a)
        xam1 = np.power(x, a - 1.0)
        denom = 2.0 + a * mb * xam1
        s3 = (x - s) * sigma_ci / mb
        sn_est = s3 + sigma_ci * xa / denom
        # Exact derivative d(sn_est)/dx:
        num = xa
        dnum_dx = a * xam1
        dden_dx = a * (a - 1.0) * mb * np.power(x, a - 2.0)
        dsn_dx = sigma_ci / mb + sigma_ci * (dnum_dx * denom - num * dden_dx) / (denom * denom)
        step = (sn_est - sn) / np.maximum(dsn_dx, 1e-12)
        x = np.maximum(x - step, max(s, 0.0) + 1e-11)
        if float(np.max(np.abs(sn_est - sn))) < 1e-7 * sigma_ci:
            break

    xa = np.power(x, a)
    xam1 = np.power(x, a - 1.0)
    denom = 2.0 + a * mb * xam1
    s3 = (x - s) * sigma_ci / mb
    sn_exact = s3 + sigma_ci * xa / denom
    tau = (sn_exact - s3) * np.sqrt(1.0 + a * mb * xam1)
    # Instantaneous friction angle: sin(phi_i) = h / (2 + h) where h = a * mb * x^(a - 1)
    h = a * mb * xam1
    sin_phi = h / (2.0 + h)
    cos_phi = np.sqrt(np.maximum(1.0 - sin_phi * sin_phi, 1e-12))
    tan_phi = sin_phi / cos_phi
    c_eq = np.maximum(tau - sn_exact * tan_phi, 0.0)
    return c_eq, tan_phi


@dataclass
class HoekBrown3D(Material3D):
    sigma_ci: float = 25_000.0
    gsi: float = 50.0
    mi: float = 10.0
    d: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.sigma_ci <= 0
            or not 0.0 < self.gsi <= 100.0
            or self.mi <= 0
            or not 0.0 <= self.d <= 1.0
        ):
            raise ValueError(f"{self.name}: invalid Hoek-Brown parameters (sigma_ci, GSI, mi, D).")
        self._mb = float(self.mi * np.exp((self.gsi - 100.0) / (28.0 - 14.0 * self.d)))
        self._s = float(np.exp((self.gsi - 100.0) / (9.0 - 3.0 * self.d)))
        self._a = float(0.5 + (np.exp(-self.gsi / 15.0) - np.exp(-20.0 / 3.0)) / 6.0)

    def strength(
        self,
        sigma_n: np.ndarray,
        *,
        z: np.ndarray | None = None,
        sigma_v: np.ndarray | None = None,
        normal_xyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _hoek_brown_tangent(sigma_n, self.sigma_ci, self._mb, self._s, self._a)


@dataclass
class HoekBrownDirect3D(Material3D):
    sigma_ci: float = 25_000.0
    mb: float = 1.5
    s: float = 0.004
    a: float = 0.506

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sigma_ci <= 0 or self.mb <= 0 or self.s < 0 or not 0.25 <= self.a <= 0.75:
            raise ValueError(
                f"{self.name}: invalid direct Hoek-Brown parameters (sigma_ci, mb, s, a)."
            )

    def strength(
        self,
        sigma_n: np.ndarray,
        *,
        z: np.ndarray | None = None,
        sigma_v: np.ndarray | None = None,
        normal_xyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _hoek_brown_tangent(sigma_n, self.sigma_ci, self.mb, self.s, self.a)


@dataclass
class BartonBandis3D(Material3D):
    phi_r: float = 25.0
    jrc: float = 10.0
    jcs: float = 50_000.0
    l0: float = 0.1
    ln: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not 0.0 <= self.phi_r < 70.0
            or self.jrc < 0
            or self.jcs <= 0
            or self.l0 <= 0
            or self.ln <= 0
        ):
            raise ValueError(f"{self.name}: invalid Barton-Bandis joint parameters.")

    def _tau(self, sn: np.ndarray) -> np.ndarray:
        ratio = self.ln / self.l0
        jrc_n = self.jrc * (ratio ** (-0.02 * self.jrc))
        jcs_n = self.jcs * (ratio ** (-0.03 * self.jrc))
        s_eff = np.clip(sn, 1e-4, jcs_n)
        ang_deg = np.clip(self.phi_r + jrc_n * np.log10(jcs_n / s_eff), 0.0, 70.0)
        return sn * np.tan(np.radians(ang_deg))

    def strength(
        self,
        sigma_n: np.ndarray,
        *,
        z: np.ndarray | None = None,
        sigma_v: np.ndarray | None = None,
        normal_xyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sn = np.maximum(np.asarray(sigma_n, dtype=float), 1e-3)
        ds = np.maximum(1e-4 * sn, 1e-5)
        tau0 = self._tau(sn)
        tp = np.maximum((self._tau(sn + ds) - tau0) / ds, 0.0)
        c_eq = np.maximum(tau0 - sn * tp, 0.0)
        return c_eq, tp


@dataclass
class PowerCurve3D(Material3D):
    pc_a: float = 1.5
    pc_b: float = 0.8
    pc_c: float = 0.0
    pc_d: float = 0.0
    pa: float = 100.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pc_a <= 0 or not 0.0 < self.pc_b <= 1.0 or self.pc_c < 0 or self.pc_d < 0:
            raise ValueError(f"{self.name}: invalid Power Curve parameters.")

    def strength(
        self,
        sigma_n: np.ndarray,
        *,
        z: np.ndarray | None = None,
        sigma_v: np.ndarray | None = None,
        normal_xyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sn = np.maximum(np.asarray(sigma_n, dtype=float), 1e-4)
        shifted = np.maximum((sn + self.pc_d) / self.pa, 1e-8)
        tau = self.pc_c + self.pc_a * self.pa * np.power(shifted, self.pc_b)
        tp = self.pc_a * self.pc_b * np.power(shifted, self.pc_b - 1.0)
        c_eq = np.maximum(tau - sn * tp, 0.0)
        return c_eq, tp


@dataclass
class PowerLaw3D(Material3D):
    coefficient: float = 0.5
    exponent: float = 0.8
    reference_pressure: float = 100.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.coefficient <= 0 or not 0.0 < self.exponent <= 1.0 or self.reference_pressure <= 0:
            raise ValueError(f"{self.name}: invalid Power Law parameters.")

    def strength(
        self,
        sigma_n: np.ndarray,
        *,
        z: np.ndarray | None = None,
        sigma_v: np.ndarray | None = None,
        normal_xyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sn = np.maximum(np.asarray(sigma_n, dtype=float), 1e-4)
        p = self.reference_pressure
        tau = self.coefficient * p * np.power(sn / p, self.exponent)
        tp = self.coefficient * self.exponent * np.power(sn / p, self.exponent - 1.0)
        c_eq = np.maximum(tau - sn * tp, 0.0)
        return c_eq, tp


@dataclass
class Shansep3D(Material3D):
    s_ratio: float = 0.22
    m_exp: float = 0.8
    ocr: float = 1.0
    su_min: float = 0.0
    ocr_bottom: float = 1.0
    z_top: float = 0.0
    z_bottom: float = 0.0

    def __post_init__(self) -> None:
        self.uses_vertical_stress = True
        self.uses_elevation = True
        self.total_stress = True
        super().__post_init__()
        if (
            self.s_ratio <= 0
            or self.m_exp < 0
            or self.ocr < 1.0
            or self.ocr_bottom < 1.0
            or self.su_min < 0
        ):
            raise ValueError(f"{self.name}: invalid SHANSEP parameters.")

    def strength(
        self,
        sigma_n: np.ndarray,
        *,
        z: np.ndarray | None = None,
        sigma_v: np.ndarray | None = None,
        normal_xyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sn = np.asarray(sigma_n, dtype=float)
        sv = np.maximum(np.asarray(sigma_v if sigma_v is not None else sn, dtype=float), 0.0)
        if z is not None and abs(self.z_top - self.z_bottom) > 1e-9:
            t = np.clip(
                (self.z_top - np.asarray(z, dtype=float)) / (self.z_top - self.z_bottom), 0.0, 1.0
            )
            ocr_loc = self.ocr + t * (self.ocr_bottom - self.ocr)
        else:
            ocr_loc = np.full_like(sn, self.ocr)
        su = self.su_min + self.s_ratio * np.power(ocr_loc, self.m_exp) * sv
        return su, np.zeros_like(sn)


@dataclass
class AnisotropicBedding3D(Material3D):
    """3D Anisotropic rock/soil strength with a weak bedding plane (dip & dip-direction).

    Interpolates smoothly between bedding strength ``(c_bedding, phi_bedding)`` when the column
    base normal aligns with the bedding pole and rock-mass strength ``(c_mass, phi_mass)`` when
    the angle between the column base and bedding exceeds ``transition_deg``.
    """

    c_mass: float = 40.0
    phi_mass: float = 35.0
    c_bedding: float = 5.0
    phi_bedding: float = 18.0
    dip_deg: float = 20.0
    dip_dir_deg: float = 180.0
    transition_deg: float = 20.0

    def __post_init__(self) -> None:
        self.uses_normal_vector = True
        super().__post_init__()
        if self.c_mass < 0 or self.c_bedding < 0:
            raise ValueError(f"{self.name}: cohesions must be >= 0.")
        if not (0.0 <= self.phi_mass < 89.0 and 0.0 <= self.phi_bedding < 89.0):
            raise ValueError(f"{self.name}: friction angles must lie in [0, 89) deg.")
        if self.transition_deg <= 0:
            raise ValueError(f"{self.name}: transition_deg must be > 0.")

    def bedding_pole(self) -> np.ndarray:
        dip = float(np.radians(self.dip_deg))
        az = float(np.radians(self.dip_dir_deg))
        return np.array(
            [np.sin(dip) * np.cos(az), np.sin(dip) * np.sin(az), np.cos(dip)],
            dtype=float,
        )

    def strength(
        self,
        sigma_n: np.ndarray,
        *,
        z: np.ndarray | None = None,
        sigma_v: np.ndarray | None = None,
        normal_xyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sn = np.asarray(sigma_n, dtype=float)
        tp_mass = float(np.tan(np.radians(self.phi_mass)))
        tp_bed = float(np.tan(np.radians(self.phi_bedding)))
        if normal_xyz is None:
            return np.full_like(sn, self.c_mass), np.full_like(sn, tp_mass)
        pole = self.bedding_pole()
        dots = np.clip(np.abs(np.asarray(normal_xyz, dtype=float) @ pole), 0.0, 1.0)
        ang_deg = np.degrees(np.arccos(dots))
        # Cosine blend: w = 0 on bedding (ang=0), w = 1 in intact mass (ang >= transition_deg)
        ratio = np.clip(ang_deg / self.transition_deg, 0.0, 1.0)
        w = 0.5 * (1.0 - np.cos(np.pi * ratio))
        c_eq = self.c_bedding + w * (self.c_mass - self.c_bedding)
        tp_eq = tp_bed + w * (tp_mass - tp_bed)
        return c_eq.reshape(sn.shape), tp_eq.reshape(sn.shape)


def material_from_spec_dict(d: dict[str, Any]) -> Material3D:
    """Instantiate a Material3D from a serialized MaterialSpec3D or 2D MaterialSpec dictionary."""
    name = str(d.get("name", "Soil 1"))
    model = str(d.get("model", "MC"))
    gamma = float(d.get("gamma", 19.0))
    use_sat = bool(d.get("use_sat", True))
    gamma_sat = float(d.get("gamma_sat", gamma)) if use_sat else gamma
    common: dict[str, Any] = dict(
        name=name,
        gamma=gamma,
        gamma_sat=gamma_sat,
        use_water=bool(d.get("use_water", True)),
        hu=float(d.get("hu", 1.0)),
        bbar=float(d.get("bbar", 0.0)),
        excess_load=bool(d.get("excess_load", False)),
    )
    if model in ("MC", "HS"):
        return MohrCoulomb3D(c=float(d.get("c", 10.0)), phi=float(d.get("phi", 30.0)), **common)
    if model == "UD":
        return Undrained3D(su=float(d.get("su", 40.0)), **common)
    if model == "UDG":
        z_top = float(d.get("z_top", d.get("su_y_top", 10.0)))
        z_bot = float(d.get("z_bottom", d.get("su_y_bottom", 0.0)))
        return UndrainedDepth3D(
            su_top=float(d.get("su_top", 25.0)),
            su_bottom=float(d.get("su_bottom", 50.0)),
            z_top=z_top,
            z_bottom=z_bot,
            **common,
        )
    if model == "HB":
        return HoekBrown3D(
            sigma_ci=float(d.get("sigma_ci", 25_000.0)),
            gsi=float(d.get("gsi", 50.0)),
            mi=float(d.get("mi", 10.0)),
            d=float(d.get("d", 0.0)),
            **common,
        )
    if model == "HBD":
        return HoekBrownDirect3D(
            sigma_ci=float(d.get("sigma_ci", 25_000.0)),
            mb=float(d.get("hb_mb", d.get("mb", 1.5))),
            s=float(d.get("hb_s", d.get("s", 0.004))),
            a=float(d.get("hb_a", d.get("a", 0.506))),
            **common,
        )
    if model == "BB":
        return BartonBandis3D(
            phi_r=float(d.get("bb_phir", d.get("phi_r", 25.0))),
            jrc=float(d.get("bb_jrc", d.get("jrc", 10.0))),
            jcs=float(d.get("bb_jcs", d.get("jcs", 50_000.0))),
            l0=float(d.get("bb_l0", d.get("l0", 0.1))),
            ln=float(d.get("bb_ln", d.get("ln", 0.1))),
            **common,
        )
    if model == "PC":
        return PowerCurve3D(
            pc_a=float(d.get("pc_a", 1.5)),
            pc_b=float(d.get("pc_b", 0.8)),
            pc_c=float(d.get("pc_c", 0.0)),
            pc_d=float(d.get("pc_d", 0.0)),
            **common,
        )
    if model == "PL":
        return PowerLaw3D(
            coefficient=float(d.get("coefficient", 0.5)),
            exponent=float(d.get("exponent", 0.8)),
            reference_pressure=float(d.get("reference_pressure", 100.0)),
            **common,
        )
    if model == "SH":
        return Shansep3D(
            s_ratio=float(d.get("shansep_s", d.get("s_ratio", 0.22))),
            m_exp=float(d.get("shansep_m", d.get("m_exp", 0.8))),
            ocr=float(d.get("ocr", 1.0)),
            su_min=float(d.get("shansep_su_min", d.get("su_min", 0.0))),
            ocr_bottom=float(d.get("ocr_bottom", 1.0)),
            z_top=float(d.get("z_top", d.get("ocr_y_top", 0.0))),
            z_bottom=float(d.get("z_bottom", d.get("ocr_y_bottom", 0.0))),
            **common,
        )
    if model == "ANISO3D":
        return AnisotropicBedding3D(
            c_mass=float(d.get("c_mass", d.get("c", 40.0))),
            phi_mass=float(d.get("phi_mass", d.get("phi", 35.0))),
            c_bedding=float(d.get("c_bedding", 5.0)),
            phi_bedding=float(d.get("phi_bedding", 18.0)),
            dip_deg=float(d.get("dip_deg", 20.0)),
            dip_dir_deg=float(d.get("dip_dir_deg", 180.0)),
            transition_deg=float(d.get("transition_deg", 20.0)),
            **common,
        )
    if model == "CN":
        # CNI nonlinear rock mass mapped to PowerCurve3D
        sc = float(d.get("cni_sigma_c", 10_000.0))
        ca = float(d.get("cni_a", 0.6))
        cn = float(d.get("cni_n", 0.75))
        ct = float(d.get("cni_t", 100.0))
        return PowerCurve3D(
            pc_a=ca * ((sc / 100.0) ** (1.0 - cn)),
            pc_b=cn,
            pc_c=0.0,
            pc_d=ct,
            **common,
        )
    if model == "SS":
        # SANISAND critical-state friction angle M_c = 6 sin(phi_cs) / (3 - sin(phi_cs))
        mc = float(d.get("ss_m", 1.25))
        sin_phi = 3.0 * mc / (6.0 + mc)
        phi_cs = float(np.degrees(np.arcsin(np.clip(sin_phi, 0.0, 0.98))))
        return MohrCoulomb3D(c=0.0, phi=phi_cs, **common)
    raise ValueError(f"{name}: unknown 3D material model '{model}'.")
