"""3D Probabilistic Slope Stability (Latin Hypercube / Monte Carlo) and Tornado Sensitivity."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from slipsurface3d.analysis import run_analysis_3d
from slipsurface3d.design import apply_design_to_materials
from slipsurface3d.geometry.surfaces import SlipSurface3D
from slipsurface3d.lem.columns import discretize_columns
from slipsurface3d.lem.solvers import solve_surface_3d
from slipsurface3d.project import MaterialSpec3D, Project3D


@dataclass
class RandomVarSpec3D:
    """Stochastic specification for a material property or global loading parameter."""

    param: str  # c | phi | gamma | gamma_sat | su | ru | kh | kv
    material_name: str | None = None
    distribution: str = "normal"  # normal | lognormal | uniform | triangular
    std: float = 0.0
    cov: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    mode_val: float | None = None

    @property
    def label(self) -> str:
        if self.material_name:
            return f"{self.material_name}.{self.param}"
        return self.param


@dataclass
class ProbabilisticResult3D:
    """Summary of 3D Monte Carlo / Latin Hypercube reliability analysis."""

    deterministic_fos: float
    samples_fos: np.ndarray
    sample_inputs: dict[str, np.ndarray]
    mean_fos: float
    std_fos: float
    pf: float
    beta_normal: float
    beta_lognormal: float
    p5: float
    p50: float
    p95: float
    n_converged: int
    n_total: int
    sampling: str
    fixed_surface: bool


@dataclass
class SensitivityParamSpec3D:
    """One-at-a-time parameter sweep range for 3D sensitivity and tornado ranking."""

    param: str
    material_name: str | None = None
    min_val: float = 0.0
    max_val: float = 1.0

    @property
    def label(self) -> str:
        if self.material_name:
            return f"{self.material_name}.{self.param}"
        return self.param


@dataclass
class SensitivityCurve3D:
    """Sweep result for a single parameter, including tornado swing ``delta_fos``."""

    label: str
    param: str
    material_name: str | None
    base_value: float
    values: np.ndarray
    fos_values: np.ndarray
    min_fos: float
    max_fos: float
    delta_fos: float = field(init=False)

    def __post_init__(self) -> None:
        self.delta_fos = float(self.max_fos - self.min_fos)


def _get_base_value(project: Project3D, param: str, material_name: str | None) -> float:
    if material_name is not None:
        for m in project.materials:
            if m.name == material_name:
                if not hasattr(m, param):
                    raise ValueError(f"Material {material_name!r} has no attribute {param!r}.")
                return float(getattr(m, param))
        raise ValueError(f"Unknown material {material_name!r} in variable specification.")
    if hasattr(project.analysis, param):
        return float(getattr(project.analysis, param))
    raise ValueError(f"Unknown global analysis parameter {param!r}.")


def _clamp_physical(param: str, vals: np.ndarray) -> np.ndarray:
    out = np.asarray(vals, dtype=float)
    if param in ("c", "su", "kh"):
        out = np.maximum(out, 0.0)
    elif param == "phi":
        out = np.clip(out, 0.0, 85.0)
    elif param in ("gamma", "gamma_sat"):
        out = np.maximum(out, 1.0)
    elif param == "ru":
        out = np.clip(out, 0.0, 0.99)
    elif param == "kv":
        out = np.clip(out, -0.95, 0.95)
    return out


def _inverse_cdf_column(u: np.ndarray, mean: float, spec: RandomVarSpec3D) -> np.ndarray:
    u_safe = np.clip(u, 1e-6, 1.0 - 1e-6)
    sigma = spec.std if spec.std > 0.0 else (abs(mean) * spec.cov if spec.cov else 0.1 * abs(mean))
    dist = spec.distribution.lower()

    if dist == "normal":
        raw = mean + sigma * norm.ppf(u_safe)
    elif dist == "lognormal":
        mu_pos = max(mean, 1e-6)
        cov_eff = max(sigma / mu_pos, 1e-6)
        zeta = float(np.sqrt(np.log(1.0 + cov_eff * cov_eff)))
        lam = float(np.log(mu_pos) - 0.5 * zeta * zeta)
        raw = np.exp(lam + zeta * norm.ppf(u_safe))
    elif dist == "uniform":
        lo = spec.min_val if spec.min_val is not None else mean - np.sqrt(3.0) * sigma
        hi = spec.max_val if spec.max_val is not None else mean + np.sqrt(3.0) * sigma
        if hi < lo:
            lo, hi = hi, lo
        raw = lo + (hi - lo) * u_safe
    elif dist == "triangular":
        lo = spec.min_val if spec.min_val is not None else mean - np.sqrt(6.0) * sigma
        hi = spec.max_val if spec.max_val is not None else mean + np.sqrt(6.0) * sigma
        if hi <= lo:
            hi = lo + 1e-6
        c_mode = spec.mode_val if spec.mode_val is not None else 0.5 * (lo + hi)
        c_mode = float(np.clip(c_mode, lo, hi))
        fc = (c_mode - lo) / max(hi - lo, 1e-12)
        raw = np.where(
            u_safe < fc,
            lo + np.sqrt(u_safe * (hi - lo) * max(c_mode - lo, 0.0)),
            hi - np.sqrt((1.0 - u_safe) * (hi - lo) * max(hi - c_mode, 0.0)),
        )
    else:
        raise ValueError(f"Unsupported distribution {spec.distribution!r}.")

    if spec.min_val is not None:
        raw = np.maximum(raw, spec.min_val)
    if spec.max_val is not None:
        raw = np.minimum(raw, spec.max_val)
    return _clamp_physical(spec.param, raw)


def _generate_samples(
    project: Project3D,
    var_specs: list[RandomVarSpec3D],
    n_samples: int,
    sampling: str,
    seed: int,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_dim = len(var_specs)
    if sampling.lower() == "lhs":
        u_mat = np.empty((n_samples, n_dim), dtype=float)
        for j in range(n_dim):
            perm = rng.permutation(n_samples)
            u_mat[:, j] = (perm + rng.uniform(0.0, 1.0, size=n_samples)) / n_samples
    elif sampling.lower() in ("mc", "monte_carlo"):
        u_mat = rng.uniform(0.0, 1.0, size=(n_samples, n_dim))
    else:
        raise ValueError(f"Unknown sampling method {sampling!r} (expected 'lhs' or 'mc').")

    out: dict[int, np.ndarray] = {}
    for j, spec in enumerate(var_specs):
        base_v = _get_base_value(project, spec.param, spec.material_name)
        out[j] = _inverse_cdf_column(u_mat[:, j], base_v, spec)
    return out


def _apply_sample_to_specs(
    base_mat_specs: list[MaterialSpec3D],
    base_kh: float,
    base_kv: float,
    base_ru: float,
    var_specs: list[RandomVarSpec3D],
    sample_vals: dict[int, np.ndarray],
    idx: int,
) -> tuple[list[MaterialSpec3D], float, float, float]:
    mats = [copy.copy(m) for m in base_mat_specs]
    mat_map = {m.name: m for m in mats}
    kh, kv, ru = base_kh, base_kv, base_ru
    for j, spec in enumerate(var_specs):
        val = float(sample_vals[j][idx])
        if spec.material_name is not None:
            setattr(mat_map[spec.material_name], spec.param, val)
        elif spec.param == "kh":
            kh = val
        elif spec.param == "kv":
            kv = val
        elif spec.param == "ru":
            ru = val
    return mats, kh, kv, ru


def run_probabilistic_3d(
    project: Project3D,
    var_specs: list[RandomVarSpec3D],
    n_samples: int = 200,
    sampling: str = "lhs",
    fixed_surface: bool = True,
    surface: SlipSurface3D | None = None,
    method: str | None = None,
    seed: int = 42,
    progress_cb: Callable[[int, int], None] | None = None,
) -> ProbabilisticResult3D:
    """Execute 3D LHS or Monte Carlo probabilistic reliability analysis."""
    if n_samples < 4:
        raise ValueError("Require n_samples >= 4 for probabilistic analysis.")
    if not var_specs:
        raise ValueError("Provide at least one RandomVarSpec3D.")

    project.validate()
    an = project.analysis
    use_method = method or (
        project.surface_config.primary_method
        if project.surface_config.primary_method in an.methods
        else an.methods[0]
    )

    # Obtain deterministic baseline & critical slip surface
    if surface is None:
        det_res = run_analysis_3d(project)
        crit_surf = det_res.critical_surface
        det_fos = (
            det_res.method_results[use_method].fos
            if use_method in det_res.method_results
            else det_res.critical_fos
        )
    else:
        crit_surf = surface
        mdl0 = project.build_model()
        cols0 = discretize_columns(
            mdl0,
            crit_surf,
            project.build_materials(),
            opts=project.build_column_options(),
            material_colors=project.material_colors(),
        )
        det_fos = solve_surface_3d(
            cols0,
            method=use_method,
            tolerance=an.tolerance,
            max_iter=an.max_iter,
            m_alpha_min=an.m_alpha_min,
            interslice=an.gle_interslice,
        ).fos

    sample_cols = _generate_samples(project, var_specs, n_samples, sampling, seed)
    sample_inputs = {var_specs[j].label: sample_cols[j] for j in range(len(var_specs))}
    fos_arr = np.full(n_samples, np.nan, dtype=float)

    if fixed_surface:
        fast_model = project.build_model().bake_horizon_grids(nx=121, ny=121)
        re_discretize = any(s.param in ("gamma", "gamma_sat") for s in var_specs)
        base_opts = project.build_column_options()
        base_cols = discretize_columns(
            fast_model,
            crit_surf,
            project.build_materials(),
            opts=base_opts,
            material_colors=project.material_colors(),
        )

        for i in range(n_samples):
            m_specs, kh_i, kv_i, ru_i = _apply_sample_to_specs(
                project.materials, an.kh, an.kv, an.ru, var_specs, sample_cols, i
            )
            mats_i = apply_design_to_materials([ms.to_material() for ms in m_specs], project.design)
            if re_discretize:
                opts_i = copy.copy(base_opts)
                opts_i.kh = kh_i
                opts_i.kv = kv_i
                opts_i.ru = ru_i
                cols_i = discretize_columns(
                    fast_model,
                    crit_surf,
                    mats_i,
                    opts=opts_i,
                    material_colors=project.material_colors(),
                )
            else:
                cols_i = copy.copy(base_cols)
                cols_i.materials = mats_i
                cols_i.kh = kh_i
                cols_i.kv = kv_i
                if an.pore_mode == "ru" and abs(ru_i - an.ru) > 1e-12:
                    u_new = np.maximum(ru_i * cols_i.W / np.maximum(cols_i.dA, 1e-12), 0.0)
                    cols_i.u = u_new
                    cols_i.sigma_v_eff = np.maximum(
                        cols_i.W / np.maximum(cols_i.dA, 1e-12) - u_new, 0.0
                    )

            res_i = solve_surface_3d(
                cols_i,
                method=use_method,
                tolerance=an.tolerance,
                max_iter=an.max_iter,
                m_alpha_min=an.m_alpha_min,
                interslice=an.gle_interslice,
            )
            if res_i.ok:
                fos_arr[i] = res_i.fos
            if progress_cb is not None:
                progress_cb(i + 1, n_samples)
    else:
        for i in range(n_samples):
            p_i = project.copy()
            m_specs, kh_i, kv_i, ru_i = _apply_sample_to_specs(
                p_i.materials,
                p_i.analysis.kh,
                p_i.analysis.kv,
                p_i.analysis.ru,
                var_specs,
                sample_cols,
                i,
            )
            p_i.materials = m_specs
            p_i.analysis.kh = kh_i
            p_i.analysis.kv = kv_i
            p_i.analysis.ru = ru_i
            p_i.analysis.methods = [use_method]
            p_i.surface_config.primary_method = use_method
            res_full = run_analysis_3d(p_i)
            if res_full.primary_result.ok:
                fos_arr[i] = res_full.critical_fos
            if progress_cb is not None:
                progress_cb(i + 1, n_samples)

    valid = fos_arr[np.isfinite(fos_arr)]
    n_conv = int(len(valid))
    if n_conv == 0:
        return ProbabilisticResult3D(
            deterministic_fos=float(det_fos),
            samples_fos=fos_arr,
            sample_inputs=sample_inputs,
            mean_fos=np.nan,
            std_fos=np.nan,
            pf=np.nan,
            beta_normal=np.nan,
            beta_lognormal=np.nan,
            p5=np.nan,
            p50=np.nan,
            p95=np.nan,
            n_converged=0,
            n_total=n_samples,
            sampling=sampling,
            fixed_surface=fixed_surface,
        )

    mean_f = float(np.mean(valid))
    std_f = float(np.std(valid, ddof=1)) if n_conv > 1 else 0.0
    pf = float(np.count_nonzero(valid < 1.0) / n_conv)
    beta_norm = (mean_f - 1.0) / max(std_f, 1e-9) if std_f > 0.0 else float(np.inf)
    if mean_f > 0.0 and std_f > 0.0:
        cov_f = std_f / mean_f
        denom_ln = float(np.sqrt(np.log(1.0 + cov_f * cov_f)))
        beta_ln = float(np.log(mean_f / np.sqrt(1.0 + cov_f * cov_f)) / max(denom_ln, 1e-12))
    else:
        beta_ln = float(np.inf)

    p5, p50, p95 = (float(x) for x in np.percentile(valid, [5.0, 50.0, 95.0]))
    return ProbabilisticResult3D(
        deterministic_fos=float(det_fos),
        samples_fos=fos_arr,
        sample_inputs=sample_inputs,
        mean_fos=mean_f,
        std_fos=std_f,
        pf=pf,
        beta_normal=beta_norm,
        beta_lognormal=beta_ln,
        p5=p5,
        p50=p50,
        p95=p95,
        n_converged=n_conv,
        n_total=n_samples,
        sampling=sampling,
        fixed_surface=fixed_surface,
    )


def default_sensitivity_specs_3d(
    project: Project3D, rel_span: float = 0.25
) -> list[SensitivityParamSpec3D]:
    """Build automatic +-25% sensitivity sweep specifications from ``project`` materials & loads."""
    specs: list[SensitivityParamSpec3D] = []
    for m in project.materials:
        if m.model in ("MC", "AN"):
            if m.c > 0.0:
                specs.append(
                    SensitivityParamSpec3D(
                        "c", m.name, max(0.0, m.c * (1.0 - rel_span)), m.c * (1.0 + rel_span)
                    )
                )
            if m.phi > 0.0:
                specs.append(
                    SensitivityParamSpec3D(
                        "phi",
                        m.name,
                        max(1.0, m.phi * (1.0 - rel_span)),
                        min(75.0, m.phi * (1.0 + rel_span)),
                    )
                )
        elif m.model == "UD" and m.su > 0.0:
            specs.append(
                SensitivityParamSpec3D(
                    "su", m.name, max(1.0, m.su * (1.0 - rel_span)), m.su * (1.0 + rel_span)
                )
            )
        specs.append(
            SensitivityParamSpec3D(
                "gamma", m.name, max(10.0, m.gamma * (1.0 - 0.12)), m.gamma * (1.0 + 0.12)
            )
        )
    if project.analysis.pore_mode == "ru":
        r0 = project.analysis.ru
        specs.append(SensitivityParamSpec3D("ru", None, 0.0, min(0.6, max(0.35, r0 * 1.5))))
    specs.append(SensitivityParamSpec3D("kh", None, 0.0, max(0.25, project.analysis.kh + 0.20)))
    return specs


def run_sensitivity_3d(
    project: Project3D,
    specs: list[SensitivityParamSpec3D] | None = None,
    n_points: int = 9,
    surface: SlipSurface3D | None = None,
    method: str | None = None,
) -> list[SensitivityCurve3D]:
    """Run one-at-a-time 3D parameter sweeps and return curves sorted by tornado swing."""
    project.validate()
    use_specs = specs if specs is not None else default_sensitivity_specs_3d(project)
    n_pts = max(3, int(n_points))
    an = project.analysis
    use_method = method or (
        project.surface_config.primary_method
        if project.surface_config.primary_method in an.methods
        else an.methods[0]
    )

    if surface is None:
        det = run_analysis_3d(project)
        crit_surf = det.critical_surface
    else:
        crit_surf = surface

    fast_model = project.build_model().bake_horizon_grids(nx=121, ny=121)
    curves: list[SensitivityCurve3D] = []

    for sp in use_specs:
        base_v = _get_base_value(project, sp.param, sp.material_name)
        xs = np.linspace(sp.min_val, sp.max_val, n_pts)
        ys = np.full(n_pts, np.nan, dtype=float)
        for k, x_val in enumerate(xs):
            p_loc = project.copy()
            if sp.material_name is not None:
                for m in p_loc.materials:
                    if m.name == sp.material_name:
                        setattr(m, sp.param, float(x_val))
            else:
                setattr(p_loc.analysis, sp.param, float(x_val))
            cols_k = discretize_columns(
                fast_model,
                crit_surf,
                p_loc.build_materials(),
                opts=p_loc.build_column_options(),
                material_colors=p_loc.material_colors(),
            )
            res_k = solve_surface_3d(
                cols_k,
                method=use_method,
                tolerance=an.tolerance,
                max_iter=an.max_iter,
                m_alpha_min=an.m_alpha_min,
                interslice=an.gle_interslice,
            )
            if res_k.ok:
                ys[k] = res_k.fos

        valid_y = ys[np.isfinite(ys)]
        min_f = float(np.min(valid_y)) if len(valid_y) else np.nan
        max_f = float(np.max(valid_y)) if len(valid_y) else np.nan
        curves.append(
            SensitivityCurve3D(
                label=sp.label,
                param=sp.param,
                material_name=sp.material_name,
                base_value=base_v,
                values=xs,
                fos_values=ys,
                min_fos=min_f,
                max_fos=max_f,
            )
        )

    curves.sort(
        key=lambda c: c.delta_fos if np.isfinite(c.delta_fos) else -1.0,
        reverse=True,
    )
    return curves
