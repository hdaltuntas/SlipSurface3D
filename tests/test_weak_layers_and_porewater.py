"""Tests for weak layers along arbitrary surfaces, per-material piezometric surfaces,
and scattered pore-pressure grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slipsurface3d.analysis import run_analysis_3d
from slipsurface3d.geometry import (
    EllipsoidSurface,
    HorizonSurface,
    WeakLayer3D,
    WeakLayerClippedSurface,
    surface_from_dict,
)
from slipsurface3d.porewater import PorePressureGrid3D, read_pore_pressure_csv
from slipsurface3d.project import (
    Project3D,
    zhang_1988_homog_ellipsoid_project,
    zhang_1988_weak_seam_project,
)
from slipsurface3d.scripting import create_simple_slope_project


def _wet_project() -> Project3D:
    proj = create_simple_slope_project(methods=("bishop",))
    proj.materials[0].use_water = True
    proj.surface_config.mode = "user"
    proj.surface_config.user_surface = EllipsoidSurface(
        xc=0.45 * (proj.x_min + proj.x_max), yc=20.0, zc=22.0, rx=16.0, ry=14.0, rz=16.0
    ).to_dict()
    return proj


def _flat(proj: Project3D, z0: float) -> HorizonSurface:
    return HorizonSurface.from_plane((proj.x_min, proj.x_max), (proj.y_min, proj.y_max), z0)


def test_weak_layer_matches_the_composite_ellipsoid_plane_benchmark():
    # Reference: explicit composite surface over a thin weak-seam stratum.
    ref = zhang_1988_weak_seam_project()
    ref.analysis.nx = ref.analysis.ny = 40
    seam = ref.surface_config.user_surface
    assert seam is not None and seam["kind"] == "composite_ellipsoid_plane"
    ref.layers[1] = (
        ref.layers[1][0],
        HorizonSurface.from_extruded([[ref.x_min, 1e-3], [ref.x_max, 1e-3]], name="Seam Top"),
    )
    expected = run_analysis_3d(ref)

    # Same problem posed as a plain ellipsoid in a homogeneous slope plus a weak layer.
    proj = zhang_1988_homog_ellipsoid_project()
    proj.analysis.nx = proj.analysis.ny = 40
    proj.analysis.methods = list(ref.analysis.methods)
    proj.materials.append(ref.materials[-1])
    proj.weak_layers = [
        WeakLayer3D("Seam", ref.materials[-1].name, _flat(proj, float(seam["plane_z0"])))
    ]
    proj.surface_config.mode = "user"
    proj.surface_config.user_surface = seam["ellipsoid"]
    got = run_analysis_3d(proj)

    assert isinstance(got.critical_surface, WeakLayerClippedSurface)
    for method in ("bishop", "spencer"):
        assert got.method_results[method].fos == pytest.approx(
            expected.method_results[method].fos, rel=2e-3
        )
    weak_idx = len(proj.materials) - 1
    on_layer = got.columns.mat_idx == weak_idx
    assert np.any(on_layer) and not np.all(on_layer)
    assert np.allclose(got.columns.z_base[on_layer], 0.0, atol=1e-6)

    restored = Project3D.from_dict(proj.to_dict())
    assert restored.weak_layers[0].material == ref.materials[-1].name
    clipped = surface_from_dict(got.critical_surface.to_dict())
    assert isinstance(clipped, WeakLayerClippedSurface)

    proj.weak_layers[0].material = "Missing"
    with pytest.raises(ValueError, match="Weak layer"):
        proj.validate()


def test_weak_layer_follows_a_dipping_imported_style_surface():
    proj = _wet_project()
    proj.materials[0].use_water = False
    weak = proj.materials[0].__class__(name="Slick", model="MC", c=0.0, phi=12.0, use_water=False)
    proj.materials.append(weak)
    layer = HorizonSurface.from_plane(
        (proj.x_min, proj.x_max), (proj.y_min, proj.y_max), 9.0, dip_deg=8.0, dip_dir_deg=180.0
    )
    proj.weak_layers = [WeakLayer3D("Bedding", "Slick", layer)]
    res = run_analysis_3d(proj)
    assert res.primary_result.ok
    on_layer = res.columns.mat_idx == 1
    assert np.any(on_layer)
    cols = res.columns
    assert np.allclose(
        cols.z_base[on_layer], layer.z(cols.x[on_layer], cols.y[on_layer]), atol=1e-6
    )
    # The base slope on the layer is the layer's own dip (8 deg towards -X).
    assert np.allclose(cols.dz_dx[on_layer], np.tan(np.radians(8.0)), atol=1e-6)


def test_piezometric_surface_and_pore_grid_reproduce_hydrostatic_pressure(tmp_path: Path):
    proj = _wet_project()
    dry = run_analysis_3d(proj)
    assert float(np.max(dry.columns.u)) == 0.0

    level = 10.0
    proj.piezo_surfaces = [("Perched", _flat(proj, level))]
    proj.materials[0].piezo = "Perched"
    piezo = run_analysis_3d(proj)
    cols = piezo.columns
    expected_u = proj.analysis.gamma_w * np.maximum(level - cols.z_base, 0.0)
    assert np.allclose(cols.u, expected_u, atol=1e-9)
    assert piezo.critical_fos < dry.critical_fos
    assert Project3D.from_dict(proj.to_dict()).piezo_surfaces[0][0] == "Perched"

    # The same pressure field supplied as scattered samples gives the same answer.
    xs = np.linspace(proj.x_min, proj.x_max, 7)
    ys = np.linspace(proj.y_min, proj.y_max, 5)
    zs = np.linspace(proj.z_min, 14.0, 13)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    samples = np.column_stack(
        [
            gx.ravel(),
            gy.ravel(),
            gz.ravel(),
            proj.analysis.gamma_w * np.maximum(level - gz.ravel(), 0),
        ]
    )
    csv_path = tmp_path / "pore.csv"
    np.savetxt(csv_path, samples, delimiter=",", header="x,y,z,u", comments="")
    proj.piezo_surfaces = []
    proj.materials[0].piezo = ""
    proj.pore_grid = read_pore_pressure_csv(csv_path)
    proj.analysis.pore_mode = "grid"
    grid = run_analysis_3d(proj)
    # Identical basal pore pressures. (The FoS differs slightly: a piezometric surface
    # standing above the slope face also ponds water on it, which a pressure grid cannot.)
    assert np.allclose(grid.columns.u, cols.u, atol=1e-6)
    assert grid.critical_fos == pytest.approx(piezo.critical_fos, rel=2e-2)
    assert Project3D.from_dict(proj.to_dict()).pore_grid == proj.pore_grid

    proj.pore_grid = []
    with pytest.raises(ValueError, match="pore-pressure samples"):
        proj.validate()
    proj.analysis.pore_mode = "water_table"
    proj.materials[0].piezo = "Nowhere"
    with pytest.raises(ValueError, match="piezometric surface"):
        proj.validate()


def test_pore_pressure_grid_falls_back_to_nearest_sample():
    coplanar = [[0.0, 0.0, 0.0, 10.0], [1.0, 0.0, 0.0, 20.0], [0.0, 1.0, 0.0, 30.0]]
    grid = PorePressureGrid3D(coplanar)
    assert float(grid(0.9, 0.1, 5.0)) == 20.0
    assert float(grid(-50.0, -50.0, 0.0)) == 10.0
    with pytest.raises(ValueError):
        PorePressureGrid3D([[0.0, 0.0, 0.0]])
