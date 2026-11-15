"""3D CAD / Mesh Exchange: DXF (3DFACE) and Wavefront OBJ export & import."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ezdxf
import numpy as np
import pyvista as pv

from slipsurface3d.geometry.surfaces import MeshSlipSurface
from slipsurface3d.geometry.terrain import HorizonSurface, SlopeModel3D


def _polydata_triangles(mesh: pv.DataSet) -> tuple[np.ndarray, np.ndarray]:
    """Extract (vertices [N, 3], triangle_indices [M, 3]) from any PyVista dataset."""
    surf = mesh.extract_surface().triangulate()
    if surf.n_points == 0 or surf.n_cells == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)
    pts = np.asarray(surf.points, dtype=float)
    faces_raw = np.asarray(surf.faces, dtype=np.int64).reshape(-1, 4)
    tris = faces_raw[:, 1:4]
    return pts, tris


def export_obj_mesh(path: str | Path, meshes: list[tuple[str, pv.DataSet]]) -> Path:
    """Export one or more named PyVista meshes to a Wavefront .obj file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# SlipSurfaceIQ-3D Wavefront OBJ Export"]
    v_offset = 1
    for group_name, dataset in meshes:
        pts, tris = _polydata_triangles(dataset)
        if len(pts) == 0 or len(tris) == 0:
            continue
        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in group_name)
        lines.append(f"g {safe_name}")
        for vx, vy, vz in pts:
            lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")
        for i0, i1, i2 in tris:
            lines.append(f"f {i0 + v_offset} {i1 + v_offset} {i2 + v_offset}")
        v_offset += len(pts)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def import_obj_vertices(path: str | Path) -> np.ndarray:
    """Read 3D vertex coordinates (N, 3) from a Wavefront .obj file."""
    pts: list[list[float]] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(pts) < 3:
        raise ValueError(f"OBJ file '{path}' contains fewer than 3 vertices.")
    return np.asarray(pts, dtype=float)


def import_obj_horizon(path: str | Path, name: str = "Imported OBJ Terrain") -> HorizonSurface:
    pts = import_obj_vertices(path)
    return HorizonSurface.from_scattered(pts, name=name)


def import_obj_slip_surface(path: str | Path) -> MeshSlipSurface:
    pts = import_obj_vertices(path)
    return MeshSlipSurface(vertices=pts)


def export_dxf_3d(
    path: str | Path,
    model: SlopeModel3D,
    critical_mesh: pv.DataSet | None = None,
    nx: int = 36,
    ny: int = 28,
) -> Path:
    """Export 3D ground surface, water table, and critical slip surface as DXF 3DFACE entities."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    layers_spec = [
        ("GROUND_3D", 3),
        ("WATER_3D", 5),
        ("CRITICAL_SLIP_3D", 1),
    ]
    for lname, color in layers_spec:
        if lname not in doc.layers:
            doc.layers.add(lname, color=color)

    def _add_mesh_faces(dataset: pv.DataSet, layer: str) -> None:
        pts, tris = _polydata_triangles(dataset)
        for i0, i1, i2 in tris:
            p0 = tuple(float(v) for v in pts[i0])
            p1 = tuple(float(v) for v in pts[i1])
            p2 = tuple(float(v) for v in pts[i2])
            msp.add_3dface([p0, p1, p2, p2], dxfattribs={"layer": layer})

    _add_mesh_faces(model.ground_mesh(nx=nx, ny=ny), "GROUND_3D")
    wt_m = model.water_mesh(nx=max(16, nx // 2), ny=max(12, ny // 2))
    if wt_m is not None:
        _add_mesh_faces(wt_m, "WATER_3D")
    if critical_mesh is not None and critical_mesh.n_points > 0:
        _add_mesh_faces(critical_mesh, "CRITICAL_SLIP_3D")

    doc.saveas(str(target))
    return target


def import_dxf_3d_vertices(path: str | Path, layer: str | None = None) -> np.ndarray:
    """Extract 3D vertex cloud (N, 3) from 3DFACE, MESH, POLYFACE, POINT, or LWPOLYLINE/POLYLINE in DXF."""
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    pts: list[list[float]] = []

    for ent in msp:
        if layer is not None and ent.dxf.layer != layer:
            continue
        etype = ent.dxftype()
        if etype == "3DFACE":
            for attr in ("vtx0", "vtx1", "vtx2", "vtx3"):
                if ent.dxf.hasattr(attr):
                    v = getattr(ent.dxf, attr)
                    pts.append([float(v.x), float(v.y), float(v.z)])
        elif etype == "POINT":
            loc = ent.dxf.location
            pts.append([float(loc.x), float(loc.y), float(loc.z)])
        elif etype == "MESH":
            ent_any: Any = ent
            for v in ent_any.vertices:
                pts.append([float(v[0]), float(v[1]), float(v[2])])
        elif etype == "POLYLINE":
            ent_poly: Any = ent
            for v in ent_poly.vertices:
                loc = v.dxf.location
                pts.append([float(loc.x), float(loc.y), float(loc.z)])

    if len(pts) < 3:
        raise ValueError(f"DXF file '{path}' contains fewer than 3 3D points.")
    arr = np.unique(np.round(np.asarray(pts, dtype=float), 6), axis=0)
    return arr
