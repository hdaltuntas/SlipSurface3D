"""3D CAD, GIS & Mesh Exchange: DXF (3DFACE), Wavefront OBJ, Binary/ASCII STL, ESRI ASCII DEM, and XYZ/CSV."""

from __future__ import annotations

import re
import struct
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


def export_stl_mesh(
    path: str | Path,
    meshes: list[tuple[str, pv.DataSet]] | pv.DataSet,
    binary: bool = True,
) -> Path:
    """Export one or more PyVista meshes to a Binary or ASCII .stl stereolithography file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    items = [("Mesh", meshes)] if isinstance(meshes, pv.DataSet) else list(meshes)

    tri_blocks: list[np.ndarray] = []
    for _, ds in items:
        pts, tris = _polydata_triangles(ds)
        if len(pts) > 0 and len(tris) > 0:
            tri_blocks.append(pts[tris])  # (M, 3, 3)

    all_tris = np.vstack(tri_blocks) if tri_blocks else np.zeros((0, 3, 3), dtype=float)
    n_tri = len(all_tris)
    if n_tri > 0:
        e1 = all_tris[:, 1, :] - all_tris[:, 0, :]
        e2 = all_tris[:, 2, :] - all_tris[:, 0, :]
        normals = np.cross(e1, e2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norms > 1e-12, normals / np.maximum(norms, 1e-12), 0.0)
    else:
        normals = np.zeros((0, 3), dtype=float)

    if binary:
        stl_dtype = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("v0", "<f4", (3,)),
                ("v1", "<f4", (3,)),
                ("v2", "<f4", (3,)),
                ("attr", "<u2"),
            ]
        )
        recs = np.zeros(n_tri, dtype=stl_dtype)
        if n_tri > 0:
            recs["normal"] = normals.astype(np.float32)
            recs["v0"] = all_tris[:, 0, :].astype(np.float32)
            recs["v1"] = all_tris[:, 1, :].astype(np.float32)
            recs["v2"] = all_tris[:, 2, :].astype(np.float32)
        header = b"SlipSurfaceIQ-3D Binary STL Export".ljust(80, b"\x00")
        with target.open("wb") as fh:
            fh.write(header)
            fh.write(struct.pack("<I", n_tri))
            fh.write(recs.tobytes())
    else:
        lines = ["solid SlipSurface3D"]
        for k in range(n_tri):
            nx_v, ny_v, nz_v = normals[k]
            lines.append(f"  facet normal {nx_v:.6e} {ny_v:.6e} {nz_v:.6e}")
            lines.append("    outer loop")
            for v in all_tris[k]:
                lines.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append("endsolid SlipSurface3D")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def import_stl_vertices(path: str | Path) -> np.ndarray:
    """Read unique 3D vertex coordinates (N, 3) from either a Binary or ASCII .stl file."""
    raw_bytes = Path(path).read_bytes()
    if len(raw_bytes) >= 84:
        n_tri_bin = struct.unpack("<I", raw_bytes[80:84])[0]
        expected_size = 84 + 50 * n_tri_bin
        is_ascii_solid = raw_bytes[:5].lower() == b"solid" and b"facet" in raw_bytes[:512]
        if len(raw_bytes) == expected_size and n_tri_bin > 0 and not is_ascii_solid:
            stl_dtype = np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("v0", "<f4", (3,)),
                    ("v1", "<f4", (3,)),
                    ("v2", "<f4", (3,)),
                    ("attr", "<u2"),
                ]
            )
            recs = np.frombuffer(raw_bytes, dtype=stl_dtype, offset=84, count=n_tri_bin)
            pts = np.vstack([recs["v0"], recs["v1"], recs["v2"]]).astype(float)
            uniq = np.unique(np.round(pts, 6), axis=0)
            if len(uniq) < 3:
                raise ValueError(f"Binary STL '{path}' contains fewer than 3 unique vertices.")
            return uniq

    # ASCII STL fallback
    pts_list: list[list[float]] = []
    text = raw_bytes.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("vertex"):
            parts = s.split()
            if len(parts) >= 4:
                pts_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(pts_list) < 3:
        raise ValueError(f"STL file '{path}' contains fewer than 3 vertices.")
    return np.unique(np.round(np.asarray(pts_list, dtype=float), 6), axis=0)


def _points_to_horizon(pts: np.ndarray, name: str, max_rbf_pts: int = 800) -> HorizonSurface:
    """Convert point cloud (N, 3) to HorizonSurface (direct RBF for <=800 pts, binned grid for large clouds)."""
    if len(pts) <= max_rbf_pts:
        return HorizonSurface.from_scattered(pts, name=name)
    x0, x1 = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    y0, y1 = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    nx = 64
    ny = 64
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    ix = np.clip(np.searchsorted(xs, pts[:, 0]) - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(ys, pts[:, 1]) - 1, 0, ny - 1)
    z_sum = np.zeros((nx, ny), dtype=float)
    cnt = np.zeros((nx, ny), dtype=int)
    np.add.at(z_sum, (ix, iy), pts[:, 2])
    np.add.at(cnt, (ix, iy), 1)
    populated = cnt > 0
    if int(np.count_nonzero(populated)) == nx * ny:
        return HorizonSurface.from_grid(xs, ys, z_sum / cnt, name=name)
    # Subsample populated cell centroids for smooth RBF fill
    ii, jj = np.nonzero(populated)
    sub_pts = np.column_stack([xs[ii], ys[jj], z_sum[ii, jj] / cnt[ii, jj]])
    if len(sub_pts) > max_rbf_pts:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(sub_pts), size=max_rbf_pts, replace=False)
        sub_pts = sub_pts[keep]
    return HorizonSurface.from_scattered(sub_pts, name=name)


def import_stl_horizon(path: str | Path, name: str = "Imported STL Terrain") -> HorizonSurface:
    pts = import_stl_vertices(path)
    return _points_to_horizon(pts, name=name)


def import_stl_slip_surface(path: str | Path) -> MeshSlipSurface:
    pts = import_stl_vertices(path)
    return MeshSlipSurface(vertices=pts)


def export_esri_ascii_dem(
    path: str | Path,
    horizon: HorizonSurface,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    cellsize: float = 1.0,
    nodata: float = -9999.0,
) -> Path:
    """Export a ``HorizonSurface`` to standard ESRI ASCII Raster Grid (`.asc`) DEM format."""
    if cellsize <= 0.0:
        raise ValueError("DEM cellsize must be strictly positive.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    x0, x1 = float(x_bounds[0]), float(x_bounds[1])
    y0, y1 = float(y_bounds[0]), float(y_bounds[1])
    ncols = max(2, int(np.round((x1 - x0) / cellsize)) + 1)
    nrows = max(2, int(np.round((y1 - y0) / cellsize)) + 1)
    xs = x0 + np.arange(ncols, dtype=float) * cellsize
    ys = y0 + np.arange(nrows, dtype=float) * cellsize
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = np.where(np.isfinite(horizon.z(xx, yy)), horizon.z(xx, yy), nodata)

    lines = [
        f"ncols         {ncols}",
        f"nrows         {nrows}",
        f"xllcenter     {x0:.6f}",
        f"yllcenter     {y0:.6f}",
        f"cellsize      {cellsize:.6f}",
        f"NODATA_value  {nodata:.6f}",
    ]
    # ESRI ASCII rows run from North (top = max Y) to South (bottom = min Y)
    for r in range(nrows - 1, -1, -1):
        row_vals = " ".join(f"{zz[c, r]:.4f}" for c in range(ncols))
        lines.append(row_vals)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def import_esri_ascii_dem(path: str | Path, name: str = "Imported DEM Grid") -> HorizonSurface:
    """Import an ESRI ASCII Raster Grid (`.asc`) DEM into a regular-grid ``HorizonSurface``."""
    raw_lines = [
        ln.strip()
        for ln in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        if ln.strip()
    ]
    header: dict[str, float] = {}
    idx = 0
    while idx < len(raw_lines):
        parts = raw_lines[idx].split()
        if len(parts) == 2 and parts[0].lower() in (
            "ncols",
            "nrows",
            "xllcorner",
            "yllcorner",
            "xllcenter",
            "yllcenter",
            "cellsize",
            "dx",
            "dy",
            "nodata_value",
        ):
            header[parts[0].lower()] = float(parts[1])
            idx += 1
        else:
            break

    ncols = int(header.get("ncols", 0))
    nrows = int(header.get("nrows", 0))
    if ncols < 2 or nrows < 2:
        raise ValueError(
            f"Invalid ESRI ASCII DEM header in '{path}' (ncols={ncols}, nrows={nrows})."
        )
    cellsize = float(header.get("cellsize", header.get("dx", 1.0)))
    nodata = header.get("nodata_value", -9999.0)
    x0 = (
        float(header["xllcenter"])
        if "xllcenter" in header
        else float(header.get("xllcorner", 0.0)) + 0.5 * cellsize
    )
    y0 = (
        float(header["yllcenter"])
        if "yllcenter" in header
        else float(header.get("yllcorner", 0.0)) + 0.5 * cellsize
    )

    vals_str = " ".join(raw_lines[idx:]).split()
    if len(vals_str) < ncols * nrows:
        raise ValueError(f"ESRI ASCII DEM '{path}' has fewer data values than ncols * nrows.")
    arr_ns = np.asarray([float(v) for v in vals_str[: ncols * nrows]], dtype=float).reshape(
        (nrows, ncols)
    )
    # Flip North->South rows to South->North (+Y) and transpose to (nx, ny)
    z_grid = arr_ns[::-1, :].T.copy()
    nodata_mask = np.isclose(z_grid, nodata) | (~np.isfinite(z_grid))
    if np.any(nodata_mask):
        if np.all(nodata_mask):
            raise ValueError(f"ESRI ASCII DEM '{path}' contains only NODATA values.")
        fill_v = float(np.mean(z_grid[~nodata_mask]))
        z_grid[nodata_mask] = fill_v

    xs = x0 + np.arange(ncols, dtype=float) * cellsize
    ys = y0 + np.arange(nrows, dtype=float) * cellsize
    return HorizonSurface.from_grid(xs, ys, z_grid, name=name)


def export_xyz_csv_horizon(
    path: str | Path,
    horizon: HorizonSurface,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    nx: int = 41,
    ny: int = 31,
) -> Path:
    """Export a ``HorizonSurface`` as a CSV ``x,y,z`` point cloud."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(x_bounds[0], x_bounds[1], max(2, nx))
    ys = np.linspace(y_bounds[0], y_bounds[1], max(2, ny))
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = horizon.z(xx, yy)
    lines = ["x,y,z"]
    for xv, yv, zv in zip(xx.ravel(), yy.ravel(), zz.ravel(), strict=False):
        if np.isfinite(zv):
            lines.append(f"{xv:.6f},{yv:.6f},{zv:.6f}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def import_xyz_csv_vertices(path: str | Path) -> np.ndarray:
    """Import scattered 3D ``(x, y, z)`` points from a `.csv`, `.xyz`, or `.txt` survey file."""
    pts: list[list[float]] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = [tok for tok in re.split(r"[,;\s]+", line) if tok]
        if len(tokens) < 3:
            continue
        try:
            pts.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])
        except ValueError:
            # Header row such as "x,y,z"
            continue
    if len(pts) < 3:
        raise ValueError(f"XYZ/CSV file '{path}' contains fewer than 3 valid (x, y, z) points.")
    return np.asarray(pts, dtype=float)


def import_xyz_csv_horizon(
    path: str | Path, name: str = "Imported XYZ/CSV Point Cloud"
) -> HorizonSurface:
    pts = import_xyz_csv_vertices(path)
    return _points_to_horizon(pts, name=name)


def import_xyz_csv_slip_surface(path: str | Path) -> MeshSlipSurface:
    pts = import_xyz_csv_vertices(path)
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
