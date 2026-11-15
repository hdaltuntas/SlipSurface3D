"""Vertical-ray and segment intersection helpers for general 3D slip surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_VERTICAL_HITS = 8
NZ_FLOOR_MIN = 1e-3


@dataclass
class BaseSample:
    """Basal hit of a downward vertical ray used as a Method-of-Columns column base."""

    z: np.ndarray
    dz_dx: np.ndarray
    dz_dy: np.ndarray
    normal_xyz: np.ndarray
    n_hits: np.ndarray
    jacobian: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.z) & np.isfinite(self.jacobian) & (self.jacobian > 0.0)


def empty_hit_buffers(
    shape: tuple[int, ...], max_hits: int = MAX_VERTICAL_HITS
) -> tuple[np.ndarray, np.ndarray]:
    z_hits = np.full(shape + (max_hits,), np.nan, dtype=float)
    normals = np.full(shape + (max_hits, 3), np.nan, dtype=float)
    return z_hits, normals


def pack_hits(
    z_list: list[np.ndarray],
    n_list: list[np.ndarray],
    shape: tuple[int, ...],
    max_hits: int = MAX_VERTICAL_HITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack per-sheet (z, normal) arrays into descending-z buffers of width ``max_hits``."""
    z_hits, normals = empty_hit_buffers(shape, max_hits=max_hits)
    if not z_list:
        return z_hits, normals
    stacked_z = np.stack(z_list, axis=-1)
    stacked_n = np.stack(n_list, axis=-1)
    order = np.argsort(-np.where(np.isfinite(stacked_z), stacked_z, -np.inf), axis=-1)
    n_sheets = stacked_z.shape[-1]
    take = min(max_hits, n_sheets)
    for k in range(take):
        idx = order[..., k]
        z_hits[..., k] = np.take_along_axis(stacked_z, idx[..., None], axis=-1)[..., 0]
        for c in range(3):
            normals[..., k, c] = np.take_along_axis(stacked_n[..., c, :], idx[..., None], axis=-1)[
                ..., 0
            ]
    return z_hits, normals


def normals_to_slopes(normal_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an upward-ish unit normal into (dz/dx, dz/dy, Jacobian)."""
    nz = normal_xyz[..., 2]
    safe = np.abs(nz) > NZ_FLOOR_MIN
    zx = np.where(safe, -normal_xyz[..., 0] / nz, np.nan)
    zy = np.where(safe, -normal_xyz[..., 1] / nz, np.nan)
    jac = np.where(safe, 1.0 / nz, np.nan)
    return zx, zy, jac


def pick_floor(
    z_hits: np.ndarray,
    normals: np.ndarray,
    z_top: np.ndarray,
    z_min: np.ndarray,
    nz_min: float = NZ_FLOOR_MIN,
) -> BaseSample:
    """Walk hits from high to low; keep the first floor-like hit below ``z_top``."""
    shape = z_hits.shape[:-1]
    z_base = np.full(shape, np.nan, dtype=float)
    n_base = np.full(shape + (3,), np.nan, dtype=float)
    max_hits = z_hits.shape[-1]
    last_z = np.full(shape, np.inf, dtype=float)
    unique_hits = np.zeros(shape, dtype=int)
    for k in range(max_hits):
        zk = z_hits[..., k]
        nk = normals[..., k, :]
        nz = nk[..., 2]
        seen = np.isfinite(zk) & (np.abs(zk - last_z) < 1e-5)
        cand = (
            (~np.isfinite(z_base))
            & np.isfinite(zk)
            & np.isfinite(nz)
            & (zk < z_top - 1e-6)
            & (zk >= z_min - 1e-3)
            & (nz > nz_min)
            & (~seen)
        )
        z_base = np.where(cand, zk, z_base)
        n_base = np.where(cand[..., None], nk, n_base)
        unique_hits = unique_hits + (np.isfinite(zk) & (~seen)).astype(int)
        last_z = np.where(np.isfinite(zk), zk, last_z)
    n_hits = unique_hits
    zx, zy, jac = normals_to_slopes(n_base)
    return BaseSample(
        z=z_base,
        dz_dx=zx,
        dz_dy=zy,
        normal_xyz=n_base,
        n_hits=n_hits,
        jacobian=jac,
    )


def graph_hits_from_z(
    zb: np.ndarray,
    dz_dx: np.ndarray,
    dz_dy: np.ndarray,
    max_hits: int = MAX_VERTICAL_HITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a single-hit (floor) buffer from a heightfield z(x, y) and its slopes."""
    z_hits, normals = empty_hit_buffers(zb.shape, max_hits=max_hits)
    jac = np.sqrt(1.0 + dz_dx * dz_dx + dz_dy * dz_dy)
    valid = np.isfinite(zb) & np.isfinite(jac) & (jac > 0.0)
    z_hits[..., 0] = np.where(valid, zb, np.nan)
    normals[..., 0, 0] = np.where(valid, -dz_dx / jac, np.nan)
    normals[..., 0, 1] = np.where(valid, -dz_dy / jac, np.nan)
    normals[..., 0, 2] = np.where(valid, 1.0 / jac, np.nan)
    return z_hits, normals


def ray_triangle_hits(
    origins: np.ndarray,
    direction: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    t_min: float = 1e-9,
    t_max: float = np.inf,
    max_hits: int = MAX_VERTICAL_HITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect rays ``origin + t * direction`` with triangles; return (xyz, normals).

    ``origins`` is (N, 3). Hits are packed as (N, max_hits, 3) xyz and (N, max_hits, 3)
    normals (geometric, unflipped), sorted by increasing ``t``.
    """
    orig = np.asarray(origins, dtype=float)
    d = np.asarray(direction, dtype=float).reshape(3)
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=np.int64)
    n_ray = orig.shape[0]
    hit_xyz = np.full((n_ray, max_hits, 3), np.nan, dtype=float)
    hit_n = np.full((n_ray, max_hits, 3), np.nan, dtype=float)
    if n_ray == 0 or len(tris) == 0:
        return hit_xyz, hit_n

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    n_tri = np.cross(e1, e2)
    n_len = np.linalg.norm(n_tri, axis=1)
    good_tri = n_len > 1e-14
    n_unit = np.zeros_like(n_tri)
    n_unit[good_tri] = n_tri[good_tri] / n_len[good_tri, None]

    h = np.cross(np.broadcast_to(d, e2.shape), e2)
    a = np.einsum("ij,ij->i", e1, h)
    valid = good_tri & (np.abs(a) > 1e-12)
    if not np.any(valid):
        return hit_xyz, hit_n

    f = np.zeros(len(tris), dtype=float)
    f[valid] = 1.0 / a[valid]
    # Per-ray s = orig - v0  =>  (N, T, 3)
    s = orig[:, None, :] - v0[None, :, :]
    u = f[None, :] * np.einsum("nti,ti->nt", s, h)
    q = np.cross(s, e1[None, :, :])
    v = f[None, :] * np.einsum("nti,i->nt", q, d)
    t = f[None, :] * np.einsum("nti,ti->nt", q, e2)
    hit = (
        valid[None, :]
        & (u >= -1e-9)
        & (v >= -1e-9)
        & (u + v <= 1.0 + 1e-9)
        & (t > t_min)
        & (t < t_max)
    )
    t_store = np.where(hit, t, np.inf)
    order = np.argsort(t_store, axis=1)
    take = min(max_hits, t_store.shape[1])
    for k in range(take):
        j = order[:, k]
        tk = t_store[np.arange(n_ray), j]
        ok = np.isfinite(tk) & (tk < np.inf / 2.0)
        hit_xyz[ok, k, :] = orig[ok] + tk[ok, None] * d[None, :]
        hit_n[ok, k, :] = n_unit[j[ok]]
    return hit_xyz, hit_n


def orient_triangles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return ``faces`` rewound so normals point into the sliding mass.

    Triangle files carry no reliable winding convention, while floor/roof classification
    depends on the sign of ``n_z``. The winding is therefore normalised in three steps:

    1. Within each edge-connected component, windings are made mutually consistent, keeping
       the orientation of the (area-weighted) majority so consistent input is left alone.
    2. Closed (watertight) components are wound inward: the sliding mass is the interior.
    3. Open components are flipped together if their net area-weighted ``n_z`` is negative,
       so a bowl is a floor. Deliberately opposed sheets (floor + separate roof) are kept.
    """
    verts = np.asarray(vertices, dtype=float)
    tris = np.array(faces, dtype=np.int64, copy=True)
    n_tri = len(tris)
    if n_tri == 0:
        return tris

    # Undirected edge keys for the three half-edges of every triangle.
    a = tris[:, [0, 1, 2]].ravel()
    b = tris[:, [1, 2, 0]].ravel()
    he_tri = np.repeat(np.arange(n_tri), 3)
    key = np.minimum(a, b) * np.int64(len(verts)) + np.maximum(a, b)
    forward = a < b
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    grp_start = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
    grp_size = np.diff(np.r_[grp_start, len(key_s)])
    size_per_he = np.empty(len(key), dtype=np.int64)
    size_per_he[order] = np.repeat(grp_size, grp_size)

    # Manifold edges (exactly two triangles) drive the consistency propagation.
    pair_start = grp_start[grp_size == 2]
    h1 = order[pair_start]
    h2 = order[pair_start + 1]
    t1, t2 = he_tri[h1], he_tri[h2]
    # Consistent neighbours traverse the shared edge in opposite directions.
    mismatch = forward[h1] == forward[h2]

    nbr_src = np.r_[t1, t2]
    nbr_dst = np.r_[t2, t1]
    nbr_mis = np.r_[mismatch, mismatch]
    adj_order = np.argsort(nbr_src, kind="stable")
    nbr_dst = nbr_dst[adj_order]
    nbr_mis = nbr_mis[adj_order]
    adj_start = np.r_[0, np.cumsum(np.bincount(nbr_src, minlength=n_tri))]

    comp = np.full(n_tri, -1, dtype=np.int64)
    flip = np.zeros(n_tri, dtype=bool)
    n_comp = 0
    for seed in range(n_tri):
        if comp[seed] >= 0:
            continue
        comp[seed] = n_comp
        stack = [seed]
        while stack:
            cur = stack.pop()
            for j in range(adj_start[cur], adj_start[cur + 1]):
                nb = int(nbr_dst[j])
                if comp[nb] < 0:
                    comp[nb] = n_comp
                    flip[nb] = flip[cur] ^ bool(nbr_mis[j])
                    stack.append(nb)
        n_comp += 1

    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area2 = np.linalg.norm(cross, axis=1)

    # Step 1: keep the majority winding of each component.
    flipped_area = np.bincount(comp, weights=np.where(flip, area2, 0.0), minlength=n_comp)
    total_area = np.bincount(comp, weights=area2, minlength=n_comp)
    flip ^= (flipped_area > 0.5 * total_area)[comp]
    sign = np.where(flip, -1.0, 1.0)

    # Step 2: closed components are wound inward (negative signed volume).
    tri_closed = np.all(size_per_he.reshape(n_tri, 3) == 2, axis=1)
    comp_closed = np.bincount(comp, weights=(~tri_closed).astype(float), minlength=n_comp) == 0
    ref = verts.mean(axis=0)
    vol6 = sign * np.einsum("ij,ij->i", v0 - ref, np.cross(v1 - ref, v2 - ref))
    comp_vol = np.bincount(comp, weights=vol6, minlength=n_comp)
    flip ^= (comp_closed & (comp_vol > 0.0))[comp]

    # Step 3: open components share one global flip towards a net upward (floor) normal.
    is_open = ~comp_closed[comp]
    if np.any(is_open):
        sign = np.where(flip, -1.0, 1.0)
        net_nz = float(np.sum(sign[is_open] * cross[is_open, 2]))
        scale = float(np.sum(area2[is_open]))
        if net_nz < -1e-9 * max(scale, 1e-300):
            flip ^= is_open

    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


class VerticalMeshIndex:
    """Uniform (x, y) bucket grid over a triangle mesh for fast vertical-ray queries.

    A brute-force ray/triangle test allocates ``rays x triangles`` work arrays, which does
    not scale to imported STL/OBJ meshes. Each ray here is only tested against the
    triangles whose (x, y) bounding box overlaps the ray's grid cell.
    """

    MAX_PAIRS_PER_CHUNK = 2_000_000

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        verts = np.asarray(vertices, dtype=float)
        tris = np.asarray(faces, dtype=np.int64)
        v0 = verts[tris[:, 0]]
        e1 = verts[tris[:, 1]] - v0
        e2 = verts[tris[:, 2]] - v0
        n_tri = np.cross(e1, e2)
        n_len = np.linalg.norm(n_tri, axis=1)
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        # Degenerate and vertical triangles cannot be hit by a vertical ray.
        keep = (n_len > 1e-14) & (np.abs(det) > 1e-12)
        self.v0 = v0[keep]
        self.e1 = e1[keep]
        self.e2 = e2[keep]
        self.inv_det = 1.0 / det[keep]
        self.n_unit = n_tri[keep] / n_len[keep, None]
        n_keep = int(np.count_nonzero(keep))

        tri_xy = verts[tris[keep]][:, :, :2] if n_keep else np.zeros((0, 3, 2))
        lo = tri_xy.min(axis=1) if n_keep else np.zeros((0, 2))
        hi = tri_xy.max(axis=1) if n_keep else np.zeros((0, 2))
        self.x0 = float(np.min(lo[:, 0])) if n_keep else 0.0
        self.y0 = float(np.min(lo[:, 1])) if n_keep else 0.0
        x1 = float(np.max(hi[:, 0])) if n_keep else 1.0
        y1 = float(np.max(hi[:, 1])) if n_keep else 1.0
        self.n_cells = int(np.clip(np.sqrt(max(n_keep, 1) / 2.0), 1, 512))
        self.cell_x = max(x1 - self.x0, 1e-12) / self.n_cells
        self.cell_y = max(y1 - self.y0, 1e-12) / self.n_cells
        self.x1, self.y1 = x1, y1

        g = self.n_cells
        ix0 = self._cell(lo[:, 0], self.x0, self.cell_x)
        ix1 = self._cell(hi[:, 0], self.x0, self.cell_x)
        iy0 = self._cell(lo[:, 1], self.y0, self.cell_y)
        iy1 = self._cell(hi[:, 1], self.y0, self.cell_y)
        ny_span = iy1 - iy0 + 1
        cnt = (ix1 - ix0 + 1) * ny_span
        tri_rep = np.repeat(np.arange(n_keep), cnt)
        local = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ny_rep = ny_span[tri_rep]
        cell = (ix0[tri_rep] + local // ny_rep) * g + iy0[tri_rep] + local % ny_rep
        order = np.argsort(cell, kind="stable")
        self.tri_ids = tri_rep[order]
        self.cell_count = np.bincount(cell, minlength=g * g)
        self.cell_start = np.cumsum(self.cell_count) - self.cell_count

    def _cell(self, value: np.ndarray, origin: float, size: float) -> np.ndarray:
        idx = np.floor((np.asarray(value, dtype=float) - origin) / size).astype(np.int64)
        return np.clip(idx, 0, self.n_cells - 1)

    def hits(
        self, x: np.ndarray, y: np.ndarray, max_hits: int = MAX_VERTICAL_HITS
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vertical-line hits at (x, y): ``(z_hits, normals)`` sorted by descending z."""
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        shape = xa.shape
        xf, yf = xa.ravel(), ya.ravel()
        n_ray = xf.size
        z_flat = np.full((n_ray, max_hits), np.nan, dtype=float)
        n_flat = np.full((n_ray, max_hits, 3), np.nan, dtype=float)
        if n_ray == 0 or len(self.tri_ids) == 0:
            return z_flat.reshape(shape + (max_hits,)), n_flat.reshape(shape + (max_hits, 3))

        with np.errstate(invalid="ignore"):
            inside = (
                (xf >= self.x0 - 1e-9)
                & (xf <= self.x1 + 1e-9)
                & (yf >= self.y0 - 1e-9)
                & (yf <= self.y1 + 1e-9)
            )
        cell = np.zeros(n_ray, dtype=np.int64)
        cell[inside] = self._cell(xf[inside], self.x0, self.cell_x) * self.n_cells + self._cell(
            yf[inside], self.y0, self.cell_y
        )
        cnt = np.where(inside, self.cell_count[cell], 0)
        csum = np.cumsum(cnt)

        hit_ray: list[np.ndarray] = []
        hit_z: list[np.ndarray] = []
        hit_tri: list[np.ndarray] = []
        start = 0
        while start < n_ray:
            done = int(csum[start - 1]) if start else 0
            end = int(np.searchsorted(csum, done + self.MAX_PAIRS_PER_CHUNK, side="right"))
            end = min(max(end, start + 1), n_ray)
            c = cnt[start:end]
            if int(c.sum()) > 0:
                ray = np.repeat(np.arange(start, end), c)
                off = np.arange(len(ray)) - np.repeat(np.cumsum(c) - c, c)
                tri = self.tri_ids[self.cell_start[cell[ray]] + off]
                px = xf[ray] - self.v0[tri, 0]
                py = yf[ray] - self.v0[tri, 1]
                e1, e2, inv = self.e1[tri], self.e2[tri], self.inv_det[tri]
                u = (px * e2[:, 1] - py * e2[:, 0]) * inv
                v = (e1[:, 0] * py - e1[:, 1] * px) * inv
                ok = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
                hit_ray.append(ray[ok])
                hit_tri.append(tri[ok])
                hit_z.append(self.v0[tri[ok], 2] + u[ok] * e1[ok, 2] + v[ok] * e2[ok, 2])
            start = end

        if hit_ray:
            ray_all = np.concatenate(hit_ray)
            z_all = np.concatenate(hit_z)
            tri_all = np.concatenate(hit_tri)
            order = np.lexsort((-z_all, ray_all))
            ray_s = ray_all[order]
            rank = np.arange(len(ray_s)) - np.searchsorted(ray_s, ray_s, side="left")
            keep = rank < max_hits
            z_flat[ray_s[keep], rank[keep]] = z_all[order][keep]
            n_flat[ray_s[keep], rank[keep]] = self.n_unit[tri_all[order][keep]]
        return z_flat.reshape(shape + (max_hits,)), n_flat.reshape(shape + (max_hits, 3))


def vertical_mesh_hits(
    x: np.ndarray,
    y: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    max_hits: int = MAX_VERTICAL_HITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertical-line intersections with a triangle mesh, sorted by descending z.

    Normals are the geometric triangle normals: ``n_z > 0`` is a floor, ``n_z < 0`` a roof.
    Pass faces through ``orient_triangles`` first if the winding is not trusted. Callers
    that query repeatedly should build one ``VerticalMeshIndex`` and reuse it.
    """
    return VerticalMeshIndex(vertices, faces).hits(x, y, max_hits=max_hits)
