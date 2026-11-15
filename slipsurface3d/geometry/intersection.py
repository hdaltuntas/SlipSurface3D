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


def vertical_mesh_hits(
    x: np.ndarray,
    y: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    z_ray: float,
    max_hits: int = MAX_VERTICAL_HITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Downward vertical rays from ``z_ray`` through a triangle mesh.

    Normals are flipped so they point into the sliding mass (n_z > 0 on floors).
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    shape = xa.shape
    orig = np.column_stack([xa.ravel(), ya.ravel(), np.full(xa.size, float(z_ray))])
    xyz, nrm = ray_triangle_hits(
        orig, np.array([0.0, 0.0, -1.0]), vertices, faces, max_hits=max_hits
    )
    z_hits, normals = empty_hit_buffers(shape, max_hits=max_hits)
    z_hits[...] = xyz[:, :, 2].reshape(shape + (max_hits,))
    # Keep geometric normals: n_z > 0 is a floor (hit from above), n_z < 0 is a roof.
    normals[...] = nrm.reshape(shape + (max_hits, 3))
    return z_hits, normals
