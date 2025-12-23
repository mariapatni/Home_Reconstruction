"""
Multiview point cloud filtering utilities with semantic support.

Key behavior (semantic-preserving):
- ID 0 is treated as "unlabeled/background".
- If a voxel has enough non-zero evidence, background (0) cannot outvote it.
- If a voxel does NOT have enough non-zero evidence, it falls back to 0.

This prevents "sometimes-correct labels" (clothes/books/walls) from being erased by many unlabeled votes.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from collections import defaultdict
from sklearn.cluster import DBSCAN


# -----------------------------
# Helpers
# -----------------------------

def create_o3d_pointcloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """Create Open3D point cloud from numpy arrays."""
    pcd = o3d.geometry.PointCloud()
    if len(points) == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def _argmax_bincount(ids: np.ndarray) -> int:
    """Fast argmax over non-negative int ids."""
    if ids.size == 0:
        return 0
    ids = ids.astype(np.int32, copy=False)
    return int(np.bincount(ids).argmax())


def vote_object_id(
    ids: np.ndarray,
    *,
    background_id: int = 0,
    prefer_nonzero: bool = True,
    min_nonzero_votes: int = 2,
    min_nonzero_ratio: float = 0.30,
) -> int:
    """
    Vote a single object id from a set of per-point ids.

    If prefer_nonzero=True:
      - If we have enough non-zero evidence, vote ONLY among non-zero ids.
      - Else return background_id.

    This matches "only truly unlabeled should become background".
    """
    if ids.size == 0:
        return background_id

    ids = ids.astype(np.int32, copy=False)

    if not prefer_nonzero:
        return _argmax_bincount(ids)

    nonzero = ids[ids != background_id]
    nz = int(nonzero.size)

    if nz < min_nonzero_votes:
        return background_id

    # Ratio is with respect to total observations in voxel
    ratio = nz / float(ids.size)
    if ratio < min_nonzero_ratio:
        return background_id

    # Vote among non-zero only
    # Need bincount on shifted ids or mask out background in bincount:
    # easiest: bincount full, then zero out background bin.
    bc = np.bincount(ids)
    if background_id < bc.size:
        bc[background_id] = 0
    return int(bc.argmax())


# -----------------------------
# Core: voxel multiview filtering
# -----------------------------

def multiview_filter_with_semantics(
    points_by_frame: list[np.ndarray],
    colors_by_frame: list[np.ndarray],
    object_ids_by_frame: list[np.ndarray],
    *,
    voxel_size: float = 0.01,
    min_views: int = 5,
    prefer_nonzero: bool = True,
    min_nonzero_votes: int = 2,
    min_nonzero_ratio: float = 0.30,
    background_id: int = 0,
    aggregate: str = "mean",  # "mean" or "medoid"
):
    """
    Combine per-frame points into voxels, require visibility in >=min_views frames,
    and produce one representative point per voxel with a voted object ID.

    Semantics:
      - background_id (0) treated as unlabeled/background.
      - If prefer_nonzero=True, labeled evidence is preserved.
    """
    assert len(points_by_frame) == len(colors_by_frame) == len(object_ids_by_frame)

    voxel_data = defaultdict(lambda: {
        "frames": set(),
        "points": [],
        "colors": [],
        "ids": [],
    })

    for frame_id, (pts, cols, ids) in enumerate(zip(points_by_frame, colors_by_frame, object_ids_by_frame)):
        if pts is None or len(pts) == 0:
            continue
        if cols is None or len(cols) != len(pts):
            raise ValueError(f"Frame {frame_id}: colors missing or wrong shape.")
        if ids is None or len(ids) != len(pts):
            raise ValueError(f"Frame {frame_id}: object_ids missing or wrong shape.")

        # voxel coordinates
        vox = np.floor(pts / float(voxel_size)).astype(np.int32)

        for i, key in enumerate(map(tuple, vox)):
            d = voxel_data[key]
            d["frames"].add(frame_id)
            d["points"].append(pts[i])
            d["colors"].append(cols[i])
            d["ids"].append(int(ids[i]))

    print(f"Filtering voxels (min views: {min_views})...")

    out_points = []
    out_colors = []
    out_ids = []

    for d in voxel_data.values():
        if len(d["frames"]) < min_views:
            continue

        pts = np.asarray(d["points"], dtype=np.float32)
        cols = np.asarray(d["colors"], dtype=np.float32)
        ids = np.asarray(d["ids"], dtype=np.int32)

        voted = vote_object_id(
            ids,
            background_id=background_id,
            prefer_nonzero=prefer_nonzero,
            min_nonzero_votes=min_nonzero_votes,
            min_nonzero_ratio=min_nonzero_ratio,
        )

        if aggregate == "mean":
            rep_p = pts.mean(axis=0)
            rep_c = cols.mean(axis=0)
        elif aggregate == "medoid":
            # pick point closest to centroid
            centroid = pts.mean(axis=0, keepdims=True)
            j = int(np.argmin(np.sum((pts - centroid) ** 2, axis=1)))
            rep_p = pts[j]
            rep_c = cols[j]
        else:
            raise ValueError("aggregate must be 'mean' or 'medoid'")

        out_points.append(rep_p)
        out_colors.append(rep_c)
        out_ids.append(voted)

    if len(out_points) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0,), np.int32)

    return (
        np.asarray(out_points, dtype=np.float32),
        np.asarray(out_colors, dtype=np.float32),
        np.asarray(out_ids, dtype=np.int32),
    )


# -----------------------------
# Geometry cleaning (preserve semantics)
# -----------------------------

def remove_disconnected_clusters(
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    *,
    eps: float = 0.10,
    min_samples: int = 50,
    keep_largest_n: int = 1,
):
    """
    Remove disconnected geometric clusters using DBSCAN.
    Semantics are preserved by indexing.
    """
    if len(points) == 0:
        return points, colors, object_ids

    print(f"Removing disconnected clusters (eps={eps}m, min_samples={min_samples})...")

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    unique = np.unique(labels)

    # Count clusters excluding noise (-1)
    cluster_labels = [l for l in unique.tolist() if l != -1]
    cluster_sizes = [(l, int(np.sum(labels == l))) for l in cluster_labels]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    n_noise = int(np.sum(labels == -1))
    print(f"Found {len(cluster_labels)} clusters (plus {n_noise} noise points)")

    if len(cluster_sizes) == 0:
        # Everything is noise; keep it rather than deleting everything.
        return points, colors, object_ids

    keep = set([l for l, _ in cluster_sizes[:keep_largest_n]])
    mask = np.isin(labels, list(keep))

    return points[mask], colors[mask], object_ids[mask]


def statistical_outlier_removal(
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    *,
    nb_neighbors: int = 20,
    std_ratio: float = 1.5,
):
    """Open3D statistical outlier removal. Preserves object_ids by indexing."""
    if len(points) == 0:
        return points, colors, object_ids

    print(f"Applying statistical outlier removal (neighbors={nb_neighbors}, std_ratio={std_ratio})...")

    pcd = create_o3d_pointcloud(points, colors)
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    ind = np.asarray(ind, dtype=np.int64)
    removed = len(points) - len(ind)
    pct = 100.0 * removed / max(1, len(points))
    print(f"Removed {removed:,} outliers ({pct:.1f}%)")

    return points[ind], colors[ind], object_ids[ind]


# -----------------------------
# Full pipeline
# -----------------------------

def process_pointcloud_with_semantics(
    points_by_frame: list[np.ndarray],
    colors_by_frame: list[np.ndarray],
    object_ids_by_frame: list[np.ndarray],
    *,
    voxel_size: float = 0.01,
    min_views: int = 5,
    prefer_nonzero: bool = True,
    min_nonzero_votes: int = 2,
    min_nonzero_ratio: float = 0.30,
    background_id: int = 0,
    cluster_eps: float = 0.10,
    min_cluster_size: int = 50,
    keep_largest_n: int = 1,
    use_sor: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 1.5,
):
    """
    Main processing pipeline:
      1) multiview voxel filter + semantic voting
      2) remove disconnected clusters
      3) optional statistical outlier removal
      4) print semantic summary
    """
    points, colors, obj_ids = multiview_filter_with_semantics(
        points_by_frame, colors_by_frame, object_ids_by_frame,
        voxel_size=voxel_size,
        min_views=min_views,
        prefer_nonzero=prefer_nonzero,
        min_nonzero_votes=min_nonzero_votes,
        min_nonzero_ratio=min_nonzero_ratio,
        background_id=background_id,
        aggregate="mean",
    )

    points, colors, obj_ids = remove_disconnected_clusters(
        points, colors, obj_ids,
        eps=cluster_eps,
        min_samples=min_cluster_size,
        keep_largest_n=keep_largest_n,
    )

    if use_sor:
        points, colors, obj_ids = statistical_outlier_removal(
            points, colors, obj_ids,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )

    # Semantic summary
    uniq, counts = np.unique(obj_ids, return_counts=True)
    print(f"\nSemantic summary: {len(uniq)} unique objects")
    for k, c in zip(uniq.tolist(), counts.tolist()):
        print(f"  ID {k}: {c:,} points")

    return points, colors, obj_ids


def create_raw_pointcloud(points_by_frame: list[np.ndarray], colors_by_frame: list[np.ndarray]):
    """Combine all frames into a single raw point cloud (no semantics)."""
    if len(points_by_frame) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.vstack(points_by_frame).astype(np.float32), np.vstack(colors_by_frame).astype(np.float32)
