"""
Multiview point cloud filtering utilities with semantic support.

Key behavior (semantic-preserving):
- ID 0 is treated as "unlabeled/background".
- If a voxel has enough non-zero evidence, background (0) cannot outvote it.
- If a voxel does NOT have enough non-zero evidence, it falls back to 0.

Also includes a final remap step:
- Objects with fewer than `remap_min_points` points are merged into background.
- Remaining IDs are remapped to contiguous values: 0..K-1
- Optional propagation of class mapping (id -> class info/name).

Designed for Record3D -> Gaussian Splatting pipelines.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional imports (Open3D + sklearn) are expected in your environment,
# but we keep imports explicit here.
import open3d as o3d
from sklearn.cluster import DBSCAN


# =============================================================================
# Helpers
# =============================================================================

def create_o3d_pointcloud(points: np.ndarray, colors: Optional[np.ndarray]) -> o3d.geometry.PointCloud:
    """Create Open3D point cloud from numpy arrays."""
    pcd = o3d.geometry.PointCloud()
    if points is None or len(points) == 0:
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
    min_nonzero_votes: int = 1,
    min_nonzero_ratio: float = 0.10,
) -> int:
    """
    Vote a single object id from a set of per-point ids for one voxel.

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

    ratio = nz / float(ids.size)
    if ratio < min_nonzero_ratio:
        return background_id

    # Vote among non-zero only by zeroing out background bin
    bc = np.bincount(ids)
    if background_id < bc.size:
        bc[background_id] = 0
    return int(bc.argmax())


# =============================================================================
# Core: voxel multiview filtering
# =============================================================================

def multiview_filter_with_semantics(
    points_by_frame: List[np.ndarray],
    colors_by_frame: List[np.ndarray],
    object_ids_by_frame: List[np.ndarray],
    *,
    voxel_size: float = 0.01,
    min_views: int = 2,
    prefer_nonzero: bool = True,
    min_nonzero_votes: int = 1,
    min_nonzero_ratio: float = 0.10,
    background_id: int = 0,
    aggregate: str = "mean",  # "mean" | "medoid"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        vox = np.floor(pts / float(voxel_size)).astype(np.int32)

        for i, key in enumerate(map(tuple, vox)):
            d = voxel_data[key]
            d["frames"].add(frame_id)
            d["points"].append(pts[i])
            d["colors"].append(cols[i])
            d["ids"].append(int(ids[i]))

    total_voxels = len(voxel_data)
    print(f"Total voxels before filtering: {total_voxels:,}")
    print(f"Filtering voxels (min views: {min_views})...")

    out_points: List[np.ndarray] = []
    out_colors: List[np.ndarray] = []
    out_ids: List[int] = []

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
            centroid = pts.mean(axis=0, keepdims=True)
            j = int(np.argmin(np.sum((pts - centroid) ** 2, axis=1)))
            rep_p = pts[j]
            rep_c = cols[j]
        else:
            raise ValueError("aggregate must be 'mean' or 'medoid'")

        out_points.append(rep_p)
        out_colors.append(rep_c)
        out_ids.append(voted)

    kept = len(out_points)
    print(f"Voxels after min_views filter: {kept:,} ({100*kept/max(1,total_voxels):.1f}% retained)")

    if kept == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0,), np.int32)

    return (
        np.asarray(out_points, dtype=np.float32),
        np.asarray(out_colors, dtype=np.float32),
        np.asarray(out_ids, dtype=np.int32),
    )


# =============================================================================
# Geometry cleaning (preserve semantics by indexing)
# =============================================================================

def remove_small_clusters_only(
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    *,
    eps: float = 0.05,
    min_cluster_size: int = 50,
    min_samples: int = 5,
    background_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove only tiny noise clusters, keep ALL substantial clusters.

    Also keeps labeled noise points (object_id != background_id) even if DBSCAN marks them as noise.
    """
    if len(points) == 0:
        return points, colors, object_ids

    print(f"Removing small clusters (eps={eps}m, min_cluster_size={min_cluster_size})...")

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    unique = np.unique(labels)

    cluster_sizes: Dict[int, int] = {}
    for l in unique:
        if l == -1:
            continue
        cluster_sizes[int(l)] = int(np.sum(labels == l))

    n_noise = int(np.sum(labels == -1))
    n_clusters = len(cluster_sizes)

    keep_labels = set(l for l, size in cluster_sizes.items() if size >= min_cluster_size)

    mask = np.isin(labels, list(keep_labels))

    # keep labeled noise
    labeled_noise = (labels == -1) & (object_ids != background_id)
    mask = mask | labeled_noise

    kept = int(np.sum(mask))
    removed = len(points) - kept

    print(f"Found {n_clusters} clusters (plus {n_noise} noise points)")
    print(f"Keeping {len(keep_labels)} clusters with >= {min_cluster_size} points")
    print(f"Removed {removed:,} points ({100*removed/max(1,len(points)):.1f}%)")

    return points[mask], colors[mask], object_ids[mask]


def remove_disconnected_clusters(
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    *,
    eps: float = 0.10,
    min_samples: int = 50,
    keep_largest_n: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove disconnected geometric clusters using DBSCAN.

    WARNING: keep_largest_n=1 will remove all objects except the main structure!
    Prefer remove_small_clusters_only() for multi-object scenes.
    """
    if len(points) == 0:
        return points, colors, object_ids

    print(f"Removing disconnected clusters (eps={eps}m, min_samples={min_samples}, keep_largest_n={keep_largest_n})...")

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    unique = np.unique(labels)

    cluster_labels = [int(l) for l in unique.tolist() if l != -1]
    cluster_sizes = [(l, int(np.sum(labels == l))) for l in cluster_labels]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    n_noise = int(np.sum(labels == -1))
    print(f"Found {len(cluster_labels)} clusters (plus {n_noise} noise points)")

    if len(cluster_sizes) == 0:
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
    std_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# =============================================================================
# Semantic post-processing: remove tiny objects + remap contiguous IDs
# =============================================================================

def remap_object_ids(
    object_ids: np.ndarray,
    class_mapping: Optional[Dict[Any, Any]] = None,
    *,
    min_points: int = 50,
    background_id: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[int, int]]:
    """
    Filter tiny objects and remap IDs to be contiguous.

    1) Merges objects with < min_points into background
    2) Remaps remaining IDs to contiguous values (0, 1, 2, ...)
    3) Produces a new class mapping keyed by *new* IDs as strings

    class_mapping can be:
      - { "1": {...}, "2": {...} } or {1: {...}, 2: {...}}
      - values can be dicts (like your current json) or strings (class names)

    Returns:
        object_ids_remapped: (N,) contiguous IDs
        new_class_mapping: dict mapping new_id(str) -> info/name
        id_remap: dict mapping old_id(int) -> new_id(int)
    """
    if object_ids is None or len(object_ids) == 0:
        return object_ids.astype(np.int32), {"0": "background"}, {background_id: 0}

    if verbose:
        print("\n" + "=" * 60)
        print("REMAPPING OBJECT IDs")
        print("=" * 60)

    object_ids = object_ids.astype(np.int32, copy=False)

    unique_ids, counts = np.unique(object_ids, return_counts=True)
    id_counts = dict(zip(unique_ids.tolist(), counts.tolist()))

    if verbose:
        print(f"Input: {len(unique_ids)} unique IDs, max ID = {int(unique_ids.max())}")

    ids_to_keep: List[int] = []
    ids_to_merge: List[Tuple[int, int]] = []

    for obj_id, count in sorted(id_counts.items(), key=lambda x: -x[1]):
        if obj_id == background_id:
            ids_to_keep.append(obj_id)
        elif count >= min_points:
            ids_to_keep.append(obj_id)
        else:
            ids_to_merge.append((obj_id, count))

    if verbose:
        print(f"Keeping {len(ids_to_keep)} objects with >= {min_points} points")
        if ids_to_merge:
            merged_points = sum(c for _, c in ids_to_merge)
            print(f"Merging {len(ids_to_merge)} tiny objects ({merged_points:,} points) into background")

    # Remapping: background stays 0, others become 1..K
    ids_to_keep_sorted = sorted(set(ids_to_keep))
    if background_id in ids_to_keep_sorted:
        ids_to_keep_sorted.remove(background_id)
    ids_to_keep_sorted = [background_id] + ids_to_keep_sorted

    id_remap: Dict[int, int] = {}
    for new_id, old_id in enumerate(ids_to_keep_sorted):
        id_remap[int(old_id)] = int(new_id)

    # tiny objects map to background
    for obj_id, _ in ids_to_merge:
        id_remap[int(obj_id)] = 0

    # apply
    object_ids_remapped = np.array([id_remap[int(i)] for i in object_ids], dtype=np.int32)

    # normalize incoming class mapping to int->value
    cm_int: Dict[int, Any] = {}
    if class_mapping is not None:
        for k, v in class_mapping.items():
            try:
                cm_int[int(k)] = v
            except Exception:
                # ignore unusable keys
                pass

    # build new mapping (string keys for json friendliness)
    new_class_mapping: Dict[str, Any] = {"0": "background"}
    for old_id, new_id in sorted(id_remap.items(), key=lambda x: x[1]):
        if new_id == 0:
            continue
        if old_id in cm_int:
            new_class_mapping[str(new_id)] = cm_int[old_id]
        else:
            new_class_mapping[str(new_id)] = f"object_{old_id}"

    if verbose:
        new_unique = np.unique(object_ids_remapped)
        print(f"Output: {len(new_unique)} unique IDs, max ID = {int(new_unique.max())}")

    return object_ids_remapped, new_class_mapping, id_remap


# =============================================================================
# Full pipeline
# =============================================================================

def process_pointcloud_with_semantics(
    points_by_frame: List[np.ndarray],
    colors_by_frame: List[np.ndarray],
    object_ids_by_frame: List[np.ndarray],
    *,
    # Voxel filtering
    voxel_size: float = 0.01,
    min_views: int = 2,

    # Semantic voting
    prefer_nonzero: bool = True,
    min_nonzero_votes: int = 1,
    min_nonzero_ratio: float = 0.10,
    background_id: int = 0,

    # Cluster filtering
    cluster_eps: float = 0.05,
    min_cluster_size: int = 20,
    keep_largest_n: Optional[int] = None,  # None => remove_small_clusters_only

    # Statistical outlier removal
    use_sor: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,

    # ✅ NEW: semantic cleanup/remap
    remap_ids: bool = True,
    remap_min_points: int = 50,
    class_mapping: Optional[Dict[Any, Any]] = None,
    verbose_remap: bool = True,

    # return extra mapping info
    return_mappings: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[int, int]]:
    """
    Main processing pipeline:
      1) multiview voxel filter + semantic voting
      2) cluster filtering
      3) optional statistical outlier removal
      4) optional remove tiny objects + remap contiguous IDs
      5) semantic summary

    If return_mappings=True, returns:
      points, colors, obj_ids, new_class_mapping, id_remap
    """
    print("\n" + "=" * 60)
    print("POINT CLOUD PROCESSING WITH SEMANTICS")
    print("=" * 60)

    # Step 1: Multiview voxel filtering
    points, colors, obj_ids = multiview_filter_with_semantics(
        points_by_frame,
        colors_by_frame,
        object_ids_by_frame,
        voxel_size=voxel_size,
        min_views=min_views,
        prefer_nonzero=prefer_nonzero,
        min_nonzero_votes=min_nonzero_votes,
        min_nonzero_ratio=min_nonzero_ratio,
        background_id=background_id,
        aggregate="mean",
    )

    if len(points) == 0:
        print("WARNING: No points remaining after voxel filtering!")
        if return_mappings:
            return points, colors, obj_ids, {"0": "background"}, {background_id: 0}
        return points, colors, obj_ids

    # Step 2: Cluster filtering
    if keep_largest_n is not None:
        points, colors, obj_ids = remove_disconnected_clusters(
            points, colors, obj_ids,
            eps=cluster_eps,
            min_samples=min_cluster_size,
            keep_largest_n=keep_largest_n,
        )
    else:
        points, colors, obj_ids = remove_small_clusters_only(
            points, colors, obj_ids,
            eps=cluster_eps,
            min_cluster_size=min_cluster_size,
            min_samples=5,
            background_id=background_id,
        )

    # Step 3: Statistical outlier removal
    if use_sor and len(points) > 0:
        points, colors, obj_ids = statistical_outlier_removal(
            points, colors, obj_ids,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )

    # Step 4: Remove tiny objects + remap contiguous
    new_class_mapping: Dict[str, Any] = {"0": "background"}
    id_remap: Dict[int, int] = {background_id: 0}
    if remap_ids and len(obj_ids) > 0:
        obj_ids, new_class_mapping, id_remap = remap_object_ids(
            obj_ids,
            class_mapping=class_mapping,
            min_points=remap_min_points,
            background_id=background_id,
            verbose=verbose_remap,
        )

    # Step 5: Semantic summary (final)
    if len(obj_ids) > 0:
        uniq, counts = np.unique(obj_ids, return_counts=True)
        print("\n" + "=" * 60)
        print(f"FINAL SEMANTIC SUMMARY: {len(uniq)} unique object IDs")
        print("=" * 60)
        for k, c in sorted(zip(uniq.tolist(), counts.tolist()), key=lambda x: -x[1]):
            pct = 100 * c / len(obj_ids)
            name = "background"
            if str(k) in new_class_mapping:
                v = new_class_mapping[str(k)]
                if isinstance(v, dict) and "class_name" in v:
                    name = v["class_name"]
                elif isinstance(v, str):
                    name = v
                else:
                    name = str(v)
            label = f"{name}" if k == 0 else f"{name}"
            print(f"  ID {k:3d} ({label:16s}): {c:,} points ({pct:.1f}%)")
    else:
        print("\nWARNING: No points in final point cloud!")

    if return_mappings:
        return points, colors, obj_ids, new_class_mapping, id_remap
    return points, colors, obj_ids


def create_raw_pointcloud(points_by_frame: List[np.ndarray], colors_by_frame: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Combine all frames into a single raw point cloud (no semantics)."""
    if len(points_by_frame) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.vstack(points_by_frame).astype(np.float32), np.vstack(colors_by_frame).astype(np.float32)
