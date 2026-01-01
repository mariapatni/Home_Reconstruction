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
    min_nonzero_votes: int = 1,
    min_nonzero_ratio: float = 0.10,
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
    min_views: int = 2,
    prefer_nonzero: bool = True,
    min_nonzero_votes: int = 1,
    min_nonzero_ratio: float = 0.10,
    background_id: int = 0,
    aggregate: str = "mean",
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

    total_voxels = len(voxel_data)
    print(f"Total voxels before filtering: {total_voxels:,}")
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

def remove_small_clusters_only(
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    *,
    eps: float = 0.05,
    min_cluster_size: int = 50,
):
    """
    Remove only tiny noise clusters, keep ALL substantial clusters.
    
    This is better for multi-object scenes than keep_largest_n=1.
    """
    if len(points) == 0:
        return points, colors, object_ids

    print(f"Removing small clusters (eps={eps}m, min_size={min_cluster_size})...")

    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(points)
    unique = np.unique(labels)

    # Count points per cluster
    cluster_sizes = {}
    for l in unique:
        if l == -1:
            continue
        cluster_sizes[l] = int(np.sum(labels == l))

    n_noise = int(np.sum(labels == -1))
    n_clusters = len(cluster_sizes)
    
    # Keep clusters with at least min_cluster_size points
    keep_labels = set(l for l, size in cluster_sizes.items() if size >= min_cluster_size)
    
    # Also keep noise points that have semantic labels (non-zero object_id)
    # This prevents losing small labeled objects that didn't form clusters
    mask = np.isin(labels, list(keep_labels))
    
    # Optionally: also keep labeled noise points
    labeled_noise = (labels == -1) & (object_ids != 0)
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
):
    """
    Remove disconnected geometric clusters using DBSCAN.
    Semantics are preserved by indexing.
    
    WARNING: keep_largest_n=1 will remove all objects except the main structure!
    For multi-object scenes, use remove_small_clusters_only() instead.
    """
    if len(points) == 0:
        return points, colors, object_ids

    print(f"Removing disconnected clusters (eps={eps}m, min_samples={min_samples})...")

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    unique = np.unique(labels)

    cluster_labels = [l for l in unique.tolist() if l != -1]
    cluster_sizes = [(l, int(np.sum(labels == l))) for l in cluster_labels]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    n_noise = int(np.sum(labels == -1))
    print(f"Found {len(cluster_labels)} clusters (plus {n_noise} noise points)")

    if len(cluster_sizes) == 0:
        return points, colors, object_ids

    keep = set([l for l, _ in cluster_sizes[:keep_largest_n]])
    mask = np.isin(labels, list(keep))

    return points[mask], colors[mask], object_ids[mask]

def remap_object_ids(
    object_ids: np.ndarray,
    class_mapping: dict = None,
    *,
    min_points: int = 50,
    background_id: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, dict, dict]:
    """
    Filter tiny objects and remap IDs to be contiguous.
    
    This:
    1. Merges objects with < min_points into background
    2. Remaps remaining IDs to contiguous values (0, 1, 2, ...)
    
    Call this AFTER all point cloud filtering is done.
    
    Args:
        object_ids: (N,) integer object IDs
        class_mapping: Optional dict mapping ID -> class name (keys as strings)
        min_points: Minimum points for an object to be kept (smaller → background)
        background_id: ID to use for background (default 0)
        verbose: Whether to print summary
        
    Returns:
        object_ids_remapped: (N,) remapped contiguous IDs
        new_class_mapping: dict mapping new ID -> class name
        id_remap: dict mapping old ID -> new ID
        
    Example:
        >>> # After processing point cloud:
        >>> points, colors, obj_ids = process_pointcloud_with_semantics(...)
        >>> 
        >>> # Remap IDs to be contiguous and filter tiny objects:
        >>> obj_ids, class_mapping, id_remap = remap_object_ids(
        ...     obj_ids, 
        ...     class_mapping=original_class_mapping,
        ...     min_points=50
        ... )
        >>> 
        >>> # Now use for training:
        >>> model = ObjectGSModel(points, colors, obj_ids, ...)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("REMAPPING OBJECT IDs")
        print("=" * 60)
    
    # Count points per object
    unique_ids, counts = np.unique(object_ids, return_counts=True)
    id_counts = dict(zip(unique_ids.tolist(), counts.tolist()))
    
    if verbose:
        print(f"Input: {len(unique_ids)} unique IDs, max ID = {unique_ids.max()}")
    
    # Identify objects to keep vs merge to background
    ids_to_keep = []
    ids_to_merge = []
    
    for obj_id, count in sorted(id_counts.items(), key=lambda x: -x[1]):
        if obj_id == background_id:
            ids_to_keep.append(obj_id)  # Always keep background
        elif count >= min_points:
            ids_to_keep.append(obj_id)
        else:
            ids_to_merge.append((obj_id, count))
    
    if verbose:
        print(f"Keeping {len(ids_to_keep)} objects with >= {min_points} points")
        if ids_to_merge:
            merged_points = sum(c for _, c in ids_to_merge)
            print(f"Merging {len(ids_to_merge)} tiny objects ({merged_points:,} points) into background")
    
    # Create remapping: old_id -> new_id
    # Ensure background is ID 0, others get sequential IDs
    ids_to_keep_sorted = sorted(ids_to_keep)
    if background_id in ids_to_keep_sorted:
        ids_to_keep_sorted.remove(background_id)
    ids_to_keep_sorted = [background_id] + ids_to_keep_sorted
    
    id_remap = {}
    for new_id, old_id in enumerate(ids_to_keep_sorted):
        id_remap[old_id] = new_id
    
    # Tiny objects map to background
    for obj_id, _ in ids_to_merge:
        id_remap[obj_id] = 0
    
    # Apply remapping
    object_ids_remapped = np.array([id_remap[i] for i in object_ids], dtype=np.int32)
    
    # Create new class mapping
    new_class_mapping = {}
    for old_id, new_id in id_remap.items():
        if new_id == 0:
            new_class_mapping[str(new_id)] = "background"
        elif class_mapping and str(old_id) in class_mapping:
            new_class_mapping[str(new_id)] = class_mapping[str(old_id)]
        else:
            new_class_mapping[str(new_id)] = f"object_{old_id}"
    
    if verbose:
        new_unique = np.unique(object_ids_remapped)
        print(f"Output: {len(new_unique)} unique IDs, max ID = {new_unique.max()}")
        print(f"One-hot dimension: {unique_ids.max() + 1} → {new_unique.max() + 1}")
    
    return object_ids_remapped, new_class_mapping, id_remap



def statistical_outlier_removal(
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    *,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
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
    # Voxel filtering
    voxel_size: float = 0.01,
    min_views: int = 2,
    
    # Semantic voting
    prefer_nonzero: bool = True,
    min_nonzero_votes: int = 1,
    min_nonzero_ratio: float = 0.10,
    background_id: int = 0,
    
    # Cluster filtering - NEW DEFAULTS for multi-object scenes
    cluster_eps: float = 0.05,
    min_cluster_size: int = 20,
    keep_largest_n: int = None,  # None = use remove_small_clusters_only
    
    # Statistical outlier removal
    use_sor: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
):
    """
    Main processing pipeline:
      1) multiview voxel filter + semantic voting
      2) remove small clusters (keeps all substantial clusters for multi-object scenes)
      3) optional statistical outlier removal
      4) print semantic summary
    
    UPDATED DEFAULTS:
      - min_views=2 (was 5) - better for sparse frame sets
      - min_nonzero_votes=1 (was 2) - preserve single-view labels
      - min_nonzero_ratio=0.10 (was 0.30) - lower threshold
      - keep_largest_n=None - keeps ALL substantial clusters (was 1, which killed multi-object scenes)
      - std_ratio=2.0 (was 1.5) - less aggressive outlier removal
    """
    print("\n" + "="*60)
    print("POINT CLOUD PROCESSING WITH SEMANTICS")
    print("="*60)
    
    # Step 1: Multiview voxel filtering
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
    
    if len(points) == 0:
        print("WARNING: No points remaining after voxel filtering!")
        return points, colors, obj_ids

    # Step 2: Cluster filtering
    if keep_largest_n is not None:
        # Legacy behavior: keep only N largest clusters
        points, colors, obj_ids = remove_disconnected_clusters(
            points, colors, obj_ids,
            eps=cluster_eps,
            min_samples=min_cluster_size,
            keep_largest_n=keep_largest_n,
        )
    else:
        # NEW: Keep all substantial clusters, only remove tiny noise
        points, colors, obj_ids = remove_small_clusters_only(
            points, colors, obj_ids,
            eps=cluster_eps,
            min_cluster_size=min_cluster_size,
        )

    # Step 3: Statistical outlier removal
    if use_sor and len(points) > 0:
        points, colors, obj_ids = statistical_outlier_removal(
            points, colors, obj_ids,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )

    new_obj_ids = remap_object_ids(obj_ids)

    # Semantic summary
    if len(obj_ids) > 0:
        uniq, counts = np.unique(obj_ids, return_counts=True)
        print(f"\n" + "="*60)
        print(f"FINAL SEMANTIC SUMMARY: {len(uniq)} unique object IDs")
        print("="*60)
        for k, c in sorted(zip(uniq.tolist(), counts.tolist()), key=lambda x: -x[1]):
            pct = 100 * c / len(obj_ids)
            label = "background" if k == 0 else f"object {k}"
            print(f"  ID {k:3d} ({label:12s}): {c:,} points ({pct:.1f}%)")
    else:
        print("\nWARNING: No points in final point cloud!")

    return points, colors, obj_ids


def create_raw_pointcloud(points_by_frame: list[np.ndarray], colors_by_frame: list[np.ndarray]):
    """Combine all frames into a single raw point cloud (no semantics)."""
    if len(points_by_frame) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.vstack(points_by_frame).astype(np.float32), np.vstack(colors_by_frame).astype(np.float32)