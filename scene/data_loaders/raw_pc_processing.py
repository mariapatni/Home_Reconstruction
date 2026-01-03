"""
Multiview point cloud filtering utilities with semantic support.
STREAMING IMPLEMENTATION - processes frames incrementally to avoid memory blowup.

OPEN3D REWRITE (full file):
- Uses Open3D point clouds for:
    * DBSCAN clustering (pcd.cluster_dbscan)
    * Statistical outlier removal (pcd.remove_statistical_outlier)
- Maintains the SAME public function names, signatures, input types, and output types
  as your streaming file.

Key behavior (semantic-preserving):
- ID 0 is treated as "unlabeled/background".
- Semantic voting is per-frame-per-voxel (not per-point), preventing dense textures from dominating.
- If a voxel has enough non-zero evidence, background (0) cannot outvote it.
- If a voxel does NOT have enough non-zero evidence, it falls back to 0.

Also includes a final remap step:
- Objects with fewer than `remap_min_points` points are merged into background.
- Remaining IDs are remapped to contiguous values: 0..K-1
- Optional propagation of class mapping (id -> class info/name).

Label-aware cleaning:
- DBSCAN clustering is applied gently to labeled points to preserve small objects.
- Background points receive more aggressive geometric cleaning.

Designed for Record3D -> Gaussian Splatting pipelines.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np

# =============================================================================
# Open3D dependency
# =============================================================================
try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None


# =============================================================================
# Type aliases
# =============================================================================
FrameData = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (points, colors, object_ids)
FrameGenerator = Generator[FrameData, None, None]


# =============================================================================
# Helpers
# =============================================================================
def create_o3d_pointcloud(points: np.ndarray, colors: Optional[np.ndarray] = None):
    """
    Create Open3D point cloud from numpy arrays.

    Inputs:
      points: (N,3) float32/float64
      colors: (N,3) float32/float64 in [0,1] (optional)

    Output:
      o3d.geometry.PointCloud
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required for this function. Install with: pip install open3d")

    pcd = o3d.geometry.PointCloud()
    if points is None or len(points) == 0:
        return pcd

    # Open3D stores float64 internally; this mirrors your earlier helper.
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))

    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64, copy=False))

    return pcd


def _open3d_dbscan_labels(
    points_xyz: np.ndarray,
    *,
    eps: float,
    min_points: int,
    verbose: bool = False,
) -> np.ndarray:
    """
    Run Open3D DBSCAN on (N,3) points. Returns labels (N,) int32.
    Noise is labeled as -1 (same convention as sklearn DBSCAN).
    """
    if points_xyz is None or len(points_xyz) == 0:
        return np.zeros((0,), dtype=np.int32)

    pcd = create_o3d_pointcloud(points_xyz, colors=None)
    labels = np.array(
        pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=bool(verbose)),
        dtype=np.int32,
    )
    return labels


def _vote_semantic_label(
    frame_votes: Dict[int, int],
    *,
    background_id: int = 0,
    prefer_nonzero: bool = True,
    min_nonzero_votes: int = 1,
    min_nonzero_ratio: float = 0.05,
) -> int:
    """
    Vote a single object id from per-frame votes for one voxel.

    Inputs:
      frame_votes: {frame_id: label} - one vote per frame that saw this voxel

    Output:
      int label (background_id or a non-zero class id)

    Behavior matches your current implementation:
      - If prefer_nonzero=True, require enough non-zero evidence, else return background_id.
      - Otherwise do a simple majority vote.
    """
    if not frame_votes:
        return background_id

    labels = list(frame_votes.values())
    total_votes = len(labels)

    if not prefer_nonzero:
        counts = defaultdict(int)
        for lbl in labels:
            counts[lbl] += 1
        return max(counts.keys(), key=lambda k: counts[k])

    nonzero_labels = [lbl for lbl in labels if lbl != background_id]
    n_nonzero = len(nonzero_labels)

    if n_nonzero < min_nonzero_votes:
        return background_id

    ratio = n_nonzero / float(total_votes)
    if ratio < min_nonzero_ratio:
        return background_id

    counts = defaultdict(int)
    for lbl in nonzero_labels:
        counts[lbl] += 1
    return max(counts.keys(), key=lambda k: counts[k])


def _intra_frame_vote(ids: np.ndarray, background_id: int = 0) -> int:
    """
    Vote for a single label within a frame for points landing in the same voxel.
    Prefers non-zero labels if any exist.

    Inputs:
      ids: (M,) int labels for points within a voxel within a single frame

    Output:
      int label for that voxel for that frame
    """
    if ids.size == 0:
        return background_id

    ids = ids.astype(np.int32, copy=False)
    nonzero = ids[ids != background_id]

    if nonzero.size > 0:
        return int(np.bincount(nonzero).argmax())
    return background_id


# =============================================================================
# Core: Streaming voxel multiview filtering
# =============================================================================
def streaming_voxel_filter(
    frame_generator: Union[FrameGenerator, Iterator[FrameData]],
    *,
    voxel_size: float = 0.01,
    min_views: int = 2,
    prefer_nonzero: bool = True,
    min_nonzero_votes: int = 1,
    min_nonzero_ratio: float = 0.05,
    background_id: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STREAMING multiview voxel filter with semantic voting.

    Input:
      frame_generator yields tuples:
        pts: (N,3) float32/float64
        cols: (N,3) float32/float64
        ids: (N,) int

    Output:
      points: (K,3) float32
      colors: (K,3) float32
      object_ids: (K,) int32

    Same output contract as your file.
    """
    # Voxel accumulators keyed by (vx,vy,vz)
    voxels: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    inv_voxel_size = 1.0 / float(voxel_size)
    total_raw_points = 0
    n_frames_seen = 0

    if verbose:
        print(f"\nStreaming voxel filter (voxel_size={voxel_size}m, min_views={min_views})")
        print("Processing frames...")

    for frame_id, (pts, cols, ids) in enumerate(frame_generator):
        if pts is None or len(pts) == 0:
            continue

        n_frames_seen += 1
        n_pts = int(len(pts))
        total_raw_points += n_pts

        # Defensive shape checks (keeps bugs from silently corrupting outputs)
        if cols is None or len(cols) != len(pts):
            raise ValueError(f"Frame {frame_id}: colors missing or wrong shape.")
        if ids is None or len(ids) != len(pts):
            raise ValueError(f"Frame {frame_id}: object_ids missing or wrong shape.")
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Frame {frame_id}: points must be (N,3). Got {pts.shape}")

        vox_coords = np.floor(pts * inv_voxel_size).astype(np.int32)

        # Group within-frame to enforce ONE semantic vote per voxel per frame
        frame_voxel_data: Dict[Tuple[int, int, int], Dict[str, Any]] = defaultdict(
            lambda: {"pts": [], "cols": [], "ids": []}
        )

        for i in range(n_pts):
            key = (int(vox_coords[i, 0]), int(vox_coords[i, 1]), int(vox_coords[i, 2]))
            frame_voxel_data[key]["pts"].append(pts[i])
            frame_voxel_data[key]["cols"].append(cols[i])
            frame_voxel_data[key]["ids"].append(int(ids[i]))

        for key, data in frame_voxel_data.items():
            pts_arr = np.asarray(data["pts"], dtype=np.float32)
            cols_arr = np.asarray(data["cols"], dtype=np.float32)
            ids_arr = np.asarray(data["ids"], dtype=np.int32)

            frame_label = _intra_frame_vote(ids_arr, background_id)

            if key not in voxels:
                voxels[key] = {
                    "view_count": 0,  # number of frames that saw this voxel
                    "pos_sum": np.zeros(3, dtype=np.float64),
                    "color_sum": np.zeros(3, dtype=np.float64),
                    "n_points": 0,
                    "frame_votes": {},  # frame_id -> label (ONE vote per frame)
                }

            v = voxels[key]
            v["view_count"] += 1
            v["pos_sum"] += pts_arr.sum(axis=0).astype(np.float64)
            v["color_sum"] += cols_arr.sum(axis=0).astype(np.float64)
            v["n_points"] += int(len(pts_arr))
            v["frame_votes"][frame_id] = int(frame_label)

    if verbose:
        print(f"Processed {n_frames_seen} frames, {total_raw_points:,} raw points")
        print(f"Total voxels before filtering: {len(voxels):,}")

    out_points: List[np.ndarray] = []
    out_colors: List[np.ndarray] = []
    out_ids: List[int] = []

    for _, v in voxels.items():
        if int(v["view_count"]) < int(min_views):
            continue

        n = int(v["n_points"])
        if n <= 0:
            continue

        mean_pos = (v["pos_sum"] / n).astype(np.float32)
        mean_color = (v["color_sum"] / n).astype(np.float32)

        voted_label = _vote_semantic_label(
            v["frame_votes"],
            background_id=background_id,
            prefer_nonzero=prefer_nonzero,
            min_nonzero_votes=min_nonzero_votes,
            min_nonzero_ratio=min_nonzero_ratio,
        )

        out_points.append(mean_pos)
        out_colors.append(mean_color)
        out_ids.append(int(voted_label))

    kept = len(out_points)
    if verbose:
        pct = 100.0 * kept / max(1, len(voxels))
        print(f"Voxels after min_views filter: {kept:,} ({pct:.1f}% retained)")

    if kept == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    return (
        np.asarray(out_points, dtype=np.float32),
        np.asarray(out_colors, dtype=np.float32),
        np.asarray(out_ids, dtype=np.int32),
    )


# =============================================================================
# Geometry cleaning (label-aware) - Open3D DBSCAN
# =============================================================================
def remove_small_clusters_label_aware(
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    *,
    eps: float = 0.05,
    min_cluster_size: int = 20,
    min_samples: int = 5,
    background_id: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove small geometric clusters, but PROTECT labeled points.

    SAME signature + outputs as your file.

    Strategy (same as your docstring):
      - Background points (object_id == background_id): DBSCAN + keep clusters >= min_cluster_size
      - Labeled points (object_id != background_id): keep all (skip DBSCAN)

    Implementation detail:
      - Uses Open3D DBSCAN. Open3D uses `min_points` instead of sklearn `min_samples`.
        We map your `min_samples` -> Open3D `min_points`.
    """
    if len(points) == 0:
        return points, colors, object_ids

    if not HAS_OPEN3D:
        if verbose:
            print("Warning: Open3D not available, skipping DBSCAN clustering")
        return points, colors, object_ids

    if verbose:
        print(f"\nLabel-aware cluster cleaning (eps={eps}m, min_cluster_size={min_cluster_size})")

    labeled_mask = object_ids != background_id
    bg_mask = ~labeled_mask

    n_labeled = int(np.sum(labeled_mask))
    n_bg = int(np.sum(bg_mask))

    if verbose:
        print(f"  Labeled points: {n_labeled:,} (protected)")
        print(f"  Background points: {n_bg:,} (will be cleaned)")

    keep_mask = labeled_mask.copy()

    if n_bg > 0:
        bg_points = points[bg_mask]

        labels = _open3d_dbscan_labels(
            bg_points,
            eps=eps,
            min_points=min_samples,
            verbose=verbose,
        )

        unique_labels = np.unique(labels)
        cluster_sizes: Dict[int, int] = {}
        for lbl in unique_labels:
            if int(lbl) == -1:
                continue
            cluster_sizes[int(lbl)] = int(np.sum(labels == lbl))

        keep_clusters = set(lbl for lbl, size in cluster_sizes.items() if size >= min_cluster_size)

        bg_keep = np.isin(labels, list(keep_clusters))

        bg_indices = np.where(bg_mask)[0]
        keep_mask[bg_indices] = bg_keep

        if verbose:
            n_bg_kept = int(np.sum(bg_keep))
            n_bg_removed = n_bg - n_bg_kept
            print(f"  Background: kept {n_bg_kept:,}, removed {n_bg_removed:,} ({100*n_bg_removed/max(1,n_bg):.1f}%)")

    total_kept = int(np.sum(keep_mask))
    total_removed = len(points) - total_kept

    if verbose:
        print(f"  Total: {total_kept:,} points kept, {total_removed:,} removed")

    return points[keep_mask], colors[keep_mask], object_ids[keep_mask]


def statistical_outlier_removal(
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    *,
    nb_neighbors: int = 20,
    std_ratio: float = 2.5,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Open3D statistical outlier removal. Preserves object_ids by indexing.

    SAME signature + outputs as your file.
    """
    if len(points) == 0:
        return points, colors, object_ids

    if not HAS_OPEN3D:
        if verbose:
            print("Warning: Open3D not available, skipping statistical outlier removal")
        return points, colors, object_ids

    if verbose:
        print(f"\nStatistical outlier removal (neighbors={nb_neighbors}, std_ratio={std_ratio})")

    pcd = create_o3d_pointcloud(points, colors)
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))

    ind = np.asarray(ind, dtype=np.int64)
    removed = len(points) - len(ind)

    if verbose:
        pct = 100.0 * removed / max(1, len(points))
        print(f"  Removed {removed:,} outliers ({pct:.1f}%)")

    return points[ind], colors[ind], object_ids[ind]


def remove_disconnected_chunks(
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    *,
    eps: float = 0.15,
    min_samples: int = 10,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove floating chunks by keeping only the largest connected component.

    SAME signature + outputs as your file.

    NOTE: Your pasted version had a bug referencing `pcd` without defining it.
    This rewrite fixes that while preserving the intended behavior.
    """
    if len(points) == 0:
        return points, colors, object_ids

    if not HAS_OPEN3D:
        if verbose:
            print("Warning: Open3D not available, skipping disconnected chunk removal")
        return points, colors, object_ids

    if verbose:
        print(f"\nRemoving disconnected chunks (eps={eps}m, min_samples={min_samples})")

    labels = _open3d_dbscan_labels(
        points,
        eps=eps,
        min_points=min_samples,
        verbose=verbose,
    )

    unique_labels = np.unique(labels)
    cluster_labels = [int(l) for l in unique_labels.tolist() if int(l) != -1]

    if len(cluster_labels) == 0:
        if verbose:
            print("  Warning: No clusters found, returning original points")
        return points, colors, object_ids

    cluster_sizes = {l: int(np.sum(labels == l)) for l in cluster_labels}
    largest_label = max(cluster_sizes.keys(), key=lambda l: cluster_sizes[l])
    largest_size = cluster_sizes[largest_label]

    if verbose:
        print(f"  Found {len(cluster_labels)} clusters")
        for l in sorted(cluster_sizes.keys(), key=lambda x: -cluster_sizes[x])[:5]:
            print(f"    Cluster {l}: {cluster_sizes[l]:,} points")
        if len(cluster_labels) > 5:
            print(f"    ... and {len(cluster_labels) - 5} smaller clusters")

    keep_mask = labels == largest_label
    n_removed = len(points) - int(np.sum(keep_mask))

    if verbose:
        print(f"  Keeping largest cluster: {largest_size:,} points")
        print(f"  Removed {n_removed:,} points from {len(cluster_labels) - 1} smaller clusters + noise")

    return points[keep_mask], colors[keep_mask], object_ids[keep_mask]


# =============================================================================
# Semantic post-processing: remove tiny objects + remap contiguous IDs
# =============================================================================
def remap_object_ids(
    object_ids: np.ndarray,
    class_mapping: Optional[Dict[Any, Any]] = None,
    *,
    min_points: int = 15,
    background_id: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[int, int]]:
    """
    Filter tiny objects and remap IDs to be contiguous.

    SAME signature + outputs as your file:
      returns (object_ids_remapped, new_class_mapping, id_remap)
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

    ids_to_keep_sorted = sorted(set(ids_to_keep))
    if background_id in ids_to_keep_sorted:
        ids_to_keep_sorted.remove(background_id)
    ids_to_keep_sorted = [background_id] + ids_to_keep_sorted

    id_remap: Dict[int, int] = {}
    for new_id, old_id in enumerate(ids_to_keep_sorted):
        id_remap[int(old_id)] = int(new_id)

    for obj_id, _ in ids_to_merge:
        id_remap[int(obj_id)] = 0

    object_ids_remapped = np.array([id_remap[int(i)] for i in object_ids], dtype=np.int32)

    cm_int: Dict[int, Any] = {}
    if class_mapping is not None:
        for k, v in class_mapping.items():
            try:
                cm_int[int(k)] = v
            except Exception:
                pass

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
# Semantic summary (standalone utility)
# =============================================================================
def print_semantic_summary(
    object_ids: np.ndarray,
    class_mapping: Optional[Dict[str, Any]] = None,
) -> None:
    """Print a summary of semantic labels in the point cloud. Same as your file."""
    if len(object_ids) == 0:
        print("\nWARNING: No points in point cloud!")
        return

    uniq, counts = np.unique(object_ids, return_counts=True)

    print("\n" + "=" * 60)
    print(f"FINAL SEMANTIC SUMMARY: {len(uniq)} unique object IDs")
    print("=" * 60)

    for k, c in sorted(zip(uniq.tolist(), counts.tolist()), key=lambda x: -x[1]):
        pct = 100 * c / len(object_ids)
        name = "background"
        if class_mapping and str(k) in class_mapping:
            v = class_mapping[str(k)]
            if isinstance(v, dict) and "class_name" in v:
                name = v["class_name"]
            elif isinstance(v, str):
                name = v
            else:
                name = str(v)
        print(f"  ID {k:3d} ({name:20s}): {c:,} points ({pct:.1f}%)")


# =============================================================================
# Legacy compatibility wrapper (unchanged interface)
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
    min_nonzero_ratio: float = 0.05,
    background_id: int = 0,
    # Cluster filtering
    cluster_eps: float = 0.05,
    min_cluster_size: int = 20,
    keep_largest_n: Optional[int] = None,  # Ignored in this implementation (kept for signature compatibility)
    # Statistical outlier removal
    use_sor: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.5,
    # Semantic cleanup/remap
    remap_ids: bool = True,
    remap_min_points: int = 15,
    class_mapping: Optional[Dict[Any, Any]] = None,
    verbose_remap: bool = True,
    # return extra mapping info
    return_mappings: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[int, int]],
]:
    """
    Legacy wrapper: accepts pre-built lists, converts to generator internally.

    SAME signature + outputs as your file.
    """
    print("\n" + "=" * 60)
    print("POINT CLOUD PROCESSING WITH SEMANTICS")
    print("=" * 60)
    print("WARNING: Using legacy list interface. Consider using streaming_voxel_filter() directly.")

    def frame_gen():
        for pts, cols, ids in zip(points_by_frame, colors_by_frame, object_ids_by_frame):
            yield (pts, cols, ids)

    # Step 1: Streaming voxel filter
    points, colors, obj_ids = streaming_voxel_filter(
        frame_gen(),
        voxel_size=voxel_size,
        min_views=min_views,
        prefer_nonzero=prefer_nonzero,
        min_nonzero_votes=min_nonzero_votes,
        min_nonzero_ratio=min_nonzero_ratio,
        background_id=background_id,
        verbose=True,
    )

    if len(points) == 0:
        print("WARNING: No points remaining after voxel filtering!")
        if return_mappings:
            return points, colors, obj_ids, {"0": "background"}, {background_id: 0}
        return points, colors, obj_ids

    # Step 2: Label-aware cluster cleaning
    points, colors, obj_ids = remove_small_clusters_label_aware(
        points,
        colors,
        obj_ids,
        eps=cluster_eps,
        min_cluster_size=min_cluster_size,
        background_id=background_id,
        verbose=True,
    )

    # Step 3: Statistical outlier removal
    if use_sor and len(points) > 0:
        points, colors, obj_ids = statistical_outlier_removal(
            points,
            colors,
            obj_ids,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
            verbose=True,
        )

    # Step 4: Remap object IDs
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

    # Summary
    print_semantic_summary(obj_ids, new_class_mapping)

    if return_mappings:
        return points, colors, obj_ids, new_class_mapping, id_remap
    return points, colors, obj_ids


# =============================================================================
# Utility: create raw point cloud (unchanged interface)
# =============================================================================
def create_raw_pointcloud(
    points_by_frame: List[np.ndarray],
    colors_by_frame: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine all frames into a single raw point cloud (no semantics).

    SAME signature + outputs as your file.
    """
    if len(points_by_frame) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    return (
        np.vstack(points_by_frame).astype(np.float32),
        np.vstack(colors_by_frame).astype(np.float32),
    )
