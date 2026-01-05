"""
Multiview point cloud filtering utilities with semantic support.
STREAMING IMPLEMENTATION - processes frames incrementally to avoid memory blowup.

OPTIMIZED VERSION with DETAILED TIMING:
- Uses vectorized numpy with voxel key hashing (no Python loops over points)
- Open3D for DBSCAN and statistical outlier removal
- Comprehensive timing logs for performance analysis

Key behavior (semantic-preserving):
- ID 0 is treated as "unlabeled/background".
- Semantic voting is per-frame-per-voxel (not per-point), preventing dense textures from dominating.
- If a voxel has enough non-zero evidence, background (0) cannot outvote it.
"""

from __future__ import annotations

import time
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
# Timing utilities
# =============================================================================
class TimingStats:
    """Accumulate timing statistics across multiple calls."""
    
    def __init__(self):
        self.times: Dict[str, List[float]] = defaultdict(list)
    
    def add(self, name: str, elapsed: float):
        self.times[name].append(elapsed)
    
    def summary(self) -> str:
        lines = ["", "=" * 70, "TIMING SUMMARY", "=" * 70]
        
        total_time = sum(sum(t) for t in self.times.values())
        
        for name in sorted(self.times.keys()):
            times = self.times[name]
            step_total = sum(times)
            count = len(times)
            avg = step_total / count if count > 0 else 0
            pct = 100 * step_total / total_time if total_time > 0 else 0
            lines.append(f"  {name:45s}: {step_total:8.2f}s ({pct:5.1f}%)  [{count} calls, {avg:.4f}s avg]")
        
        lines.append("-" * 70)
        lines.append(f"  {'TOTAL':45s}: {total_time:8.2f}s")
        lines.append("=" * 70)
        
        return "\n".join(lines)


# Global timing stats
_timing_stats = TimingStats()


def reset_timing():
    """Reset timing statistics."""
    global _timing_stats
    _timing_stats = TimingStats()


def print_timing_summary():
    """Print the accumulated timing summary."""
    global _timing_stats
    print(_timing_stats.summary())


# =============================================================================
# Helpers
# =============================================================================
def create_o3d_pointcloud(points: np.ndarray, colors: Optional[np.ndarray] = None):
    """Create Open3D point cloud from numpy arrays."""
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required. Install with: pip install open3d")

    pcd = o3d.geometry.PointCloud()
    if points is None or len(points) == 0:
        return pcd

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
    """Run Open3D DBSCAN. Returns labels (N,) int32. Noise = -1."""
    if points_xyz is None or len(points_xyz) == 0:
        return np.zeros((0,), dtype=np.int32)

    pcd = create_o3d_pointcloud(points_xyz, colors=None)
    labels = np.array(
        pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=bool(verbose)),
        dtype=np.int32,
    )
    return labels


# =============================================================================
# Vectorized voxel operations
# =============================================================================
def _compute_voxel_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Compute integer voxel keys for points. VECTORIZED.
    """
    vox_coords = np.floor(points / voxel_size).astype(np.int64)
    offset = 2**20
    vox_coords = vox_coords + offset
    stride = 2**21
    keys = vox_coords[:, 2] * (stride * stride) + vox_coords[:, 1] * stride + vox_coords[:, 0]
    return keys


def _vectorized_intra_frame_vote_fast(
    voxel_keys: np.ndarray,
    object_ids: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    background_id: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    VECTORIZED intra-frame aggregation.
    """
    n_points = len(voxel_keys)
    
    sort_idx = np.argsort(voxel_keys)
    sorted_keys = voxel_keys[sort_idx]
    sorted_ids = object_ids[sort_idx]
    sorted_pts = points[sort_idx]
    sorted_cols = colors[sort_idx]
    
    unique_keys, group_starts, counts = np.unique(
        sorted_keys, return_index=True, return_counts=True
    )
    n_voxels = len(unique_keys)
    
    pts_cumsum = np.zeros((n_points + 1, 3), dtype=np.float64)
    pts_cumsum[1:] = np.cumsum(sorted_pts.astype(np.float64), axis=0)
    
    cols_cumsum = np.zeros((n_points + 1, 3), dtype=np.float64)
    cols_cumsum[1:] = np.cumsum(sorted_cols.astype(np.float64), axis=0)
    
    group_ends = group_starts + counts
    
    pos_sums = pts_cumsum[group_ends] - pts_cumsum[group_starts]
    col_sums = cols_cumsum[group_ends] - cols_cumsum[group_starts]
    
    voted_labels = np.zeros(n_voxels, dtype=np.int32)
    
    for i in range(n_voxels):
        start = group_starts[i]
        end = group_ends[i]
        group_ids = sorted_ids[start:end]
        
        nonzero = group_ids[group_ids != background_id]
        if len(nonzero) > 0:
            voted_labels[i] = np.bincount(nonzero).argmax()
        else:
            voted_labels[i] = background_id
    
    return unique_keys, voted_labels, pos_sums, col_sums, counts.astype(np.int32)


# =============================================================================
# Core: Streaming voxel multiview filtering (OPTIMIZED + TIMED)
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
    STREAMING multiview voxel filter with semantic voting. OPTIMIZED + TIMED.
    """
    global _timing_stats
    reset_timing()
    
    voxels: Dict[int, Dict[str, Any]] = {}
    
    total_raw_points = 0
    n_frames_seen = 0
    
    time_voxel_keys = 0.0
    time_intra_frame = 0.0
    time_merge = 0.0
    time_frame_load = 0.0

    if verbose:
        print(f"\n{'='*70}")
        print("STREAMING VOXEL FILTER")
        print(f"{'='*70}")
        print(f"  voxel_size={voxel_size}m, min_views={min_views}")
        print("Processing frames...")

    t_total_start = time.perf_counter()
    t_last_frame_end = t_total_start

    for frame_id, (pts, cols, ids) in enumerate(frame_generator):
        t_frame_received = time.perf_counter()
        time_frame_load += t_frame_received - t_last_frame_end
        
        if pts is None or len(pts) == 0:
            t_last_frame_end = time.perf_counter()
            continue

        n_frames_seen += 1
        n_pts = len(pts)
        total_raw_points += n_pts

        if cols is None or len(cols) != n_pts:
            raise ValueError(f"Frame {frame_id}: colors missing or wrong shape.")
        if ids is None or len(ids) != n_pts:
            raise ValueError(f"Frame {frame_id}: object_ids missing or wrong shape.")

        pts = np.asarray(pts, dtype=np.float32)
        cols = np.asarray(cols, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int32)

        t0 = time.perf_counter()
        voxel_keys = _compute_voxel_keys(pts, voxel_size)
        time_voxel_keys += time.perf_counter() - t0
        
        t0 = time.perf_counter()
        unique_keys, voted_labels, pos_sums, col_sums, counts = _vectorized_intra_frame_vote_fast(
            voxel_keys, ids, pts, cols, background_id
        )
        time_intra_frame += time.perf_counter() - t0
        
        t0 = time.perf_counter()
        for i in range(len(unique_keys)):
            key = int(unique_keys[i])
            
            if key not in voxels:
                voxels[key] = {
                    "view_count": 0,
                    "pos_sum": np.zeros(3, dtype=np.float64),
                    "color_sum": np.zeros(3, dtype=np.float64),
                    "n_points": 0,
                    "frame_votes": {},
                }
            
            v = voxels[key]
            v["view_count"] += 1
            v["pos_sum"] += pos_sums[i]
            v["color_sum"] += col_sums[i]
            v["n_points"] += int(counts[i])
            v["frame_votes"][frame_id] = int(voted_labels[i])
        time_merge += time.perf_counter() - t0
        
        t_last_frame_end = time.perf_counter()

    t_frame_processing = time.perf_counter() - t_total_start

    _timing_stats.add("01. Frame loading (generator)", time_frame_load)
    _timing_stats.add("02. Voxel key computation", time_voxel_keys)
    _timing_stats.add("03. Intra-frame aggregation", time_intra_frame)
    _timing_stats.add("04. Global voxel merge", time_merge)

    if verbose:
        print(f"\nFrame processing complete:")
        print(f"  Frames: {n_frames_seen}, Points: {total_raw_points:,}, Voxels: {len(voxels):,}")
        print(f"\n  Timing breakdown:")
        print(f"    {'Frame loading (I/O + camera):':<35} {time_frame_load:8.2f}s ({100*time_frame_load/t_frame_processing:5.1f}%)")
        print(f"    {'Voxel key computation:':<35} {time_voxel_keys:8.2f}s ({100*time_voxel_keys/t_frame_processing:5.1f}%)")
        print(f"    {'Intra-frame aggregation:':<35} {time_intra_frame:8.2f}s ({100*time_intra_frame/t_frame_processing:5.1f}%)")
        print(f"    {'Global voxel merge:':<35} {time_merge:8.2f}s ({100*time_merge/t_frame_processing:5.1f}%)")
        print(f"    {'TOTAL:':<35} {t_frame_processing:8.2f}s")

    # Build output arrays
    if verbose:
        print(f"\nBuilding output arrays...")
    
    t_output_start = time.perf_counter()
    
    out_points = []
    out_colors = []
    out_ids = []

    t_voting_start = time.perf_counter()
    
    for key, v in voxels.items():
        if v["view_count"] < min_views:
            continue

        n = v["n_points"]
        if n <= 0:
            continue

        mean_pos = (v["pos_sum"] / n).astype(np.float32)
        mean_color = (v["color_sum"] / n).astype(np.float32)

        frame_votes = v["frame_votes"]
        labels = list(frame_votes.values())
        
        if prefer_nonzero:
            nonzero_labels = [lbl for lbl in labels if lbl != background_id]
            n_nonzero = len(nonzero_labels)
            
            if n_nonzero < min_nonzero_votes:
                voted_label = background_id
            elif n_nonzero / len(labels) < min_nonzero_ratio:
                voted_label = background_id
            else:
                voted_label = max(set(nonzero_labels), key=nonzero_labels.count)
        else:
            voted_label = max(set(labels), key=labels.count)

        out_points.append(mean_pos)
        out_colors.append(mean_color)
        out_ids.append(voted_label)

    time_voting = time.perf_counter() - t_voting_start
    _timing_stats.add("05. Cross-frame voting", time_voting)
    
    t_convert_start = time.perf_counter()
    
    if len(out_points) == 0:
        result = (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    else:
        result = (
            np.array(out_points, dtype=np.float32),
            np.array(out_colors, dtype=np.float32),
            np.array(out_ids, dtype=np.int32),
        )
    
    time_convert = time.perf_counter() - t_convert_start
    _timing_stats.add("06. Array conversion", time_convert)
    
    time_output = time.perf_counter() - t_output_start

    kept = len(out_points)
    if verbose:
        pct = 100.0 * kept / max(1, len(voxels))
        print(f"  Voxels kept: {kept:,} / {len(voxels):,} ({pct:.1f}%)")
        print(f"\n  Output timing:")
        print(f"    {'Cross-frame voting:':<35} {time_voting:8.2f}s")
        print(f"    {'Array conversion:':<35} {time_convert:8.2f}s")
        print(f"    {'TOTAL output:':<35} {time_output:8.2f}s")

    return result


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
    """Remove small geometric clusters, but PROTECT labeled points."""
    global _timing_stats
    
    if len(points) == 0:
        return points, colors, object_ids

    if not HAS_OPEN3D:
        if verbose:
            print("Warning: Open3D not available, skipping DBSCAN clustering")
        return points, colors, object_ids

    if verbose:
        print(f"\n{'='*70}")
        print("LABEL-AWARE CLUSTER CLEANING")
        print(f"{'='*70}")
        print(f"  eps={eps}m, min_cluster_size={min_cluster_size}")

    t_total_start = time.perf_counter()

    t_mask_start = time.perf_counter()
    labeled_mask = object_ids != background_id
    bg_mask = ~labeled_mask
    n_labeled = int(np.sum(labeled_mask))
    n_bg = int(np.sum(bg_mask))
    time_mask = time.perf_counter() - t_mask_start

    if verbose:
        print(f"  Labeled: {n_labeled:,} (protected), Background: {n_bg:,} (to clean)")

    keep_mask = labeled_mask.copy()

    time_dbscan = 0.0
    time_filter = 0.0

    if n_bg > 0:
        t_extract_start = time.perf_counter()
        bg_points = points[bg_mask]
        time_extract = time.perf_counter() - t_extract_start
        
        t_dbscan_start = time.perf_counter()
        labels = _open3d_dbscan_labels(bg_points, eps=eps, min_points=min_samples, verbose=verbose)
        time_dbscan = time.perf_counter() - t_dbscan_start

        t_filter_start = time.perf_counter()
        unique_labels = np.unique(labels)
        cluster_sizes = {int(l): int(np.sum(labels == l)) for l in unique_labels if l != -1}
        keep_clusters = {l for l, size in cluster_sizes.items() if size >= min_cluster_size}
        bg_keep = np.isin(labels, list(keep_clusters))
        bg_indices = np.where(bg_mask)[0]
        keep_mask[bg_indices] = bg_keep
        time_filter = time.perf_counter() - t_filter_start

        _timing_stats.add("07. DBSCAN (background)", time_dbscan)

        if verbose:
            n_bg_kept = int(np.sum(bg_keep))
            n_bg_removed = n_bg - n_bg_kept
            print(f"\n  Timing:")
            print(f"    {'Extract background:':<35} {time_extract:8.2f}s")
            print(f"    {'DBSCAN clustering:':<35} {time_dbscan:8.2f}s")
            print(f"    {'Filter small clusters:':<35} {time_filter:8.2f}s")
            print(f"  Result: kept {n_bg_kept:,}, removed {n_bg_removed:,} ({100*n_bg_removed/max(1,n_bg):.1f}%)")

    t_apply_start = time.perf_counter()
    result = points[keep_mask], colors[keep_mask], object_ids[keep_mask]
    time_apply = time.perf_counter() - t_apply_start

    time_total = time.perf_counter() - t_total_start

    if verbose:
        total_kept = int(np.sum(keep_mask))
        total_removed = len(points) - total_kept
        print(f"\n  Total: {total_kept:,} kept, {total_removed:,} removed")
        print(f"  ⏱ Step total: {time_total:.2f}s")

    return result

def remove_small_clusters_label_aware_v2(
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    *,
    eps: float = 0.05,
    min_cluster_size: int = 20,
    min_samples: int = 5,
    background_id: int = 0,
    protect_labeled: bool = False,  # NEW: option to protect or not
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove small geometric clusters, optionally per semantic label."""
    
    if len(points) == 0 or not HAS_OPEN3D:
        return points, colors, object_ids
    
    keep_mask = np.ones(len(points), dtype=bool)
    
    unique_labels = np.unique(object_ids)
    
    for label in unique_labels:
        # Optionally skip labeled objects
        if protect_labeled and label != background_id:
            continue
        
        label_mask = object_ids == label
        label_points = points[label_mask]
        
        if len(label_points) < min_cluster_size:
            # Entire label is too small, remove it
            keep_mask[label_mask] = False
            continue
        
        # DBSCAN within this label
        cluster_labels = _open3d_dbscan_labels(
            label_points, eps=eps, min_points=min_samples
        )
        
        # Find which clusters are large enough
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        small_clusters = set(
            c for c, cnt in zip(unique_clusters, counts) 
            if c == -1 or cnt < min_cluster_size  # -1 is DBSCAN noise
        )
        
        # Mark small clusters for removal
        label_indices = np.where(label_mask)[0]
        for i, cluster in enumerate(cluster_labels):
            if cluster in small_clusters:
                keep_mask[label_indices[i]] = False
    
    if verbose:
        n_removed = len(points) - np.sum(keep_mask)
        print(f"  Removed {n_removed:,} points from small clusters")
    
    return points[keep_mask], colors[keep_mask], object_ids[keep_mask]


def statistical_outlier_removal(
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    *,
    nb_neighbors: int = 20,
    std_ratio: float = 1.5,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open3D statistical outlier removal."""
    global _timing_stats
    
    if len(points) == 0:
        return points, colors, object_ids

    if not HAS_OPEN3D:
        if verbose:
            print("Warning: Open3D not available, skipping statistical outlier removal")
        return points, colors, object_ids

    if verbose:
        print(f"\n{'='*70}")
        print("STATISTICAL OUTLIER REMOVAL")
        print(f"{'='*70}")
        print(f"  neighbors={nb_neighbors}, std_ratio={std_ratio}, points={len(points):,}")

    t_start = time.perf_counter()
    
    t_pcd_start = time.perf_counter()
    pcd = create_o3d_pointcloud(points, colors)
    time_pcd = time.perf_counter() - t_pcd_start
    
    t_sor_start = time.perf_counter()
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
    time_sor = time.perf_counter() - t_sor_start
    
    t_index_start = time.perf_counter()
    ind = np.asarray(ind, dtype=np.int64)
    result = points[ind], colors[ind], object_ids[ind]
    time_index = time.perf_counter() - t_index_start
    
    time_total = time.perf_counter() - t_start
    removed = len(points) - len(ind)

    _timing_stats.add("08. Statistical outlier removal", time_sor)

    if verbose:
        pct = 100.0 * removed / max(1, len(points))
        print(f"\n  Timing:")
        print(f"    {'Create point cloud:':<35} {time_pcd:8.2f}s")
        print(f"    {'SOR algorithm:':<35} {time_sor:8.2f}s")
        print(f"    {'Apply indices:':<35} {time_index:8.2f}s")
        print(f"  Result: removed {removed:,} outliers ({pct:.1f}%)")
        print(f"  ⏱ Step total: {time_total:.2f}s")

    return result


def remove_disconnected_chunks(
    points: np.ndarray,
    colors: np.ndarray,
    object_ids: np.ndarray,
    *,
    eps: float = 0.15,
    min_samples: int = 10,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove floating chunks by keeping only the largest connected component."""
    global _timing_stats
    
    if len(points) == 0:
        return points, colors, object_ids

    if not HAS_OPEN3D:
        if verbose:
            print("Warning: Open3D not available, skipping disconnected chunk removal")
        return points, colors, object_ids

    if verbose:
        print(f"\n{'='*70}")
        print("REMOVING DISCONNECTED CHUNKS")
        print(f"{'='*70}")
        print(f"  eps={eps}m, min_samples={min_samples}, points={len(points):,}")

    t_start = time.perf_counter()
    
    t_dbscan_start = time.perf_counter()
    labels = _open3d_dbscan_labels(points, eps=eps, min_points=min_samples, verbose=verbose)
    time_dbscan = time.perf_counter() - t_dbscan_start
    
    _timing_stats.add("09. DBSCAN (chunk detection)", time_dbscan)

    t_analysis_start = time.perf_counter()
    unique_labels = np.unique(labels)
    cluster_labels = [int(l) for l in unique_labels if l != -1]

    if len(cluster_labels) == 0:
        if verbose:
            print("  Warning: No clusters found, returning original points")
        return points, colors, object_ids

    cluster_sizes = {l: int(np.sum(labels == l)) for l in cluster_labels}
    largest_label = max(cluster_sizes.keys(), key=lambda l: cluster_sizes[l])
    largest_size = cluster_sizes[largest_label]
    time_analysis = time.perf_counter() - t_analysis_start

    t_mask_start = time.perf_counter()
    keep_mask = labels == largest_label
    result = points[keep_mask], colors[keep_mask], object_ids[keep_mask]
    time_mask = time.perf_counter() - t_mask_start
    
    n_removed = len(points) - int(np.sum(keep_mask))
    time_total = time.perf_counter() - t_start

    if verbose:
        print(f"\n  Timing:")
        print(f"    {'DBSCAN clustering:':<35} {time_dbscan:8.2f}s")
        print(f"    {'Cluster analysis:':<35} {time_analysis:8.2f}s")
        print(f"    {'Apply mask:':<35} {time_mask:8.2f}s")
        print(f"  Found {len(cluster_labels)} clusters, keeping largest: {largest_size:,} points")
        print(f"  Removed {n_removed:,} points from smaller clusters + noise")
        print(f"  ⏱ Step total: {time_total:.2f}s")

    return result


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
    """Filter tiny objects and remap IDs to be contiguous."""
    global _timing_stats
    
    if object_ids is None or len(object_ids) == 0:
        return object_ids.astype(np.int32), {"0": {"class_name": "background", "instance_id": 0}}, {background_id: 0}

    if verbose:
        print(f"\n{'='*70}")
        print("REMAPPING OBJECT IDs")
        print(f"{'='*70}")

    t_start = time.perf_counter()
    object_ids = np.asarray(object_ids, dtype=np.int32)

    unique_ids, counts = np.unique(object_ids, return_counts=True)
    id_counts = dict(zip(unique_ids.tolist(), counts.tolist()))

    if verbose:
        print(f"  Input: {len(unique_ids)} unique IDs, max={int(unique_ids.max())}")

    # Build lookup
    cm_lookup: Dict[int, Dict[str, Any]] = {}
    if class_mapping is not None:
        for k, v in class_mapping.items():
            try:
                key = int(k)
                if isinstance(v, dict):
                    cm_lookup[key] = {"class_name": v.get("class_name", f"object_{key}"), "instance_id": v.get("instance_id", 0)}
                elif isinstance(v, str):
                    cm_lookup[key] = {"class_name": v, "instance_id": 0}
                else:
                    cm_lookup[key] = {"class_name": str(v), "instance_id": 0}
            except (ValueError, TypeError):
                pass

    # Filter
    ids_to_keep = []
    ids_to_merge = []
    for obj_id, count in sorted(id_counts.items(), key=lambda x: -x[1]):
        if obj_id == background_id or count >= min_points:
            ids_to_keep.append(obj_id)
        else:
            ids_to_merge.append((obj_id, count))

    if verbose:
        print(f"  Keeping {len(ids_to_keep)} objects with >= {min_points} points")
        if ids_to_merge:
            print(f"  Merging {len(ids_to_merge)} tiny objects into background")

    # Build remap
    ids_to_keep_sorted = sorted(set(ids_to_keep))
    if background_id in ids_to_keep_sorted:
        ids_to_keep_sorted.remove(background_id)
    ids_to_keep_sorted = [background_id] + ids_to_keep_sorted

    id_remap: Dict[int, int] = {}
    for new_id, old_id in enumerate(ids_to_keep_sorted):
        id_remap[int(old_id)] = int(new_id)
    for obj_id, _ in ids_to_merge:
        id_remap[int(obj_id)] = 0

    # Apply VECTORIZED
    max_old_id = max(id_remap.keys()) + 1
    remap_table = np.zeros(max_old_id, dtype=np.int32)
    for old_id, new_id in id_remap.items():
        if old_id < max_old_id:
            remap_table[old_id] = new_id
    
    object_ids_clipped = np.clip(object_ids, 0, max_old_id - 1)
    object_ids_remapped = remap_table[object_ids_clipped]

    # Build new mapping
    new_class_mapping: Dict[str, Dict[str, Any]] = {"0": {"class_name": "background", "instance_id": 0}}
    for old_id, new_id in sorted(id_remap.items(), key=lambda x: x[1]):
        if new_id == 0:
            continue
        if old_id in cm_lookup:
            new_class_mapping[str(new_id)] = cm_lookup[old_id].copy()
        else:
            new_class_mapping[str(new_id)] = {"class_name": f"object_{old_id}", "instance_id": 0}

    time_total = time.perf_counter() - t_start
    _timing_stats.add("10. ID remapping", time_total)

    if verbose:
        new_unique = np.unique(object_ids_remapped)
        print(f"  Output: {len(new_unique)} unique IDs, max={int(new_unique.max())}")
        print(f"  ⏱ Step total: {time_total:.2f}s")

    return object_ids_remapped, new_class_mapping, id_remap


# =============================================================================
# Semantic summary
# =============================================================================
def print_semantic_summary(
    object_ids: np.ndarray,
    class_mapping: Optional[Dict[str, Any]] = None,
) -> None:
    """Print a summary of semantic labels in the point cloud."""
    if len(object_ids) == 0:
        print("\nWARNING: No points in point cloud!")
        return

    uniq, counts = np.unique(object_ids, return_counts=True)

    print("\n" + "=" * 70)
    print(f"FINAL SEMANTIC SUMMARY: {len(uniq)} unique object IDs")
    print("=" * 70)

    for k, c in sorted(zip(uniq.tolist(), counts.tolist()), key=lambda x: -x[1]):
        pct = 100 * c / len(object_ids)
        name = "background"
        
        if class_mapping and str(k) in class_mapping:
            v = class_mapping[str(k)]
            if isinstance(v, dict):
                class_name = v.get("class_name", f"object_{k}")
                instance_id = v.get("instance_id", 0)
                name = f"{class_name} (inst {instance_id})"
            elif isinstance(v, str):
                name = v
            else:
                name = str(v)
        elif k != 0:
            name = f"object_{k}"
            
        print(f"  ID {k:3d} ({name:30s}): {c:,} points ({pct:.1f}%)")


# =============================================================================
# Legacy wrapper
# =============================================================================
def process_pointcloud_with_semantics(
    points_by_frame: List[np.ndarray],
    colors_by_frame: List[np.ndarray],
    object_ids_by_frame: List[np.ndarray],
    *,
    voxel_size: float = 0.01,
    min_views: int = 2,
    prefer_nonzero: bool = True,
    min_nonzero_votes: int = 1,
    min_nonzero_ratio: float = 0.05,
    background_id: int = 0,
    cluster_eps: float = 0.05,
    min_cluster_size: int = 20,
    keep_largest_n: Optional[int] = None,
    use_sor: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.5,
    remap_ids: bool = True,
    remap_min_points: int = 15,
    class_mapping: Optional[Dict[Any, Any]] = None,
    verbose_remap: bool = True,
    return_mappings: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[int, int]],
]:
    """Legacy wrapper for list-based input."""
    print("\n" + "=" * 70)
    print("POINT CLOUD PROCESSING WITH SEMANTICS")
    print("=" * 70)

    def frame_gen():
        for pts, cols, ids in zip(points_by_frame, colors_by_frame, object_ids_by_frame):
            yield (pts, cols, ids)

    points, colors, obj_ids = streaming_voxel_filter(
        frame_gen(),
        voxel_size=voxel_size, min_views=min_views,
        prefer_nonzero=prefer_nonzero, min_nonzero_votes=min_nonzero_votes,
        min_nonzero_ratio=min_nonzero_ratio, background_id=background_id,
        verbose=True,
    )

    if len(points) == 0:
        if return_mappings:
            return points, colors, obj_ids, {"0": {"class_name": "background", "instance_id": 0}}, {background_id: 0}
        return points, colors, obj_ids

    points, colors, obj_ids = remove_small_clusters_label_aware(
        points, colors, obj_ids,
        eps=cluster_eps, min_cluster_size=min_cluster_size,
        background_id=background_id, verbose=True,
    )

    if use_sor and len(points) > 0:
        points, colors, obj_ids = statistical_outlier_removal(
            points, colors, obj_ids,
            nb_neighbors=nb_neighbors, std_ratio=std_ratio, verbose=True,
        )

    new_class_mapping = {"0": {"class_name": "background", "instance_id": 0}}
    id_remap = {background_id: 0}
    
    if remap_ids and len(obj_ids) > 0:
        obj_ids, new_class_mapping, id_remap = remap_object_ids(
            obj_ids, class_mapping=class_mapping,
            min_points=remap_min_points, background_id=background_id,
            verbose=verbose_remap,
        )

    print_semantic_summary(obj_ids, new_class_mapping)
    print_timing_summary()

    if return_mappings:
        return points, colors, obj_ids, new_class_mapping, id_remap
    return points, colors, obj_ids


def create_raw_pointcloud(
    points_by_frame: List[np.ndarray],
    colors_by_frame: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine all frames into a single raw point cloud."""
    if len(points_by_frame) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return (
        np.vstack(points_by_frame).astype(np.float32),
        np.vstack(colors_by_frame).astype(np.float32),
    )