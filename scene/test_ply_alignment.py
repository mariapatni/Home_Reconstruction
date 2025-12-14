"""
Diagnostic test to compare PLY point clouds with RGB-D point clouds
and verify coordinate transformation correctness.

Uses only numpy/torch - no open3d or visualizers.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData

# Add scene directory to path
sys.path.append(str(Path(__file__).parent))
from data_loaders.record3d_loader import (
    Record3DScene, 
    load_record3d_metadata,
    parse_pose
)
from test import create_point_cloud_from_rgbd


def load_ply_points(ply_path):
    """Load points from PLY file using plyfile (no open3d)"""
    plydata = PlyData.read(str(ply_path))
    
    # PLY files typically have 'vertex' element with 'x', 'y', 'z' properties
    vertex = plydata['vertex']
    
    # Extract x, y, z coordinates
    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])
    
    points = np.stack([x, y, z], axis=1)
    
    # Try to get colors if available
    colors = None
    if 'red' in vertex.dtype.names and 'green' in vertex.dtype.names and 'blue' in vertex.dtype.names:
        r = np.array(vertex['red'])
        g = np.array(vertex['green'])
        b = np.array(vertex['blue'])
        colors = np.stack([r, g, b], axis=1) / 255.0  # Normalize to [0, 1]
    
    return points, colors


def compute_point_cloud_overlap(points1, points2, radius=0.05):
    """
    Compute overlap between two point clouds by counting points within radius.
    Returns: overlap ratio (0-1), where 1 means perfect overlap.
    """
    from scipy.spatial import cKDTree
    
    if len(points1) == 0 or len(points2) == 0:
        return 0.0
    
    # Build KD-tree for faster nearest neighbor search
    tree = cKDTree(points2)
    
    # For each point in points1, find nearest point in points2
    distances, _ = tree.query(points1, k=1)
    
    # Count points within radius
    within_radius = np.sum(distances < radius)
    overlap_ratio = within_radius / len(points1)
    
    return overlap_ratio, np.mean(distances)


def test_single_frame_alignment(scene_path, frame_idx=0):
    """
    Compare PLY point cloud with RGB-D point cloud for a single frame.
    Uses numerical metrics only - no visualization.
    """
    print(f"\n{'='*60}")
    print(f"Testing Frame {frame_idx} Alignment")
    print(f"{'='*60}")
    
    scene_path = Path(scene_path)
    
    # Load metadata
    meta = load_record3d_metadata(scene_path)
    c2w = parse_pose(meta, frame_idx)
    camera_center = c2w[:3, 3]
    
    print(f"\nCamera center (from c2w): {camera_center}")
    
    # Load PLY file using plyfile
    ply_path = scene_path / "PLYs" / f"{frame_idx:05d}.ply"
    if not ply_path.exists():
        print(f"PLY file not found: {ply_path}")
        return None
    
    points_ply, colors_ply = load_ply_points(ply_path)
    
    print(f"\nPLY point cloud (raw):")
    print(f"  Number of points: {len(points_ply)}")
    print(f"  Bounding box: min={points_ply.min(axis=0)}, max={points_ply.max(axis=0)}")
    print(f"  Centroid: {points_ply.mean(axis=0)}")
    print(f"  Distance from camera center: {np.linalg.norm(points_ply.mean(axis=0) - camera_center):.3f}m")
    
    # Create RGB-D point cloud from same frame
    scene = Record3DScene(
        scene_path=scene_path,
        train_frames=[frame_idx],
        test_frames=[],
        gaussians=None
    )
    
    if len(scene.train_cameras) == 0:
        print("Could not create camera for RGB-D point cloud")
        return None
    
    cam = scene.train_cameras[0]
    points_rgbd, colors_rgbd = create_point_cloud_from_rgbd(cam, subsample=4)
    
    if points_rgbd is None or len(points_rgbd) == 0:
        print("Could not create RGB-D point cloud (no depth map?)")
        return None
    
    print(f"\nRGB-D point cloud:")
    print(f"  Number of points: {len(points_rgbd)}")
    print(f"  Bounding box: min={points_rgbd.min(axis=0)}, max={points_rgbd.max(axis=0)}")
    print(f"  Centroid: {points_rgbd.mean(axis=0)}")
    print(f"  Distance from camera center: {np.linalg.norm(points_rgbd.mean(axis=0) - camera_center):.3f}m")
    
    # Test different transformation approaches
    results = {}
    
    # Approach 1: No transformation (PLY already in world coords)
    points_ply_no_transform = points_ply.copy()
    centroid_diff_no_transform = np.linalg.norm(points_ply_no_transform.mean(axis=0) - points_rgbd.mean(axis=0))
    overlap_no_transform, avg_dist_no_transform = compute_point_cloud_overlap(
        points_ply_no_transform, points_rgbd, radius=0.05
    )
    
    # Approach 2: Current c2w transformation
    ones = np.ones((len(points_ply), 1))
    points_ply_hom = np.hstack([points_ply, ones])
    points_ply_c2w = (c2w @ points_ply_hom.T).T[:, :3]
    centroid_diff_c2w = np.linalg.norm(points_ply_c2w.mean(axis=0) - points_rgbd.mean(axis=0))
    overlap_c2w, avg_dist_c2w = compute_point_cloud_overlap(
        points_ply_c2w, points_rgbd, radius=0.05
    )
    
    # Approach 3: Transposed c2w
    c2w_transposed = c2w.T
    points_ply_transposed = (c2w_transposed @ points_ply_hom.T).T[:, :3]
    centroid_diff_transposed = np.linalg.norm(points_ply_transposed.mean(axis=0) - points_rgbd.mean(axis=0))
    overlap_transposed, avg_dist_transposed = compute_point_cloud_overlap(
        points_ply_transposed, points_rgbd, radius=0.05
    )
    
    # Approach 4: Inverse c2w (w2c) - maybe PLY is in world and needs inverse?
    w2c = np.linalg.inv(c2w)
    points_ply_w2c = (w2c @ points_ply_hom.T).T[:, :3]
    centroid_diff_w2c = np.linalg.norm(points_ply_w2c.mean(axis=0) - points_rgbd.mean(axis=0))
    overlap_w2c, avg_dist_w2c = compute_point_cloud_overlap(
        points_ply_w2c, points_rgbd, radius=0.05
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("Transformation Comparison:")
    print(f"{'='*60}")
    print(f"{'Approach':<25} {'Centroid Diff':<15} {'Overlap':<12} {'Avg Distance':<15}")
    print(f"{'-'*70}")
    print(f"{'No transformation':<25} {centroid_diff_no_transform:>10.3f}m    {overlap_no_transform:>8.2%}    {avg_dist_no_transform:>10.3f}m")
    print(f"{'c2w (current)':<25} {centroid_diff_c2w:>10.3f}m    {overlap_c2w:>8.2%}    {avg_dist_c2w:>10.3f}m")
    print(f"{'c2w transposed':<25} {centroid_diff_transposed:>10.3f}m    {overlap_transposed:>8.2%}    {avg_dist_transposed:>10.3f}m")
    print(f"{'w2c (inverse)':<25} {centroid_diff_w2c:>10.3f}m    {overlap_w2c:>8.2%}    {avg_dist_w2c:>10.3f}m")
    
    # Determine best approach
    approaches = [
        ('no_transform', centroid_diff_no_transform, overlap_no_transform, avg_dist_no_transform),
        ('c2w', centroid_diff_c2w, overlap_c2w, avg_dist_c2w),
        ('c2w_transposed', centroid_diff_transposed, overlap_transposed, avg_dist_transposed),
        ('w2c', centroid_diff_w2c, overlap_w2c, avg_dist_w2c),
    ]
    
    # Best is highest overlap and lowest distance
    best = min(approaches, key=lambda x: (x[2] * -1, x[3]))  # Maximize overlap, minimize distance
    
    print(f"\n{'='*60}")
    print(f"Best approach: {best[0]}")
    print(f"  Centroid difference: {best[1]:.3f}m")
    print(f"  Overlap ratio: {best[2]:.2%}")
    print(f"  Average distance: {best[3]:.3f}m")
    
    return {
        'frame_idx': frame_idx,
        'camera_center': camera_center,
        'approaches': {
            'no_transform': {
                'centroid_diff': centroid_diff_no_transform,
                'overlap': overlap_no_transform,
                'avg_distance': avg_dist_no_transform
            },
            'c2w': {
                'centroid_diff': centroid_diff_c2w,
                'overlap': overlap_c2w,
                'avg_distance': avg_dist_c2w
            },
            'c2w_transposed': {
                'centroid_diff': centroid_diff_transposed,
                'overlap': overlap_transposed,
                'avg_distance': avg_dist_transposed
            },
            'w2c': {
                'centroid_diff': centroid_diff_w2c,
                'overlap': overlap_w2c,
                'avg_distance': avg_dist_w2c
            }
        },
        'best_approach': best[0]
    }


def test_multiple_frames(scene_path, frame_indices=[0, 10, 20, 30]):
    """Test alignment for multiple frames and determine best approach"""
    results = []
    for idx in frame_indices:
        result = test_single_frame_alignment(scene_path, idx)
        if result:
            results.append(result)
    
    if len(results) == 0:
        print("No valid results to summarize")
        return
    
    print(f"\n{'='*60}")
    print("Summary across frames:")
    print(f"{'='*60}")
    
    # Aggregate statistics for each approach
    approaches = ['no_transform', 'c2w', 'c2w_transposed', 'w2c']
    stats = {approach: {'centroid_diffs': [], 'overlaps': [], 'avg_distances': []} 
             for approach in approaches}
    
    for result in results:
        for approach in approaches:
            stats[approach]['centroid_diffs'].append(result['approaches'][approach]['centroid_diff'])
            stats[approach]['overlaps'].append(result['approaches'][approach]['overlap'])
            stats[approach]['avg_distances'].append(result['approaches'][approach]['avg_distance'])
    
    print(f"\n{'Approach':<25} {'Avg Centroid Diff':<18} {'Avg Overlap':<15} {'Avg Distance':<15}")
    print(f"{'-'*75}")
    
    for approach in approaches:
        avg_centroid = np.mean(stats[approach]['centroid_diffs'])
        avg_overlap = np.mean(stats[approach]['overlaps'])
        avg_dist = np.mean(stats[approach]['avg_distances'])
        print(f"{approach:<25} {avg_centroid:>12.3f}m      {avg_overlap:>10.2%}      {avg_dist:>10.3f}m")
    
    # Determine best approach (highest overlap, lowest distance)
    best_scores = {}
    for approach in approaches:
        # Score = overlap - normalized_distance (higher is better)
        avg_overlap = np.mean(stats[approach]['overlaps'])
        avg_dist = np.mean(stats[approach]['avg_distances'])
        # Normalize distance (assuming max reasonable distance is 10m)
        normalized_dist = min(avg_dist / 10.0, 1.0)
        score = avg_overlap - normalized_dist * 0.5
        best_scores[approach] = score
    
    best_approach = max(best_scores.items(), key=lambda x: x[1])[0]
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDED APPROACH: {best_approach}")
    print(f"  Average centroid difference: {np.mean(stats[best_approach]['centroid_diffs']):.3f}m")
    print(f"  Average overlap: {np.mean(stats[best_approach]['overlaps']):.2%}")
    print(f"  Average distance: {np.mean(stats[best_approach]['avg_distances']):.3f}m")
    print(f"{'='*60}")


if __name__ == "__main__":
    scene_path = Path("/workspace/Home_Reconstruction/data_scenes/maria_bedroom")
    
    # Test single frame
    test_single_frame_alignment(scene_path, frame_idx=0)
    
    # Test multiple frames
    # test_multiple_frames(scene_path, frame_indices=[0, 10, 20, 30])

