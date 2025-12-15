"""
Multiview point cloud filtering utilities using Open3D.
"""

import open3d as o3d
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import DBSCAN



def create_o3d_pointcloud(points, colors):
    """
    Create Open3D point cloud from numpy arrays.
    
    Args:
        points: numpy array (N, 3) - XYZ coordinates
        colors: numpy array (N, 3) - RGB colors [0-1]
    
    Returns:
        o3d.geometry.PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def per_frame_voxel_downsample(points_by_frame, colors_by_frame, voxel_size):
    """
    Voxelize each frame individually using Open3D's voxel downsampling.
    
    Args:
        points_by_frame: list of numpy arrays (N, 3)
        colors_by_frame: list of numpy arrays (N, 3)
        voxel_size: voxel size in meters (e.g., 0.01 = 1cm)
    
    Returns:
        downsampled_points, downsampled_colors (lists of numpy arrays)
    """
    downsampled_points = []
    downsampled_colors = []
    
    print(f"Voxelizing {len(points_by_frame)} frames (voxel size: {voxel_size}m)...")
    for points, colors in tqdm(zip(points_by_frame, colors_by_frame), 
                                total=len(points_by_frame)):
        pcd = create_o3d_pointcloud(points, colors)
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        downsampled_points.append(np.asarray(pcd_down.points))
        downsampled_colors.append(np.asarray(pcd_down.colors))
    
    return downsampled_points, downsampled_colors

def statistical_outlier_removal(points, colors, nb_neighbors=20, std_ratio=2.0):
    """
    Remove statistical outliers (isolated points).
    
    Args:
        points: numpy array (N, 3)
        colors: numpy array (N, 3)
        nb_neighbors: number of neighbors to analyze (higher = more aggressive)
        std_ratio: standard deviation threshold (lower = more aggressive)
    
    Returns:
        filtered_points, filtered_colors (numpy arrays)
    """
    print(f"Applying statistical outlier removal (neighbors={nb_neighbors}, std_ratio={std_ratio})...")
    
    pcd = create_o3d_pointcloud(points, colors)
    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    clean_points = np.asarray(pcd_clean.points)
    clean_colors = np.asarray(pcd_clean.colors)
    
    removed = len(points) - len(clean_points)
    print(f"Removed {removed:,} outliers ({100*removed/len(points):.1f}%)")
    print(f"Remaining: {len(clean_points):,} points")
    
    return clean_points, clean_colors

def multiview_filter(points_by_frame, colors_by_frame, voxel_size, min_views):
    """
    Filter points using multiview consistency with spatial hashing.
    Only keeps voxels that appear in at least min_views frames.
    
    Args:
        points_by_frame: list of numpy arrays (N, 3)
        colors_by_frame: list of numpy arrays (N, 3)
        voxel_size: voxel size for spatial hashing (meters)
        min_views: minimum number of frames a voxel must appear in
    
    Returns:
        filtered_points, filtered_colors (numpy arrays)
    """
    voxel_data = defaultdict(lambda: {'frames': set(), 'points': [], 'colors': []})
    
    print(f"Accumulating votes across {len(points_by_frame)} frames...")
    for frame_id, (points, colors) in enumerate(tqdm(
            zip(points_by_frame, colors_by_frame), 
            total=len(points_by_frame))):
        
        # Convert to voxel grid coordinates
        voxel_coords = np.floor(points / voxel_size).astype(np.int32)
        
        # Accumulate per voxel
        for i, voxel_key in enumerate(map(tuple, voxel_coords)):
            voxel_data[voxel_key]['frames'].add(frame_id)
            voxel_data[voxel_key]['points'].append(points[i])
            voxel_data[voxel_key]['colors'].append(colors[i])
    
    # Filter by view count and average
    print(f"Filtering voxels (min views: {min_views})...")
    filtered_points = []
    filtered_colors = []
    
    for data in voxel_data.values():
        if len(data['frames']) >= min_views:
            filtered_points.append(np.mean(data['points'], axis=0))
            filtered_colors.append(np.mean(data['colors'], axis=0))
    
    total_voxels = len(voxel_data)
    kept_voxels = len(filtered_points)
    print(f"Kept {kept_voxels:,} / {total_voxels:,} voxels "
          f"({100*kept_voxels/total_voxels:.1f}%)")
    
    return np.array(filtered_points), np.array(filtered_colors)


def create_raw_pointcloud(points_by_frame, colors_by_frame):
    """
    Combine all frames into a single raw point cloud.
    
    Args:
        points_by_frame: list of numpy arrays (N, 3)
        colors_by_frame: list of numpy arrays (N, 3)
    
    Returns:
        raw_points, raw_colors (numpy arrays)
    """
    print(f"\n{'='*60}")
    print("CREATING RAW UNFILTERED POINT CLOUD")
    print(f"{'='*60}")
    
    raw_points = np.vstack(points_by_frame)
    raw_colors = np.vstack(colors_by_frame)
    
    print(f"Raw combined points: {len(raw_points):,}")
    
    return raw_points, raw_colors

def remove_disconnected_clusters(points, colors, eps=0.1, min_samples=50, 
                                 keep_largest_n=1, min_cluster_size=1000):
    """
    Remove disconnected clusters (like mirror reflections).
    Keeps only the largest cluster(s) representing the main scene.
    
    Args:
        points: numpy array (N, 3)
        colors: numpy array (N, 3)
        eps: maximum distance between points in a cluster (meters)
        min_samples: minimum points to form a dense region
        keep_largest_n: number of largest clusters to keep (default: 1)
        min_cluster_size: minimum size for a cluster to be considered
    
    Returns:
        filtered_points, filtered_colors (numpy arrays)
    """
    from sklearn.cluster import DBSCAN
    
    print(f"Removing disconnected clusters (eps={eps}m, min_samples={min_samples})...")
    
    # Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = clustering.fit_predict(points)
    
    # Analyze clusters
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label
    
    print(f"Found {len(unique_labels)} clusters (plus {sum(labels == -1)} noise points)")
    
    # Get cluster sizes
    cluster_sizes = {}
    for label in unique_labels:
        cluster_sizes[label] = sum(labels == label)
    
    # Sort by size
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # Print cluster info
    for i, (label, size) in enumerate(sorted_clusters[:10]):  # Top 10
        print(f"  Cluster {i+1}: {size:,} points")
    
    # Keep only the largest N clusters above minimum size
    keep_labels = set()
    for label, size in sorted_clusters[:keep_largest_n]:
        if size >= min_cluster_size:
            keep_labels.add(label)
    
    print(f"Keeping {len(keep_labels)} cluster(s)")
    
    # Filter points
    mask = np.isin(labels, list(keep_labels))
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    
    removed = len(points) - len(filtered_points)
    print(f"Removed {removed:,} points from disconnected clusters ({100*removed/len(points):.1f}%)")
    
    return filtered_points, filtered_colors


def create_filtered_pointcloud(points_by_frame, colors_by_frame, 
                               voxel_size=0.01, min_views=3, nb_neighbors=20, std_ratio=2.0, 
                              cluster_eps=0.1, keep_largest_n=1, min_cluster_size=1000):
    """
    Create filtered point cloud using per-frame voxelization and multiview filtering.
    
    Args:
        points_by_frame: list of numpy arrays (N, 3)
        colors_by_frame: list of numpy arrays (N, 3)
        voxel_size: voxel size in meters (default: 0.01 = 1cm)
        min_views: minimum frames a voxel must appear in (default: 3)
    
    Returns:
        filtered_points, filtered_colors (numpy arrays)
    """
    print(f"\n{'='*60}")
    print("CREATING FILTERED POINT CLOUD")
    print(f"{'='*60}")
    
    # Step 1: Per-frame voxelization
    points_voxelized, colors_voxelized = per_frame_voxel_downsample(
        points_by_frame, 
        colors_by_frame, 
        voxel_size
    )
    
    print(f"After per-frame voxelization: "
          f"{sum(len(p) for p in points_voxelized):,} points")
    
    # Step 2: Multiview filtering
    filtered_points, filtered_colors = multiview_filter(
        points_voxelized,
        colors_voxelized,
        voxel_size=voxel_size,
        min_views=min_views
    )

    # Steps 3: Clustering
    filtered_points, filtered_colors = remove_disconnected_clusters(
            filtered_points,
            filtered_colors,
            eps=cluster_eps,
            min_samples=50,
            keep_largest_n=keep_largest_n,
            min_cluster_size=min_cluster_size
        )

    # Step 4: SOR
    filtered_points, filtered_colors = statistical_outlier_removal(
            filtered_points,
            filtered_colors,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
    
    return filtered_points, filtered_colors


def print_summary(raw_points, filtered_points, num_cameras):
    """
    Print summary statistics of the filtering results.
    
    Args:
        raw_points: numpy array (N, 3)
        filtered_points: numpy array (M, 3)
        num_cameras: number of cameras/frames used
    """
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Raw point cloud:      {len(raw_points):,} points")
    print(f"Filtered point cloud: {len(filtered_points):,} points")
    print(f"Reduction:            {100*(1 - len(filtered_points)/len(raw_points)):.1f}%")
    print(f"Cameras used:         {num_cameras}")
    print(f"{'='*60}\n")