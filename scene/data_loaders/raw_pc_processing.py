"""
Multiview point cloud filtering utilities with semantic support.
Handles points, colors, AND object IDs throughout the pipeline.
"""

import open3d as o3d
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import DBSCAN


def create_o3d_pointcloud(points, colors):
    """Create Open3D point cloud from numpy arrays."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def majority_vote(ids):
    """Safely compute majority vote for object IDs."""
    if len(ids) == 0:
        return 0
    ids = np.asarray(ids, dtype=np.int32)
    if ids.min() < 0:
        offset = -ids.min()
        ids = ids + offset
        voted = np.bincount(ids).argmax() - offset
    else:
        voted = np.bincount(ids).argmax()
    return int(voted)


def per_frame_voxel_downsample_with_semantics(points_by_frame, colors_by_frame, 
                                               object_ids_by_frame, voxel_size):
    """Voxelize each frame individually, preserving object IDs via voting."""
    downsampled_points = []
    downsampled_colors = []
    downsampled_object_ids = []
    
    print(f"Voxelizing {len(points_by_frame)} frames with semantics (voxel size: {voxel_size}m)...")
    
    for points, colors, obj_ids in tqdm(zip(points_by_frame, colors_by_frame, object_ids_by_frame),
                                         total=len(points_by_frame)):
        if len(points) == 0:
            continue
            
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        voxel_dict = defaultdict(lambda: {'points': [], 'colors': [], 'object_ids': []})
        for i, vidx in enumerate(voxel_indices):
            key = tuple(vidx)
            voxel_dict[key]['points'].append(points[i])
            voxel_dict[key]['colors'].append(colors[i])
            voxel_dict[key]['object_ids'].append(obj_ids[i])
        
        frame_points = []
        frame_colors = []
        frame_obj_ids = []
        
        for data in voxel_dict.values():
            frame_points.append(np.mean(data['points'], axis=0))
            frame_colors.append(np.mean(data['colors'], axis=0))
            voted_id = majority_vote(data['object_ids'])
            frame_obj_ids.append(voted_id)
        
        if len(frame_points) > 0:
            downsampled_points.append(np.array(frame_points))
            downsampled_colors.append(np.array(frame_colors))
            downsampled_object_ids.append(np.array(frame_obj_ids, dtype=np.int32))
    
    return downsampled_points, downsampled_colors, downsampled_object_ids


def multiview_filter_with_semantics(points_by_frame, colors_by_frame, 
                                     object_ids_by_frame, voxel_size, min_views):
    """Filter points using multiview consistency, with object ID voting."""
    voxel_data = defaultdict(lambda: {
        'frames': set(), 'points': [], 'colors': [], 'object_ids': []
    })
    
    print(f"Accumulating votes across {len(points_by_frame)} frames...")
    for frame_id, (points, colors, obj_ids) in enumerate(tqdm(
            zip(points_by_frame, colors_by_frame, object_ids_by_frame),
            total=len(points_by_frame))):
        
        if len(points) == 0:
            continue
        
        voxel_coords = np.floor(points / voxel_size).astype(np.int32)
        
        for i, voxel_key in enumerate(map(tuple, voxel_coords)):
            voxel_data[voxel_key]['frames'].add(frame_id)
            voxel_data[voxel_key]['points'].append(points[i])
            voxel_data[voxel_key]['colors'].append(colors[i])
            voxel_data[voxel_key]['object_ids'].append(obj_ids[i])
    
    print(f"Filtering voxels (min views: {min_views})...")
    filtered_points = []
    filtered_colors = []
    filtered_object_ids = []
    
    for data in voxel_data.values():
        if len(data['frames']) >= min_views:
            filtered_points.append(np.mean(data['points'], axis=0))
            filtered_colors.append(np.mean(data['colors'], axis=0))
            voted_id = majority_vote(data['object_ids'])
            filtered_object_ids.append(voted_id)
    
    total_voxels = len(voxel_data)
    kept_voxels = len(filtered_points)
    
    if total_voxels > 0:
        print(f"Kept {kept_voxels:,} / {total_voxels:,} voxels ({100*kept_voxels/total_voxels:.1f}%)")
    
    if len(filtered_points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=np.int32)
    
    return (np.array(filtered_points), 
            np.array(filtered_colors), 
            np.array(filtered_object_ids, dtype=np.int32))


def statistical_outlier_removal_with_semantics(points, colors, object_ids,
                                                nb_neighbors=20, std_ratio=2.0):
    """Remove statistical outliers while preserving object IDs."""
    if len(points) == 0:
        return points, colors, object_ids
        
    print(f"Applying statistical outlier removal (neighbors={nb_neighbors}, std_ratio={std_ratio})...")
    
    pcd = create_o3d_pointcloud(points, colors)
    pcd_clean, indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    
    indices = np.array(indices)
    clean_points = np.asarray(pcd_clean.points)
    clean_colors = np.asarray(pcd_clean.colors)
    clean_object_ids = object_ids[indices]
    
    removed = len(points) - len(clean_points)
    print(f"Removed {removed:,} outliers ({100*removed/len(points):.1f}%)")
    
    return clean_points, clean_colors, clean_object_ids


def remove_disconnected_clusters_with_semantics(points, colors, object_ids,
                                                 eps=0.1, min_samples=50,
                                                 keep_largest_n=1, min_cluster_size=1000):
    """Remove disconnected clusters while preserving object IDs."""
    if len(points) == 0:
        return points, colors, object_ids
        
    print(f"Removing disconnected clusters (eps={eps}m, min_samples={min_samples})...")
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = clustering.fit_predict(points)
    
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    print(f"Found {len(unique_labels)} clusters (plus {sum(labels == -1)} noise points)")
    
    if len(unique_labels) == 0:
        return points, colors, object_ids
    
    cluster_sizes = {label: sum(labels == label) for label in unique_labels}
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    
    keep_labels = set()
    for label, size in sorted_clusters[:keep_largest_n]:
        if size >= min_cluster_size:
            keep_labels.add(label)
    
    if len(keep_labels) == 0:
        keep_labels.add(sorted_clusters[0][0])
    
    mask = np.isin(labels, list(keep_labels))
    
    return points[mask], colors[mask], object_ids[mask]


def create_filtered_pointcloud_with_semantics(points_by_frame, colors_by_frame, 
                                               object_ids_by_frame,
                                               voxel_size=0.01, min_views=3,
                                               nb_neighbors=20, std_ratio=2.0,
                                               cluster_eps=0.1, keep_largest_n=1,
                                               min_cluster_size=1000, use_sor=True):
    """Full pipeline with semantic preservation."""
    print(f"\\n{'='*60}")
    print("CREATING FILTERED POINT CLOUD WITH SEMANTICS")
    print(f"{'='*60}")
    
    if len(points_by_frame) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=np.int32)
    
    # Step 1: Per-frame voxelization
    points_vox, colors_vox, obj_ids_vox = per_frame_voxel_downsample_with_semantics(
        points_by_frame, colors_by_frame, object_ids_by_frame, voxel_size
    )
    
    total_after_vox = sum(len(p) for p in points_vox) if points_vox else 0
    print(f"After per-frame voxelization: {total_after_vox:,} points")
    
    if total_after_vox == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=np.int32)
    
    # Step 2: Multiview filtering
    points, colors, object_ids = multiview_filter_with_semantics(
        points_vox, colors_vox, obj_ids_vox,
        voxel_size=voxel_size, min_views=min_views
    )
    
    if len(points) == 0:
        return points, colors, object_ids
    
    # Step 3: Clustering
    points, colors, object_ids = remove_disconnected_clusters_with_semantics(
        points, colors, object_ids,
        eps=cluster_eps, min_samples=50,
        keep_largest_n=keep_largest_n, min_cluster_size=min_cluster_size
    )
    
    # Step 4: Statistical outlier removal
    if use_sor and len(points) > 0:
        points, colors, object_ids = statistical_outlier_removal_with_semantics(
            points, colors, object_ids,
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
    
    # Print summary
    if len(object_ids) > 0:
        unique_ids = np.unique(object_ids)
        print(f"\\nSemantic summary: {len(unique_ids)} unique objects")
        for obj_id in unique_ids[:10]:
            count = np.sum(object_ids == obj_id)
            print(f"  ID {obj_id}: {count:,} points")
    
    return points, colors, object_ids


def create_filtered_pointcloud(points_by_frame, colors_by_frame,
                                voxel_size=0.01, min_views=3,
                                nb_neighbors=20, std_ratio=2.0,
                                cluster_eps=0.1, keep_largest_n=1,
                                min_cluster_size=1000, use_sor=True):
    """Legacy version without semantics."""
    object_ids_by_frame = [np.zeros(len(p), dtype=np.int32) for p in points_by_frame]
    
    points, colors, _ = create_filtered_pointcloud_with_semantics(
        points_by_frame, colors_by_frame, object_ids_by_frame,
        voxel_size=voxel_size, min_views=min_views,
        nb_neighbors=nb_neighbors, std_ratio=std_ratio,
        cluster_eps=cluster_eps, keep_largest_n=keep_largest_n,
        min_cluster_size=min_cluster_size, use_sor=use_sor
    )
    
    return points, colors


def create_raw_pointcloud(points_by_frame, colors_by_frame):
    """Combine all frames into a single raw point cloud."""
    if len(points_by_frame) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    
    return np.vstack(points_by_frame), np.vstack(colors_by_frame)