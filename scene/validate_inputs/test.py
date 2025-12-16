"""
Test script for Record3D data loader
"""

import sys
sys.path.append('.')

import numpy as np
import torch
import open3d as o3d
from pathlib import Path
from data_loaders.record3d_loader import (
    load_record3d_metadata,
    parse_intrinsics,
    parse_pose,
    Record3DScene
)

def test_metadata_parsing():
    """Test loading and parsing metadata"""
    
    scene_path = Path("../data_scenes/maria_bedroom")  # Adjust path as needed
    
    print("="*60)
    print("Testing Record3D Metadata Parsing")
    print("="*60)
    
    # Load metadata
    meta = load_record3d_metadata(scene_path)
    
    # Test first frame
    print("\n" + "="*60)
    print("Frame 0 Details")
    print("="*60)
    
    K, width, height = parse_intrinsics(meta, 0)
    print(f"\nIntrinsic Matrix K:")
    print(K)
    print(f"\nImage dimensions: {width}x{height}")
    print(f"fx: {K[0,0]:.2f}, fy: {K[1,1]:.2f}")
    print(f"cx: {K[0,2]:.2f}, cy: {K[1,2]:.2f}")
    
    c2w = parse_pose(meta, 0)
    print(f"\nCamera-to-World Matrix:")
    print(c2w)
    print(f"\nCamera position: {c2w[:3, 3]}")
    
    # Test a few more frames
    print("\n" + "="*60)
    print("Testing Multiple Frames")
    print("="*60)
    
    for i in [0, 10, 20]:
        if i < len(meta['poses']):
            c2w = parse_pose(meta, i)
            print(f"Frame {i} camera position: {c2w[:3, 3]}")

def test_scene_loading():
    """Test full scene loading"""
    
    scene_path = Path("../data_scenes/maria_bedroom")
    
    print("\n" + "="*60)
    print("Testing Scene Loading")
    print("="*60)
    
    # Load scene (without Gaussians for now)
    scene = Record3DScene(
        scene_path=scene_path,
        train_frames=list(range(0, 50, 5)),  # Every 5th frame, first 50
        test_frames=[2, 7, 12],
        gaussians=None,  # Don't initialize Gaussians yet
        voxel_size=0.005
    )
    
    print(f"\nLoaded scene successfully!")
    print(f"  Training cameras: {len(scene.train_cameras)}")
    print(f"  Test cameras: {len(scene.test_cameras)}")
    
    # Check first camera
    if len(scene.train_cameras) > 0:
        cam = scene.train_cameras[0]
        print(f"\nFirst training camera:")
        print(f"  Image shape: {cam.original_image.shape}")
        print(f"  Camera position: {cam.camera_center}")
        print(f"  Has depth: {cam.depth_map is not None}")
        print(f"  Depth map shape: {cam.depth_map.shape if cam.depth_map is not None else 'None'}")
        print(f"  Depth map min/max: {cam.depth_map.min() if cam.depth_map is not None else 'None'} / {cam.depth_map.max() if cam.depth_map is not None else 'None'}")
    
    return scene

def create_point_cloud_from_rgbd(cam, subsample=4):
    """
    Create a point cloud from RGB-D camera data
    
    Args:
        cam: Record3DCamera object
        subsample: Subsample factor to reduce point count (use every Nth pixel)
    
    Returns:
        points: Nx3 numpy array of 3D points in world coordinates
        colors: Nx3 numpy array of RGB colors [0, 1]
    """
    if cam.depth_map is None:
        return None, None
    
    # Get image dimensions
    H, W = cam.depth_map.shape
    
    # Create pixel grid (subsampled)
    u = torch.arange(0, W, subsample, dtype=torch.float32)
    v = torch.arange(0, H, subsample, dtype=torch.float32)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    
    # Get depth values at these pixels
    depth_flat = cam.depth_map[v_flat.long(), u_flat.long()]
    
    # Filter out invalid depth (zero or NaN)
    valid = (depth_flat > 0) & torch.isfinite(depth_flat)
    u_valid = u_flat[valid]
    v_valid = v_flat[valid]
    depth_valid = depth_flat[valid]
    
    if len(depth_valid) == 0:
        return None, None
    
    # Convert to camera coordinates using intrinsics
    fx, fy = cam.fx, cam.fy
    cx, cy = cam.cx, cam.cy
    
    # Camera coordinates: x = (u - cx) * z / fx, y = (v - cy) * z / fy, z = depth
    x_cam = (u_valid - cx) * depth_valid / fx
    y_cam = (v_valid - cy) * depth_valid / fy
    z_cam = depth_valid
    
    # Stack into [N, 3] tensor
    points_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)
    
    # Transform to world coordinates using c2w matrix
    # Add homogeneous coordinate
    ones = torch.ones(len(points_cam), 1)
    points_cam_hom = torch.cat([points_cam, ones], dim=1)
    
    # Transform: points_world = c2w @ points_cam
    points_world_hom = (cam.c2w @ points_cam_hom.T).T
    points_world = points_world_hom[:, :3]
    
    # Get RGB colors at valid pixels
    # original_image is [3, H, W]
    rgb_flat = cam.original_image[:, v_valid.long(), u_valid.long()].T  # [N, 3]
    
    return points_world.numpy(), rgb_flat.numpy()

def create_camera_frustum(cam, scale=0.1):
    """
    Create a camera frustum visualization
    
    Args:
        cam: Record3DCamera object
        scale: Scale factor for frustum size
    
    Returns:
        frustum_lines: Open3D LineSet representing camera frustum
    """
    # Camera center in world coordinates
    cam_center = cam.camera_center.numpy()
    
    # Get camera orientation from c2w matrix
    c2w = cam.c2w.numpy()
    R = c2w[:3, :3]  # Rotation matrix
    t = c2w[:3, 3]   # Translation
    
    # Camera coordinate system vectors
    # In camera space: forward = +Z, right = +X, up = -Y
    forward = R @ np.array([0, 0, 1])
    right = R @ np.array([1, 0, 0])
    up = R @ np.array([0, -1, 0])
    
    # Frustum parameters
    near = 0.1 * scale
    far = 0.5 * scale
    fov_x = cam.FoVx
    fov_y = cam.FoVy
    
    # Calculate frustum corners at near and far planes
    tan_half_fov_x = np.tan(fov_x / 2)
    tan_half_fov_y = np.tan(fov_y / 2)
    
    # Near plane corners
    near_right = near * tan_half_fov_x
    near_top = near * tan_half_fov_y
    near_corners = np.array([
        [-near_right, -near_top, near],
        [near_right, -near_top, near],
        [near_right, near_top, near],
        [-near_right, near_top, near]
    ])
    
    # Far plane corners
    far_right = far * tan_half_fov_x
    far_top = far * tan_half_fov_y
    far_corners = np.array([
        [-far_right, -far_top, far],
        [far_right, -far_top, far],
        [far_right, far_top, far],
        [-far_right, far_top, far]
    ])
    
    # Transform to world coordinates
    near_corners_world = (R @ near_corners.T).T + cam_center
    far_corners_world = (R @ far_corners.T).T + cam_center
    
    # Create line set for frustum
    points = np.vstack([
        cam_center.reshape(1, 3),  # Camera center
        near_corners_world,        # Near plane corners
        far_corners_world           # Far plane corners
    ])
    
    # Define lines: center to near corners, center to far corners, near edges, far edges
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Center to near corners
        [0, 5], [0, 6], [0, 7], [0, 8],  # Center to far corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Near plane edges
        [5, 6], [6, 7], [7, 8], [8, 5],  # Far plane edges
        [1, 5], [2, 6], [3, 7], [4, 8]   # Connecting edges
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # Red color for cameras
    
    return line_set

def visualize_combined_point_clouds(scene):
    """Create and visualize combined point clouds from all cameras"""
    
    print("\n" + "="*60)
    print("Creating Combined Point Cloud Visualization")
    print("="*60)
    
    all_points = []
    all_colors = []
    camera_frustums = []
    
    # Process training cameras
    print(f"\nProcessing {len(scene.train_cameras)} training cameras...")
    for i, cam in enumerate(scene.train_cameras):
        points, colors = create_point_cloud_from_rgbd(cam, subsample=4)
        if points is not None and len(points) > 0:
            all_points.append(points)
            all_colors.append(colors)
            print(f"  Camera {i}: {len(points)} points")
        
        # Create camera frustum
        frustum = create_camera_frustum(cam, scale=0.2)
        camera_frustums.append(frustum)
    
    # Process test cameras
    print(f"\nProcessing {len(scene.test_cameras)} test cameras...")
    for i, cam in enumerate(scene.test_cameras):
        points, colors = create_point_cloud_from_rgbd(cam, subsample=4)
        if points is not None and len(points) > 0:
            all_points.append(points)
            all_colors.append(colors)
            print(f"  Test camera {i}: {len(points)} points")
        
        # Create camera frustum (different color for test cameras)
        frustum = create_camera_frustum(cam, scale=0.2)
        frustum.paint_uniform_color([0, 1, 0])  # Green for test cameras
        camera_frustums.append(frustum)
    
    if len(all_points) == 0:
        print("  Warning: No valid point clouds created!")
        return
    
    # Combine all point clouds
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    print(f"\nCombined point cloud: {len(combined_points)} points")
    print(f"  Bounding box: min={combined_points.min(axis=0)}, max={combined_points.max(axis=0)}")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # Downsample for faster visualization
    if len(combined_points) > 100000:
        print(f"Downsampling from {len(combined_points)} to 100k points for visualization...")
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        print(f"After downsampling: {len(pcd.points)} points")
    
    # Prepare visualization objects
    vis_objects = [pcd] + camera_frustums
    
    # Save to file for inspection
    output_path = Path("../data_scenes/maria_bedroom/combined_pointcloud.ply")
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"\nSaved point cloud to: {output_path}")
    print(f"  You can open this file in MeshLab, CloudCompare, or any PLY viewer")
    print(f"  to inspect the alignment of point clouds from different cameras.")
    
    # Try to visualize (may fail in headless environments)
    try:
        print("\nAttempting to open visualization window...")
        print("  Controls:")
        print("    - Mouse: Rotate view")
        print("    - Shift + Mouse: Pan view")
        print("    - Mouse wheel: Zoom")
        print("    - Q or close window: Exit")
        print("  Red frustums = Training cameras")
        print("  Green frustums = Test cameras")
        
        o3d.visualization.draw_geometries(vis_objects, 
                                          window_name="Combined Point Cloud - Check Alignment",
                                          width=1024, 
                                          height=768)
    except Exception as e:
        print(f"  Note: Visualization window unavailable (headless environment)")
        print(f"  Point cloud saved to file for manual inspection.")

if __name__ == "__main__":
    # Run tests
    test_metadata_parsing()
    scene = test_scene_loading()
    
    # Visualize combined point clouds
    if scene is not None:
        visualize_combined_point_clouds(scene)
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)