"""
Record3D data loader for Gaussian Splatting
Parses Record3D export format and creates training data
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import OpenEXR
import Imath
from scipy.spatial import cKDTree
import random
from data_loaders.raw_pc_processing import (
            create_raw_pointcloud,
            create_filtered_pointcloud
        )


class Record3DCamera:
    """Camera object compatible with training pipeline"""
    
    def __init__(self, image_path, depth_path, c2w, K, 
                 width, height, camera_id, image_name):
        """
        Args:
            image_path: Path to RGB image
            depth_path: Path to depth map
            c2w: 4x4 camera-to-world matrix
            K: 3x3 intrinsic matrix
            width, height: Image dimensions
            camera_id: Unique camera ID
            image_name: Image filename
        """
        # Load RGB
        self.original_image = self._load_rgb(image_path)  # [3, H, W]
        
        # Load depth (optional) - resize to match RGB dimensions
        self.depth_map = self._load_depth(depth_path, width, height) if depth_path and Path(depth_path).exists() else None
        
        # Camera metadata
        self.camera_id = camera_id
        self.image_name = image_name
        self.image_width = width
        self.image_height = height
        
        # Extract intrinsics from K matrix
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        
        # Field of view
        self.FoVx = 2 * np.arctan(width / (2 * self.fx))
        self.FoVy = 2 * np.arctan(height / (2 * self.fy))
        
        # Camera pose (c2w matrix)
        self.c2w = torch.tensor(c2w, dtype=torch.float32)
        self.camera_center = self.c2w[:3, 3]
        
        # World to camera transform
        w2c = torch.inverse(self.c2w)
        self.world_view_transform = w2c.transpose(0, 1)
        
        # Projection matrix (OpenGL style)
        self.projection_matrix = self._get_projection_matrix()
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        
        # Alpha mask (all valid, or load if you have sky masks)
        self.alpha_mask = torch.ones((1, height, width))
        
        # Depth supervision
        if self.depth_map is not None:
            # Valid depth: positive and finite (not NaN, not inf)
            valid = (self.depth_map > 0) & torch.isfinite(self.depth_map)
            self.invdepthmap = torch.zeros_like(self.depth_map)
            self.invdepthmap[valid] = 1.0 / self.depth_map[valid]
            self.depth_mask = valid.float()
        else:
            self.invdepthmap = None
            self.depth_mask = None
        
        # Resolution scale (for LOD, set to 1.0 by default)
        self.resolution_scale = 1.0
    
    def _load_rgb(self, path):
        """Load RGB image and convert to [3, H, W] tensor"""
        img = Image.open(path).convert('RGB')
        img = torch.from_numpy(np.array(img)).float() / 255.0
        return img.permute(2, 0, 1)
    
    def _load_depth(self, path, target_width, target_height):
        """Load depth from EXR file and resize to match RGB dimensions"""
        try:
            import torch.nn.functional as F
            
            exr_file = OpenEXR.InputFile(str(path))
            dw = exr_file.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            
            # Check available channels
            available_channels = list(exr_file.header()['channels'].keys())
            
            # Try common depth channel names (R, Y, Z, or first available channel)
            depth_channel = None
            for channel_name in ['R', 'Y', 'Z', 'G', 'B']:
                if channel_name in available_channels:
                    depth_channel = channel_name
                    break
            
            if depth_channel is None and len(available_channels) > 0:
                depth_channel = available_channels[0]
            
            if depth_channel is None:
                raise ValueError(f"No channels found in EXR file: {path}")
            
            depth_str = exr_file.channel(depth_channel, Imath.PixelType(Imath.PixelType.FLOAT))
            depth = np.frombuffer(depth_str, dtype=np.float32)
            depth = depth.reshape(size[1], size[0])
            
            # Convert to torch tensor
            depth = torch.from_numpy(depth.copy())
            
            # Replace NaN values with 0 (invalid depth)
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Resize to match RGB image dimensions if needed
            if depth.shape[0] != target_height or depth.shape[1] != target_width:
                depth = depth.unsqueeze(0).unsqueeze(0)
                depth = F.interpolate(
                    depth, 
                    size=(target_height, target_width), 
                    mode='bilinear', 
                    align_corners=False
                )
                depth = depth.squeeze(0).squeeze(0)
            
            return depth
        except Exception as e:
            print(f"Warning: Could not load depth from {path}: {e}")
            return None
    
    def _get_projection_matrix(self, znear=0.01, zfar=100.0):
        """Create OpenGL-style projection matrix"""
        tanHalfFovY = np.tan(self.FoVy / 2)
        tanHalfFovX = np.tan(self.FoVx / 2)
        
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right
        
        P = torch.zeros(4, 4)
        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[2, 2] = -(zfar + znear) / (zfar - znear)
        P[3, 2] = -1.0
        P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
        
        return P.transpose(0, 1)
    
    def get_opencv_viewmat(self):
        """Convert camera from OpenGL to OpenCV convention for gsplat"""
        # OpenGL -> OpenCV conversion matrix
        # Flips Y and Z axes
        gl_to_cv = torch.tensor([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=torch.float32, device=self.c2w.device)
        
        # Convert camera pose to OpenCV
        c2w_cv = self.c2w @ gl_to_cv
        w2c_cv = torch.inverse(c2w_cv)
        
        return w2c_cv.transpose(0, 1)


def load_record3d_metadata(scene_path):
    """
    Load Record3D metadata.json
    
    Expected format:
    {
        "w": 720,
        "h": 960,
        "fps": 60,
        "K": [fx, 0, cx, 0, fy, cy, 0, 0, 1],  # 3x3 flattened
        "poses": [
            [m00, m01, ..., m15],  # 4x4 flattened c2w matrix
            ...
        ]
    }
    """
    metadata_path = Path(scene_path) / "EXR_RGBD" / "metadata.json"
    
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    
    print(f"Loaded metadata from {metadata_path}")
    print(f"  Image dimensions: {meta.get('w', '?')}x{meta.get('h', '?')}")
    print(f"  Number of frames: {len(meta['poses'])}")
    
    return meta


def parse_intrinsics(meta, frame_idx=0):
    """
    Parse camera intrinsics from Record3D metadata
    
    Returns:
        K: 3x3 intrinsic matrix as numpy array
        width: Image width
        height: Image height
    """
    # Get image dimensions
    width = meta.get('w', 720)
    height = meta.get('h', 960)
    
    # Get intrinsics
    if 'perFrameIntrinsicCoeffs' in meta and len(meta['perFrameIntrinsicCoeffs']) > frame_idx:
        coeffs = meta['perFrameIntrinsicCoeffs'][frame_idx]
        if len(coeffs) >= 4:
            fx, fy, cx, cy = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
        elif len(coeffs) == 9:
            K = np.array(coeffs).reshape(3, 3)
        else:
            raise ValueError(f"Unexpected perFrameIntrinsicCoeffs format: {len(coeffs)} elements")
    elif 'K' in meta:
        K_flat = meta['K']
        if len(K_flat) == 9:
            K = np.array(K_flat).reshape(3, 3)
        elif len(K_flat) == 4:
            fx, fy, cx, cy = K_flat
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
        else:
            raise ValueError(f"Unexpected K format: {len(K_flat)} elements")
    else:
        # Fallback: estimate from image size
        fx = fy = width
        cx = width / 2
        cy = height / 2
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
    
    return K, width, height


def parse_pose(meta, frame_idx):
    """
    Parse camera pose from Record3D metadata
    
    Returns:
        c2w: 4x4 numpy array (camera-to-world)
    """
    poses = meta['poses']
    
    if frame_idx >= len(poses):
        raise IndexError(f"Frame {frame_idx} out of range (only {len(poses)} poses)")
    
    pose_flat = poses[frame_idx]
    
    if len(pose_flat) == 16:
        c2w = np.array(pose_flat).reshape(4, 4)
    elif len(pose_flat) == 12:
        c2w = np.array(pose_flat).reshape(3, 4)
        c2w = np.vstack([c2w, [0, 0, 0, 1]])
    elif len(pose_flat) == 7:
        # Quaternion + translation: [qx, qy, qz, qw, tx, ty, tz]
        qx, qy, qz, qw, tx, ty, tz = pose_flat
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = [tx, ty, tz]
    else:
        raise ValueError(f"Unexpected pose format: {len(pose_flat)} elements")
    
    return c2w


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
    
    # Unproject depth to 3D camera coordinates
    fx, fy = cam.fx, cam.fy
    cx, cy = cam.cx, cam.cy
    
    x_cam = (u_valid - cx) * depth_valid / fx
    y_cam = -(v_valid - cy) * depth_valid / fy
    z_cam = -depth_valid
    
    # Stack into [N, 3] tensor
    points_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)
    
    # Transform to world coordinates using c2w matrix
    ones = torch.ones(len(points_cam), 1)
    points_cam_hom = torch.cat([points_cam, ones], dim=1)  # [N, 4]
    points_world_hom = points_cam_hom @ cam.c2w.T  # [N, 4]
    points_world = points_world_hom[:, :3]
    
    # Get RGB colors at valid pixels
    rgb_flat = cam.original_image[:, v_valid.long(), u_valid.long()].T  # [N, 3]
    
    return points_world.numpy(), rgb_flat.numpy()


def load_processed_pointcloud(scene_path, custom_path=None):
    """
    Load pre-processed point cloud
    
    Args:
        scene_path: Path to scene directory
        custom_path: Optional custom path to PLY file
    
    Returns:
        points, colors if found, else (None, None)
    """
    import open3d as o3d
    
    # Use custom path if provided, otherwise default to processed.ply
    if custom_path is not None:
        ply_path = Path(custom_path)
    else:
        ply_path = Path(scene_path) / "processed.ply"
    
    if not ply_path.exists():
        return None, None
    
    print(f"Loading point cloud from {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    print(f"  Loaded {len(points):,} points")
    
    return points, colors


def save_processed_pointcloud(scene_path, points, colors):
    """
    Save processed point cloud as processed.ply
    
    Args:
        scene_path: Path to scene directory
        points: Nx3 numpy array
        colors: Nx3 numpy array
    """
    import open3d as o3d
    
    output_path = Path(scene_path) / "processed.ply"
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Saved processed point cloud to {output_path}")


def reconstruct_from_rgbd(scene_path, frame_indices, subsample=4):
    """
    Reconstruct point cloud from RGBD frames
    
    Args:
        scene_path: Path to scene directory
        frame_indices: List of frame indices to use
        subsample: Subsampling factor for depth unprojection
    
    Returns:
        points: Nx3 numpy array
        colors: Nx3 numpy array
    """    
    meta = load_record3d_metadata(scene_path)
    
    points_by_frame = []
    colors_by_frame = []
    
    print(f"\nReconstructing from {len(frame_indices)} RGBD frames...")
    
    for idx in frame_indices:
        # Parse camera parameters
        K, width, height = parse_intrinsics(meta, idx)
        c2w = parse_pose(meta, idx)
        
        # File paths
        file_id = str(idx)
        rgb_path = Path(scene_path) / "EXR_RGBD" / "rgb" / f"{file_id}.png"
        depth_path = Path(scene_path) / "EXR_RGBD" / "depth" / f"{file_id}.exr"
        
        if not rgb_path.exists():
            rgb_path = Path(scene_path) / "EXR_RGBD" / "rgb" / f"{file_id}.jpg"
        
        if not rgb_path.exists():
            print(f"  Warning: RGB not found for frame {idx}, skipping")
            continue
        
        # Create temporary camera object
        cam = Record3DCamera(
            image_path=rgb_path,
            depth_path=depth_path,
            c2w=c2w,
            K=K,
            width=width,
            height=height,
            camera_id=idx,
            image_name=file_id
        )
        
        # Unproject to 3D
        points, colors = create_point_cloud_from_rgbd(cam, subsample=subsample)
        
        if points is not None and len(points) > 0:
            points_by_frame.append(points)
            colors_by_frame.append(colors)
        
        if len(points_by_frame) % 10 == 0:
            print(f"  Processed {len(points_by_frame)} frames...")
    
    print(f"Total raw points: {sum(len(p) for p in points_by_frame):,}")

    print("\n✓ Applying multiview filtering...")
    points, colors = create_filtered_pointcloud(
            points_by_frame,
            colors_by_frame,
            voxel_size=0.01,
            min_views=5,
            use_sor=True,
            nb_neighbors=20,
            std_ratio=1.5
        )
    
    return points, colors


class Record3DScene:
    """Scene loader for Record3D data"""
    
    def __init__(self, scene_path, train_frames=None, test_frames=None,
                 subsample=4, pointcloud_path=None):
        """
        Args:
            scene_path: Path to scene directory (e.g., "data_scenes/maria_bedroom")
            train_frames: List of frame indices for training (or None for auto)
            test_frames: List of frame indices for testing (or None for auto)
            subsample: Subsampling factor for RGBD reconstruction
            pointcloud_path: Optional path to custom PLY file
        """
        self.scene_path = Path(scene_path)
        self.model_path = scene_path  # For compatibility
        
        # Load metadata
        meta = load_record3d_metadata(self.scene_path)
        n_frames = len(meta['poses'])
        
        # Auto-select frames if not specified
        if train_frames is None:
            train_frames = list(range(0, n_frames, 20))  # Every 20th frame
        
        if test_frames is None:
            all_frames = set(range(n_frames))
            available_test_frames = list(all_frames - set(train_frames))
            num_test_frames = max(1, int(len(train_frames) * 0.2))
            test_frames = sorted(random.sample(available_test_frames, 
                                             min(num_test_frames, len(available_test_frames))))
        
        print(f"\nDataset split:")
        print(f"  Training: {len(train_frames)} frames")
        print(f"  Testing: {len(test_frames)} frames")
        
        # Create cameras
        self.train_cameras = self._create_cameras(meta, train_frames)
        self.test_cameras = self._create_cameras(meta, test_frames)
        
        # Load point cloud
        print("\nLoading point cloud...")
        
        if pointcloud_path is not None:
            # Custom path provided - load from there
            print(f"Using custom point cloud path: {pointcloud_path}")
            points, colors = load_processed_pointcloud(self.scene_path, custom_path=pointcloud_path)
            
            if points is None:
                raise FileNotFoundError(f"Custom point cloud not found: {pointcloud_path}")
        
        else:
            # No custom path - use processed.ply (create if needed)
            points, colors = load_processed_pointcloud(self.scene_path, custom_path=None)
            
            if points is None:
                # processed.ply doesn't exist - create it
                print("processed.ply not found, creating filtered point cloud...")
                points, colors = reconstruct_from_rgbd(
                    self.scene_path,
                    frame_indices=train_frames,
                    subsample=subsample
                )
                
                # Save as processed.ply
                save_processed_pointcloud(self.scene_path, points, colors)
        
        # Store point cloud
        self.points = points
        self.colors = colors
        print(f"Scene initialized with {len(points):,} points")
        
        # Background color
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    def _create_cameras(self, meta, frame_indices):
        """Create camera objects for specified frames"""
        cameras = []
        
        print(f"\nCreating {len(frame_indices)} cameras...")
        
        for idx in frame_indices:
            K, width, height = parse_intrinsics(meta, idx)
            c2w = parse_pose(meta, idx)
            
            file_id = str(idx)
            rgb_path = self.scene_path / "EXR_RGBD" / "rgb" / f"{file_id}.png"
            depth_path = self.scene_path / "EXR_RGBD" / "depth" / f"{file_id}.exr"
            
            if not rgb_path.exists():
                rgb_path = self.scene_path / "EXR_RGBD" / "rgb" / f"{file_id}.jpg"
            
            if not rgb_path.exists():
                print(f"  Warning: RGB image not found for frame {idx}, skipping")
                continue
            
            cam = Record3DCamera(
                image_path=rgb_path,
                depth_path=depth_path,
                c2w=c2w,
                K=K,
                width=width,
                height=height,
                camera_id=idx,
                image_name=file_id
            )
            
            cameras.append(cam)
        
        print(f"Created {len(cameras)} cameras")
        return cameras
    
    def getTrainCameras(self):
        return self.train_cameras.copy()
    
    def getTestCameras(self):
        return self.test_cameras.copy()
    
    def save(self, iteration):
        """Save checkpoint (stub for compatibility)"""
        pass