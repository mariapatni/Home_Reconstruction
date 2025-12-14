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
from plyfile import PlyData
from scipy.spatial import cKDTree
import random

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
                # Use first available channel if none of the common ones exist
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
                # Add batch and channel dimensions for interpolation: [1, 1, H, W]
                depth = depth.unsqueeze(0).unsqueeze(0)
                # Resize using bilinear interpolation
                depth = F.interpolate(
                    depth, 
                    size=(target_height, target_width), 
                    mode='bilinear', 
                    align_corners=False
                )
                # Remove batch and channel dimensions: [H, W]
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
        ],
        "initPose": {...}
    }
    """
    metadata_path = Path(scene_path) / "EXR_RGBD" / "metadata.json"
    
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    
    print(f"Loaded metadata from {metadata_path}")
    w = meta.get('w', '?')
    h = meta.get('h', '?')
    print(f"  Image dimensions: {w}x{h}")
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
    if 'w' in meta and 'h' in meta:
        width = meta['w']
        height = meta['h']
    else:
        # Fallback defaults
        width = 720
        height = 960
    
    # Get intrinsics
    if 'perFrameIntrinsicCoeffs' in meta and len(meta['perFrameIntrinsicCoeffs']) > frame_idx:
        # Per-frame intrinsics: [fx, fy, cx, cy] format
        coeffs = meta['perFrameIntrinsicCoeffs'][frame_idx]
        if len(coeffs) >= 4:
            fx, fy, cx, cy = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
        elif len(coeffs) == 9:
            # Full 3x3 matrix flattened
            K = np.array(coeffs).reshape(3, 3)
        else:
            raise ValueError(f"Unexpected perFrameIntrinsicCoeffs format: {len(coeffs)} elements")
    elif 'K' in meta:
        # Global intrinsics (flattened 3x3 matrix)
        K_flat = meta['K']
        if len(K_flat) == 9:
            K = np.array(K_flat).reshape(3, 3)
        elif len(K_flat) == 4:
            # [fx, fy, cx, cy] format
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
    
    Record3D typically stores 4x4 camera-to-world (c2w) matrices
    
    Returns:
        c2w: 4x4 numpy array
    """
    poses = meta['poses']
    
    if frame_idx >= len(poses):
        raise IndexError(f"Frame {frame_idx} out of range (only {len(poses)} poses)")
    
    pose_flat = poses[frame_idx]
    
    # Record3D can store poses in different formats
    if len(pose_flat) == 16:
        # Flattened 4x4 matrix
        c2w = np.array(pose_flat).reshape(4, 4)
    elif len(pose_flat) == 12:
        # 3x4 matrix
        c2w = np.array(pose_flat).reshape(3, 4)
        c2w = np.vstack([c2w, [0, 0, 0, 1]])
    elif len(pose_flat) == 7:
        # Quaternion + translation format: [qx, qy, qz, qw, tx, ty, tz]
        qx, qy, qz, qw, tx, ty, tz = pose_flat
        
        # Convert quaternion to rotation matrix
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        
        # Create 4x4 transformation matrix
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = [tx, ty, tz]
    else:
        raise ValueError(f"Unexpected pose format: {len(pose_flat)} elements")
    
    return c2w


def load_and_merge_plys(scene_path, frame_indices, voxel_size=0.005):
    """
    Load and merge partial point clouds from Record3D
    
    FIX: Transform each PLY from camera space to world space before merging.
    Record3D PLY files are in camera space (relative to each camera), not world space.
    
    Args:
        scene_path: Path to scene directory
        frame_indices: Which frames to include
        voxel_size: Voxel size for downsampling (in meters)
    
    Returns:
        points: Nx3 numpy array of positions in world coordinates
        colors: Nx3 numpy array of RGB colors [0, 1]
    """
    from plyfile import PlyData
    
    ply_dir = Path(scene_path) / "PLYs"
    
    if not ply_dir.exists():
        raise ValueError(f"PLY directory not found: {ply_dir}")
    
    # Load metadata for camera poses
    meta = load_record3d_metadata(scene_path)
    
    all_points = []
    all_colors = []
    
    print(f"Loading {len(frame_indices)} point clouds...")
    
    for idx in frame_indices:
        ply_path = ply_dir / f"{idx:05d}.ply"
        
        if not ply_path.exists():
            print(f"  Warning: {ply_path} not found, skipping")
            continue
        
        # Load PLY file using plyfile (in camera space)
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']
        
        # Extract x, y, z coordinates
        x = np.array(vertex['x'])
        y = np.array(vertex['y'])
        z = np.array(vertex['z'])
        points_cam = np.stack([x, y, z], axis=1)  # Camera space
        
        # Extract colors if available
        if 'red' in vertex.dtype.names and 'green' in vertex.dtype.names and 'blue' in vertex.dtype.names:
            r = np.array(vertex['red'])
            g = np.array(vertex['green'])
            b = np.array(vertex['blue'])
            colors = np.stack([r, g, b], axis=1) / 255.0  # Normalize to [0, 1]
        else:
            colors = np.ones((len(points_cam), 3)) * 0.5
        
        # FIX: Get camera pose and transform from camera space to world space
        c2w = parse_pose(meta, idx)
        
        # Transform: points_world = R @ points_cam + t
        # Extract rotation and translation from c2w matrix
        R = c2w[:3, :3]  # Rotation matrix
        t = c2w[:3, 3]    # Translation vector
        
        # Transform points from camera space to world space
        points_world = (R @ points_cam.T).T + t
        
        all_points.append(points_world)  # Now in world space!
        all_colors.append(colors)
        
        if (len(all_points) % 10) == 0:
            print(f"  Loaded {len(all_points)} point clouds...")
    
    # Merge all point clouds (now all in same world coordinate system)
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    print(f"Merged {len(all_points):,} points")
    
    # Downsample using voxel grid (no open3d needed)
    print(f"\nDownsampling with voxel_size={voxel_size}...")
    points, colors = voxel_downsample(all_points, all_colors, voxel_size)
    print(f"  → {len(points):,} points after downsampling")
    
    # Remove statistical outliers
    print("Removing outliers...")
    points, colors = remove_statistical_outliers(points, colors, nb_neighbors=20, std_ratio=2.0)
    print(f"  → {len(points):,} points after outlier removal")
    
    print(f"\nFinal point cloud: {len(points):,} points")
    
    return points, colors


def voxel_downsample(points, colors, voxel_size):
    """
    Downsample point cloud using voxel grid (no open3d).
    
    Args:
        points: Nx3 array of points
        colors: Nx3 array of colors
        voxel_size: Size of voxel grid
    
    Returns:
        downsampled_points, downsampled_colors
    """
    if len(points) == 0:
        return points, colors
    
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Use dictionary to keep one point per voxel (first point in each voxel)
    voxel_dict = {}
    for i in range(len(points)):
        key = tuple(voxel_indices[i])
        if key not in voxel_dict:
            voxel_dict[key] = i
    
    # Extract downsampled points and colors
    keep_indices = list(voxel_dict.values())
    return points[keep_indices], colors[keep_indices]


def remove_statistical_outliers(points, colors, nb_neighbors=20, std_ratio=2.0):
    """
    Remove statistical outliers using nearest neighbor distances (no open3d).
    
    Args:
        points: Nx3 array of points
        colors: Nx3 array of colors
        nb_neighbors: Number of neighbors to consider
        std_ratio: Standard deviation ratio threshold
    
    Returns:
        filtered_points, filtered_colors
    """
    if len(points) == 0:
        return points, colors
    
    from scipy.spatial import cKDTree
    
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(points)
    
    # Find k nearest neighbors (including the point itself)
    k = min(nb_neighbors + 1, len(points))
    distances, _ = tree.query(points, k=k)
    
    # Compute mean distance to neighbors (excluding self)
    if k > 1:
        mean_distances = distances[:, 1:].mean(axis=1)
    else:
        mean_distances = np.zeros(len(points))
    
    # Compute mean and std of mean distances
    overall_mean = np.mean(mean_distances)
    overall_std = np.std(mean_distances)
    
    # Keep points within threshold
    threshold = overall_mean + std_ratio * overall_std
    mask = mean_distances < threshold
    
    return points[mask], colors[mask]


class Record3DScene:
    """Scene loader for Record3D data"""
    
    def __init__(self, scene_path, train_frames=None, test_frames=None,
                 gaussians=None, voxel_size=0.005):
        """
        Args:
            scene_path: Path to scene directory (e.g., "data_scenes/maria_bedroom")
            train_frames: List of frame indices for training (or None for auto)
            test_frames: List of frame indices for testing (or None for auto)
            gaussians: Gaussian model to initialize
            voxel_size: Point cloud downsampling resolution
        """
        self.scene_path = Path(scene_path)
        self.model_path = scene_path  # For compatibility
        
        # Load metadata
        meta = load_record3d_metadata(self.scene_path)
        n_frames = len(meta['poses'])
        
        # Auto-select frames if not specified
        if train_frames is None:
            # Every 2nd frame for training
            train_frames = list(range(0, n_frames, 20))
        
        if test_frames is None:
            
            # Frames not in training set
            all_frames = set(range(n_frames))
            available_test_frames = list(all_frames - set(train_frames))
            
            # Randomly sample from available frames
            num_test_frames = int(len(train_frames) * 0.2)
            test_frames = sorted(random.sample(available_test_frames, num_test_frames))
        
        
        print(f"\nDataset split:")
        print(f"  Training: {len(train_frames)} frames")
        print(f"  Testing: {len(test_frames)} frames")
        
        # Create cameras
        self.train_cameras = self._create_cameras(meta, train_frames)
        self.test_cameras = self._create_cameras(meta, test_frames)
        
        # Merge point clouds
        if gaussians is not None:
            print("\nInitializing Gaussians from point clouds...")
            points, colors = load_and_merge_plys(
                self.scene_path,
                frame_indices=train_frames,
                voxel_size=voxel_size
            )
            self._initialize_gaussians(gaussians, points, colors)
        
        # Background color
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    def _create_cameras(self, meta, frame_indices):
        """Create camera objects for specified frames"""
        cameras = []
        
        print(f"\nCreating {len(frame_indices)} cameras...")
        
        for idx in frame_indices:
            # Parse intrinsics
            K, width, height = parse_intrinsics(meta, idx)
            
            # Parse pose
            c2w = parse_pose(meta, idx)
            
            # File paths - use non-padded format (0.png, 0.jpg, 1.png, etc.)
            file_id = str(idx)
            rgb_path = self.scene_path / "EXR_RGBD" / "rgb" / f"{file_id}.png"
            depth_path = self.scene_path / "EXR_RGBD" / "depth" / f"{file_id}.exr"
            
            if not rgb_path.exists():
                # Try .jpg
                rgb_path = self.scene_path / "EXR_RGBD" / "rgb" / f"{file_id}.jpg"
            
            if not rgb_path.exists():
                print(f"  Warning: RGB image not found for frame {idx}, skipping")
                continue
            
            # depth_path can be None if not found (optional)
            
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
    
    def _initialize_gaussians(self, gaussians, points, colors):
        """Initialize Gaussian parameters from point cloud"""
        N = len(points)
        
        # Positions
        gaussians._xyz = nn.Parameter(torch.tensor(points, dtype=torch.float32).cuda())
        
        # Colors (convert to SH)
        SH_C0 = 0.28209479177387814
        colors_sh = (colors - 0.5) / SH_C0
        gaussians._features_dc = nn.Parameter(
            torch.tensor(colors_sh, dtype=torch.float32).unsqueeze(1).cuda()
        )
        gaussians._features_rest = nn.Parameter(
            torch.zeros((N, 15, 3), dtype=torch.float32).cuda()
        )
        
        # Opacity (start mostly transparent)
        gaussians._opacity = nn.Parameter(
            torch.ones((N, 1), dtype=torch.float32).cuda() * -2.0  # sigmoid(-2) ≈ 0.12
        )
        
        # Scales (from nearest neighbor distances)
        print("Computing initial scales from point cloud...")
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=4)
        avg_dist = distances[:, 1:].mean(axis=1)
        scales = np.log(avg_dist * 0.5)  # Log space
        gaussians._scaling = nn.Parameter(
            torch.tensor(np.stack([scales, scales, scales], axis=1), dtype=torch.float32).cuda()
        )
        
        # Rotations (identity)
        gaussians._rotation = nn.Parameter(
            torch.tensor([[1, 0, 0, 0]], dtype=torch.float32).repeat(N, 1).cuda()
        )
        
        print(f"Initialized {N} Gaussians")
    
    def getTrainCameras(self):
        return self.train_cameras.copy()
    
    def getTestCameras(self):
        return self.test_cameras.copy()
    
    def save(self, iteration):
        """Save checkpoint (stub for compatibility)"""
        pass
