"""
Record3D data loader for Gaussian Splatting
WITH SEMANTIC SEGMENTATION SUPPORT

Uses new semantic-preserving point cloud filtering from:
  scene.data_loaders.raw_pc_processing.process_pointcloud_with_semantics
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Optional imports with fallbacks
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("Warning: OpenEXR not installed. Depth loading will be limited.")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not installed. Point cloud I/O will be limited.")

# ✅ NEW: import the new processing entry point (no legacy names)
from scene.data_loaders.raw_pc_processing import (
    create_raw_pointcloud,
    process_pointcloud_with_semantics,
)


# =============================================================================
# Camera
# =============================================================================

class Record3DCamera:
    """Camera object compatible with training pipeline."""

    def __init__(
        self,
        image_path,
        depth_path,
        c2w,
        K,
        width,
        height,
        camera_id,
        image_name,
        mask_path=None,
    ):
        # Load RGB
        self.original_image = self._load_rgb(image_path)  # [3, H, W]

        # Load depth
        self.depth_map = None
        if depth_path and Path(depth_path).exists():
            self.depth_map = self._load_depth(depth_path, width, height)

        # Load semantic mask (IDs)
        self.object_mask = None
        if mask_path and Path(mask_path).exists():
            self.object_mask = self._load_mask(mask_path, width, height)

        # Camera metadata
        self.camera_id = camera_id
        self.image_name = image_name
        self.image_width = width
        self.image_height = height

        # Intrinsics
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])

        # FoV
        self.FoVx = 2 * np.arctan(width / (2 * self.fx))
        self.FoVy = 2 * np.arctan(height / (2 * self.fy))

        # Pose
        self.c2w = torch.tensor(c2w, dtype=torch.float32)
        self.camera_center = self.c2w[:3, 3]

        # World to camera
        w2c = torch.inverse(self.c2w)
        self.world_view_transform = w2c.transpose(0, 1)

        # Projection (OpenGL style)
        self.projection_matrix = self._get_projection_matrix()
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix

        # Alpha mask
        self.alpha_mask = torch.ones((1, height, width))

        # Depth supervision
        if self.depth_map is not None:
            valid = (self.depth_map > 0) & torch.isfinite(self.depth_map)
            self.invdepthmap = torch.zeros_like(self.depth_map)
            self.invdepthmap[valid] = 1.0 / self.depth_map[valid]
            self.depth_mask = valid.float()
        else:
            self.invdepthmap = None
            self.depth_mask = None

        self.resolution_scale = 1.0

    def _load_rgb(self, path):
        img = Image.open(path).convert("RGB")
        img = torch.from_numpy(np.array(img)).float() / 255.0
        return img.permute(2, 0, 1)

    def _load_mask(self, path, target_width, target_height):
        """
        Load semantic mask as [H, W] tensor of integer object IDs.
        Expects grayscale PNG where pixel value = object ID.
        """
        try:
            mask_img = Image.open(path)

            if mask_img.mode in ["RGB", "RGBA", "P"]:
                mask_img = mask_img.convert("L")

            # Convert to tensor
            mask = torch.from_numpy(np.array(mask_img)).long()

            # Resize with NEAREST (labels)
            if mask.shape[0] != target_height or mask.shape[1] != target_width:
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(target_height, target_width),
                    mode="nearest",
                ).squeeze().long()

            return mask
        except Exception as e:
            print(f"Warning: Could not load mask from {path}: {e}")
            return None

    def _load_depth(self, path, target_width, target_height):
        if not HAS_OPENEXR:
            print(f"Warning: OpenEXR not installed, cannot load {path}")
            return None

        try:
            exr_file = OpenEXR.InputFile(str(path))
            dw = exr_file.header()["dataWindow"]
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

            available_channels = list(exr_file.header()["channels"].keys())
            depth_channel = None
            for channel_name in ["R", "Y", "Z", "G", "B"]:
                if channel_name in available_channels:
                    depth_channel = channel_name
                    break
            if depth_channel is None and len(available_channels) > 0:
                depth_channel = available_channels[0]
            if depth_channel is None:
                raise ValueError(f"No channels found in EXR file: {path}")

            depth_str = exr_file.channel(depth_channel, Imath.PixelType(Imath.PixelType.FLOAT))
            depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])

            depth = torch.from_numpy(depth.copy())
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            if depth.shape[0] != target_height or depth.shape[1] != target_width:
                depth = depth.unsqueeze(0).unsqueeze(0)
                depth = F.interpolate(
                    depth,
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                )
                depth = depth.squeeze(0).squeeze(0)

            return depth
        except Exception as e:
            print(f"Warning: Could not load depth from {path}: {e}")
            return None

    def _get_projection_matrix(self, znear=0.01, zfar=100.0):
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
        """Convert camera from OpenGL to OpenCV convention for gsplat."""
        gl_to_cv = torch.tensor(
            [[1, 0, 0, 0],
             [0, -1, 0, 0],
             [0, 0, -1, 0],
             [0, 0, 0, 1]],
            dtype=torch.float32,
            device=self.c2w.device,
        )
        c2w_cv = self.c2w @ gl_to_cv
        w2c_cv = torch.inverse(c2w_cv)
        return w2c_cv


# =============================================================================
# Metadata and Pose Parsing
# =============================================================================

def load_record3d_metadata(scene_path):
    metadata_path = Path(scene_path) / "EXR_RGBD" / "metadata.json"
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    print(f"Loaded metadata from {metadata_path}")
    print(f"  Image dimensions: {meta.get('w', '?')}x{meta.get('h', '?')}")
    print(f"  Number of frames: {len(meta['poses'])}")
    return meta


def parse_intrinsics(meta, frame_idx=0):
    width = meta.get("w", 720)
    height = meta.get("h", 960)

    if "perFrameIntrinsicCoeffs" in meta and len(meta["perFrameIntrinsicCoeffs"]) > frame_idx:
        coeffs = meta["perFrameIntrinsicCoeffs"][frame_idx]
        if len(coeffs) >= 4:
            fx, fy, cx, cy = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        elif len(coeffs) == 9:
            K = np.array(coeffs).reshape(3, 3)
        else:
            raise ValueError(f"Unexpected perFrameIntrinsicCoeffs format: {len(coeffs)} elements")
    elif "K" in meta:
        K_flat = meta["K"]
        if len(K_flat) == 9:
            K = np.array(K_flat).reshape(3, 3)
        elif len(K_flat) == 4:
            fx, fy, cx, cy = K_flat
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        else:
            raise ValueError(f"Unexpected K format: {len(K_flat)} elements")
    else:
        fx = fy = width
        cx = width / 2
        cy = height / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K, width, height


def parse_pose(meta, frame_idx):
    poses = meta["poses"]
    if frame_idx >= len(poses):
        raise IndexError(f"Frame {frame_idx} out of range (only {len(poses)} poses)")

    pose_flat = poses[frame_idx]

    if len(pose_flat) == 16:
        c2w = np.array(pose_flat).reshape(4, 4)
    elif len(pose_flat) == 12:
        c2w = np.array(pose_flat).reshape(3, 4)
        c2w = np.vstack([c2w, [0, 0, 0, 1]])
    elif len(pose_flat) == 7:
        qx, qy, qz, qw, tx, ty, tz = pose_flat
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = [tx, ty, tz]
    else:
        raise ValueError(f"Unexpected pose format: {len(pose_flat)} elements")

    return c2w


# =============================================================================
# Point Cloud Creation
# =============================================================================

def create_point_cloud_from_rgbd(cam: Record3DCamera, subsample: int = 4):
    """
    Create a point cloud from RGB-D camera data WITH SEMANTIC LABELS.
    Returns: points (Nx3), colors (Nx3), object_ids (N,)
    """
    if cam.depth_map is None:
        return None, None, None

    H, W = cam.depth_map.shape

    u = torch.arange(0, W, subsample, dtype=torch.float32)
    v = torch.arange(0, H, subsample, dtype=torch.float32)
    u_grid, v_grid = torch.meshgrid(u, v, indexing="xy")
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()

    depth_flat = cam.depth_map[v_flat.long(), u_flat.long()]
    valid = (depth_flat > 0) & torch.isfinite(depth_flat)

    u_valid = u_flat[valid]
    v_valid = v_flat[valid]
    depth_valid = depth_flat[valid]

    if len(depth_valid) == 0:
        return None, None, None

    fx, fy = cam.fx, cam.fy
    cx, cy = cam.cx, cam.cy

    x_cam = (u_valid - cx) * depth_valid / fx
    y_cam = -(v_valid - cy) * depth_valid / fy
    z_cam = -depth_valid
    points_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)

    ones = torch.ones(len(points_cam), 1)
    points_cam_hom = torch.cat([points_cam, ones], dim=1)
    points_world_hom = points_cam_hom @ cam.c2w.T
    points_world = points_world_hom[:, :3]

    rgb_flat = cam.original_image[:, v_valid.long(), u_valid.long()].T

    if cam.object_mask is not None:
        object_ids = cam.object_mask[v_valid.long(), u_valid.long()].cpu().numpy().astype(np.int32)
    else:
        object_ids = np.zeros(len(points_world), dtype=np.int32)

    return points_world.cpu().numpy(), rgb_flat.cpu().numpy(), object_ids


# =============================================================================
# Point Cloud I/O
# =============================================================================

def load_processed_pointcloud_with_semantics(scene_path):
    if not HAS_OPEN3D:
        print("Warning: Open3D not installed, cannot load point cloud")
        return None, None, None

    scene_path = Path(scene_path)
    ply_path = scene_path / "processed_semantic.ply"
    ids_path = scene_path / "object_ids.npy"

    if not ply_path.exists():
        return None, None, None

    print(f"Loading semantic point cloud from {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if ids_path.exists():
        object_ids = np.load(ids_path).astype(np.int32)
        print(f"  Loaded {len(points):,} points with {len(np.unique(object_ids))} unique objects")
    else:
        print(f"  Warning: {ids_path} not found, using zeros")
        object_ids = np.zeros(len(points), dtype=np.int32)

    return points, colors, object_ids


def save_processed_pointcloud_with_semantics(scene_path, points, colors, object_ids):
    if not HAS_OPEN3D:
        print("Warning: Open3D not installed, cannot save point cloud")
        return

    scene_path = Path(scene_path)
    ply_path = scene_path / "processed_semantic.ply"
    ids_path = scene_path / "object_ids.npy"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(ply_path), pcd)

    np.save(ids_path, object_ids.astype(np.int32))

    print(f"Saved semantic point cloud to {ply_path}")
    print(f"Saved object IDs to {ids_path}")


# =============================================================================
# Reconstruction
# =============================================================================

def reconstruct_from_rgbd_with_semantics(
    scene_path,
    frame_indices,
    subsample=4,
    voxel_size=0.01,
    min_views=5,
    # semantic-preserving parameters
    prefer_nonzero=True,
    min_nonzero_votes=2,
    min_nonzero_ratio=0.20,  # good default for small objects
    background_id=0,
):
    """
    Reconstruct point cloud from RGBD frames WITH semantic labels,
    then filter using the new semantic-preserving pc_processing.
    """
    from tqdm import tqdm

    meta = load_record3d_metadata(scene_path)
    scene_path = Path(scene_path)

    masks_dir = scene_path / "EXR_RGBD" / "masks"
    has_masks = masks_dir.exists()

    if has_masks:
        print(f"Found masks directory: {masks_dir}")
        mapping_path = masks_dir / "object_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                mapping = json.load(f)
            print(f"  Prompts: {mapping.get('prompts', 'N/A')}")
            print(f"  Objects: {mapping.get('num_objects', 'N/A')}")
    else:
        print("No masks directory found - using object_id=0 for all points")

    points_by_frame, colors_by_frame, object_ids_by_frame = [], [], []

    print(f"\nReconstructing from {len(frame_indices)} RGBD frames...")

    for idx in tqdm(frame_indices, desc="Processing frames"):
        K, width, height = parse_intrinsics(meta, idx)
        c2w = parse_pose(meta, idx)

        file_id = str(idx)
        rgb_path = scene_path / "EXR_RGBD" / "rgb" / f"{file_id}.png"
        if not rgb_path.exists():
            rgb_path = scene_path / "EXR_RGBD" / "rgb" / f"{file_id}.jpg"

        depth_path = scene_path / "EXR_RGBD" / "depth" / f"{file_id}.exr"
        mask_path = (masks_dir / f"{file_id}.png") if has_masks else None

        if not rgb_path.exists():
            continue

        cam = Record3DCamera(
            image_path=rgb_path,
            depth_path=depth_path,
            c2w=c2w,
            K=K,
            width=width,
            height=height,
            camera_id=idx,
            image_name=file_id,
            mask_path=mask_path,
        )

        points, colors, obj_ids = create_point_cloud_from_rgbd(cam, subsample=subsample)
        if points is not None and len(points) > 0:
            points_by_frame.append(points)
            colors_by_frame.append(colors)
            object_ids_by_frame.append(obj_ids)

    if len(points_by_frame) == 0:
        print("Warning: No valid frames processed")
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0,), dtype=np.int32)

    print(f"Total raw points: {sum(len(p) for p in points_by_frame):,}")

    print("\n✓ Applying multiview filtering with semantic-preserving voting...")

    points, colors, object_ids = process_pointcloud_with_semantics(
        points_by_frame,
        colors_by_frame,
        object_ids_by_frame,
        voxel_size=voxel_size,
        min_views=min_views,
        prefer_nonzero=prefer_nonzero,
        min_nonzero_votes=min_nonzero_votes,
        min_nonzero_ratio=min_nonzero_ratio,
        background_id=background_id,
        cluster_eps=0.10,
        min_cluster_size=50,
        keep_largest_n=1,
        use_sor=True,
        nb_neighbors=20,
        std_ratio=1.5,
    )

    return points, colors, object_ids


# =============================================================================
# Scene class
# =============================================================================

class Record3DScene:
    """Scene loader for Record3D data WITH SEMANTIC SUPPORT."""

    def __init__(
        self,
        scene_path,
        train_frames=None,
        test_frames=None,
        subsample=4,
        pointcloud_path=None,
        use_semantics=True,
        frame_step=20,
        test_ratio=0.2,
        redo_semantics=False,
        # semantic-preserving voting knobs
        prefer_nonzero=True,
        min_nonzero_votes=2,
        min_nonzero_ratio=0.20,
        background_id=0,
    ):
        self.scene_path = Path(scene_path)
        self.model_path = str(scene_path)
        self.use_semantics = use_semantics

        meta = load_record3d_metadata(self.scene_path)
        n_frames = len(meta["poses"])

        if train_frames is None:
            train_frames = list(range(0, n_frames, frame_step))

        if test_frames is None:
            all_frames = set(range(n_frames))
            available_test_frames = list(all_frames - set(train_frames))
            num_test_frames = max(1, int(len(train_frames) * test_ratio))
            test_frames = sorted(
                random.sample(
                    available_test_frames,
                    min(num_test_frames, len(available_test_frames)),
                )
            )

        print("\nDataset split:")
        print(f"  Training: {len(train_frames)} frames")
        print(f"  Testing: {len(test_frames)} frames")

        masks_dir = self.scene_path / "EXR_RGBD" / "masks"
        self.has_masks = masks_dir.exists() and use_semantics

        if use_semantics and not masks_dir.exists():
            print(f"\n⚠️  Warning: use_semantics=True but no masks found at {masks_dir}")
            print("   Run generate_masks first, or set use_semantics=False")
            self.has_masks = False

        # cameras
        self.train_cameras = self._create_cameras(meta, train_frames)
        self.test_cameras = self._create_cameras(meta, test_frames)

        print("\nLoading point cloud...")

        if pointcloud_path is not None:
            raise NotImplementedError("Custom pointcloud_path path not wired in this rewrite (easy to add).")

        if use_semantics and self.has_masks:
            semantic_exists = (self.scene_path / "processed_semantic.ply").exists() and \
                              (self.scene_path / "object_ids.npy").exists()

            if semantic_exists and not redo_semantics:
                points, colors, object_ids = load_processed_pointcloud_with_semantics(self.scene_path)
            else:
                print("Creating semantic point cloud...")
                points, colors, object_ids = reconstruct_from_rgbd_with_semantics(
                    self.scene_path,
                    frame_indices=train_frames,
                    subsample=subsample,
                    voxel_size=0.01,
                    min_views=5,
                    prefer_nonzero=prefer_nonzero,
                    min_nonzero_votes=min_nonzero_votes,
                    min_nonzero_ratio=min_nonzero_ratio,
                    background_id=background_id,
                )
                save_processed_pointcloud_with_semantics(self.scene_path, points, colors, object_ids)
        else:
            raise NotImplementedError("Non-semantic path omitted in this rewrite (you can keep your old legacy path).")

        self.points = points
        self.colors = colors
        self.object_ids = object_ids
        self.num_objects = int(object_ids.max()) + 1 if len(object_ids) > 0 else 1

    def _create_cameras(self, meta, frame_indices):
        from tqdm import tqdm

        cameras = []
        masks_dir = self.scene_path / "EXR_RGBD" / "masks"

        print(f"\nCreating {len(frame_indices)} cameras...")

        for idx in tqdm(frame_indices, desc="Loading cameras"):
            K, width, height = parse_intrinsics(meta, idx)
            c2w = parse_pose(meta, idx)

            file_id = str(idx)
            rgb_path = self.scene_path / "EXR_RGBD" / "rgb" / f"{file_id}.png"
            if not rgb_path.exists():
                rgb_path = self.scene_path / "EXR_RGBD" / "rgb" / f"{file_id}.jpg"

            depth_path = self.scene_path / "EXR_RGBD" / "depth" / f"{file_id}.exr"

            mask_path = None
            if self.has_masks:
                candidate = masks_dir / f"{file_id}.png"
                if candidate.exists():
                    mask_path = candidate

            if not rgb_path.exists():
                print(f"  Warning: RGB not found for frame {idx}, skipping")
                continue

            cam = Record3DCamera(
                image_path=rgb_path,
                depth_path=depth_path,
                c2w=c2w,
                K=K,
                width=width,
                height=height,
                camera_id=idx,
                image_name=file_id,
                mask_path=mask_path,
            )
            cameras.append(cam)

        print(f"Created {len(cameras)} cameras")
        return cameras

    def getTrainCameras(self):
        return self.train_cameras.copy()

    def getTestCameras(self):
        return self.test_cameras.copy()

    def get_object_points(self, object_id):
        m = self.object_ids == object_id
        return self.points[m], self.colors[m]

    def get_object_ids_list(self):
        return np.unique(self.object_ids).tolist()

    def save(self, iteration):
        pass
