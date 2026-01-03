"""
Record3D data loader for Gaussian Splatting
WITH SEMANTIC SEGMENTATION SUPPORT (STREAMING VERSION)

Expected scene structure:

scene_path/
  EXR_RGBD/
    metadata.json
    rgb/0.png, 1.png, ...
    depth/0.exr, 1.exr, ...
  object_id_masks/
    frame_0000.png, frame_0020.png, ...   (or fallback: {idx}.png)
  class_mapping.json                       (optional)

Outputs produced:
  processed_semantic.ply
  object_ids.npy
  class_mapping_remapped.json              (optional)
  id_remap.json                            (optional)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

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

# ✅ Processing entry points (streaming version)
from scene.data_loaders.raw_pc_processing import (
    create_raw_pointcloud,
    streaming_voxel_filter,
    remove_small_clusters_label_aware,
    statistical_outlier_removal,
    remove_disconnected_chunks,
    remap_object_ids,
    print_semantic_summary,
    # Legacy wrapper still available if needed
    process_pointcloud_with_semantics,
)


# =============================================================================
# Camera
# =============================================================================
class Record3DCamera:
    """Camera object compatible with the training pipeline."""

    def __init__(
        self,
        image_path: Path,
        depth_path: Optional[Path],
        c2w: np.ndarray,
        K: np.ndarray,
        width: int,
        height: int,
        camera_id: int,
        image_name: str,
        mask_path: Optional[Path] = None,
    ):
        # ------------------------------------------------------------
        # Device (trainer expects GPU-cached tensors)
        # ------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------
        # Load data (CPU first)
        # ------------------------------------------------------------
        self.original_image = self._load_rgb(image_path)  # [3,H,W] float in [0,1], CPU

        self.depth_map = None
        if depth_path is not None and Path(depth_path).exists():
            self.depth_map = self._load_depth(depth_path, width, height)  # [H,W], CPU (float)

        self.object_mask = None
        if mask_path is not None and Path(mask_path).exists():
            self.object_mask = self._load_mask(mask_path, width, height)  # [H,W] long, CPU

        # ------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------
        self.camera_id = int(camera_id)
        self.image_name = str(image_name)
        self.image_width = int(width)
        self.image_height = int(height)

        # Intrinsics (floats)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])

        # FoV (used by projection builder)
        self.FoVx = 2 * np.arctan(self.image_width / (2 * self.fx))
        self.FoVy = 2 * np.arctan(self.image_height / (2 * self.fy))

        # ------------------------------------------------------------
        # Pose / transforms (put on device)
        # ------------------------------------------------------------
        self.c2w = torch.tensor(c2w, dtype=torch.float32, device=device)  # [4,4]
        self.camera_center = self.c2w[:3, 3]  # [3]

        # World->camera in "GL-ish" convention used elsewhere in code
        w2c = torch.inverse(self.c2w)
        self.world_view_transform = w2c.transpose(0, 1)  # [4,4]

        # Projection + full projection (OpenGL-style matrix as you had)
        self.projection_matrix = self._get_projection_matrix().to(device)  # [4,4]
        self.full_proj_transform = (self.world_view_transform @ self.projection_matrix).contiguous()

        # ------------------------------------------------------------
        # Trainer-compatibility caches (scene/train.py expects these)
        # ------------------------------------------------------------
        # GT image + intrinsics on device
        self._gt_image_gpu = self.original_image.to(device).contiguous()          # [3,H,W]
        self._K_gpu = torch.tensor(K, dtype=torch.float32, device=device).contiguous()  # [3,3]

        # Optional GT mask on device
        self._gt_mask_gpu = self.object_mask.to(device).contiguous() if self.object_mask is not None else None  # [H,W] long or None

        # View matrix in OpenCV convention for gsplat (world->camera)
        # render_with_semantics() accesses camera._viewmat_gpu
        self._viewmat_gpu = self.get_opencv_viewmat().to(device).contiguous()  # [4,4]

        # These are frequently accessed by your model/trainer; safe to cache
        self._campos_gpu = self.camera_center.contiguous()  # [3]
        self._projmat_gpu = self.projection_matrix.contiguous()  # [4,4]
        self._fullproj_gpu = (self._viewmat_gpu @ self._projmat_gpu).contiguous()  # [4,4]

        # Width/height cache (some trainers stash these)
        self._W = self.image_width
        self._H = self.image_height

        # ------------------------------------------------------------
        # Depth supervision helpers (keep on CPU unless your trainer needs GPU)
        # ------------------------------------------------------------
        self.alpha_mask = torch.ones((1, self.image_height, self.image_width), dtype=torch.float32)

        if self.depth_map is not None:
            valid = (self.depth_map > 0) & torch.isfinite(self.depth_map)
            self.invdepthmap = torch.zeros_like(self.depth_map)
            self.invdepthmap[valid] = 1.0 / self.depth_map[valid]
            self.depth_mask = valid.float()
        else:
            self.invdepthmap = None
            self.depth_mask = None

        self.resolution_scale = 1.0


    def _load_rgb(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = torch.from_numpy(np.array(img)).float() / 255.0
        return img.permute(2, 0, 1)

    def _load_mask(self, path: Path, target_width: int, target_height: int) -> Optional[torch.Tensor]:
        """
        Load semantic mask as [H, W] tensor of integer object IDs.
        Expects grayscale PNG where pixel value = object ID.
        """
        try:
            mask_img = Image.open(path)
            if mask_img.mode in ["RGB", "RGBA", "P"]:
                mask_img = mask_img.convert("L")
            mask = torch.from_numpy(np.array(mask_img)).long()

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

    def _load_depth(self, path: Path, target_width: int, target_height: int) -> Optional[torch.Tensor]:
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

    def _get_projection_matrix(self, znear: float = 0.01, zfar: float = 100.0) -> torch.Tensor:
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

    def get_opencv_viewmat(self) -> torch.Tensor:
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

def load_record3d_metadata(scene_path: Path) -> Dict[str, Any]:
    metadata_path = Path(scene_path) / "EXR_RGBD" / "metadata.json"
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    print(f"Loaded metadata from {metadata_path}")
    print(f"  Image dimensions: {meta.get('w', '?')}x{meta.get('h', '?')}")
    print(f"  Number of frames: {len(meta['poses'])}")
    return meta


def parse_intrinsics(meta: Dict[str, Any], frame_idx: int = 0) -> Tuple[np.ndarray, int, int]:
    width = int(meta.get("w", 720))
    height = int(meta.get("h", 960))

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
        fx = fy = float(width)
        cx = width / 2.0
        cy = height / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K, width, height


def parse_pose(meta: Dict[str, Any], frame_idx: int) -> np.ndarray:
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
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
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

def create_point_cloud_from_rgbd(
    cam: "Record3DCamera",
    subsample: int = 4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Create a point cloud from RGB-D camera data WITH SEMANTIC LABELS.
    GPU-first implementation: keeps all math on cam.depth_map/c2w device,
    only moves to CPU at the end for numpy outputs.

    Returns:
        points:    (N,3) float32 numpy
        colors:    (N,3) float32 numpy in [0,1] (or whatever your original_image uses)
        object_ids:(N,)  int32 numpy
    """
    if cam.depth_map is None:
        return None, None, None

    # Choose device/dtype from depth_map (or c2w). This prevents CPU/CUDA mixing.
    device = cam.depth_map.device if isinstance(cam.depth_map, torch.Tensor) else torch.device("cpu")
    depth = cam.depth_map.to(device)

    H, W = depth.shape

    # Build subsampled pixel grid ON DEVICE
    u = torch.arange(0, W, subsample, device=device, dtype=torch.float32)
    v = torch.arange(0, H, subsample, device=device, dtype=torch.float32)
    u_grid, v_grid = torch.meshgrid(u, v, indexing="xy")
    u_flat = u_grid.reshape(-1)
    v_flat = v_grid.reshape(-1)

    # Sample depth
    depth_flat = depth[v_flat.long(), u_flat.long()]
    valid = (depth_flat > 0) & torch.isfinite(depth_flat)

    if not torch.any(valid):
        return None, None, None

    u_valid = u_flat[valid]
    v_valid = v_flat[valid]
    depth_valid = depth_flat[valid]

    # Intrinsics to device/dtype
    fx = torch.as_tensor(cam.fx, device=device, dtype=torch.float32)
    fy = torch.as_tensor(cam.fy, device=device, dtype=torch.float32)
    cx = torch.as_tensor(cam.cx, device=device, dtype=torch.float32)
    cy = torch.as_tensor(cam.cy, device=device, dtype=torch.float32)

    # Camera-space points
    x_cam = (u_valid - cx) * depth_valid / fx
    y_cam = -(v_valid - cy) * depth_valid / fy
    z_cam = -depth_valid
    points_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)  # [N,3]

    # Homogeneous transform (ALL ON DEVICE)
    c2w = cam.c2w
    if not isinstance(c2w, torch.Tensor):
        c2w = torch.as_tensor(c2w, device=device, dtype=torch.float32)
    else:
        c2w = c2w.to(device=device, dtype=torch.float32)

    ones = torch.ones((points_cam.shape[0], 1), device=device, dtype=points_cam.dtype)
    points_cam_hom = torch.cat([points_cam, ones], dim=1)  # [N,4]
    points_world_hom = points_cam_hom @ c2w.T
    points_world = points_world_hom[:, :3]  # [N,3]

    # Colors (keep on device, move at end)
    img = cam.original_image
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)
    img = img.to(device)
    rgb_flat = img[:, v_valid.long(), u_valid.long()].T.contiguous()  # [N,3]

    # Object IDs
    if cam.object_mask is not None:
        om = cam.object_mask
        if not isinstance(om, torch.Tensor):
            om = torch.as_tensor(om)
        om = om.to(device)
        object_ids = om[v_valid.long(), u_valid.long()].to(torch.int32)
    else:
        object_ids = torch.zeros((points_world.shape[0],), device=device, dtype=torch.int32)

    # Return CPU numpy
    return (
        points_world.detach().cpu().numpy().astype(np.float32, copy=False),
        rgb_flat.detach().cpu().numpy().astype(np.float32, copy=False),
        object_ids.detach().cpu().numpy().astype(np.int32, copy=False),
    )


# =============================================================================
# Point Cloud I/O
# =============================================================================

def load_processed_pointcloud_with_semantics(scene_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
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


def save_processed_pointcloud_with_semantics(scene_path: Path, points: np.ndarray, colors: np.ndarray, object_ids: np.ndarray) -> None:
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
# Frame Generator (STREAMING)
# =============================================================================

def _find_rgb_path(scene_path: Path, idx: int) -> Optional[Path]:
    file_id = str(idx)
    rgb_path = scene_path / "EXR_RGBD" / "rgb" / f"{file_id}.png"
    if rgb_path.exists():
        return rgb_path
    rgb_path = scene_path / "EXR_RGBD" / "rgb" / f"{file_id}.jpg"
    if rgb_path.exists():
        return rgb_path
    return None


def _find_mask_path(masks_dir: Path, idx: int) -> Optional[Path]:
    # preferred naming: frame_XXXX.png
    candidate = masks_dir / f"frame_{idx:04d}.png"
    if candidate.exists():
        return candidate
    # fallback: {idx}.png
    candidate = masks_dir / f"{idx}.png"
    if candidate.exists():
        return candidate
    # fallback: {idx}.jpg (unlikely but harmless)
    candidate = masks_dir / f"{idx}.jpg"
    if candidate.exists():
        return candidate
    return None


def _load_class_mapping(scene_path: Path) -> Optional[Dict[Any, Any]]:
    mapping_path = scene_path / "class_mapping.json"
    if not mapping_path.exists():
        return None
    with open(mapping_path, "r") as f:
        cm = json.load(f)
    # keep as-is; downstream code tolerates str/int keys
    return cm


def create_frame_generator(
    scene_path: Path,
    frame_indices: List[int],
    meta: Dict[str, Any],
    masks_dir: Optional[Path],
    subsample: int = 4,
    verbose: bool = True,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Generator that yields (points, colors, object_ids) per frame.
    Loads and processes one frame at a time to minimize memory.
    """
    from tqdm import tqdm
    
    iterator = tqdm(frame_indices, desc="Processing frames") if verbose else frame_indices
    
    for idx in iterator:
        K, width, height = parse_intrinsics(meta, idx)
        c2w = parse_pose(meta, idx)
        
        rgb_path = _find_rgb_path(scene_path, idx)
        if rgb_path is None:
            continue
        
        depth_path = scene_path / "EXR_RGBD" / "depth" / f"{idx}.exr"
        mask_path = _find_mask_path(masks_dir, idx) if masks_dir else None
        
        cam = Record3DCamera(
            image_path=rgb_path,
            depth_path=depth_path,
            c2w=c2w,
            K=K,
            width=width,
            height=height,
            camera_id=idx,
            image_name=str(idx),
            mask_path=mask_path,
        )
        
        pts, cols, obj_ids = create_point_cloud_from_rgbd(cam, subsample=subsample)
        
        if pts is not None and len(pts) > 0:
            yield (pts, cols, obj_ids)
        
        # Camera and frame data go out of scope here, memory freed


# =============================================================================
# Reconstruction (STREAMING)
# =============================================================================

def reconstruct_from_rgbd_with_semantics(
    scene_path: Path,
    frame_indices: List[int],
    *,
    subsample: int = 4,

    # voxel filtering
    voxel_size: float = 0.01,
    min_views: int = 2,

    # semantic voting
    prefer_nonzero: bool = True,
    min_nonzero_votes: int = 1,
    min_nonzero_ratio: float = 0.05,
    background_id: int = 0,

    # cluster + outlier
    cluster_eps: float = 0.05,
    min_cluster_size: int = 20,
    use_sor: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.5,

    # remap
    remap_min_points: int = 15,
    save_remapped_mapping: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct point cloud from RGBD frames WITH semantic labels.
    STREAMING IMPLEMENTATION - processes frames incrementally.

    Expects masks in: scene_path/object_id_masks/
    Expects class mapping in: scene_path/class_mapping.json (optional)
    
    Pipeline (each step can be commented out for testing):
        1. streaming_voxel_filter - multiview filtering with semantic voting
        2. remove_small_clusters_label_aware - DBSCAN (protects labeled points)
        3. statistical_outlier_removal - SOR cleanup
        4. remove_disconnected_chunks - remove floating artifacts
        5. remap_object_ids - contiguous ID assignment
    """
    scene_path = Path(scene_path)
    meta = load_record3d_metadata(scene_path)

    masks_dir = scene_path / "object_id_masks"
    has_masks = masks_dir.exists()

    class_mapping = _load_class_mapping(scene_path) if has_masks else None
    
    if has_masks:
        print(f"✓ Found masks directory: {masks_dir}")
        if class_mapping is not None:
            try:
                cm_keys = [k for k in class_mapping.keys()]
                print(f"  Loaded class mapping with {len(cm_keys)} entries")
            except Exception:
                print("  Loaded class mapping")
        else:
            print("  Warning: class_mapping.json not found - proceeding without mapping")
    else:
        print("⚠️  No masks directory found - using object_id=0 for all points")

    print(f"\n{'='*60}")
    print("POINT CLOUD PROCESSING WITH SEMANTICS (STREAMING)")
    print(f"{'='*60}")
    print(f"Processing {len(frame_indices)} frames...")

    # =========================================================================
    # Step 1: Streaming voxel filter with semantic voting
    # =========================================================================
    frame_gen = create_frame_generator(
        scene_path,
        frame_indices,
        meta,
        masks_dir if has_masks else None,
        subsample=subsample,
        verbose=True,
    )
    
    points, colors, obj_ids = streaming_voxel_filter(
        frame_gen,
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
        return points, colors, obj_ids

    # =========================================================================
    # Step 2: Label-aware cluster cleaning (DBSCAN)
    # Comment out this block to skip DBSCAN
    # =========================================================================
    points, colors, obj_ids = remove_small_clusters_label_aware(
        points, colors, obj_ids,
        eps=cluster_eps,
        min_cluster_size=min_cluster_size,
        background_id=background_id,
        verbose=True,
    )

    # =========================================================================
    # Step 3: Statistical outlier removal
    # Comment out this block to skip SOR
    # =========================================================================
    if use_sor and len(points) > 0:
        points, colors, obj_ids = statistical_outlier_removal(
            points, colors, obj_ids,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
            verbose=True,
        )

    # =========================================================================
    # Step 4: Remove disconnected chunks (floating artifacts)
    # Comment out this block to skip chunk removal
    # =========================================================================
    points, colors, obj_ids = remove_disconnected_chunks(
        points, colors, obj_ids,
        eps=0.15,        # Points within 15cm are neighbors
        min_samples=10,  # Need 10+ points to form a cluster core
        verbose=True,
    )

    # =========================================================================
    # Step 5: Remap object IDs to contiguous values
    # Comment out this block to skip remapping
    # =========================================================================
    new_class_mapping: Dict[str, Any] = {"0": "background"}
    id_remap: Dict[int, int] = {background_id: 0}
    
    if len(obj_ids) > 0:
        obj_ids, new_class_mapping, id_remap = remap_object_ids(
            obj_ids,
            class_mapping=class_mapping,
            min_points=remap_min_points,
            background_id=background_id,
            verbose=True,
        )

    # =========================================================================
    # Save remapped mapping files
    # =========================================================================
    if save_remapped_mapping and new_class_mapping is not None:
        try:
            out_map_path = scene_path / "class_mapping_remapped.json"
            with open(out_map_path, "w") as f:
                json.dump(new_class_mapping, f, indent=2)

            out_remap_path = scene_path / "id_remap.json"
            with open(out_remap_path, "w") as f:
                json.dump({str(k): int(v) for k, v in id_remap.items()}, f, indent=2)

            print(f"\nSaved remapped class mapping to {out_map_path}")
            print(f"Saved id remap to {out_remap_path}")
        except Exception as e:
            print(f"Warning: could not save remapped mapping files: {e}")

    # Print final summary
    print_semantic_summary(obj_ids, new_class_mapping)

    return points, colors, obj_ids


# =============================================================================
# Scene class
# =============================================================================

class Record3DScene:
    """Scene loader for Record3D data WITH semantic support (STREAMING)."""

    def __init__(
        self,
        scene_path: str | Path,
        *,
        train_frames: Optional[List[int]] = None,
        test_frames: Optional[List[int]] = None,
        subsample: int = 4,
        use_semantics: bool = True,
        frame_step: int = 20,
        test_ratio: float = 0.2,
        redo_semantics: bool = False,

        # semantic voting knobs
        prefer_nonzero: bool = True,
        min_nonzero_votes: int = 1,
        min_nonzero_ratio: float = 0.05,
        background_id: int = 0,

        # processing knobs
        voxel_size: float = 0.01,
        min_views: int = 2,
        cluster_eps: float = 0.05,
        min_cluster_size: int = 20,
        use_sor: bool = True,
        nb_neighbors: int = 20,
        std_ratio: float = 2.5,

        # semantic cleanup/remap
        remap_min_points: int = 15,
    ):
        self.scene_path = Path(scene_path)
        self.model_path = str(self.scene_path)
        self.use_semantics = bool(use_semantics)

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
        print(f"  Testing:  {len(test_frames)} frames")

        masks_dir = self.scene_path / "object_id_masks"
        self.has_masks = masks_dir.exists() and self.use_semantics

        if self.use_semantics and not masks_dir.exists():
            print(f"\n⚠️  Warning: use_semantics=True but no masks found at {masks_dir}")
            print("   Run your mask generation first, or set use_semantics=False")
            self.has_masks = False

        # cameras
        self.train_cameras = self._create_cameras(meta, train_frames)
        self.test_cameras = self._create_cameras(meta, test_frames)

        print("\nLoading point cloud...")

        if not self.use_semantics or not self.has_masks:
            raise NotImplementedError("Non-semantic path omitted - add legacy code if needed.")

        semantic_exists = (
            (self.scene_path / "processed_semantic.ply").exists() 
            and (self.scene_path / "object_ids.npy").exists()
        )

        if semantic_exists and not redo_semantics:
            points, colors, object_ids = load_processed_pointcloud_with_semantics(self.scene_path)
            if points is None:
                points, colors, object_ids = (
                    np.zeros((0, 3), np.float32),
                    np.zeros((0, 3), np.float32),
                    np.zeros((0,), np.int32),
                )
        else:
            print("Creating semantic point cloud (streaming)...")
            points, colors, object_ids = reconstruct_from_rgbd_with_semantics(
                self.scene_path,
                frame_indices=train_frames,
                subsample=subsample,

                voxel_size=voxel_size,
                min_views=min_views,

                prefer_nonzero=prefer_nonzero,
                min_nonzero_votes=min_nonzero_votes,
                min_nonzero_ratio=min_nonzero_ratio,
                background_id=background_id,

                cluster_eps=cluster_eps,
                min_cluster_size=min_cluster_size,

                use_sor=use_sor,
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio,

                remap_min_points=remap_min_points,
                save_remapped_mapping=True,
            )
            save_processed_pointcloud_with_semantics(self.scene_path, points, colors, object_ids)

        self.points = points
        self.colors = colors
        self.object_ids = object_ids
        self.num_objects = int(object_ids.max()) + 1 if len(object_ids) > 0 else 1

    def _create_cameras(self, meta: Dict[str, Any], frame_indices: List[int]) -> List[Record3DCamera]:
        from tqdm import tqdm

        cameras: List[Record3DCamera] = []
        masks_dir = self.scene_path / "object_id_masks"

        print(f"\nCreating {len(frame_indices)} cameras...")

        for idx in tqdm(frame_indices, desc="Loading cameras"):
            K, width, height = parse_intrinsics(meta, idx)
            c2w = parse_pose(meta, idx)

            rgb_path = _find_rgb_path(self.scene_path, idx)
            if rgb_path is None:
                print(f"  Warning: RGB not found for frame {idx}, skipping")
                continue

            depth_path = self.scene_path / "EXR_RGBD" / "depth" / f"{idx}.exr"
            mask_path = _find_mask_path(masks_dir, idx) if self.has_masks else None

            cam = Record3DCamera(
                image_path=rgb_path,
                depth_path=depth_path,
                c2w=c2w,
                K=K,
                width=width,
                height=height,
                camera_id=idx,
                image_name=str(idx),
                mask_path=mask_path,
            )
            cameras.append(cam)

        print(f"Created {len(cameras)} cameras")
        return cameras

    def getTrainCameras(self) -> List[Record3DCamera]:
        return self.train_cameras.copy()

    def getTestCameras(self) -> List[Record3DCamera]:
        return self.test_cameras.copy()

    def get_object_points(self, object_id: int) -> Tuple[np.ndarray, np.ndarray]:
        m = self.object_ids == int(object_id)
        return self.points[m], self.colors[m]

    def get_object_ids_list(self) -> List[int]:
        return np.unique(self.object_ids).tolist()

    def save(self, iteration: int) -> None:
        # Training code can hook into this if desired
        pass