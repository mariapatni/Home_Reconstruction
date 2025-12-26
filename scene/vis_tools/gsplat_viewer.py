# File: scene/vis_tools/gsplat_ply_viewer.py
"""
gsplat_ply_viewer.py

A small, notebook-friendly renderer that loads a 3DGS-style PLY (your training export)
and renders frames using gsplat.rasterization with Record3D cameras (metadata.json).

Designed to match your training conventions:
- opacity is stored as LOGIT -> sigmoid(opacity)
- scales are stored as LOG-SCALE -> exp(scale)
- quaternion is normalized
- colors are taken from f_dc_0..2 by default (RGB-ish DC term)

Usage (in a Jupyter notebook):
    from scene.vis_tools.gsplat_ply_viewer import (
        load_gaussians_from_ply,
        Record3DMeta,
        render_record3d_frame,
        render_orbit,
        show_image,
    )

    gauss = load_gaussians_from_ply(ply_path, device="cuda")
    meta = Record3DMeta(scene_path)

    img = render_record3d_frame(gauss, meta, frame_idx=0)
    show_image(img)

Notes:
- Requires: gsplat, plyfile, torch, numpy, PIL
- If gsplat CUDA extension isn't available, you’ll hit the same build path as training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union, List, Dict

import json
import math
import numpy as np
import torch

from plyfile import PlyData

# gsplat import (will trigger CUDA extension load if not already built)
import gsplat


# ---------------------------
# Utilities
# ---------------------------

def _to_device(x: torch.Tensor, device: str) -> torch.Tensor:
    return x.to(device) if x.device.type != device else x


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def safe_normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # q: [N,4]
    n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
    return q / n


def ensure_float_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32, copy=False)).to(device)


def ensure_int_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.int64, copy=False)).to(device)


# ---------------------------
# Data containers
# ---------------------------

@dataclass
class Gaussians:
    means: torch.Tensor      # [N,3]
    quats: torch.Tensor      # [N,4]
    scales: torch.Tensor     # [N,3]  (linear)
    opacities: torch.Tensor  # [N]    (linear [0,1])
    colors: torch.Tensor     # [N,3]  (float, typically [0,1] but depends on export)

    @property
    def device(self) -> torch.device:
        return self.means.device


@dataclass
class Record3DMeta:
    scene_path: Union[str, Path]
    metadata_path: Optional[Union[str, Path]] = None

    def __post_init__(self):
        self.scene_path = Path(self.scene_path)
        if self.metadata_path is None:
            self.metadata_path = self.scene_path / "EXR_RGBD" / "metadata.json"
        else:
            self.metadata_path = Path(self.metadata_path)

        meta = json.loads(self.metadata_path.read_text())
        self._meta = meta
        self.width = int(meta.get("w", 720))
        self.height = int(meta.get("h", 960))
        self.num_frames = len(meta["poses"])

    def get_intrinsics(self, frame_idx: int = 0) -> np.ndarray:
        meta = self._meta
        w, h = self.width, self.height

        if "perFrameIntrinsicCoeffs" in meta and len(meta["perFrameIntrinsicCoeffs"]) > frame_idx:
            coeffs = meta["perFrameIntrinsicCoeffs"][frame_idx]
            fx, fy, cx, cy = coeffs[:4]
        elif "K" in meta and isinstance(meta["K"], list):
            K_flat = meta["K"]
            if len(K_flat) == 4:
                fx, fy, cx, cy = K_flat
            elif len(K_flat) == 9:
                K = np.array(K_flat, dtype=np.float32).reshape(3, 3)
                return K
            else:
                raise ValueError(f"Unexpected K length: {len(K_flat)}")
        else:
            # fallback
            fx = fy = float(w)
            cx = w / 2.0
            cy = h / 2.0

        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        return K

    def get_c2w(self, frame_idx: int) -> np.ndarray:
        poses = self._meta["poses"]
        if frame_idx >= len(poses):
            raise IndexError(f"frame_idx {frame_idx} out of range (num_frames={len(poses)})")

        pose_flat = poses[frame_idx]
        if len(pose_flat) == 16:
            c2w = np.array(pose_flat, dtype=np.float32).reshape(4, 4)
        elif len(pose_flat) == 12:
            c2w = np.array(pose_flat, dtype=np.float32).reshape(3, 4)
            c2w = np.vstack([c2w, [0, 0, 0, 1]]).astype(np.float32)
        else:
            raise ValueError(f"Unexpected pose format length: {len(pose_flat)}")
        return c2w

    def get_w2c_opencv(self, frame_idx: int) -> np.ndarray:
        """
        Match the convention used in your loader:
            gl_to_cv = diag([1,-1,-1,1])
            c2w_cv = c2w @ gl_to_cv
            w2c_cv = inv(c2w_cv)
        """
        c2w = self.get_c2w(frame_idx)
        gl_to_cv = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1],
        ], dtype=np.float32)
        c2w_cv = c2w @ gl_to_cv
        w2c_cv = np.linalg.inv(c2w_cv).astype(np.float32)
        return w2c_cv


# ---------------------------
# PLY Loading
# ---------------------------

def load_gaussians_from_ply(
    ply_path: Union[str, Path],
    device: str = "cuda",
    *,
    assume_opacity_is_logit: bool = True,
    assume_scale_is_log: bool = True,
    color_fields: Tuple[str, str, str] = ("f_dc_0", "f_dc_1", "f_dc_2"),
    clamp_scales: Tuple[float, float] = (1e-6, 1e2),
    clamp_opacity: Tuple[float, float] = (1e-6, 1.0),
) -> Gaussians:
    """
    Load gaussians from a 3DGS-style PLY with fields:
        x,y,z, opacity, scale_0..2, rot_0..3, and color fields (default f_dc_0..2)

    Your export appears to store:
      - opacity as logit (can be negative/positive)
      - scale as log-scale (negative/positive)
    """
    ply_path = Path(ply_path)
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    names = v.dtype.names

    required = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    for k in required:
        if k not in names:
            raise ValueError(f"PLY missing required field '{k}'. Found: {names}")

    for k in color_fields:
        if k not in names:
            raise ValueError(f"PLY missing color field '{k}'. Found: {names}")

    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32, copy=False)
    scales_raw = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32, copy=False)
    op_raw = v["opacity"].astype(np.float32, copy=False)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32, copy=False)
    cols = np.stack([v[color_fields[0]], v[color_fields[1]], v[color_fields[2]]], axis=1).astype(np.float32, copy=False)

    means_t = ensure_float_tensor(means, device)
    scales_raw_t = ensure_float_tensor(scales_raw, device)
    op_raw_t = ensure_float_tensor(op_raw, device)
    quats_t = ensure_float_tensor(quats, device)
    cols_t = ensure_float_tensor(cols, device)

    if assume_scale_is_log:
        scales_t = torch.exp(scales_raw_t)
    else:
        scales_t = scales_raw_t
    scales_t = scales_t.clamp(clamp_scales[0], clamp_scales[1])

    if assume_opacity_is_logit:
        op_t = sigmoid(op_raw_t)
    else:
        op_t = op_raw_t
    op_t = op_t.clamp(clamp_opacity[0], clamp_opacity[1])

    quats_t = safe_normalize_quat(quats_t)

    return Gaussians(
        means=means_t,
        quats=quats_t,
        scales=scales_t,
        opacities=op_t,
        colors=cols_t,
    )


# ---------------------------
# Rendering
# ---------------------------

@torch.no_grad()
def render_gsplat(
    gauss: Gaussians,
    *,
    viewmat_w2c: torch.Tensor,   # [4,4]
    K: torch.Tensor,             # [3,3]
    width: int,
    height: int,
    background: Optional[Union[float, Tuple[float, float, float]]] = None,
    packed: bool = False,
) -> torch.Tensor:
    """
    Returns:
        image: torch.FloatTensor [H,W,3] in [0,1] (as produced by gsplat; clamped for safety)
    """
    device = gauss.device
    viewmat_w2c = _to_device(viewmat_w2c, device)
    K = _to_device(K, device)

    if background is None:
        backgrounds = None
    else:
        if isinstance(background, (int, float)):
            b = float(background)
            backgrounds = torch.tensor([[b, b, b]], dtype=torch.float32, device=device)
        else:
            backgrounds = torch.tensor([list(background)], dtype=torch.float32, device=device)

    renders, alphas, info = gsplat.rasterization(
        means=gauss.means,
        quats=gauss.quats,
        scales=gauss.scales,
        opacities=gauss.opacities,
        colors=gauss.colors,
        viewmats=viewmat_w2c.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=int(width),
        height=int(height),
        packed=packed,
        backgrounds=backgrounds,
    )

    img = renders[0].clamp(0.0, 1.0)  # [H,W,3]
    return img


@torch.no_grad()
def render_record3d_frame(
    gauss: Gaussians,
    meta: Record3DMeta,
    frame_idx: int,
    *,
    device: Optional[str] = None,
    background: Optional[Union[float, Tuple[float, float, float]]] = None,
    packed: bool = False,
) -> np.ndarray:
    """
    Render a specific Record3D frame index using metadata.json cameras.

    Returns:
        img_np uint8 [H,W,3]
    """
    if device is not None:
        # move gaussians
        gauss = Gaussians(
            means=gauss.means.to(device),
            quats=gauss.quats.to(device),
            scales=gauss.scales.to(device),
            opacities=gauss.opacities.to(device),
            colors=gauss.colors.to(device),
        )

    K_np = meta.get_intrinsics(frame_idx)
    w2c_np = meta.get_w2c_opencv(frame_idx)

    K = torch.from_numpy(K_np).float().to(gauss.device)
    w2c = torch.from_numpy(w2c_np).float().to(gauss.device)

    img = render_gsplat(
        gauss,
        viewmat_w2c=w2c,
        K=K,
        width=meta.width,
        height=meta.height,
        background=background,
        packed=packed,
    )

    img_u8 = (img.detach().cpu().numpy() * 255.0).astype(np.uint8)
    return img_u8


# ---------------------------
# Orbit camera (independent of Record3D)
# ---------------------------

def _look_at_w2c_opencv(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0, 1, 0], dtype=np.float32),
) -> np.ndarray:
    """
    Build a simple OpenCV-style w2c look-at matrix.
    OpenCV camera looks along +Z? Many conventions differ; gsplat expects a standard w2c.
    We'll construct with camera forward pointing from eye->target as +Z in camera space,
    which is common in some CV pipelines.

    If the orbit looks flipped, swap signs on forward or up.
    """
    eye = eye.astype(np.float32)
    target = target.astype(np.float32)
    up = up.astype(np.float32)

    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)

    true_up = np.cross(right, forward)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-8)

    # Camera-to-world (c2w) basis columns (right, up, forward)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye

    w2c = np.linalg.inv(c2w).astype(np.float32)
    return w2c


def _make_K_from_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    """
    Simple pinhole intrinsics from horizontal fov (approx).
    """
    fov = math.radians(float(fov_deg))
    fx = (width / 2.0) / math.tan(fov / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


@torch.no_grad()
def render_orbit(
    gauss: Gaussians,
    *,
    width: int = 960,
    height: int = 720,
    fov_deg: float = 60.0,
    center: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    n_frames: int = 120,
    elevation_deg: float = 10.0,
    background: Optional[Union[float, Tuple[float, float, float]]] = None,
    packed: bool = False,
) -> List[np.ndarray]:
    """
    Render an orbit (turntable) sequence using a synthetic camera.

    Returns:
        list of uint8 images [H,W,3]
    """
    device = gauss.device

    means = gauss.means.detach().cpu().numpy()
    if center is None:
        center = means.mean(axis=0).astype(np.float32)
    if radius is None:
        mn = means.min(axis=0)
        mx = means.max(axis=0)
        diag = float(np.linalg.norm(mx - mn))
        radius = max(0.5, diag * 0.7)

    elev = math.radians(float(elevation_deg))
    K_np = _make_K_from_fov(width, height, fov_deg)
    K = torch.from_numpy(K_np).float().to(device)

    frames: List[np.ndarray] = []
    for i in range(int(n_frames)):
        theta = 2.0 * math.pi * (i / float(n_frames))
        eye = np.array([
            center[0] + radius * math.cos(theta) * math.cos(elev),
            center[1] + radius * math.sin(elev),
            center[2] + radius * math.sin(theta) * math.cos(elev),
        ], dtype=np.float32)

        w2c_np = _look_at_w2c_opencv(eye, center)
        w2c = torch.from_numpy(w2c_np).float().to(device)

        img = render_gsplat(
            gauss,
            viewmat_w2c=w2c,
            K=K,
            width=width,
            height=height,
            background=background,
            packed=packed,
        )
        frames.append((img.cpu().numpy() * 255.0).astype(np.uint8))

    return frames


# ---------------------------
# Notebook helpers
# ---------------------------

def show_image(img_u8: np.ndarray):
    """
    Display uint8 image in a notebook without needing cv2.
    """
    from PIL import Image
    from IPython.display import display
    display(Image.fromarray(img_u8))


def save_gif(frames_u8: List[np.ndarray], out_path: Union[str, Path], fps: int = 24):
    """
    Save a list of uint8 frames [H,W,3] as GIF.
    """
    from PIL import Image
    out_path = Path(out_path)
    imgs = [Image.fromarray(f) for f in frames_u8]
    duration_ms = int(1000 / max(1, fps))
    imgs[0].save(
        out_path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
    )
    return str(out_path)
