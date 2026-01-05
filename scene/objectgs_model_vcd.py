"""
ObjectGS: Anchor-based Gaussian Splatting with Object Awareness
Enhanced with FastGS-inspired View-Consistent Densification (VCD)

Key additions:
- Multi-view consistent densification (VCD) using footprint-based error aggregation
- Multi-view consistent pruning (VCP) 
- Comprehensive logging for understanding densification/pruning decisions
- Configurable hybrid mode (gradient + VCD) or pure VCD mode

Based on FastGS paper: "FastGS: Training 3D Gaussian Splatting in 100 Seconds"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewDependentAttributeMLP(nn.Module):
    """
    Generates Gaussian attributes from anchor features.
    (Unchanged from original)
    """

    def __init__(
        self,
        feature_dim: int = 32,
        k: int = 10,
        view_dim: int = 4,
        color_delta_scale: float = 0.25,
    ):
        super().__init__()
        self.k = int(k)
        self.feature_dim = int(feature_dim)
        self.view_dim = int(view_dim)
        self.color_delta_scale = float(color_delta_scale)

        view_input_dim = self.feature_dim + self.view_dim

        self.opacity_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.k),
        )

        self.scale_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.k * 3),
        )

        self.rotation_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.k * 4),
        )

        self.color_mlp = nn.Sequential(
            nn.Linear(view_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.k * 3),
        )

        self._init_stable()

    def _init_stable(self):
        with torch.no_grad():
            scale_out = self.scale_mlp[-1]
            scale_out.weight.mul_(0.1)
            scale_out.bias.fill_(-4.0)

            op_out = self.opacity_mlp[-1]
            op_out.weight.mul_(0.1)
            op_out.bias.fill_(-2.0)

            rot_out = self.rotation_mlp[-1]
            rot_out.weight.mul_(0.1)
            rot_out.bias.zero_()
            rot_out.bias[3::4] = 1.0

            col_out = self.color_mlp[-1]
            col_out.weight.mul_(0.1)
            col_out.bias.zero_()

    def forward(
        self,
        anchor_features: torch.Tensor,
        view_dirs: Optional[torch.Tensor] = None,
        view_dists: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = anchor_features.shape[0]

        opacity_raw = self.opacity_mlp(anchor_features)
        scale_raw = self.scale_mlp(anchor_features).view(N, self.k, 3)

        rotation = self.rotation_mlp(anchor_features).view(N, self.k, 4)
        rotation = rotation / (rotation.norm(dim=-1, keepdim=True) + 1e-9)

        if view_dirs is not None and view_dists is not None:
            view_info = torch.cat([view_dists, view_dirs], dim=-1)
        else:
            view_info = torch.zeros(
                (N, self.view_dim),
                device=anchor_features.device,
                dtype=anchor_features.dtype,
            )

        col_in = torch.cat([anchor_features, view_info], dim=-1)
        color_delta = self.color_mlp(col_in).view(N, self.k, 3)
        color_delta = torch.tanh(color_delta) * self.color_delta_scale

        return opacity_raw, scale_raw, rotation, color_delta


class VCDStats:
    """
    Container for View-Consistent Densification statistics and logging.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.reset()
    
    def reset(self):
        """Reset all accumulated statistics."""
        self.views_sampled = 0
        self.total_high_error_pixels = 0
        self.total_pixels_evaluated = 0
        self.per_view_stats: List[Dict] = []
        self.anchor_score_histogram: Optional[torch.Tensor] = None
        self.gaussian_score_histogram: Optional[torch.Tensor] = None
    
    def log_view_stats(self, view_idx: int, stats: Dict):
        """Log statistics for a single view."""
        self.per_view_stats.append(stats)
        self.views_sampled += 1
        self.total_high_error_pixels += stats.get('high_error_pixels', 0)
        self.total_pixels_evaluated += stats.get('total_pixels', 0)
        
        if self.logger:
            self.logger.debug(
                f"  [VCD View {view_idx}] "
                f"L1={stats.get('l1_loss', 0):.4f} "
                f"high_err_px={stats.get('high_error_pixels', 0):,}/{stats.get('total_pixels', 0):,} "
                f"({stats.get('high_error_fraction', 0)*100:.1f}%) "
                f"gaussians_in_high_err={stats.get('gaussians_in_high_error', 0):,}"
            )
    
    def log_summary(self, anchor_scores: torch.Tensor, gaussian_scores: torch.Tensor, 
                    densify_threshold: float, prune_threshold: float):
        """Log summary statistics after all views processed."""
        if self.logger is None:
            return
        
        # Compute histograms
        anchor_scores_np = anchor_scores.detach().cpu().numpy()
        gaussian_scores_np = gaussian_scores.detach().cpu().numpy()
        
        # Anchor score statistics
        a_min, a_max = float(anchor_scores.min()), float(anchor_scores.max())
        a_mean, a_std = float(anchor_scores.mean()), float(anchor_scores.std())
        a_median = float(anchor_scores.median())
        
        # How many anchors above/below thresholds
        anchors_above_densify = int((anchor_scores > densify_threshold).sum())
        anchors_below_prune = int((anchor_scores < prune_threshold).sum())
        
        self.logger.info(f"  [VCD Summary] Views sampled: {self.views_sampled}")
        self.logger.info(f"  [VCD Summary] Total high-error pixels: {self.total_high_error_pixels:,}/{self.total_pixels_evaluated:,} "
                        f"({self.total_high_error_pixels/max(1, self.total_pixels_evaluated)*100:.1f}%)")
        self.logger.info(f"  [VCD Summary] Anchor scores: min={a_min:.4f}, max={a_max:.4f}, "
                        f"mean={a_mean:.4f}, std={a_std:.4f}, median={a_median:.4f}")
        self.logger.info(f"  [VCD Summary] Anchors above densify threshold ({densify_threshold}): "
                        f"{anchors_above_densify:,} ({anchors_above_densify/len(anchor_scores)*100:.1f}%)")
        self.logger.info(f"  [VCD Summary] Anchors below prune threshold ({prune_threshold}): "
                        f"{anchors_below_prune:,} ({anchors_below_prune/len(anchor_scores)*100:.1f}%)")
        
        # Log percentile distribution
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        pct_values = np.percentile(anchor_scores_np, percentiles)
        pct_str = ", ".join([f"p{p}={v:.4f}" for p, v in zip(percentiles, pct_values)])
        self.logger.info(f"  [VCD Summary] Anchor score percentiles: {pct_str}")


class ObjectGSModelVCD(nn.Module):
    """
    ObjectGS Model with FastGS-inspired View-Consistent Densification (VCD).
    
    Key differences from original ObjectGSModel:
    1. Adds VCD score computation using 2D footprint error aggregation
    2. Adds VCP score computation for pruning decisions
    3. Comprehensive logging of densification/pruning decisions
    4. Configurable hybrid (gradient + VCD) or pure VCD mode
    """
    
    def __init__(
        self,
        point_cloud: Optional[Union[np.ndarray, torch.Tensor]],
        colors: Optional[Union[np.ndarray, torch.Tensor]],
        object_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        voxel_size: float = 0.01,
        k: int = 10,
        feature_dim: int = 32,
        object_names: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        color_delta_scale: float = 0.25,
        num_objects_override: Optional[int] = None,
        # VCD-specific parameters
        vcd_num_views: int = 10,
        vcd_error_threshold: float = 0.5,
        vcd_mode: str = "hybrid",  # "hybrid", "pure_vcd", "gradient_only"
        vcd_gradient_weight: float = 0.5,  # Weight for gradient component in hybrid mode
    ):
        super().__init__()
        self.logger = logger
        self.k = int(k)
        self.feature_dim = int(feature_dim)
        self.voxel_size = float(voxel_size)
        
        # VCD configuration
        self.vcd_num_views = int(vcd_num_views)
        self.vcd_error_threshold = float(vcd_error_threshold)
        self.vcd_mode = str(vcd_mode)
        self.vcd_gradient_weight = float(vcd_gradient_weight)
        self.vcd_stats = VCDStats(logger)

        # num_objects
        if num_objects_override is not None:
            self.num_objects = int(num_objects_override)
        else:
            if object_ids is None:
                self.num_objects = 1
            else:
                oid_np = (
                    object_ids.detach().cpu().numpy()
                    if isinstance(object_ids, torch.Tensor)
                    else np.asarray(object_ids)
                )
                self.num_objects = int(oid_np.max()) + 1 if oid_np.size > 0 else 1

        if object_names is None:
            object_names = [f"object_{i}" for i in range(self.num_objects)]
        else:
            if len(object_names) < self.num_objects:
                object_names = list(object_names) + [
                    f"object_{i}" for i in range(len(object_names), self.num_objects)
                ]
        if len(object_names) > 0:
            object_names = list(object_names)
            object_names[0] = "background"
        self.object_names = object_names

        # placeholder mode (for checkpoint reconstruction)
        if point_cloud is None or colors is None:
            anchor_positions = np.zeros((1, 3), dtype=np.float32)
            anchor_colors = np.zeros((1, 3), dtype=np.float32)
            anchor_object_ids = np.zeros((1,), dtype=np.int32)
            if self.logger:
                self.logger.info("Initialized placeholder anchors (checkpoint reconstruction mode).")
        else:
            pc = (
                point_cloud.detach().cpu().numpy()
                if isinstance(point_cloud, torch.Tensor)
                else np.asarray(point_cloud)
            )
            col = (
                colors.detach().cpu().numpy()
                if isinstance(colors, torch.Tensor)
                else np.asarray(colors)
            )

            if object_ids is None:
                object_ids = np.zeros(len(pc), dtype=np.int32)
                if self.logger:
                    self.logger.warning("No object_ids provided; using all zeros (background).")
            oid = (
                object_ids.detach().cpu().numpy()
                if isinstance(object_ids, torch.Tensor)
                else np.asarray(object_ids)
            )
            oid = oid.astype(np.int32)
            if num_objects_override is not None and self.num_objects > 0:
                oid = np.clip(oid, 0, self.num_objects - 1)

            if self.logger:
                self._log_init(pc, oid)

            anchor_positions, anchor_colors, anchor_object_ids = self._voxelize_instance_aware(
                pc, col, oid
            )

        num_anchors = int(anchor_positions.shape[0])

        if self.logger:
            self._log_anchor_info(anchor_object_ids, num_anchors)
            self.logger.info(f"One-hot semantic dimension: {self.num_objects}")
            self.logger.info(f"VCD Configuration: mode={self.vcd_mode}, num_views={self.vcd_num_views}, "
                           f"error_threshold={self.vcd_error_threshold}, gradient_weight={self.vcd_gradient_weight}")
            self.logger.info("=" * 70)

        # Fixed anchor positions (offsets learnable)
        self.anchor_positions = nn.Parameter(
            torch.tensor(anchor_positions, dtype=torch.float32),
            requires_grad=False,
        )

        # Buffers that can change on densify/prune
        self.register_buffer(
            "anchor_colors",
            torch.tensor(anchor_colors, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "anchor_object_ids",
            torch.tensor(anchor_object_ids, dtype=torch.long),
            persistent=True,
        )

        # Learnables
        self.anchor_features = nn.Parameter(self._init_features(anchor_colors, num_anchors))
        self.anchor_scalings = nn.Parameter(torch.ones(num_anchors, dtype=torch.float32))
        self.anchor_offsets = nn.Parameter(
            torch.randn(num_anchors, self.k, 3, dtype=torch.float32) * 0.01
        )

        # Original gradient-based densification stats
        self.register_buffer(
            "anchor_gradient_accum",
            torch.zeros(num_anchors, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "anchor_gradient_count",
            torch.zeros(num_anchors, dtype=torch.int32),
            persistent=True,
        )
        self.register_buffer(
            "anchor_lowgrad_streak",
            torch.zeros(num_anchors, dtype=torch.int32),
            persistent=True,
        )
        
        # NEW: VCD-specific buffers
        self.register_buffer(
            "anchor_vcd_score",
            torch.zeros(num_anchors, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "anchor_vcp_score",
            torch.zeros(num_anchors, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "gaussian_vcd_score",
            torch.zeros(num_anchors * self.k, dtype=torch.float32),
            persistent=True,
        )

        self.attribute_mlp = ViewDependentAttributeMLP(
            feature_dim=self.feature_dim,
            k=self.k,
            view_dim=4,
            color_delta_scale=float(color_delta_scale),
        )

        self._num_anchors = num_anchors
        self._num_gaussians = num_anchors * self.k

    @property
    def num_anchors(self) -> int:
        return int(self._num_anchors)

    @property
    def num_gaussians(self) -> int:
        return int(self._num_gaussians)

    def state_metadata(self) -> Dict:
        return {
            "k": int(self.k),
            "feature_dim": int(self.feature_dim),
            "voxel_size": float(self.voxel_size),
            "num_objects": int(self.num_objects),
            "object_names": list(self.object_names) if self.object_names is not None else [],
            "num_anchors": int(self._num_anchors),
            "num_gaussians": int(self._num_gaussians),
            "vcd_mode": self.vcd_mode,
            "vcd_num_views": self.vcd_num_views,
            "vcd_error_threshold": self.vcd_error_threshold,
        }

    def _log_init(self, point_cloud: np.ndarray, object_ids: np.ndarray):
        self.logger.info("=" * 70)
        self.logger.info("INITIALIZING ObjectGSModelVCD (with FastGS VCD)")
        self.logger.info("=" * 70)
        self.logger.info(f"Input points: {len(point_cloud):,}")
        self.logger.info(f"Voxel size: {self.voxel_size}m ({self.voxel_size*100:.1f}cm)")
        self.logger.info(f"k (Gaussians/anchor): {self.k}")
        self.logger.info(f"Feature dim: {self.feature_dim}")
        self.logger.info(f"Number of instances: {int(object_ids.max()) + 1}")

        unique_ids, counts = np.unique(object_ids, return_counts=True)
        self.logger.info("Instance distribution (top 10):")
        top = np.argsort(-counts)[:10]
        for idx in top:
            oid = int(unique_ids[idx])
            cnt = int(counts[idx])
            name = self.object_names[oid] if oid < len(self.object_names) else f"instance_{oid}"
            self.logger.info(f"  ID {oid}: {name} ({cnt:,} points)")
        if len(unique_ids) > 10:
            self.logger.info(f"  ... and {len(unique_ids) - 10} more instances")

    def _log_anchor_info(self, anchor_object_ids: np.ndarray, num_anchors: int):
        self.logger.info(f"Created {num_anchors:,} anchors → {num_anchors * self.k:,} Gaussians")
        unique, counts = np.unique(anchor_object_ids, return_counts=True)
        self.logger.info("Anchors per instance (top 10):")
        top = np.argsort(-counts)[:10]
        for idx in top:
            oid = int(unique[idx])
            cnt = int(counts[idx])
            name = self.object_names[oid] if oid < len(self.object_names) else f"instance_{oid}"
            self.logger.info(f"  {name}: {cnt:,} anchors ({cnt * self.k:,} Gaussians)")

    def _init_features(self, anchor_colors: np.ndarray, num_anchors: int) -> torch.Tensor:
        feats = torch.randn(num_anchors, self.feature_dim, dtype=torch.float32) * 0.1
        if num_anchors > 0:
            feats[:, :3] = torch.tensor(anchor_colors, dtype=torch.float32)
        return feats

    def _voxelize_instance_aware(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        object_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        object_ids = np.asarray(object_ids, dtype=np.int32)

        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)

        voxel_dict: Dict[Tuple[Tuple[int, int, int], int], Dict[str, List]] = {}
        for i, vidx in enumerate(voxel_indices):
            key = (tuple(vidx.tolist()), int(object_ids[i]))
            if key not in voxel_dict:
                voxel_dict[key] = {"points": [], "colors": []}
            voxel_dict[key]["points"].append(points[i])
            voxel_dict[key]["colors"].append(colors[i])

        anchor_positions = []
        anchor_colors = []
        anchor_object_ids = []

        for (vxyz, instance_id), data in voxel_dict.items():
            pos = np.mean(np.asarray(data["points"]), axis=0)
            col = np.mean(np.asarray(data["colors"]), axis=0)
            anchor_positions.append(pos)
            anchor_colors.append(col)
            anchor_object_ids.append(instance_id)

        if self.logger:
            n_spatial_voxels = len(set(k[0] for k in voxel_dict.keys()))
            n_instance_anchors = len(voxel_dict)
            overlap = n_instance_anchors - n_spatial_voxels
            if overlap > 0:
                self.logger.info(
                    f"Instance-aware voxelization: {n_spatial_voxels:,} spatial voxels → {n_instance_anchors:,} anchors "
                    f"({overlap:,} from overlapping instances)"
                )

        return (
            np.asarray(anchor_positions, dtype=np.float32),
            np.asarray(anchor_colors, dtype=np.float32),
            np.asarray(anchor_object_ids, dtype=np.int32),
        )

    def compute_view_info(self, camera_center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dirs = camera_center.unsqueeze(0) - self.anchor_positions
        dists = dirs.norm(dim=-1, keepdim=True)
        view_dirs = dirs / (dists + 1e-8)
        view_dists = torch.log(dists + 1e-8) / 10.0
        return view_dirs, view_dists

    def get_parameters_as_tensors(
        self,
        camera_center: Optional[torch.Tensor] = None,
        object_mask: Optional[List[int]] = None,
    ) -> Dict:
        if camera_center is not None:
            view_dirs, view_dists = self.compute_view_info(camera_center)
        else:
            view_dirs, view_dists = None, None

        pos = self.anchor_positions.unsqueeze(1) + (
            self.anchor_offsets * self.anchor_scalings.unsqueeze(1).unsqueeze(2)
        )
        pos = pos.reshape(-1, 3)

        opacity_raw, scale_raw, rotation, color_delta = self.attribute_mlp(
            self.anchor_features, view_dirs, view_dists
        )

        opacity_raw = opacity_raw.reshape(-1, 1)
        scale_raw = scale_raw.reshape(-1, 3)
        rotation = rotation.reshape(-1, 4)
        color_delta = color_delta.reshape(-1, 3)

        anchor_cols = self.anchor_colors.unsqueeze(1).expand(-1, self.k, -1).reshape(-1, 3)
        color = torch.clamp(anchor_cols + color_delta, 0.0, 1.0)

        gaussian_object_ids = self.anchor_object_ids.unsqueeze(1).expand(-1, self.k).reshape(-1)
        semantics = F.one_hot(gaussian_object_ids, num_classes=self.num_objects).float()

        out = {
            "pos": pos,
            "opacity_raw": opacity_raw,
            "scale_raw": scale_raw,
            "rotation": rotation,
            "color": color,
            "object_ids": gaussian_object_ids,
            "semantics": semantics,
            "num_gaussians": int(self._num_gaussians),
            "num_anchors": int(self._num_anchors),
            "num_objects": int(self.num_objects),
        }

        if object_mask is not None:
            out = self._apply_object_mask(out, object_mask)
        return out

    def _apply_object_mask(self, params: Dict, object_ids_to_keep: List[int]) -> Dict:
        device = params["object_ids"].device
        N = int(params["num_gaussians"])
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        for oid in object_ids_to_keep:
            mask |= (params["object_ids"] == int(oid))

        return {
            "pos": params["pos"][mask],
            "opacity_raw": params["opacity_raw"][mask],
            "scale_raw": params["scale_raw"][mask],
            "rotation": params["rotation"][mask],
            "color": params["color"][mask],
            "object_ids": params["object_ids"][mask],
            "semantics": params["semantics"][mask],
            "num_gaussians": int(mask.sum().item()),
            "num_anchors": int(self._num_anchors),
            "num_objects": int(self.num_objects),
            "object_mask": mask,
        }

    # =========================================================================
    # GRADIENT-BASED STATS (Original)
    # =========================================================================

    def update_gradient_stats(self, viewspace_gradients: torch.Tensor, visibility_mask: torch.Tensor):
        grad_norms = viewspace_gradients.norm(dim=-1)
        grad_norms = grad_norms.reshape(self._num_anchors, self.k)
        vis = visibility_mask.reshape(self._num_anchors, self.k)

        vis_count = vis.sum(dim=1).clamp(min=1)
        anchor_grads = (grad_norms * vis.float()).sum(dim=1) / vis_count
        anchor_visible = vis.any(dim=1)

        self.anchor_gradient_accum[anchor_visible] += anchor_grads[anchor_visible]
        self.anchor_gradient_count[anchor_visible] += 1

    def reset_gradient_stats(self, reset_streak: bool = False):
        self.anchor_gradient_accum.zero_()
        self.anchor_gradient_count.zero_()
        if reset_streak:
            self.anchor_lowgrad_streak.zero_()

    # =========================================================================
    # VCD: VIEW-CONSISTENT DENSIFICATION (FastGS-inspired)
    # =========================================================================

    @torch.no_grad()
    def compute_gaussian_footprints(
        self,
        means3d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        img_width: int,
        img_height: int,
        n_sigma: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute 2D footprint (bounding box) for each Gaussian.
        
        Returns:
            footprint_min: [N, 2] (x_min, y_min) in pixel coords
            footprint_max: [N, 2] (x_max, y_max) in pixel coords
            valid_mask: [N] bool - True if Gaussian is in front of camera and in view
        """
        N = means3d.shape[0]
        device = means3d.device
        
        # Transform to camera space
        R = viewmat[:3, :3]  # [3, 3]
        t = viewmat[:3, 3]   # [3]
        
        means_cam = means3d @ R.T + t  # [N, 3]
        
        # Check if in front of camera
        valid_mask = means_cam[:, 2] > 0.1  # z > 0.1m
        
        # Project to image plane
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        z = means_cam[:, 2].clamp(min=0.1)
        means2d_x = (means_cam[:, 0] * fx / z) + cx
        means2d_y = (means_cam[:, 1] * fy / z) + cy
        means2d = torch.stack([means2d_x, means2d_y], dim=-1)  # [N, 2]
        
        # Compute projected scale (approximate)
        # Use max scale as radius estimate
        max_scale = scales.max(dim=-1).values  # [N]
        projected_radius = (max_scale * fx / z) * n_sigma  # [N]
        
        # Compute footprint bounds
        footprint_min = means2d - projected_radius.unsqueeze(-1)  # [N, 2]
        footprint_max = means2d + projected_radius.unsqueeze(-1)  # [N, 2]
        
        # Clamp to image bounds
        footprint_min[:, 0].clamp_(min=0, max=img_width - 1)
        footprint_min[:, 1].clamp_(min=0, max=img_height - 1)
        footprint_max[:, 0].clamp_(min=0, max=img_width - 1)
        footprint_max[:, 1].clamp_(min=0, max=img_height - 1)
        
        # Update valid mask: footprint must have non-zero area
        valid_mask = valid_mask & (footprint_max[:, 0] > footprint_min[:, 0]) & (footprint_max[:, 1] > footprint_min[:, 1])
        
        return footprint_min, footprint_max, valid_mask

    @torch.no_grad()
    def compute_vcd_scores_for_view(
        self,
        error_map: torch.Tensor,
        footprint_min: torch.Tensor,
        footprint_max: torch.Tensor,
        valid_mask: torch.Tensor,
        opacities: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute VCD score for each Gaussian based on high-error pixels in its footprint.
        
        FastGS Equation 10:
        s_plus_i = (1/K) * sum_views(count_high_error_pixels_in_footprint)
        
        Args:
            error_map: [H, W] normalized error map (0-1)
            footprint_min: [N, 2] bounding box min
            footprint_max: [N, 2] bounding box max
            valid_mask: [N] bool
            opacities: [N] opacity values (for weighting)
            
        Returns:
            gaussian_scores: [N] VCD score per Gaussian
            stats: Dict with logging info
        """
        device = error_map.device
        H, W = error_map.shape
        N = footprint_min.shape[0]
        
        # Create high-error mask
        high_error_mask = (error_map > self.vcd_error_threshold).float()
        
        # Count high-error pixels in each Gaussian's footprint
        gaussian_scores = torch.zeros(N, device=device)
        
        # For efficiency, we'll use a grid-based approach
        # Convert footprints to integer pixel indices
        fp_min_int = footprint_min.long()
        fp_max_int = footprint_max.long()
        
        # Count high-error pixels in each footprint
        # This is O(N * avg_footprint_area), which can be expensive
        # Optimization: Use cumsum for O(N) approach
        
        # Build integral image of high-error mask
        integral = torch.zeros((H + 1, W + 1), device=device)
        integral[1:, 1:] = torch.cumsum(torch.cumsum(high_error_mask, dim=0), dim=1)
        
        # Compute sum in each rectangle using integral image
        # sum = integral[y2+1, x2+1] - integral[y1, x2+1] - integral[y2+1, x1] + integral[y1, x1]
        x1 = fp_min_int[:, 0].clamp(0, W)
        y1 = fp_min_int[:, 1].clamp(0, H)
        x2 = fp_max_int[:, 0].clamp(0, W - 1)
        y2 = fp_max_int[:, 1].clamp(0, H - 1)
        
        # Handle edge cases
        x2 = torch.maximum(x2, x1)
        y2 = torch.maximum(y2, y1)
        
        high_error_counts = (
            integral[y2 + 1, x2 + 1]
            - integral[y1, x2 + 1]
            - integral[y2 + 1, x1]
            + integral[y1, x1]
        )
        
        # Normalize by footprint area to get density
        footprint_areas = ((x2 - x1 + 1) * (y2 - y1 + 1)).float().clamp(min=1)
        
        # Score = high_error_count (not normalized, following FastGS)
        # But we can also weight by opacity for better discrimination
        gaussian_scores = high_error_counts * valid_mask.float()
        
        # Optional: weight by opacity (Gaussians with low opacity contribute less)
        # gaussian_scores = gaussian_scores * opacities.clamp(min=0.01)
        
        stats = {
            "high_error_pixels": int(high_error_mask.sum().item()),
            "total_pixels": int(H * W),
            "high_error_fraction": float(high_error_mask.mean().item()),
            "gaussians_in_high_error": int((gaussian_scores > 0).sum().item()),
            "mean_footprint_area": float(footprint_areas.mean().item()),
            "valid_gaussians": int(valid_mask.sum().item()),
        }
        
        return gaussian_scores, stats

    @torch.no_grad()
    def compute_vcd_scores(
        self,
        train_cameras: List,
        render_fn,
        num_views: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute VCD and VCP scores across multiple views.
        
        Args:
            train_cameras: List of camera objects
            render_fn: Function(camera, params) -> (rgb, gt_image, info)
            num_views: Number of views to sample (default: self.vcd_num_views)
            
        Returns:
            anchor_vcd_scores: [N_anchors] densification scores
            anchor_vcp_scores: [N_anchors] pruning scores
        """
        if num_views is None:
            num_views = self.vcd_num_views
        
        num_views = min(num_views, len(train_cameras))
        device = self.anchor_positions.device
        
        # Reset VCD stats
        self.vcd_stats.reset()
        
        # Sample random views
        indices = torch.randperm(len(train_cameras))[:num_views].tolist()
        
        # Accumulators
        gaussian_vcd_accum = torch.zeros(self._num_gaussians, device=device)
        gaussian_vcp_accum = torch.zeros(self._num_gaussians, device=device)
        view_count = torch.zeros(self._num_gaussians, device=device)
        
        if self.logger:
            self.logger.info(f"[VCD] Computing scores across {num_views} views...")
        
        for view_idx, cam_idx in enumerate(indices):
            camera = train_cameras[cam_idx]
            
            # Get camera parameters
            gt_image = camera._gt_image_gpu  # [3, H, W]
            viewmat = camera._viewmat_gpu    # [4, 4]
            K = camera._K_gpu                # [3, 3]
            W = int(camera.image_width)
            H = int(camera.image_height)
            
            # Get Gaussian parameters
            params = self.get_parameters_as_tensors(
                camera_center=getattr(camera, "_campos_gpu", None)
            )
            
            # Render
            rgb, gt, info = render_fn(camera, params)
            
            # Compute error map (FastGS Eq. 6-8)
            error_map = (rgb - gt_image).abs().mean(dim=0)  # [H, W]
            e_min, e_max = error_map.min(), error_map.max()
            normalized_error = (error_map - e_min) / (e_max - e_min + 1e-8)
            
            # Compute photometric loss for VCP (FastGS Eq. 11)
            l1_loss = float(F.l1_loss(rgb, gt_image).item())
            
            # Get Gaussian properties
            means3d = params["pos"]
            scales = torch.exp(params["scale_raw"])
            rotations = params["rotation"]
            opacities = torch.sigmoid(params["opacity_raw"]).squeeze(-1)
            
            # Compute footprints
            footprint_min, footprint_max, valid_mask = self.compute_gaussian_footprints(
                means3d, scales, rotations, viewmat, K, W, H
            )
            
            # Compute VCD scores for this view
            gaussian_scores, view_stats = self.compute_vcd_scores_for_view(
                normalized_error, footprint_min, footprint_max, valid_mask, opacities
            )
            
            view_stats["l1_loss"] = l1_loss
            view_stats["cam_idx"] = cam_idx
            
            # Accumulate
            gaussian_vcd_accum += gaussian_scores
            gaussian_vcp_accum += gaussian_scores * l1_loss  # VCP: weight by photometric loss
            view_count += valid_mask.float()
            
            # Log
            self.vcd_stats.log_view_stats(view_idx, view_stats)
        
        # Average across views (FastGS Eq. 10: divide by K)
        view_count = view_count.clamp(min=1)
        gaussian_vcd_scores = gaussian_vcd_accum / num_views
        gaussian_vcp_scores = gaussian_vcp_accum / num_views
        
        # Normalize VCP scores (FastGS Eq. 12)
        vcp_min, vcp_max = gaussian_vcp_scores.min(), gaussian_vcp_scores.max()
        gaussian_vcp_scores = (gaussian_vcp_scores - vcp_min) / (vcp_max - vcp_min + 1e-8)
        
        # Aggregate to anchor level (max across k Gaussians per anchor)
        anchor_vcd_scores = gaussian_vcd_scores.view(self._num_anchors, self.k).max(dim=1).values
        anchor_vcp_scores = gaussian_vcp_scores.view(self._num_anchors, self.k).max(dim=1).values
        
        # Store in buffers
        self.gaussian_vcd_score.copy_(gaussian_vcd_scores)
        self.anchor_vcd_score.copy_(anchor_vcd_scores)
        self.anchor_vcp_score.copy_(anchor_vcp_scores)
        
        return anchor_vcd_scores, anchor_vcp_scores

    # =========================================================================
    # DENSIFY AND PRUNE (Enhanced with VCD)
    # =========================================================================

    @torch.no_grad()
    def densify_and_prune(
        self,
        iteration: int,
        grad_threshold: float = 5e-5,
        min_opacity: float = 0.01,
        prune_warmup_iters: int = 0,
        prune_grad_factor: float = 0.25,
        min_cycles: int = 5,
        # VCD-specific thresholds
        vcd_densify_threshold: float = 10.0,
        vcd_prune_threshold: float = 0.1,
        # Optional: train_cameras and render_fn for VCD computation
        train_cameras: Optional[List] = None,
        render_fn=None,
    ) -> Dict:
        """
        Densify and prune anchors using hybrid gradient + VCD approach.
        
        VCD modes:
        - "gradient_only": Original behavior
        - "pure_vcd": Only use VCD scores
        - "hybrid": Combine gradient and VCD scores
        """
        stats: Dict = {
            "anchors_before": int(self._num_anchors),
            "pruned": 0,
            "grown": 0,
            "mode": self.vcd_mode,
        }
        
        if self.logger:
            self.logger.info(f"[DENSIFY/PRUNE] iter={iteration} mode={self.vcd_mode}")
        
        # Compute VCD scores if needed and data available
        vcd_available = False
        if self.vcd_mode in ("pure_vcd", "hybrid") and train_cameras is not None and render_fn is not None:
            anchor_vcd_scores, anchor_vcp_scores = self.compute_vcd_scores(
                train_cameras, render_fn
            )
            vcd_available = True
            stats["vcd_computed"] = True
        else:
            anchor_vcd_scores = self.anchor_vcd_score
            anchor_vcp_scores = self.anchor_vcp_score
            stats["vcd_computed"] = False
            if self.vcd_mode in ("pure_vcd", "hybrid"):
                if self.logger:
                    self.logger.warning("[VCD] Cannot compute VCD scores: train_cameras or render_fn not provided")

        # Compute opacity
        opacity_raw, _, _, _ = self.attribute_mlp(self.anchor_features, None, None)
        anchor_op = torch.sigmoid(opacity_raw).mean(dim=1)

        # Gradient statistics
        visible_this_cycle = self.anchor_gradient_count > 0
        counts = self.anchor_gradient_count.clamp(min=1).float()
        avg_grads = self.anchor_gradient_accum / counts

        # =====================================================================
        # GROWTH DECISION
        # =====================================================================
        
        if self.vcd_mode == "gradient_only":
            grow_mask = avg_grads > float(grad_threshold)
            stats["grow_method"] = "gradient_only"
            
        elif self.vcd_mode == "pure_vcd":
            grow_mask = anchor_vcd_scores > float(vcd_densify_threshold)
            stats["grow_method"] = "pure_vcd"
            
        else:  # hybrid
            # Normalize gradient scores to [0, 1]
            grad_min, grad_max = avg_grads.min(), avg_grads.max()
            grad_normalized = (avg_grads - grad_min) / (grad_max - grad_min + 1e-8)
            
            # Normalize VCD scores to [0, 1]
            vcd_min, vcd_max = anchor_vcd_scores.min(), anchor_vcd_scores.max()
            vcd_normalized = (anchor_vcd_scores - vcd_min) / (vcd_max - vcd_min + 1e-8)
            
            # Combine
            w = self.vcd_gradient_weight
            combined_score = w * grad_normalized + (1 - w) * vcd_normalized
            
            # Threshold on combined score
            grow_mask = combined_score > 0.5  # Top 50% of combined score
            
            # Also require gradient above minimum threshold
            grow_mask = grow_mask & (avg_grads > float(grad_threshold) * 0.5)
            
            stats["grow_method"] = "hybrid"
            stats["hybrid_weight"] = w

        # =====================================================================
        # PRUNING DECISION
        # =====================================================================
        
        # Update low-gradient streak
        low_grad_now = visible_this_cycle & (
            avg_grads < (float(grad_threshold) * float(prune_grad_factor))
        )
        inc = self.anchor_lowgrad_streak + 1
        zero = torch.zeros_like(self.anchor_lowgrad_streak)
        self.anchor_lowgrad_streak = torch.where(
            low_grad_now,
            inc,
            torch.where(visible_this_cycle, zero, self.anchor_lowgrad_streak),
        )

        # Base pruning: low opacity + sustained low gradient
        low_opacity = anchor_op < float(min_opacity)
        base_prune = (self.anchor_lowgrad_streak >= int(min_cycles)) & low_opacity
        
        if self.vcd_mode == "gradient_only":
            prune_mask = base_prune
            stats["prune_method"] = "gradient_only"
            
        elif self.vcd_mode == "pure_vcd":
            # VCP: prune if high VCP score (contributes to quality degradation) AND low opacity
            prune_mask = (anchor_vcp_scores > float(vcd_prune_threshold)) & low_opacity
            stats["prune_method"] = "pure_vcd"
            
        else:  # hybrid
            # Combine: prune if (base_prune OR high_vcp) AND low_opacity
            vcp_prune = anchor_vcp_scores > float(vcd_prune_threshold)
            prune_mask = (base_prune | vcp_prune) & low_opacity
            stats["prune_method"] = "hybrid"

        # Warmup: disable pruning early
        if int(iteration) < int(prune_warmup_iters):
            prune_mask = torch.zeros_like(prune_mask, dtype=torch.bool)
            stats["prune_warmup_active"] = True
        else:
            stats["prune_warmup_active"] = False

        # Never grow anchors we are pruning
        grow_mask = grow_mask & (~prune_mask)

        # =====================================================================
        # LOGGING
        # =====================================================================
        
        if self.logger:
            # Log detailed statistics
            self.logger.info(f"  [GROWTH] Candidates: {grow_mask.sum().item():,} anchors")
            if vcd_available:
                self.vcd_stats.log_summary(
                    anchor_vcd_scores, self.gaussian_vcd_score,
                    vcd_densify_threshold, vcd_prune_threshold
                )
            
            # Gradient stats
            g_min, g_max = float(avg_grads.min()), float(avg_grads.max())
            g_mean, g_median = float(avg_grads.mean()), float(avg_grads.median())
            self.logger.info(f"  [GRADIENT] min={g_min:.2e}, max={g_max:.2e}, mean={g_mean:.2e}, median={g_median:.2e}")
            self.logger.info(f"  [GRADIENT] Above threshold ({grad_threshold}): {(avg_grads > grad_threshold).sum().item():,}")
            
            # Opacity stats
            op_min, op_max = float(anchor_op.min()), float(anchor_op.max())
            op_mean = float(anchor_op.mean())
            self.logger.info(f"  [OPACITY] min={op_min:.4f}, max={op_max:.4f}, mean={op_mean:.4f}")
            self.logger.info(f"  [OPACITY] Below min ({min_opacity}): {(anchor_op < min_opacity).sum().item():,}")
            
            # Streak stats
            streak_max = int(self.anchor_lowgrad_streak.max().item())
            streak_mean = float(self.anchor_lowgrad_streak.float().mean().item())
            above_min_cycles = int((self.anchor_lowgrad_streak >= min_cycles).sum().item())
            self.logger.info(f"  [STREAK] max={streak_max}, mean={streak_mean:.1f}, above_min_cycles({min_cycles}): {above_min_cycles:,}")
            
            # Final decisions
            self.logger.info(f"  [DECISION] grow={grow_mask.sum().item():,}, prune={prune_mask.sum().item():,}")

        # =====================================================================
        # APPLY DENSIFICATION
        # =====================================================================
        
        stats["pruned"] = int(prune_mask.sum().item())
        stats["grown"] = int(grow_mask.sum().item())

        if stats["pruned"] > 0 or stats["grown"] > 0:
            mapping = self._apply_densification(prune_mask, grow_mask)
            stats.update(mapping)

        # Reset cycle accumulators
        self.anchor_gradient_accum.zero_()
        self.anchor_gradient_count.zero_()

        stats["anchors_after"] = int(self._num_anchors)
        
        if self.logger:
            delta = stats["anchors_after"] - stats["anchors_before"]
            self.logger.info(f"  [RESULT] {stats['anchors_before']:,} → {stats['anchors_after']:,} ({delta:+,}) anchors")
        
        return stats

    @torch.no_grad()
    def _apply_densification(self, prune_mask: torch.Tensor, grow_mask: torch.Tensor) -> Dict:
        device = self.anchor_positions.device
        keep_mask = ~prune_mask

        keep_indices_old = torch.where(keep_mask)[0]
        grow_indices_old = torch.where(grow_mask & keep_mask)[0]

        n_keep = int(keep_indices_old.numel())
        n_grow = int(grow_indices_old.numel())
        new_n = n_keep + n_grow
        if new_n <= 0:
            return {"keep_indices_old": keep_indices_old, "grow_parent_old": grow_indices_old}

        # Kept anchors
        new_pos = self.anchor_positions.data[keep_mask]
        new_col = self.anchor_colors[keep_mask]
        new_oid = self.anchor_object_ids[keep_mask]
        new_feat = self.anchor_features.data[keep_mask]
        new_scl = self.anchor_scalings.data[keep_mask]
        new_off = self.anchor_offsets.data[keep_mask]

        new_grad_accum = self.anchor_gradient_accum[keep_mask]
        new_grad_count = self.anchor_gradient_count[keep_mask]
        new_streak = self.anchor_lowgrad_streak[keep_mask].to(dtype=torch.int32)
        
        # VCD buffers
        new_vcd_score = self.anchor_vcd_score[keep_mask]
        new_vcp_score = self.anchor_vcp_score[keep_mask]

        # Grown children
        if n_grow > 0:
            mapping = {int(keep_indices_old[i]): i for i in range(n_keep)}
            grow_in_kept = torch.tensor(
                [mapping[int(g)] for g in grow_indices_old if int(g) in mapping],
                device=device,
                dtype=torch.long,
            )

            if grow_in_kept.numel() > 0:
                child_scale = 0.6

                gpos = new_pos[grow_in_kept] + torch.randn_like(new_pos[grow_in_kept]) * self.voxel_size * 0.1
                gcol = new_col[grow_in_kept]
                goid = new_oid[grow_in_kept]
                gfeat = new_feat[grow_in_kept] + torch.randn_like(new_feat[grow_in_kept]) * 0.01
                gscl = new_scl[grow_in_kept] * child_scale
                goff = new_off[grow_in_kept] + torch.randn_like(new_off[grow_in_kept]) * 0.0005

                new_pos = torch.cat([new_pos, gpos], dim=0)
                new_col = torch.cat([new_col, gcol], dim=0)
                new_oid = torch.cat([new_oid, goid], dim=0)
                new_feat = torch.cat([new_feat, gfeat], dim=0)
                new_scl = torch.cat([new_scl, gscl], dim=0)
                new_off = torch.cat([new_off, goff], dim=0)

                new_grad_accum = torch.cat(
                    [new_grad_accum, torch.zeros(len(grow_in_kept), device=device, dtype=new_grad_accum.dtype)],
                    dim=0,
                )
                new_grad_count = torch.cat(
                    [new_grad_count, torch.zeros(len(grow_in_kept), device=device, dtype=torch.int32)],
                    dim=0,
                )
                new_streak = torch.cat(
                    [new_streak, torch.zeros(len(grow_in_kept), device=device, dtype=torch.int32)],
                    dim=0,
                )
                new_vcd_score = torch.cat(
                    [new_vcd_score, torch.zeros(len(grow_in_kept), device=device, dtype=new_vcd_score.dtype)],
                    dim=0,
                )
                new_vcp_score = torch.cat(
                    [new_vcp_score, torch.zeros(len(grow_in_kept), device=device, dtype=new_vcp_score.dtype)],
                    dim=0,
                )

        # Reassign tensors
        self.anchor_positions = nn.Parameter(new_pos, requires_grad=False)
        self.anchor_features = nn.Parameter(new_feat)
        self.anchor_scalings = nn.Parameter(new_scl)
        self.anchor_offsets = nn.Parameter(new_off)

        self._buffers["anchor_colors"] = new_col
        self._buffers["anchor_object_ids"] = new_oid
        self._buffers["anchor_gradient_accum"] = new_grad_accum
        self._buffers["anchor_gradient_count"] = new_grad_count
        self._buffers["anchor_lowgrad_streak"] = new_streak
        self._buffers["anchor_vcd_score"] = new_vcd_score
        self._buffers["anchor_vcp_score"] = new_vcp_score
        
        # Update Gaussian-level VCD buffer
        new_gaussian_vcd = torch.zeros(new_n * self.k, device=device, dtype=torch.float32)
        self._buffers["gaussian_vcd_score"] = new_gaussian_vcd

        self._num_anchors = int(new_n)
        self._num_gaussians = int(new_n * self.k)

        return {
            "keep_indices_old": keep_indices_old.detach().cpu(),
            "grow_parent_old": grow_indices_old.detach().cpu(),
        }

    # =========================================================================
    # CHECKPOINT COMPATIBILITY
    # =========================================================================

    @staticmethod
    def _infer_num_anchors_from_state_dict(state_dict: Dict) -> Optional[int]:
        for key in ("anchor_positions", "anchor_features", "anchor_offsets", "anchor_colors", "anchor_object_ids"):
            t = state_dict.get(key, None)
            if t is not None and hasattr(t, "shape") and len(t.shape) > 0:
                return int(t.shape[0])
        return None

    def _buffer_assign(self, name: str, tensor: torch.Tensor):
        if name in self._buffers:
            self._buffers[name] = tensor
        else:
            self.register_buffer(name, tensor, persistent=True)

    def rebuild_anchors_from_state_dict(self, state_dict: Dict) -> bool:
        N = self._infer_num_anchors_from_state_dict(state_dict)
        if N is None:
            return False

        curN = int(self.anchor_positions.shape[0]) if hasattr(self, "anchor_positions") else None
        if curN == N:
            return False

        device = next(self.parameters(), torch.empty(0)).device

        def dt(name, default):
            t = state_dict.get(name, None)
            return t.dtype if t is not None else default

        ap = state_dict.get("anchor_positions", None)
        ap_shape = tuple(ap.shape) if ap is not None else (N, 3)
        self.anchor_positions = nn.Parameter(
            torch.empty(ap_shape, device=device, dtype=dt("anchor_positions", torch.float32)),
            requires_grad=False,
        )

        af = state_dict.get("anchor_features", None)
        af_shape = tuple(af.shape) if af is not None else (N, self.feature_dim)
        self.anchor_features = nn.Parameter(
            torch.empty(af_shape, device=device, dtype=dt("anchor_features", torch.float32)),
            requires_grad=True,
        )

        asc = state_dict.get("anchor_scalings", None)
        asc_shape = tuple(asc.shape) if asc is not None else (N,)
        self.anchor_scalings = nn.Parameter(
            torch.empty(asc_shape, device=device, dtype=dt("anchor_scalings", torch.float32)),
            requires_grad=True,
        )

        ao = state_dict.get("anchor_offsets", None)
        ao_shape = tuple(ao.shape) if ao is not None else (N, self.k, 3)
        self.anchor_offsets = nn.Parameter(
            torch.empty(ao_shape, device=device, dtype=dt("anchor_offsets", torch.float32)),
            requires_grad=True,
        )

        ac = state_dict.get("anchor_colors", None)
        ac_shape = tuple(ac.shape) if ac is not None else (N, 3)
        self._buffer_assign(
            "anchor_colors",
            torch.empty(ac_shape, device=device, dtype=dt("anchor_colors", torch.float32)),
        )

        oid = state_dict.get("anchor_object_ids", None)
        oid_shape = tuple(oid.shape) if oid is not None else (N,)
        self._buffer_assign(
            "anchor_object_ids",
            torch.empty(oid_shape, device=device, dtype=dt("anchor_object_ids", torch.long)),
        )

        # Original stats buffers
        for buf_name in ("anchor_gradient_accum", "anchor_gradient_count", "anchor_lowgrad_streak"):
            buf = state_dict.get(buf_name, None)
            buf_shape = tuple(buf.shape) if buf is not None else (N,)
            buf_dtype = dt(buf_name, torch.float32 if "accum" in buf_name else torch.int32)
            self._buffer_assign(buf_name, torch.zeros(buf_shape, device=device, dtype=buf_dtype))
        
        # VCD buffers
        for buf_name in ("anchor_vcd_score", "anchor_vcp_score"):
            buf = state_dict.get(buf_name, None)
            buf_shape = tuple(buf.shape) if buf is not None else (N,)
            self._buffer_assign(buf_name, torch.zeros(buf_shape, device=device, dtype=torch.float32))
        
        # Gaussian-level VCD buffer
        gaussian_vcd = state_dict.get("gaussian_vcd_score", None)
        gaussian_vcd_shape = tuple(gaussian_vcd.shape) if gaussian_vcd is not None else (N * self.k,)
        self._buffer_assign("gaussian_vcd_score", torch.zeros(gaussian_vcd_shape, device=device, dtype=torch.float32))

        self._num_anchors = int(N)
        self._num_gaussians = int(N * self.k)
        return True

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError as e:
            msg = str(e)
            if "size mismatch for anchor" in msg:
                if self.rebuild_anchors_from_state_dict(state_dict):
                    return super().load_state_dict(state_dict, strict=strict, assign=assign)
            raise

    @classmethod
    def from_checkpoint(
        cls,
        ckpt: Union[str, Path, Dict],
        device: Union[str, torch.device] = "cuda",
        map_location=None,
    ):
        if isinstance(ckpt, (str, Path)):
            ckpt_dict = torch.load(str(ckpt), map_location=map_location)
        else:
            ckpt_dict = ckpt

        sd = ckpt_dict.get("model_state_dict", ckpt_dict)

        meta = ckpt_dict.get("model_meta", None)
        if meta is None:
            meta = {
                "k": ckpt_dict.get("k", 10),
                "feature_dim": ckpt_dict.get("feature_dim", 32),
                "voxel_size": ckpt_dict.get("voxel_size", 0.01),
                "num_objects": ckpt_dict.get("num_objects", 1),
                "object_names": ckpt_dict.get("object_names", None),
            }

        model = cls(
            point_cloud=None,
            colors=None,
            object_ids=None,
            voxel_size=float(meta.get("voxel_size", 0.01)),
            k=int(meta.get("k", 10)),
            feature_dim=int(meta.get("feature_dim", 32)),
            object_names=meta.get("object_names", None),
            logger=None,
            num_objects_override=int(meta.get("num_objects", 1)),
            vcd_mode=meta.get("vcd_mode", "hybrid"),
            vcd_num_views=int(meta.get("vcd_num_views", 10)),
            vcd_error_threshold=float(meta.get("vcd_error_threshold", 0.5)),
        ).to(device)

        model.rebuild_anchors_from_state_dict(sd)
        model.load_state_dict(sd, strict=True)

        model.num_objects = int(meta.get("num_objects", model.num_objects))
        obj_names = meta.get("object_names", None)
        if obj_names is not None:
            obj_names = list(obj_names)
            if len(obj_names) < model.num_objects:
                obj_names += [f"object_{i}" for i in range(len(obj_names), model.num_objects)]
            if len(obj_names) > 0:
                obj_names[0] = "background"
            model.object_names = obj_names

        return model, ckpt_dict