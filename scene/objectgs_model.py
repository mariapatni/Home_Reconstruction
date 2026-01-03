# =========================
# file: scene/objectgs_model.py
# =========================
"""
ObjectGS: Anchor-based Gaussian Splatting with Object Awareness (FIXED)

Key fixes:
- Densify/prune keeps buffers registered as buffers (no accidental de-registration).
- load_state_dict() supports variable anchor counts by rebuilding anchor tensors on mismatch.
- from_checkpoint() reconstructs a FRESH model for exporting best ckpt (avoids topology mismatch).
- state_metadata() is saved into checkpoint to reconstruct reliably.

Trainer expects model.get_parameters_as_tensors() returns:
  pos, opacity_raw, scale_raw, rotation, color, object_ids, semantics
semantics is ONE-HOT [N_gauss, num_objects]

NEW in this revision:
- Proper "min_cycles" semantics: low-gradient streak counted in *densify cycles*, not iterations.
- Removes any accidental double-resets by making resets explicit: accum/count reset per cycle, streak persists.
- Remaps anchor_lowgrad_streak correctly across prune/grow.
- Adds checkpoint rebuild support for anchor_lowgrad_streak.
- Adds optional hard reset of streak via reset_gradient_stats(reset_streak=True).
- Streak update only considers anchors visible at least once in the last cycle (optional safety behavior).
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

    Opacity/scale/rotation: view-independent
    Color: view-dependent delta in [-delta_scale, +delta_scale]
    """

    def __init__(
        self,
        feature_dim: int = 32,
        k: int = 10,
        view_dim: int = 4,  # 1 dist + 3 dir
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
            # scale small => exp(scale_raw) near tiny
            scale_out = self.scale_mlp[-1]
            scale_out.weight.mul_(0.1)
            scale_out.bias.fill_(-4.0)

            # low opacity initially
            op_out = self.opacity_mlp[-1]
            op_out.weight.mul_(0.1)
            op_out.bias.fill_(-2.0)

            # identity quaternion bias
            rot_out = self.rotation_mlp[-1]
            rot_out.weight.mul_(0.1)
            rot_out.bias.zero_()
            rot_out.bias[3::4] = 1.0

            # color delta small
            col_out = self.color_mlp[-1]
            col_out.weight.mul_(0.1)
            col_out.bias.zero_()

    def forward(
        self,
        anchor_features: torch.Tensor,  # [N, F]
        view_dirs: Optional[torch.Tensor] = None,   # [N,3]
        view_dists: Optional[torch.Tensor] = None,  # [N,1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = anchor_features.shape[0]

        opacity_raw = self.opacity_mlp(anchor_features)  # [N,k]
        scale_raw = self.scale_mlp(anchor_features).view(N, self.k, 3)  # [N,k,3]

        rotation = self.rotation_mlp(anchor_features).view(N, self.k, 4)  # [N,k,4]
        rotation = rotation / (rotation.norm(dim=-1, keepdim=True) + 1e-9)

        if view_dirs is not None and view_dists is not None:
            view_info = torch.cat([view_dists, view_dirs], dim=-1)  # [N,4]
        else:
            view_info = torch.zeros(
                (N, self.view_dim),
                device=anchor_features.device,
                dtype=anchor_features.dtype,
            )

        col_in = torch.cat([anchor_features, view_info], dim=-1)
        color_delta = self.color_mlp(col_in).view(N, self.k, 3)
        color_delta = torch.tanh(color_delta) * self.color_delta_scale  # true delta

        return opacity_raw, scale_raw, rotation, color_delta


class ObjectGSModel(nn.Module):
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
    ):
        super().__init__()
        self.logger = logger
        self.k = int(k)
        self.feature_dim = int(feature_dim)
        self.voxel_size = float(voxel_size)

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
            self.logger.info("=" * 70)

        # Fixed anchor positions (offsets learnable)
        self.anchor_positions = nn.Parameter(
            torch.tensor(anchor_positions, dtype=torch.float32),
            requires_grad=False,
        )

        # Buffers that can change on densify/prune MUST remain buffers
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

        # Densification stats buffers
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
        # NEW: per-anchor low-gradient streak counted in *densify cycles*
        self.register_buffer(
            "anchor_lowgrad_streak",
            torch.zeros(num_anchors, dtype=torch.int32),
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

    # -------------------------------------------------------------------------
    # Metadata for checkpoint reconstruction
    # -------------------------------------------------------------------------

    def state_metadata(self) -> Dict:
        return {
            "k": int(self.k),
            "feature_dim": int(self.feature_dim),
            "voxel_size": float(self.voxel_size),
            "num_objects": int(self.num_objects),
            "object_names": list(self.object_names) if self.object_names is not None else [],
            "num_anchors": int(self._num_anchors),
            "num_gaussians": int(self._num_gaussians),
        }

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------

    def _log_init(self, point_cloud: np.ndarray, object_ids: np.ndarray):
        self.logger.info("=" * 70)
        self.logger.info("INITIALIZING ObjectGSModel (Instance Segmentation)")
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

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

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
        """
        Instance-aware voxelization: key = (voxel_xyz, instance_id).
        """
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

    # -------------------------------------------------------------------------
    # View info
    # -------------------------------------------------------------------------

    def compute_view_info(self, camera_center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        camera_center: [3]
        returns:
          view_dirs: [N,3]
          view_dists: [N,1] log-normalized
        """
        dirs = camera_center.unsqueeze(0) - self.anchor_positions  # [N,3]
        dists = dirs.norm(dim=-1, keepdim=True)                    # [N,1]
        view_dirs = dirs / (dists + 1e-8)
        view_dists = torch.log(dists + 1e-8) / 10.0
        return view_dirs, view_dists

    # -------------------------------------------------------------------------
    # Parameters for renderer
    # -------------------------------------------------------------------------

    def get_parameters_as_tensors(
        self,
        camera_center: Optional[torch.Tensor] = None,
        object_mask: Optional[List[int]] = None,
    ) -> Dict:
        """
        Returns dict:
          pos: [N_g,3]
          opacity_raw: [N_g,1]
          scale_raw: [N_g,3]
          rotation: [N_g,4]
          color: [N_g,3]
          object_ids: [N_g]
          semantics: [N_g,num_objects] one-hot
        """
        if camera_center is not None:
            view_dirs, view_dists = self.compute_view_info(camera_center)
        else:
            view_dirs, view_dists = None, None

        # positions [N_anchor,k,3] -> [N_g,3]
        pos = self.anchor_positions.unsqueeze(1) + (
            self.anchor_offsets * self.anchor_scalings.unsqueeze(1).unsqueeze(2)
        )
        pos = pos.reshape(-1, 3)

        # MLP outputs at anchor-level
        opacity_raw, scale_raw, rotation, color_delta = self.attribute_mlp(
            self.anchor_features, view_dirs, view_dists
        )

        # flatten to gaussian-level
        opacity_raw = opacity_raw.reshape(-1, 1)
        scale_raw = scale_raw.reshape(-1, 3)
        rotation = rotation.reshape(-1, 4)
        color_delta = color_delta.reshape(-1, 3)

        # anchor color baseline + delta
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

    # -------------------------------------------------------------------------
    # Densification / pruning stats
    # -------------------------------------------------------------------------

    def update_gradient_stats(self, viewspace_gradients: torch.Tensor, visibility_mask: torch.Tensor):
        """
        viewspace_gradients: [N_g,3]
        visibility_mask: [N_g] bool
        """
        grad_norms = viewspace_gradients.norm(dim=-1)  # [N_g]
        grad_norms = grad_norms.reshape(self._num_anchors, self.k)
        vis = visibility_mask.reshape(self._num_anchors, self.k)

        vis_count = vis.sum(dim=1).clamp(min=1)
        anchor_grads = (grad_norms * vis.float()).sum(dim=1) / vis_count
        anchor_visible = vis.any(dim=1)

        self.anchor_gradient_accum[anchor_visible] += anchor_grads[anchor_visible]
        self.anchor_gradient_count[anchor_visible] += 1

    def reset_gradient_stats(self, reset_streak: bool = False):
        """
        Reset cycle accumulators. By default we do NOT reset streak because streak is meant
        to accumulate across densify cycles. Use reset_streak=True for a full hard reset.
        """
        self.anchor_gradient_accum.zero_()
        self.anchor_gradient_count.zero_()
        if reset_streak:
            self.anchor_lowgrad_streak.zero_()

    @torch.no_grad()
    def densify_and_prune(
        self,
        iteration: int,
        grad_threshold: float = 5e-5,
        min_opacity: float = 0.01,
        prune_warmup_iters: int = 0,
        prune_grad_factor: float = 0.25,
        min_cycles: int = 5,  # number of *densify cycles* of sustained low-grad
    ) -> Dict:
        """
        Densify and prune anchors based on gradient statistics and opacity.

        Behavior:
        - Growth allowed at all times
        - Pruning disabled until iteration >= prune_warmup_iters
        - Pruning only if:
            (low opacity) AND (low gradient sustained for min_cycles densify cycles)

        Streak update behavior:
        - Only anchors visible at least once in the last cycle contribute to (reset/increment) streak.
          Anchors never visible keep their current streak (no change).
        """
        stats: Dict = {
            "anchors_before": int(self._num_anchors),
            "pruned": 0,
            "grown": 0,
        }

        # ------------------------------------------------------------
        # Compute opacity
        # ------------------------------------------------------------
        opacity_raw, _, _, _ = self.attribute_mlp(self.anchor_features, None, None)
        anchor_op = torch.sigmoid(opacity_raw).mean(dim=1)  # [N_anchor]

        # ------------------------------------------------------------
        # Gradient statistics for *last densify cycle*
        # ------------------------------------------------------------
        visible_this_cycle = self.anchor_gradient_count > 0
        counts = self.anchor_gradient_count.clamp(min=1).float()
        avg_grads = self.anchor_gradient_accum / counts

        # ------------------------------------------------------------
        # Growth rule (unchanged)
        # ------------------------------------------------------------
        grow_mask = avg_grads > float(grad_threshold)

        # ------------------------------------------------------------
        # Update low-gradient streak ONCE per densify cycle
        # Only update streak for anchors visible in this cycle.
        # ------------------------------------------------------------
        low_grad_now = visible_this_cycle & (
            avg_grads < (float(grad_threshold) * float(prune_grad_factor))
        )

        # If visible and low-grad -> streak+1
        # If visible and NOT low-grad -> streak=0
        # If NOT visible -> streak unchanged
        inc = self.anchor_lowgrad_streak + 1
        zero = torch.zeros_like(self.anchor_lowgrad_streak)
        self.anchor_lowgrad_streak = torch.where(
            low_grad_now,
            inc,
            torch.where(visible_this_cycle, zero, self.anchor_lowgrad_streak),
        )

        # ------------------------------------------------------------
        # Pruning rule: sustained low-grad for min_cycles + low opacity
        # ------------------------------------------------------------
        low_opacity = anchor_op < float(min_opacity)
        prune_mask = (self.anchor_lowgrad_streak >= int(min_cycles)) & low_opacity

        # Warmup override: disable pruning early (streak may still accumulate)
        if int(iteration) < int(prune_warmup_iters):
            prune_mask = torch.zeros_like(prune_mask, dtype=torch.bool)

        # Never grow anchors we are pruning
        grow_mask = grow_mask & (~prune_mask)

        # ------------------------------------------------------------
        # Apply densification
        # ------------------------------------------------------------
        stats["pruned"] = int(prune_mask.sum().item())
        stats["grown"] = int(grow_mask.sum().item())

        if stats["pruned"] > 0 or stats["grown"] > 0:
            mapping = self._apply_densification(prune_mask, grow_mask)
            stats.update(mapping)

        # IMPORTANT: reset cycle accumulators (but NOT the streak)
        self.anchor_gradient_accum.zero_()
        self.anchor_gradient_count.zero_()

        stats["anchors_after"] = int(self._num_anchors)
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

        # --------------------------------------------------
        # Kept anchors
        # --------------------------------------------------
        new_pos = self.anchor_positions.data[keep_mask]
        new_col = self.anchor_colors[keep_mask]
        new_oid = self.anchor_object_ids[keep_mask]
        new_feat = self.anchor_features.data[keep_mask]
        new_scl = self.anchor_scalings.data[keep_mask]
        new_off = self.anchor_offsets.data[keep_mask]

        # Preserve gradient history for kept anchors
        new_grad_accum = self.anchor_gradient_accum[keep_mask]
        new_grad_count = self.anchor_gradient_count[keep_mask]
        # Preserve streak for kept anchors (force int32 for safety)
        new_streak = self.anchor_lowgrad_streak[keep_mask].to(dtype=torch.int32)

        # --------------------------------------------------
        # Grown children
        # --------------------------------------------------
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

                # children start with ZERO history / ZERO streak
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

        # --------------------------------------------------
        # Reassign tensors
        # --------------------------------------------------
        self.anchor_positions = nn.Parameter(new_pos, requires_grad=False)
        self.anchor_features = nn.Parameter(new_feat)
        self.anchor_scalings = nn.Parameter(new_scl)
        self.anchor_offsets = nn.Parameter(new_off)

        self._buffers["anchor_colors"] = new_col
        self._buffers["anchor_object_ids"] = new_oid
        self._buffers["anchor_gradient_accum"] = new_grad_accum
        self._buffers["anchor_gradient_count"] = new_grad_count
        self._buffers["anchor_lowgrad_streak"] = new_streak

        self._num_anchors = int(new_n)
        self._num_gaussians = int(new_n * self.k)

        return {
            "keep_indices_old": keep_indices_old.detach().cpu(),
            "grow_parent_old": grow_indices_old.detach().cpu(),
        }

    # ---------------------------------------------------------------------
    # Checkpoint compatibility (variable anchor counts)
    # ---------------------------------------------------------------------

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
        """
        Rebuild anchor-dependent Parameters/Buffers to match checkpoint sizes.
        """
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

        aga = state_dict.get("anchor_gradient_accum", None)
        aga_shape = tuple(aga.shape) if aga is not None else (N,)
        self._buffer_assign(
            "anchor_gradient_accum",
            torch.zeros(aga_shape, device=device, dtype=dt("anchor_gradient_accum", torch.float32)),
        )

        agc = state_dict.get("anchor_gradient_count", None)
        agc_shape = tuple(agc.shape) if agc is not None else (N,)
        self._buffer_assign(
            "anchor_gradient_count",
            torch.zeros(agc_shape, device=device, dtype=dt("anchor_gradient_count", torch.int32)),
        )

        als = state_dict.get("anchor_lowgrad_streak", None)
        als_shape = tuple(als.shape) if als is not None else (N,)
        self._buffer_assign(
            "anchor_lowgrad_streak",
            torch.zeros(als_shape, device=device, dtype=dt("anchor_lowgrad_streak", torch.int32)),
        )

        self._num_anchors = int(N)
        self._num_gaussians = int(N * self.k)
        return True

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """
        Auto-rebuild anchor tensors if checkpoint anchor count differs.
        """
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
        """
        Build a FRESH model instance whose anchor tensors match the checkpoint.
        Returns (model, ckpt_dict).
        """
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
        ).to(device)

        model.rebuild_anchors_from_state_dict(sd)
        model.load_state_dict(sd, strict=True)

        # restore names/count
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
