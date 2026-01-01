"""
ObjectGS: Anchor-based Gaussian Splatting with Object Awareness

INSTANCE SEGMENTATION VERSION with:
- Instance-aware voxelization (no majority voting that merges instances)
- View-dependent attribute generation (direction + distance)
- Dynamic anchor growing and pruning
- Full instance segmentation support with one-hot encoding
- Anchor color baseline for proper initialization

Based on the ObjectGS paper: "ObjectGS: Object-aware Scene Reconstruction 
and Scene Understanding via Gaussian Splatting"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple


class ViewDependentAttributeMLP(nn.Module):
    """
    MLP to generate Gaussian attributes from anchor features.
    
    Includes view-dependence following the paper:
    {c_0, ..., c_{k-1}} = MLP(f, δ, d)
    
    Where:
    - f = anchor feature
    - δ = viewing distance (scalar)
    - d = viewing direction (3D unit vector)
    
    Color is output as a DELTA to be added to anchor colors.
    """
    
    def __init__(self, feature_dim: int = 32, k: int = 10, view_dim: int = 4):
        """
        Args:
            feature_dim: Dimension of anchor features
            k: Number of Gaussians per anchor
            view_dim: Dimension of view encoding (1 distance + 3 direction = 4)
        """
        super().__init__()
        self.k = k
        self.feature_dim = feature_dim
        self.view_dim = view_dim
        
        # Input dim for view-dependent MLPs
        view_input_dim = feature_dim + view_dim
        
        # Opacity: view-independent (doesn't change with viewpoint)
        self.opacity_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, k)
        )
        
        # Scale: view-independent
        self.scale_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, k * 3)
        )
        
        # Rotation: view-independent
        self.rotation_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, k * 4)
        )
        
        # Color: VIEW-DEPENDENT, outputs DELTA (not absolute color)
        self.color_mlp = nn.Sequential(
            nn.Linear(view_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, k * 3)
        )
    
    def forward(self, anchor_features: torch.Tensor, 
                view_dirs: Optional[torch.Tensor] = None,
                view_dists: Optional[torch.Tensor] = None) -> Tuple:
        """
        Generate Gaussian attributes.
        
        Args:
            anchor_features: [N, feature_dim] anchor features
            view_dirs: [N, 3] unit vectors from anchor to camera (optional)
            view_dists: [N, 1] distances from anchor to camera (optional)
        
        Returns:
            opacity_raw, scale_raw, rotation, color_delta
        """
        batch_size = anchor_features.shape[0]
        
        # View-independent attributes
        opacity_raw = self.opacity_mlp(anchor_features)
        
        scale_raw = self.scale_mlp(anchor_features)
        scale_raw = scale_raw.reshape(batch_size, self.k, 3)
        
        rotation = self.rotation_mlp(anchor_features)
        rotation = rotation.reshape(batch_size, self.k, 4)
        rotation = rotation / (rotation.norm(dim=-1, keepdim=True) + 1e-9)
        
        # View-dependent color delta
        if view_dirs is not None and view_dists is not None:
            # Concatenate features with view information
            view_info = torch.cat([view_dists, view_dirs], dim=-1)  # [N, 4]
            color_input = torch.cat([anchor_features, view_info], dim=-1)  # [N, feature_dim + 4]
        else:
            # Fallback: pad with zeros if no view info (for export, etc.)
            zeros = torch.zeros(batch_size, self.view_dim, device=anchor_features.device)
            color_input = torch.cat([anchor_features, zeros], dim=-1)
        
        color_delta = self.color_mlp(color_input)
        color_delta = color_delta.reshape(batch_size, self.k, 3)
        # Use tanh to get delta in range [-1, 1], then scale
        color_delta = torch.tanh(color_delta) * 0.5 + 0.5  # Now in [0, 1] range
        
        return opacity_raw, scale_raw, rotation, color_delta


class ObjectGSModel(nn.Module):
    """
    ObjectGS Model - Instance Segmentation Version
    
    Features:
    - Per-Gaussian instance IDs from SAM3 segmentation
    - Instance-aware voxelization (overlapping instances get separate anchors)
    - View-dependent color generation with anchor color baseline
    - Dynamic anchor growing and pruning
    - One-hot semantic encoding for classification-based loss
    - Object-aware rendering and editing
    
    Key difference from semantic segmentation version:
    - Voxelization uses (voxel_x, voxel_y, voxel_z, instance_id) keys
    - This ensures different instances in the same spatial location get separate anchors
    - No majority voting that could merge instance boundaries
    """
    
    def __init__(self, point_cloud: np.ndarray, colors: np.ndarray, 
                 object_ids: Optional[np.ndarray] = None,
                 voxel_size: float = 0.01, k: int = 10, feature_dim: int = 32,
                 object_names: Optional[List[str]] = None, 
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            point_cloud: (N, 3) numpy array of points
            colors: (N, 3) numpy array of RGB colors [0, 1]
            object_ids: (N,) numpy array of integer instance IDs (0=background)
            voxel_size: Voxel size for anchor creation (default 1cm)
            k: Number of Gaussians per anchor
            feature_dim: Anchor feature dimension
            object_names: List of object names (from class_mapping.json)
            logger: Optional logger
        """
        super().__init__()
        
        self.logger = logger
        self.k = k
        self.feature_dim = feature_dim
        self.voxel_size = voxel_size
        
        # Handle missing object_ids
        if object_ids is None:
            object_ids = np.zeros(len(point_cloud), dtype=np.int32)
            if self.logger:
                self.logger.warning("No object_ids provided, using all zeros (background)")
        
        # num_objects is the total number of instance classes (including background at 0)
        self.num_objects = int(object_ids.max()) + 1
        self.object_names = object_names or [f"object_{i}" for i in range(self.num_objects)]
        if len(self.object_names) > 0:
            self.object_names[0] = "background"
        
        if self.logger:
            self._log_init_info(point_cloud, object_ids)
        
        # Instance-aware voxelization (KEY CHANGE from semantic version)
        anchor_positions, anchor_colors, anchor_object_ids = self._voxelize_instance_aware(
            point_cloud, colors, object_ids
        )
        
        num_anchors = len(anchor_positions)
        
        if self.logger:
            self._log_anchor_info(anchor_object_ids, num_anchors)
        
        # =====================================================================
        # DYNAMIC ANCHOR STORAGE (can grow/prune)
        # =====================================================================
        
        # Anchor positions - fixed (offsets are learnable)
        self.anchor_positions = nn.Parameter(
            torch.tensor(anchor_positions, dtype=torch.float32),
            requires_grad=False
        )
        
        # Anchor colors - buffer (used as baseline for Gaussian colors)
        self.register_buffer(
            'anchor_colors', 
            torch.tensor(anchor_colors, dtype=torch.float32)
        )
        
        # Anchor object IDs - buffer (instance IDs, not learnable)
        self.register_buffer(
            'anchor_object_ids',
            torch.tensor(anchor_object_ids, dtype=torch.long)
        )
        
        # Learnable parameters
        self.anchor_features = nn.Parameter(
            self._init_features(anchor_colors, num_anchors)
        )
        
        self.anchor_scalings = nn.Parameter(torch.ones(num_anchors))
        
        self.anchor_offsets = nn.Parameter(
            torch.randn(num_anchors, k, 3) * 0.01
        )
        
        # =====================================================================
        # TRACKING FOR GROW/PRUNE
        # =====================================================================
        
        self.register_buffer(
            'anchor_gradient_accum',
            torch.zeros(num_anchors)
        )
        self.register_buffer(
            'anchor_gradient_count',
            torch.zeros(num_anchors, dtype=torch.int32)
        )
        
        # View-dependent attribute MLP
        self.attribute_mlp = ViewDependentAttributeMLP(
            feature_dim=feature_dim, k=k, view_dim=4
        )
        self._init_mlp()
        
        # Track counts
        self._num_anchors = num_anchors
        self._num_gaussians = num_anchors * k
        
        if self.logger:
            self.logger.info(f"One-hot semantic dimension: {self.num_objects}")
            self.logger.info("=" * 70)
    
    @property
    def num_anchors(self) -> int:
        return self._num_anchors
    
    @property
    def num_gaussians(self) -> int:
        return self._num_gaussians
    
    def _log_init_info(self, point_cloud, object_ids):
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
        sorted_idx = np.argsort(-counts)[:10]
        for idx in sorted_idx:
            obj_id = unique_ids[idx]
            count = counts[idx]
            name = self.object_names[obj_id] if obj_id < len(self.object_names) else f"instance_{obj_id}"
            self.logger.info(f"  ID {obj_id}: {name} ({count:,} points)")
        if len(unique_ids) > 10:
            self.logger.info(f"  ... and {len(unique_ids) - 10} more instances")
    
    def _log_anchor_info(self, anchor_object_ids, num_anchors):
        self.logger.info(f"Created {num_anchors:,} anchors → {num_anchors * self.k:,} Gaussians")
        unique, counts = np.unique(anchor_object_ids, return_counts=True)
        self.logger.info("Anchors per instance (top 10):")
        sorted_idx = np.argsort(-counts)[:10]
        for idx in sorted_idx:
            obj_id = unique[idx]
            count = counts[idx]
            name = self.object_names[obj_id] if obj_id < len(self.object_names) else f"instance_{obj_id}"
            self.logger.info(f"  {name}: {count:,} anchors ({count * self.k:,} Gaussians)")
    
    def _init_features(self, anchor_colors, num_anchors):
        """Initialize anchor features with color info embedded"""
        features = torch.randn(num_anchors, self.feature_dim) * 0.1
        # Embed color information in first 3 dimensions
        features[:, :3] = torch.tensor(anchor_colors, dtype=torch.float32)
        return features
    
    def _voxelize_instance_aware(self, points: np.ndarray, colors: np.ndarray, 
                                  object_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Instance-aware voxelization.
        
        KEY DIFFERENCE from semantic version:
        - Uses (voxel_x, voxel_y, voxel_z, instance_id) as keys
        - Different instances in the same spatial voxel get SEPARATE anchors
        - No majority voting that could merge instance boundaries
        
        This is critical for instance segmentation where you need to preserve
        the identity of each instance even when objects are touching or overlapping.
        
        Args:
            points: (N, 3) point positions
            colors: (N, 3) RGB colors
            object_ids: (N,) instance IDs
            
        Returns:
            anchor_positions: (M, 3) anchor positions
            anchor_colors: (M, 3) anchor colors
            anchor_object_ids: (M,) anchor instance IDs
        """
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        
        # KEY CHANGE: Use (voxel, instance_id) as key
        voxel_dict = {}
        for i, vidx in enumerate(voxel_indices):
            # Include instance_id in the key - this prevents merging different instances
            key = (tuple(vidx), int(object_ids[i]))
            if key not in voxel_dict:
                voxel_dict[key] = {'colors': [], 'points': []}
            voxel_dict[key]['colors'].append(colors[i])
            voxel_dict[key]['points'].append(points[i])
        
        anchor_positions = []
        anchor_colors = []
        anchor_object_ids = []
        
        for (vidx, instance_id), data in voxel_dict.items():
            # Use centroid of actual points (more accurate than voxel center)
            pos = np.mean(data['points'], axis=0)
            avg_color = np.mean(data['colors'], axis=0)
            
            anchor_positions.append(pos)
            anchor_colors.append(avg_color)
            anchor_object_ids.append(instance_id)
        
        if self.logger:
            n_spatial_voxels = len(set(k[0] for k in voxel_dict.keys()))
            n_instance_anchors = len(voxel_dict)
            overlap_anchors = n_instance_anchors - n_spatial_voxels
            if overlap_anchors > 0:
                self.logger.info(f"Instance-aware voxelization: {n_spatial_voxels:,} spatial voxels → {n_instance_anchors:,} anchors")
                self.logger.info(f"  ({overlap_anchors:,} anchors from overlapping instances)")
        
        return (np.array(anchor_positions), 
                np.array(anchor_colors), 
                np.array(anchor_object_ids, dtype=np.int32))
    
    def _init_mlp(self):
        """Initialize MLP weights for stable training"""
        with torch.no_grad():
            # Scale: start small
            scale_out = self.attribute_mlp.scale_mlp[-1]
            scale_out.bias.data.fill_(-4.0)
            scale_out.weight.data *= 0.1
            
            # Opacity: start slightly negative (low opacity initially)
            opacity_out = self.attribute_mlp.opacity_mlp[-1]
            opacity_out.bias.data.fill_(-2.0)
            opacity_out.weight.data *= 0.1
            
            # Rotation: identity quaternion
            rot_out = self.attribute_mlp.rotation_mlp[-1]
            rot_out.bias.data.zero_()
            rot_out.bias.data[3::4] = 1.0  # w component
            
            # Color delta: start at zero delta (output centered at 0.5 after tanh transform)
            color_out = self.attribute_mlp.color_mlp[-1]
            color_out.weight.data *= 0.1
            color_out.bias.data.zero_()
    
    # =========================================================================
    # CORE FORWARD PASS
    # =========================================================================
    
    def compute_view_info(self, camera_center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute view direction and distance for each anchor.
        
        Args:
            camera_center: [3] camera position in world coordinates
        
        Returns:
            view_dirs: [N, 3] unit vectors from anchor to camera
            view_dists: [N, 1] normalized distances
        """
        # Direction from anchor to camera
        dirs = camera_center.unsqueeze(0) - self.anchor_positions  # [N, 3]
        dists = dirs.norm(dim=-1, keepdim=True)  # [N, 1]
        
        # Normalize direction
        view_dirs = dirs / (dists + 1e-8)  # [N, 3]
        
        # Normalize distance (log scale for stability)
        view_dists = torch.log(dists + 1e-8) / 10.0  # [N, 1]
        
        return view_dirs, view_dists
    
    def get_parameters_as_tensors(self, camera_center: Optional[torch.Tensor] = None,
                                   object_mask: Optional[List[int]] = None) -> Dict:
        """
        Get all Gaussian parameters.
        
        Args:
            camera_center: [3] camera position for view-dependent colors
            object_mask: Optional list of object IDs to include
        
        Returns:
            Dictionary of Gaussian parameters including:
            - pos: [N_gaussians, 3] positions
            - opacity_raw: [N_gaussians, 1] raw opacities (apply sigmoid)
            - scale_raw: [N_gaussians, 3] raw scales (apply exp)
            - rotation: [N_gaussians, 4] quaternions
            - color: [N_gaussians, 3] RGB colors
            - object_ids: [N_gaussians] integer instance IDs
            - semantics: [N_gaussians, num_objects] one-hot encoding
        """
        # Compute view info if camera provided
        if camera_center is not None:
            view_dirs, view_dists = self.compute_view_info(camera_center)
        else:
            view_dirs, view_dists = None, None
        
        # Compute positions from anchors + offsets
        positions = (
            self.anchor_positions.unsqueeze(1) + 
            self.anchor_offsets * self.anchor_scalings.unsqueeze(1).unsqueeze(2)
        )
        positions = positions.reshape(-1, 3)
        
        # MLP forward with view dependence
        opacity_raw, scale_raw, rotation, color_delta = self.attribute_mlp(
            self.anchor_features, view_dirs, view_dists
        )
        
        # Flatten from [N_anchors, k, ...] to [N_gaussians, ...]
        opacity_raw = opacity_raw.reshape(-1, 1)
        scale_raw = scale_raw.reshape(-1, 3)
        rotation = rotation.reshape(-1, 4)
        color_delta = color_delta.reshape(-1, 3)
        
        # =====================================================================
        # COLOR: Anchor color baseline + learned delta
        # This ensures colors start from the point cloud and MLP learns adjustments
        # =====================================================================
        anchor_colors_expanded = self.anchor_colors.unsqueeze(1).expand(-1, self.k, -1).reshape(-1, 3)
        
        # color_delta is in [0, 1] range (from tanh * 0.5 + 0.5)
        # Convert to adjustment: (delta - 0.5) gives us [-0.5, 0.5] range
        # Scale by factor to control adjustment magnitude
        color = anchor_colors_expanded + (color_delta - 0.5) * 0.5
        color = torch.clamp(color, 0, 1)
        
        # Expand object IDs to Gaussians
        gaussian_object_ids = self.anchor_object_ids.unsqueeze(1).expand(-1, self.k).reshape(-1)
        
        # One-hot semantics for classification-based rendering
        # This is the key to the paper's approach - semantics are discrete, not continuous
        semantics = F.one_hot(gaussian_object_ids, num_classes=self.num_objects).float()
        
        result = {
            'pos': positions,
            'opacity_raw': opacity_raw,
            'scale_raw': scale_raw,
            'rotation': rotation,
            'color': color,
            'color_delta': color_delta,
            'anchor_colors': anchor_colors_expanded,
            'object_ids': gaussian_object_ids,
            'semantics': semantics,
            'num_gaussians': self._num_gaussians,
            'num_anchors': self._num_anchors,
            'num_objects': self.num_objects,
        }
        
        # Apply object mask if specified
        if object_mask is not None:
            result = self._apply_object_mask(result, object_mask)
        
        return result
    
    def _apply_object_mask(self, params: Dict, object_ids_to_keep: List[int]) -> Dict:
        """Filter parameters to only include specified objects"""
        device = params['object_ids'].device
        mask = torch.zeros(params['num_gaussians'], dtype=torch.bool, device=device)
        
        for obj_id in object_ids_to_keep:
            mask |= (params['object_ids'] == obj_id)
        
        return {
            'pos': params['pos'][mask],
            'opacity_raw': params['opacity_raw'][mask],
            'scale_raw': params['scale_raw'][mask],
            'rotation': params['rotation'][mask],
            'color': params['color'][mask],
            'color_delta': params['color_delta'][mask],
            'anchor_colors': params['anchor_colors'][mask],
            'object_ids': params['object_ids'][mask],
            'semantics': params['semantics'][mask],
            'num_gaussians': mask.sum().item(),
            'num_anchors': self._num_anchors,
            'num_objects': self.num_objects,
            'object_mask': mask,
        }
    
    # =========================================================================
    # ANCHOR GROWING AND PRUNING
    # =========================================================================
    
    def update_gradient_stats(self, viewspace_gradients: torch.Tensor, 
                               visibility_mask: torch.Tensor):
        """
        Accumulate gradient statistics for densification decisions.
        
        Called after each training step.
        
        Args:
            viewspace_gradients: Gradients of Gaussian positions in view space
            visibility_mask: Boolean mask of which Gaussians were visible
        """
        # Convert Gaussian-level stats to anchor-level
        grad_norms = viewspace_gradients.norm(dim=-1)  # [N_gaussians]
        
        # Reshape to [N_anchors, k] and take mean per anchor
        grad_norms = grad_norms.reshape(self._num_anchors, self.k)
        visibility = visibility_mask.reshape(self._num_anchors, self.k)
        
        # Mean gradient per anchor (only for visible Gaussians)
        visible_count = visibility.sum(dim=1).clamp(min=1)
        anchor_grads = (grad_norms * visibility.float()).sum(dim=1) / visible_count
        
        # Any anchor with at least one visible Gaussian
        anchor_visible = visibility.any(dim=1)
        
        # Accumulate
        self.anchor_gradient_accum[anchor_visible] += anchor_grads[anchor_visible]
        self.anchor_gradient_count[anchor_visible] += 1
    
    def reset_gradient_stats(self):
        """Reset gradient accumulators after densification"""
        self.anchor_gradient_accum.zero_()
        self.anchor_gradient_count.zero_()
    
    def densify_and_prune(self, 
                          grad_threshold: float = 0.0002,
                          opacity_threshold: float = 0.005,
                          scale_threshold: float = 0.01,
                          max_screen_size: Optional[float] = None,
                          min_opacity: float = 0.005) -> Dict:
        """
        Perform anchor densification (growing) and pruning.
        
        This is the key dynamic adaptation mechanism from Scaffold-GS.
        Grown anchors inherit their parent's instance ID.
        
        Args:
            grad_threshold: Gradient threshold for densification
            opacity_threshold: Minimum opacity to keep anchor
            scale_threshold: Scale threshold for splitting vs cloning
            max_screen_size: Maximum screen-space size (optional)
            min_opacity: Minimum opacity to keep
        
        Returns:
            Dictionary with statistics about the operation
        """
        stats = {
            'anchors_before': self._num_anchors,
            'pruned': 0,
            'grown': 0,
        }
        
        with torch.no_grad():
            # Get current opacities
            opacity_raw, _, _, _ = self.attribute_mlp(self.anchor_features, None, None)
            anchor_opacities = torch.sigmoid(opacity_raw).mean(dim=1)  # Mean over k Gaussians
            
            # Average gradients
            avg_grads = self.anchor_gradient_accum / self.anchor_gradient_count.clamp(min=1)
            
            # ===== PRUNING =====
            prune_mask = anchor_opacities < min_opacity
            stats['pruned'] = prune_mask.sum().item()
            
            # ===== GROWING =====
            grow_mask = avg_grads > grad_threshold
            grow_mask &= ~prune_mask  # Don't grow anchors we're about to prune
            stats['grown'] = grow_mask.sum().item()
            
            # ===== APPLY CHANGES =====
            if stats['pruned'] > 0 or stats['grown'] > 0:
                self._apply_densification(prune_mask, grow_mask)
            
            # Reset stats
            self.reset_gradient_stats()
        
        stats['anchors_after'] = self._num_anchors
        
        if self.logger:
            self.logger.info(
                f"Densification: {stats['anchors_before']} → {stats['anchors_after']} anchors "
                f"(pruned {stats['pruned']}, grew {stats['grown']})"
            )
        
        return stats
    
    def _apply_densification(self, prune_mask: torch.Tensor, grow_mask: torch.Tensor):
        """
        Apply pruning and growing to anchor tensors.
        
        IMPORTANT: Grown anchors inherit their parent's instance ID.
        This maintains instance consistency during densification.
        """
        device = self.anchor_positions.device
        
        # Keep mask (inverse of prune)
        keep_mask = ~prune_mask
        
        # Indices of anchors to grow (in original indexing)
        grow_indices = torch.where(grow_mask & keep_mask)[0]
        
        # New anchor count
        n_keep = keep_mask.sum().item()
        n_grow = len(grow_indices)
        new_n_anchors = n_keep + n_grow
        
        if new_n_anchors == 0:
            if self.logger:
                self.logger.warning("Densification would remove all anchors, skipping")
            return
        
        # ===== GATHER KEPT ANCHORS =====
        new_positions = self.anchor_positions.data[keep_mask]
        new_colors = self.anchor_colors[keep_mask]
        new_object_ids = self.anchor_object_ids[keep_mask]
        new_features = self.anchor_features.data[keep_mask]
        new_scalings = self.anchor_scalings.data[keep_mask]
        new_offsets = self.anchor_offsets.data[keep_mask]
        
        # ===== ADD GROWN ANCHORS =====
        if n_grow > 0:
            # Map grow_indices to their position in the kept array
            kept_indices = torch.where(keep_mask)[0]
            
            grow_in_kept = []
            for gi in grow_indices:
                match = (kept_indices == gi).nonzero(as_tuple=True)[0]
                if len(match) > 0:
                    grow_in_kept.append(match[0].item())
            
            if len(grow_in_kept) > 0:
                grow_in_kept = torch.tensor(grow_in_kept, device=device)
                
                # Clone with small perturbation
                grown_positions = new_positions[grow_in_kept] + torch.randn(len(grow_in_kept), 3, device=device) * self.voxel_size * 0.1
                grown_colors = new_colors[grow_in_kept]
                # IMPORTANT: Grown anchors inherit parent's instance ID
                grown_object_ids = new_object_ids[grow_in_kept]
                grown_features = new_features[grow_in_kept] + torch.randn(len(grow_in_kept), self.feature_dim, device=device) * 0.01
                grown_scalings = new_scalings[grow_in_kept]
                grown_offsets = new_offsets[grow_in_kept] + torch.randn(len(grow_in_kept), self.k, 3, device=device) * 0.001
                
                # Concatenate
                new_positions = torch.cat([new_positions, grown_positions], dim=0)
                new_colors = torch.cat([new_colors, grown_colors], dim=0)
                new_object_ids = torch.cat([new_object_ids, grown_object_ids], dim=0)
                new_features = torch.cat([new_features, grown_features], dim=0)
                new_scalings = torch.cat([new_scalings, grown_scalings], dim=0)
                new_offsets = torch.cat([new_offsets, grown_offsets], dim=0)
        
        # ===== UPDATE MODEL PARAMETERS =====
        self.anchor_positions = nn.Parameter(new_positions, requires_grad=False)
        self.register_buffer('anchor_colors', new_colors)
        self.register_buffer('anchor_object_ids', new_object_ids)
        self.anchor_features = nn.Parameter(new_features)
        self.anchor_scalings = nn.Parameter(new_scalings)
        self.anchor_offsets = nn.Parameter(new_offsets)
        
        # Update gradient tracking buffers
        self.register_buffer('anchor_gradient_accum', torch.zeros(new_n_anchors, device=device))
        self.register_buffer('anchor_gradient_count', torch.zeros(new_n_anchors, dtype=torch.int32, device=device))
        
        # Update counts
        self._num_anchors = new_n_anchors
        self._num_gaussians = new_n_anchors * self.k
    
    def prune_low_opacity(self, min_opacity: float = 0.01) -> int:
        """
        Simple pruning: remove anchors where all Gaussians have low opacity.
        
        Returns:
            Number of pruned anchors
        """
        with torch.no_grad():
            opacity_raw, _, _, _ = self.attribute_mlp(self.anchor_features, None, None)
            anchor_opacities = torch.sigmoid(opacity_raw).max(dim=1)[0]
            
            prune_mask = anchor_opacities < min_opacity
            n_prune = prune_mask.sum().item()
            
            if n_prune > 0 and n_prune < self._num_anchors:
                grow_mask = torch.zeros_like(prune_mask)
                self._apply_densification(prune_mask, grow_mask)
            
            return n_prune
    
    # =========================================================================
    # OPTIMIZER HELPERS
    # =========================================================================
    
    def get_optimizer_param_groups(self, config: Dict) -> List[Dict]:
        """Get parameter groups for optimizer"""
        return [
            {
                'params': [self.anchor_features], 
                'lr': config.get('lr_feature', 0.0025), 
                'name': 'features'
            },
            {
                'params': [self.anchor_scalings], 
                'lr': config.get('lr_scaling', 0.005), 
                'name': 'scalings'
            },
            {
                'params': [self.anchor_offsets], 
                'lr': config.get('lr_position', 0.00016), 
                'name': 'offsets'
            },
            {
                'params': self.attribute_mlp.parameters(), 
                'lr': config.get('lr', 0.001), 
                'name': 'mlp'
            }
        ]
    
    def get_optimizer_param_groups_after_densify(self, optimizer: torch.optim.Optimizer, 
                                                   config: Dict) -> torch.optim.Optimizer:
        """
        Recreate optimizer after densification changes parameter shapes.
        """
        new_optimizer = torch.optim.Adam(self.get_optimizer_param_groups(config))
        return new_optimizer
    
    # =========================================================================
    # OBJECT MANIPULATION
    # =========================================================================
    
    def get_object_gaussians(self, object_id: int) -> Dict:
        """Get parameters for a single object"""
        return self.get_parameters_as_tensors(object_mask=[object_id])
    
    def get_object_stats(self) -> Dict:
        """Get statistics per object"""
        stats = {}
        
        for obj_id in range(self.num_objects):
            mask = self.anchor_object_ids == obj_id
            n_anchors = mask.sum().item()
            
            if n_anchors == 0:
                continue
            
            name = self.object_names[obj_id] if obj_id < len(self.object_names) else f"object_{obj_id}"
            positions = self.anchor_positions[mask]
            
            stats[obj_id] = {
                'name': name,
                'num_anchors': n_anchors,
                'num_gaussians': n_anchors * self.k,
                'center': positions.mean(dim=0).cpu().numpy(),
                'bbox_min': positions.min(dim=0)[0].cpu().numpy(),
                'bbox_max': positions.max(dim=0)[0].cpu().numpy(),
            }
        
        return stats
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def export_object_ply(self, object_id: int, output_path: str, 
                          min_opacity: float = 0.1):
        """
        Export a single object as a point cloud.
        
        Args:
            object_id: Object ID to export
            output_path: Path to save PLY
            min_opacity: Minimum opacity threshold for export
        """
        import open3d as o3d
        
        with torch.no_grad():
            params = self.get_object_gaussians(object_id)
            
            positions = params['pos']
            colors = params['color']
            opacities = torch.sigmoid(params['opacity_raw']).squeeze(-1)
            
            # Filter by opacity
            mask = opacities > min_opacity
            positions = positions[mask].cpu().numpy()
            colors = colors[mask].cpu().numpy()
            
            if len(positions) == 0:
                if self.logger:
                    self.logger.warning(f"No Gaussians with opacity > {min_opacity} for object {object_id}")
                return
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
        
        o3d.io.write_point_cloud(str(output_path), pcd)
        
        name = self.object_names[object_id] if object_id < len(self.object_names) else f"object_{object_id}"
        
        if self.logger:
            self.logger.info(f"Exported {name}: {len(positions):,} Gaussians to {output_path}")
    
    def export_all_objects_ply(self, output_dir: str, min_opacity: float = 0.1):
        """
        Export all objects as separate PLY files.
        
        Args:
            output_dir: Directory to save PLY files
            min_opacity: Minimum opacity threshold for export
        """
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for obj_id in range(self.num_objects):
            mask = self.anchor_object_ids == obj_id
            if mask.sum() == 0:
                continue
            
            name = self.object_names[obj_id] if obj_id < len(self.object_names) else f"object_{obj_id}"
            # Sanitize filename
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
            output_path = output_dir / f"{obj_id:03d}_{safe_name}.ply"
            
            self.export_object_ply(obj_id, str(output_path), min_opacity)