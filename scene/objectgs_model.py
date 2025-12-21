"""
ObjectGS: Anchor-based Gaussian Splatting with Object Awareness

UPDATED: Full semantic support with object-aware losses and rendering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


class AttributeMLP(nn.Module):
    """MLP to generate Gaussian attributes from anchor features"""
    
    def __init__(self, feature_dim=32, k=10):
        super().__init__()
        self.k = k
        self.feature_dim = feature_dim
        
        self.opacity_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, k)
        )
        
        self.scale_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, k * 3)
        )
        
        self.rotation_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, k * 4)
        )
        
        self.color_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, k * 3)
        )
    
    def forward(self, anchor_features):
        batch_size = anchor_features.shape[0]
        
        opacity_raw = self.opacity_mlp(anchor_features)
        
        scale_raw = self.scale_mlp(anchor_features)
        scale_raw = scale_raw.reshape(batch_size, self.k, 3)
        
        rotation = self.rotation_mlp(anchor_features)
        rotation = rotation.reshape(batch_size, self.k, 4)
        rotation = rotation / (rotation.norm(dim=-1, keepdim=True) + 1e-9)
        
        color = self.color_mlp(anchor_features)
        color = color.reshape(batch_size, self.k, 3)
        color = torch.tanh(color) * 0.5 + 0.5
        
        return opacity_raw, scale_raw, rotation, color


class ObjectGSModel(nn.Module):
    """
    ObjectGS Model with Full Semantic Support
    
    Features:
    - Per-Gaussian object IDs from SAM3 segmentation
    - Object-aware rendering (can render specific objects)
    - Semantic loss support
    - Object isolation for editing
    """
    
    def __init__(self, point_cloud, colors, object_ids=None, 
                 voxel_size=0.02, k=10, feature_dim=32, 
                 object_names=None, logger=None):
        """
        Args:
            point_cloud: (N, 3) numpy array of points
            colors: (N, 3) numpy array of RGB colors [0, 1]
            object_ids: (N,) numpy array of integer object IDs (0=background)
            voxel_size: Voxel size for anchor creation
            k: Number of Gaussians per anchor
            feature_dim: Anchor feature dimension
            object_names: List of object names (e.g., ["bed", "dresser", ...])
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
        
        self.num_objects = int(object_ids.max()) + 1
        self.object_names = object_names or [f"object_{i}" for i in range(self.num_objects)]
        self.object_names[0] = "background"  # ID 0 is always background
        
        if self.logger:
            self.logger.info("=" * 70)
            self.logger.info("INITIALIZING ObjectGSModel WITH SEMANTICS")
            self.logger.info("=" * 70)
            self.logger.info(f"Input points: {len(point_cloud):,}")
            self.logger.info(f"Voxel size: {voxel_size}")
            self.logger.info(f"k (Gaussians/anchor): {k}")
            self.logger.info(f"Feature dim: {feature_dim}")
            self.logger.info(f"Number of objects: {self.num_objects}")
            for i, name in enumerate(self.object_names[:10]):
                count = np.sum(object_ids == i)
                self.logger.info(f"  ID {i}: {name} ({count:,} points)")
            if self.num_objects > 10:
                self.logger.info(f"  ... and {self.num_objects - 10} more")
        
        # Voxelize with semantic voting
        anchor_positions, anchor_colors, anchor_object_ids = self._voxelize(
            point_cloud, colors, object_ids
        )
        
        num_anchors = len(anchor_positions)
        if self.logger:
            self.logger.info(f"Created {num_anchors:,} anchors → {num_anchors * k:,} Gaussians")
            
            # Log anchor distribution per object
            unique, counts = np.unique(anchor_object_ids, return_counts=True)
            self.logger.info("Anchors per object:")
            for obj_id, count in zip(unique, counts):
                name = self.object_names[obj_id] if obj_id < len(self.object_names) else f"object_{obj_id}"
                self.logger.info(f"  {name}: {count:,} anchors ({count * k:,} Gaussians)")
        
        # Fixed buffers
        self.register_buffer('anchor_positions', torch.tensor(anchor_positions, dtype=torch.float32))
        self.register_buffer('anchor_colors', torch.tensor(anchor_colors, dtype=torch.float32))
        self.register_buffer('anchor_object_ids', torch.tensor(anchor_object_ids, dtype=torch.long))
        
        # Learnable parameters
        features_init = torch.randn(num_anchors, feature_dim) * 0.1
        features_init[:, :3] = self.anchor_colors
        self.anchor_features = nn.Parameter(features_init)
        
        self.anchor_scalings = nn.Parameter(torch.ones(num_anchors))
        self.anchor_offsets = nn.Parameter(torch.randn(num_anchors, k, 3) * 0.01)
        
        # Pre-expanded buffers for Gaussians
        expanded_colors = self.anchor_colors.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3)
        self.register_buffer('gaussian_anchor_colors', expanded_colors)
        
        expanded_obj_ids = self.anchor_object_ids.unsqueeze(1).expand(-1, k).reshape(-1)
        self.register_buffer('gaussian_object_ids', expanded_obj_ids)
        
        # Attribute MLP
        self.attribute_mlp = AttributeMLP(feature_dim=feature_dim, k=k)
        self._init_mlp()
        
        self.num_anchors = num_anchors
        self.num_gaussians = num_anchors * k
        
        if self.logger:
            self.logger.info("=" * 70)
    
    def _voxelize(self, points, colors, object_ids):
        """Voxelize point cloud with semantic majority voting"""
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        
        voxel_dict = {}
        for i, vidx in enumerate(voxel_indices):
            key = tuple(vidx)
            if key not in voxel_dict:
                voxel_dict[key] = {'colors': [], 'object_ids': []}
            voxel_dict[key]['colors'].append(colors[i])
            voxel_dict[key]['object_ids'].append(object_ids[i])
        
        anchor_positions = []
        anchor_colors = []
        anchor_object_ids = []
        
        for vidx, data in voxel_dict.items():
            pos = (np.array(vidx) + 0.5) * self.voxel_size
            avg_color = np.mean(data['colors'], axis=0)
            
            # Majority vote for object ID
            ids = np.array(data['object_ids'])
            if len(ids) > 0:
                obj_id = int(np.bincount(ids).argmax())
            else:
                obj_id = 0
            
            anchor_positions.append(pos)
            anchor_colors.append(avg_color)
            anchor_object_ids.append(obj_id)
        
        return (np.array(anchor_positions), 
                np.array(anchor_colors), 
                np.array(anchor_object_ids))
    
    def _init_mlp(self):
        """Initialize MLP weights"""
        with torch.no_grad():
            scale_out = self.attribute_mlp.scale_mlp[-1]
            scale_out.bias.data.fill_(-4.0)
            scale_out.weight.data *= 0.1
            
            opacity_out = self.attribute_mlp.opacity_mlp[-1]
            opacity_out.bias.data.fill_(-2.2)
            opacity_out.weight.data *= 0.1
            
            color_out = self.attribute_mlp.color_mlp[-1]
            color_out.weight.data *= 2.0
            color_out.bias.data.zero_()
            
            rot_out = self.attribute_mlp.rotation_mlp[-1]
            rot_out.bias.data.zero_()
            rot_out.bias.data[3::4] = 1.0
    
    def get_parameters_as_tensors(self, object_mask=None):
        """
        Get all Gaussian parameters.
        
        Args:
            object_mask: Optional list of object IDs to include.
                        If None, returns all Gaussians.
                        e.g., [1, 2] returns only Gaussians from objects 1 and 2
        
        Returns:
            Dictionary of Gaussian parameters
        """
        # Compute positions
        positions = (self.anchor_positions.unsqueeze(1) + 
                    self.anchor_offsets * self.anchor_scalings.unsqueeze(1).unsqueeze(2))
        positions = positions.reshape(-1, 3)
        
        # MLP forward
        opacity_raw, scale_raw, rotation, color_delta = self.attribute_mlp(self.anchor_features)
        
        # Flatten
        opacity_raw = opacity_raw.reshape(-1, 1)
        scale_raw = scale_raw.reshape(-1, 3)
        rotation = rotation.reshape(-1, 4)
        color_delta = color_delta.reshape(-1, 3)
        
        # Final color
        color = self.gaussian_anchor_colors + (color_delta - 0.5) * 1.0
        color = torch.clamp(color, 0, 1)
        
        # Semantics
        semantics = F.one_hot(self.gaussian_object_ids, num_classes=self.num_objects).float()
        
        result = {
            'pos': positions,
            'opacity_raw': opacity_raw,
            'scale_raw': scale_raw,
            'rotation': rotation,
            'color': color,
            'color_delta': color_delta,
            'anchor_colors': self.gaussian_anchor_colors,
            'object_ids': self.gaussian_object_ids,
            'semantics': semantics,
            'num_gaussians': self.num_gaussians,
            'num_anchors': self.num_anchors,
            'num_objects': self.num_objects
        }
        
        # Apply object mask if specified
        if object_mask is not None:
            result = self._apply_object_mask(result, object_mask)
        
        return result
    
    def _apply_object_mask(self, params, object_ids_to_keep):
        """Filter parameters to only include specified objects"""
        mask = torch.zeros(self.num_gaussians, dtype=torch.bool, device=self.gaussian_object_ids.device)
        for obj_id in object_ids_to_keep:
            mask |= (self.gaussian_object_ids == obj_id)
        
        filtered = {
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
            'num_anchors': self.num_anchors,  # Original count
            'num_objects': self.num_objects,
            'object_mask': mask,  # Include mask for reference
        }
        return filtered
    
    def get_object_gaussians(self, object_id):
        """Get parameters for a single object"""
        return self.get_parameters_as_tensors(object_mask=[object_id])
    
    def get_object_stats(self):
        """Get statistics per object"""
        stats = {}
        for obj_id in range(self.num_objects):
            mask = self.gaussian_object_ids == obj_id
            count = mask.sum().item()
            name = self.object_names[obj_id] if obj_id < len(self.object_names) else f"object_{obj_id}"
            
            if count > 0:
                positions = (self.anchor_positions.unsqueeze(1) + 
                            self.anchor_offsets * self.anchor_scalings.unsqueeze(1).unsqueeze(2))
                positions = positions.reshape(-1, 3)[mask]
                
                stats[obj_id] = {
                    'name': name,
                    'num_gaussians': count,
                    'center': positions.mean(dim=0).cpu().numpy(),
                    'bbox_min': positions.min(dim=0)[0].cpu().numpy(),
                    'bbox_max': positions.max(dim=0)[0].cpu().numpy(),
                }
        
        return stats
    
    def get_optimizer_param_groups(self, config):
        """Get parameter groups for optimizer"""
        return [
            {'params': [self.anchor_features], 'lr': config.get('lr_feature', 0.0025), 'name': 'features'},
            {'params': [self.anchor_scalings], 'lr': config.get('lr_scaling', 0.005), 'name': 'scalings'},
            {'params': [self.anchor_offsets], 'lr': config.get('lr_position', 0.00016), 'name': 'offsets'},
            {'params': self.attribute_mlp.parameters(), 'lr': config.get('lr', 0.001), 'name': 'mlp'}
        ]
    
    # =========================================================================
    # OBJECT MANIPULATION METHODS
    # =========================================================================
    
    def hide_object(self, object_id):
        """
        'Hide' an object by zeroing its opacity (non-destructive).
        Note: This modifies the model state. Call unhide_object to restore.
        """
        if not hasattr(self, '_hidden_objects'):
            self._hidden_objects = set()
        self._hidden_objects.add(object_id)
    
    def unhide_object(self, object_id):
        """Restore a hidden object"""
        if hasattr(self, '_hidden_objects'):
            self._hidden_objects.discard(object_id)
    
    def get_visible_parameters(self):
        """Get parameters excluding hidden objects"""
        if not hasattr(self, '_hidden_objects') or len(self._hidden_objects) == 0:
            return self.get_parameters_as_tensors()
        
        visible_objects = [i for i in range(self.num_objects) if i not in self._hidden_objects]
        return self.get_parameters_as_tensors(object_mask=visible_objects)
    
    def export_object_ply(self, object_id, output_path):
        """Export a single object as a point cloud"""
        import open3d as o3d
        
        params = self.get_object_gaussians(object_id)
        positions = params['pos'].detach().cpu().numpy()
        colors = params['color'].detach().cpu().numpy()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(str(output_path), pcd)
        
        name = self.object_names[object_id] if object_id < len(self.object_names) else f"object_{object_id}"
        print(f"Exported {name} ({len(positions):,} Gaussians) to {output_path}")