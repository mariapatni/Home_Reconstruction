"""
ObjectGS: Anchor-based Gaussian Splatting with Object Awareness

PERFORMANCE FIX: Batched parameters - no Python loops in forward pass!
This should give ~50-100x speedup.

Original issue: get_parameters_as_tensors() iterated through all anchors in Python
Fix: Store all anchor data as batched tensors, update in-place
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
from datetime import datetime
import pytz


# =============================================================================
# ATTRIBUTE MLP (unchanged)
# =============================================================================

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
        """
        Args:
            anchor_features: [N, F] batched anchor features
        Returns:
            opacity_raw, scale_raw, rotation, color
        """
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


# =============================================================================
# FAST MODEL - BATCHED PARAMETERS
# =============================================================================

class ObjectGSModel(nn.Module):
    """
    ObjectGS Model - FAST VERSION
    
    Key difference: All anchor parameters stored as batched tensors
    No Python loops in forward pass!
    """
    
    def __init__(self, point_cloud, colors, object_ids=None, 
                 voxel_size=0.02, k=10, feature_dim=32, logger=None):
        super().__init__()
        
        self.logger = logger
        self.k = k
        self.feature_dim = feature_dim
        self.voxel_size = voxel_size
        
        if object_ids is None:
            object_ids = np.ones(len(point_cloud), dtype=np.int32)
        
        self.num_objects = int(object_ids.max()) + 1
        
        if self.logger:
            self.logger.info("=" * 70)
            self.logger.info("INITIALIZING ObjectGSModelFast (BATCHED)")
            self.logger.info("=" * 70)
            self.logger.info(f"Input points: {len(point_cloud):,}")
            self.logger.info(f"Voxel size: {voxel_size}")
            self.logger.info(f"k (Gaussians/anchor): {k}")
            self.logger.info(f"Feature dim: {feature_dim}")
        
        # Voxelize and create anchor data
        anchor_positions, anchor_colors, anchor_object_ids = self._voxelize(
            point_cloud, colors, object_ids
        )
        
        num_anchors = len(anchor_positions)
        if self.logger:
            self.logger.info(f"Created {num_anchors} anchors → {num_anchors * k:,} Gaussians")
        
        # =====================================================================
        # BATCHED PARAMETERS - This is the key fix!
        # =====================================================================
        
        # Fixed buffers (not learnable)
        self.register_buffer('anchor_positions', torch.tensor(anchor_positions, dtype=torch.float32))
        self.register_buffer('anchor_colors', torch.tensor(anchor_colors, dtype=torch.float32))
        self.register_buffer('anchor_object_ids', torch.tensor(anchor_object_ids, dtype=torch.long))
        
        # Learnable parameters - ALL BATCHED
        # Features: [num_anchors, feature_dim]
        features_init = torch.randn(num_anchors, feature_dim) * 0.1
        features_init[:, :3] = self.anchor_colors  # Initialize with anchor colors
        self.anchor_features = nn.Parameter(features_init)
        
        # Scalings: [num_anchors] - single scale per anchor
        self.anchor_scalings = nn.Parameter(torch.ones(num_anchors))
        
        # Offsets: [num_anchors, k, 3]
        self.anchor_offsets = nn.Parameter(torch.randn(num_anchors, k, 3) * 0.01)
        
        # Pre-expand anchor colors for Gaussians: [num_anchors * k, 3]
        expanded_colors = self.anchor_colors.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3)
        self.register_buffer('gaussian_anchor_colors', expanded_colors)
        
        # Pre-expand object IDs: [num_anchors * k]
        expanded_obj_ids = self.anchor_object_ids.unsqueeze(1).expand(-1, k).reshape(-1)
        self.register_buffer('gaussian_object_ids', expanded_obj_ids)
        
        # Attribute MLP
        self.attribute_mlp = AttributeMLP(feature_dim=feature_dim, k=k)
        self._init_mlp()
        
        self.num_anchors = num_anchors
        self.num_gaussians = num_anchors * k
        
        if self.logger:
            self.logger.info("-" * 50)
            self.logger.info("BATCHED PARAMETER SHAPES:")
            self.logger.info(f"  anchor_features: {list(self.anchor_features.shape)}")
            self.logger.info(f"  anchor_scalings: {list(self.anchor_scalings.shape)}")
            self.logger.info(f"  anchor_offsets:  {list(self.anchor_offsets.shape)}")
            self.logger.info("=" * 70)
    
    def _voxelize(self, points, colors, object_ids):
        """Voxelize point cloud into anchors"""
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
            obj_id = int(np.bincount(data['object_ids']).argmax())
            
            anchor_positions.append(pos)
            anchor_colors.append(avg_color)
            anchor_object_ids.append(obj_id)
        
        return (np.array(anchor_positions), 
                np.array(anchor_colors), 
                np.array(anchor_object_ids))
    
    def _init_mlp(self):
        """Initialize MLP with good defaults"""
        with torch.no_grad():
            # Scale: exp(-4) ≈ 0.018
            scale_out = self.attribute_mlp.scale_mlp[-1]
            scale_out.bias.data.fill_(-4.0)
            scale_out.weight.data *= 0.1
            
            # Opacity: sigmoid(-2.2) ≈ 0.10
            opacity_out = self.attribute_mlp.opacity_mlp[-1]
            opacity_out.bias.data.fill_(-2.2)
            opacity_out.weight.data *= 0.1
            
            # Color
            color_out = self.attribute_mlp.color_mlp[-1]
            color_out.weight.data *= 2.0
            color_out.bias.data.zero_()
            
            # Rotation: identity quaternion
            rot_out = self.attribute_mlp.rotation_mlp[-1]
            rot_out.bias.data.zero_()
            rot_out.bias.data[3::4] = 1.0
    
    def get_parameters_as_tensors(self):
        """
        Get all Gaussian parameters - FAST BATCHED VERSION
        
        No Python loops! All tensor operations.
        """
        # Positions: [num_anchors, 1, 3] + [num_anchors, k, 3] * [num_anchors, 1, 1]
        positions = (self.anchor_positions.unsqueeze(1) + 
                    self.anchor_offsets * self.anchor_scalings.unsqueeze(1).unsqueeze(2))
        positions = positions.reshape(-1, 3)  # [num_gaussians, 3]
        
        # MLP forward pass - single batched call
        opacity_raw, scale_raw, rotation, color_delta = self.attribute_mlp(self.anchor_features)
        
        # Flatten: [num_anchors, k, ...] -> [num_gaussians, ...]
        opacity_raw = opacity_raw.reshape(-1, 1)
        scale_raw = scale_raw.reshape(-1, 3)
        rotation = rotation.reshape(-1, 4)
        color_delta = color_delta.reshape(-1, 3)
        
        # Final color
        color = self.gaussian_anchor_colors + (color_delta - 0.5) * 1.0
        color = torch.clamp(color, 0, 1)
        
        # Object semantics
        semantics = F.one_hot(self.gaussian_object_ids, num_classes=self.num_objects).float()
        
        return {
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
    
    def get_optimizer_param_groups(self, config):
        """Get parameter groups for optimizer"""
        return [
            {'params': [self.anchor_features], 'lr': config.get('lr_feature', 0.0025), 'name': 'features'},
            {'params': [self.anchor_scalings], 'lr': config.get('lr_scaling', 0.005), 'name': 'scalings'},
            {'params': [self.anchor_offsets], 'lr': config.get('lr_position', 0.00016), 'name': 'offsets'},
            {'params': self.attribute_mlp.parameters(), 'lr': config.get('lr', 0.001), 'name': 'mlp'}
        ]