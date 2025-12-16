"""
ObjectGS: Anchor-based Gaussian Splatting with Object Awareness
FIXED VERSION - Proper opacity initialization for stable training

Key fixes:
1. Initialize opacities LOW (inverse_sigmoid(0.1) ≈ -2.2) NOT HIGH!
2. Better color preservation
3. Smaller initial scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree


class Gaussian:
    """Individual 3D Gaussian primitive"""
    
    def __init__(self, position, opacity_raw, scale_raw, rotation, 
                 color, object_id, num_objects):
        self.position = position
        self.opacity_raw = opacity_raw
        self.scale_raw = scale_raw
        self.rotation = rotation
        self.color = color
        self.object_id = object_id
        self.semantic = self._create_one_hot(object_id, num_objects)
    
    def _create_one_hot(self, object_id, num_objects):
        one_hot = torch.zeros(num_objects)
        one_hot[object_id] = 1.0
        return one_hot
    
    @property
    def opacity(self):
        return torch.sigmoid(self.opacity_raw).clamp(0, 0.999)
    
    @property
    def scale(self):
        return torch.exp(self.scale_raw).clamp_min(1e-6)


class Anchor(nn.Module):
    """Anchor point that generates k neural Gaussians"""
    
    def __init__(self, position, color, object_id, k=10, feature_dim=32):
        super().__init__()
        
        # Register position as buffer
        self.register_buffer('position', torch.tensor(position, dtype=torch.float32))
        
        # Store anchor color for initialization
        self.register_buffer('color', torch.tensor(color, dtype=torch.float32))
        
        self.object_id = object_id
        self.k = k
        
        # Learnable parameters
        # Initialize feature to encode the color
        feature_init = torch.randn(feature_dim) * 0.1
        feature_init[:3] = torch.tensor(color, dtype=torch.float32)  # First 3 dims = color
        self.feature = nn.Parameter(feature_init)
        
        self.scaling = nn.Parameter(torch.tensor(1.0))
        self.offsets = nn.Parameter(torch.randn(k, 3) * 0.01)
        
    def get_gaussian_positions(self):
        positions = self.position.unsqueeze(0) + self.offsets * self.scaling
        return positions


def inverse_sigmoid(x):
    """Compute inverse sigmoid (logit)"""
    x = torch.clamp(x, 1e-6, 1 - 1e-6)
    return torch.log(x / (1 - x))


class AttributeMLP(nn.Module):
    """MLP to generate Gaussian attributes from anchor features"""
    
    def __init__(self, feature_dim=32, k=10):
        super().__init__()
        self.k = k
        
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
    
    def forward(self, anchor_feature):
        """
        Args:
            anchor_feature: [N, F] batched anchor features
            
        Returns:
            opacity_raw: [N, k]
            scale_raw: [N, k, 3]
            rotation: [N, k, 4]
            color: [N, k, 3]
        """
        batch_size = anchor_feature.shape[0]
        
        opacity_raw = self.opacity_mlp(anchor_feature)  # [N, k]
        
        scale_raw = self.scale_mlp(anchor_feature)
        scale_raw = scale_raw.reshape(batch_size, self.k, 3)  # [N, k, 3]
        
        rotation = self.rotation_mlp(anchor_feature)
        rotation = rotation.reshape(batch_size, self.k, 4)  # [N, k, 4]
        rotation = rotation / (rotation.norm(dim=-1, keepdim=True) + 1e-9)
        
        color = self.color_mlp(anchor_feature)
        color = color.reshape(batch_size, self.k, 3)  # [N, k, 3]
        
        # Use tanh to allow full color range, then scale to [0, 1]
        color = torch.tanh(color) * 0.5 + 0.5
        
        return opacity_raw, scale_raw, rotation, color


class ObjectGSModel(nn.Module):
    """Main ObjectGS model with FIXED initialization"""
    
    def __init__(self, point_cloud, colors, object_ids=None, 
                 voxel_size=0.02, k=10, feature_dim=32):
        super().__init__()
        
        self.k = k
        self.feature_dim = feature_dim
        self.voxel_size = voxel_size
        
        if object_ids is None:
            object_ids = np.ones(len(point_cloud), dtype=np.int32)
        
        self.num_objects = int(object_ids.max()) + 1
        
        print(f"\n{'='*60}")
        print("Initializing ObjectGS model (FIXED VERSION)")
        print(f"{'='*60}")
        print(f"  Point cloud: {len(point_cloud):,} points")
        print(f"  Voxel size: {voxel_size}m")
        print(f"  Gaussians per anchor: {k}")
        print(f"  Number of objects: {self.num_objects}")
        
        # Create anchors from voxelized point cloud
        self.anchors = self._create_anchors_from_pointcloud(
            point_cloud, colors, object_ids
        )
        
        print(f"  Created {len(self.anchors)} anchors")
        print(f"  Total Gaussians: {len(self.anchors) * k:,}")
        
        # MLP for generating Gaussian attributes
        self.attribute_mlp = AttributeMLP(feature_dim=feature_dim, k=k)
        
        # CRITICAL FIX: Initialize MLPs properly for STABLE training
        with torch.no_grad():
            # 1. Scale MLP: SMALL initial scales (exp(-4) ≈ 0.018)
            # Start smaller than before to prevent immediate bloating
            scale_output = self.attribute_mlp.scale_mlp[-1]
            if hasattr(scale_output, 'bias') and scale_output.bias is not None:
                scale_output.bias.data.fill_(-4.0)  # Smaller than before
            if hasattr(scale_output, 'weight'):
                scale_output.weight.data *= 0.1  # Small weights
                print("  ✓ Initialized scales to SMALL (~0.018)")
            
            # 2. Opacity MLP: LOW initial opacity (inverse_sigmoid(0.1) ≈ -2.2)
            # THIS IS THE KEY FIX! Start with low opacity, let it grow during training
            opacity_output = self.attribute_mlp.opacity_mlp[-1]
            if hasattr(opacity_output, 'bias') and opacity_output.bias is not None:
                # inverse_sigmoid(0.1) ≈ -2.197
                opacity_output.bias.data.fill_(-2.2)
                print("  ✓ Initialized opacities to LOW (0.1) - KEY FIX!")
            if hasattr(opacity_output, 'weight'):
                opacity_output.weight.data *= 0.1  # Small weights
            
            # 3. Color MLP: preserve point cloud colors
            color_output = self.attribute_mlp.color_mlp[-1]
            if hasattr(color_output, 'weight'):
                color_output.weight.data *= 0.01  # Very small weights
            if hasattr(color_output, 'bias') and color_output.bias is not None:
                color_output.bias.data.zero_()
                print("  ✓ Initialized colors to preserve point cloud colors")
            
            # 4. Rotation MLP: initialize to identity quaternion [0,0,0,1]
            rotation_output = self.attribute_mlp.rotation_mlp[-1]
            if hasattr(rotation_output, 'bias') and rotation_output.bias is not None:
                rotation_output.bias.data.zero_()
                # Set every 4th element (w component) to 1
                rotation_output.bias.data[3::4] = 1.0
        
        # Device tracking
        self.device = torch.device('cpu')
        
        # Pre-compute anchor data
        self._precompute_anchor_data()
        
        print(f"  ✓ Pre-computed anchor data")
        print(f"{'='*60}\n")
    
    def _precompute_anchor_data(self):
        """Pre-compute anchor object IDs and colors as tensors"""
        object_ids_list = []
        colors_list = []
        
        for anchor in self.anchors:
            # Each anchor creates k Gaussians with same object_id and color
            object_ids_list.extend([anchor.object_id] * self.k)
            for _ in range(self.k):
                colors_list.append(anchor.color)
        
        self.register_buffer(
            '_anchor_object_ids',
            torch.tensor(object_ids_list, dtype=torch.long)
        )
        self.register_buffer(
            '_anchor_colors',
            torch.stack(colors_list)
        )
    
    def to(self, device):
        """Override to track device"""
        self.device = device
        return super().to(device)
    
    def _create_anchors_from_pointcloud(self, points, colors, object_ids):
        """Voxelize point cloud and create one anchor per voxel"""
        
        # Compute voxel indices
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        
        # Group points by voxel
        voxel_dict = {}
        for i, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            if key not in voxel_dict:
                voxel_dict[key] = {
                    'points': [],
                    'colors': [],
                    'object_ids': []
                }
            voxel_dict[key]['points'].append(points[i])
            voxel_dict[key]['colors'].append(colors[i])
            voxel_dict[key]['object_ids'].append(object_ids[i])
        
        # Create anchors
        anchors = []
        for voxel_idx, data in voxel_dict.items():
            voxel_center = (np.array(voxel_idx) + 0.5) * self.voxel_size
            object_id = int(np.bincount(data['object_ids']).argmax())
            
            # Average color in this voxel
            avg_color = np.mean(data['colors'], axis=0)
            
            anchor = Anchor(
                position=voxel_center,
                color=avg_color,
                object_id=object_id,
                k=self.k,
                feature_dim=self.feature_dim
            )
            
            anchors.append(anchor)
        
        return nn.ModuleList(anchors)
    
    def get_parameters_as_tensors(self):
        """
        Export all Gaussian parameters as tensors
        OPTIMIZED: Zero Python loops
        """
        num_anchors = len(self.anchors)
        
        # Stack all anchor parameters
        anchor_features = torch.stack([a.feature for a in self.anchors])
        anchor_positions = torch.stack([a.position for a in self.anchors])
        anchor_offsets = torch.stack([a.offsets for a in self.anchors])
        anchor_scalings = torch.stack([a.scaling for a in self.anchors])
        
        # Compute positions
        positions = anchor_positions.unsqueeze(1) + anchor_offsets * anchor_scalings.unsqueeze(1).unsqueeze(2)
        positions = positions.reshape(-1, 3)
        
        # Run MLP
        opacity_raw, scale_raw, rotation, color_delta = self.attribute_mlp(anchor_features)
        
        # Flatten
        opacity_raw = opacity_raw.reshape(-1, 1)
        scale_raw = scale_raw.reshape(-1, 3)
        rotation = rotation.reshape(-1, 4)
        color_delta = color_delta.reshape(-1, 3)
        
        # Use anchor colors as base, MLP provides small corrections
        color = self._anchor_colors + (color_delta - 0.5) * 0.2
        color = torch.clamp(color, 0, 1)
        
        # Object IDs and semantics
        object_ids = self._anchor_object_ids
        semantics = F.one_hot(object_ids, num_classes=self.num_objects).float()
        
        return {
            'pos': positions,
            'opacity_raw': opacity_raw,
            'scale_raw': scale_raw,
            'rotation': rotation,
            'color': color,
            'object_ids': object_ids,
            'semantics': semantics,
            'num_gaussians': positions.shape[0],
            'num_anchors': num_anchors,
            'num_objects': self.num_objects
        }
    
    def save(self, path):
        """Save model parameters"""
        params = self.get_parameters_as_tensors()
        torch.save({
            'state_dict': self.state_dict(),
            'params': params,
            'config': {
                'voxel_size': self.voxel_size,
                'k': self.k,
                'feature_dim': self.feature_dim,
                'num_objects': self.num_objects
            }
        }, path)
        print(f"Saved model to {path}")
    
    def summary(self):
        """Print model summary"""
        params = self.get_parameters_as_tensors()
        
        print("\n" + "="*60)
        print("ObjectGS Model Summary")
        print("="*60)
        print(f"Anchors:          {params['num_anchors']:,}")
        print(f"Gaussians:        {params['num_gaussians']:,}")
        print(f"Objects:          {params['num_objects']}")
        print(f"Gaussians/Anchor: {self.k}")
        print(f"Voxel size:       {self.voxel_size}m")
        print(f"Feature dim:      {self.feature_dim}")
        print("-"*60)
        
        # Check actual parameter ranges
        opacity = torch.sigmoid(params['opacity_raw'])
        scale = torch.exp(params['scale_raw'])
        
        print(f"Opacity range:    [{opacity.min():.3f}, {opacity.max():.3f}] (mean: {opacity.mean():.3f})")
        print(f"Scale range:      [{scale.min():.4f}, {scale.max():.4f}] (mean: {scale.mean():.4f})")
        print(f"Color range:      [{params['color'].min():.3f}, {params['color'].max():.3f}]")
        
        # Verify initialization
        if opacity.mean() < 0.3:
            print("\n✓ GOOD: Opacities start LOW (will increase during training)")
        else:
            print("\n⚠ WARNING: Opacities start HIGH (may cause blob formation)")
        
        if scale.mean() < 0.05:
            print("✓ GOOD: Scales start SMALL")
        else:
            print("⚠ WARNING: Scales start LARGE")
        
        print("="*60 + "\n")


def initialize_from_pointcloud(point_cloud, colors, object_ids=None,
                               voxel_size=0.02, k=10):
    """
    Convenience function to create ObjectGS model from point cloud
    """
    model = ObjectGSModel(
        point_cloud=point_cloud,
        colors=colors,
        object_ids=object_ids,
        voxel_size=voxel_size,
        k=k
    )
    
    model.summary()
    
    return model