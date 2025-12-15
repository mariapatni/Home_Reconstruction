"""
ObjectGS: Anchor-based Gaussian Splatting with Object Awareness
Implements hierarchical representation: Point Cloud → Anchors → Gaussians
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree


class Gaussian:
    """Individual 3D Gaussian primitive"""
    
    def __init__(self, position, opacity_raw, scale_raw, rotation, 
                 color, object_id, num_objects):
        """
        Args:
            position: [3] 3D position in world space
            opacity_raw: scalar - raw opacity (before sigmoid)
            scale_raw: [3] - raw scale (before exp)
            rotation: [4] - quaternion [w, x, y, z]
            color: [3] - RGB color [0, 1]
            object_id: int - which object this belongs to
            num_objects: int - total number of objects in scene
        """
        self.position = position
        self.opacity_raw = opacity_raw
        self.scale_raw = scale_raw
        self.rotation = rotation
        self.color = color
        self.object_id = object_id
        
        # One-hot semantic encoding (NOT learnable!)
        self.semantic = self._create_one_hot(object_id, num_objects)
    
    def _create_one_hot(self, object_id, num_objects):
        """Create one-hot encoding for object ID"""
        one_hot = torch.zeros(num_objects)
        one_hot[object_id] = 1.0
        return one_hot
    
    @property
    def opacity(self):
        """Actual opacity (after sigmoid): [0, 0.999]"""
        return torch.sigmoid(self.opacity_raw).clamp(0, 0.999)
    
    @property
    def scale(self):
        """Actual scale (after exp): positive values"""
        return torch.exp(self.scale_raw).clamp_min(1e-6)


class Anchor(nn.Module):
    """Anchor point that generates k neural Gaussians"""
    
    def __init__(self, position, object_id, k=10, feature_dim=32):
        """
        Args:
            position: [3] center of voxel in world space
            object_id: int - which object this anchor belongs to
            k: int - number of Gaussians to generate per anchor
            feature_dim: int - dimensionality of anchor feature
        """
        super().__init__()
        
        # Register position as buffer (not a parameter, but moves with model)
        self.register_buffer('position', torch.tensor(position, dtype=torch.float32))
        self.object_id = object_id
        self.k = k
        
        # Learnable parameters
        self.feature = nn.Parameter(torch.randn(feature_dim) * 0.1)
        self.scaling = nn.Parameter(torch.tensor(1.0))
        self.offsets = nn.Parameter(torch.randn(k, 3) * 0.01)
        
    def get_gaussian_positions(self):
        """
        Compute positions of k Gaussians from anchor
        
        Returns:
            [k, 3] positions
        """
        # Position = anchor_center + offset * scaling
        # position is now a buffer so it's on the same device
        positions = self.position.unsqueeze(0) + self.offsets * self.scaling
        return positions


class AttributeMLP(nn.Module):
    """MLP to generate Gaussian attributes from anchor features"""
    
    def __init__(self, feature_dim=32, k=10):
        super().__init__()
        self.k = k
        
        # Separate MLPs for different attributes
        # Opacity: k values
        self.opacity_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, k)
        )
        
        # Scale: k × 3 values (width, height, depth for each Gaussian)
        self.scale_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, k * 3)
        )
        
        # Rotation: k × 4 values (quaternion for each)
        self.rotation_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, k * 4)
        )
        
        # Color: k × 3 values (RGB for each)
        self.color_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, k * 3)
        )
    
    def forward(self, anchor_feature):
        """
        Generate attributes for k Gaussians from anchor feature
        
        Args:
            anchor_feature: [F] anchor feature vector
            
        Returns:
            opacity_raw: [k] raw opacity values
            scale_raw: [k, 3] raw scale values
            rotation: [k, 4] quaternions
            color: [k, 3] RGB colors
        """
        opacity_raw = self.opacity_mlp(anchor_feature)  # [k]
        
        scale_raw = self.scale_mlp(anchor_feature)
        scale_raw = scale_raw.reshape(self.k, 3)  # [k, 3]
        
        rotation = self.rotation_mlp(anchor_feature)
        rotation = rotation.reshape(self.k, 4)  # [k, 4]
        # Normalize quaternions
        rotation = rotation / (rotation.norm(dim=-1, keepdim=True) + 1e-9)
        
        color = self.color_mlp(anchor_feature)
        color = color.reshape(self.k, 3)  # [k, 3]
        color = torch.sigmoid(color)  # [0, 1]
        
        return opacity_raw, scale_raw, rotation, color


class ObjectGSModel(nn.Module):
    """Main ObjectGS model: manages anchors and generates Gaussians"""
    
    def __init__(self, point_cloud, colors, object_ids=None, 
                 voxel_size=0.02, k=10, feature_dim=32):
        """
        Args:
            point_cloud: [N, 3] numpy array of points
            colors: [N, 3] numpy array of RGB colors [0, 1]
            object_ids: [N] numpy array of object IDs (or None for all 1s)
            voxel_size: float - size of voxel grid for anchor placement
            k: int - number of Gaussians per anchor
            feature_dim: int - anchor feature dimensionality
        """
        super().__init__()
        
        self.k = k
        self.feature_dim = feature_dim
        self.voxel_size = voxel_size
        
        # Set all object IDs to 1 if not provided (placeholder)
        if object_ids is None:
            object_ids = np.ones(len(point_cloud), dtype=np.int32)
        
        self.num_objects = int(object_ids.max()) + 1  # +1 for background (ID=0)
        
        print(f"\nInitializing ObjectGS model:")
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
        
        # Device will be set when model is moved
        self.device = torch.device('cpu')
    
    def to(self, device):
        """Override to track device"""
        self.device = device
        return super().to(device)
    
    def _create_anchors_from_pointcloud(self, points, colors, object_ids):
        """Voxelize point cloud and create one anchor per voxel"""
        
        # Compute voxel indices for each point
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
        
        # Create one anchor per voxel
        anchors = []
        for voxel_idx, data in voxel_dict.items():
            # Voxel center
            voxel_center = (np.array(voxel_idx) + 0.5) * self.voxel_size
            
            # Object ID: majority vote from points in voxel
            object_id = int(np.bincount(data['object_ids']).argmax())
            
            # Create anchor
            anchor = Anchor(
                position=voxel_center,
                object_id=object_id,
                k=self.k,
                feature_dim=self.feature_dim
            )
            
            anchors.append(anchor)
        
        return nn.ModuleList(anchors)
    
    def generate_gaussians(self):
        """
        Generate all Gaussians from all anchors
        
        Returns:
            List of Gaussian objects
        """
        all_gaussians = []
        
        for anchor in self.anchors:
            # Get positions for k Gaussians
            positions = anchor.get_gaussian_positions()  # [k, 3]
            
            # Generate attributes using MLP
            opacity_raw, scale_raw, rotation, color = self.attribute_mlp(
                anchor.feature
            )
            
            # Create k Gaussians
            for i in range(self.k):
                gaussian = Gaussian(
                    position=positions[i],
                    opacity_raw=opacity_raw[i],
                    scale_raw=scale_raw[i],
                    rotation=rotation[i],
                    color=color[i],
                    object_id=anchor.object_id,
                    num_objects=self.num_objects
                )
                all_gaussians.append(gaussian)
        
        return all_gaussians
    
    def get_parameters_as_tensors(self):
        """
        Export all Gaussian parameters as tensors (for rendering/saving)
        
        Returns:
            dict with keys:
                pos: [N, 3] - positions
                opacity_raw: [N, 1] - raw opacity
                scale_raw: [N, 3] - raw scale
                rotation: [N, 4] - quaternions
                color: [N, 3] - RGB colors
                object_ids: [N] - object IDs
                semantics: [N, num_objects] - one-hot encodings
                num_gaussians: int
                num_anchors: int
                num_objects: int
        """
        gaussians = self.generate_gaussians()
        
        N = len(gaussians)
        
        pos = torch.stack([g.position for g in gaussians])  # [N, 3]
        opacity_raw = torch.stack([g.opacity_raw for g in gaussians]).unsqueeze(1)  # [N, 1]
        scale_raw = torch.stack([g.scale_raw for g in gaussians])  # [N, 3]
        rotation = torch.stack([g.rotation for g in gaussians])  # [N, 4]
        color = torch.stack([g.color for g in gaussians])  # [N, 3]
        object_ids = torch.tensor([g.object_id for g in gaussians], dtype=torch.long)  # [N]
        semantics = torch.stack([g.semantic for g in gaussians])  # [N, num_objects]
        
        return {
            'pos': pos.to(self.device),
            'opacity_raw': opacity_raw.to(self.device),
            'scale_raw': scale_raw.to(self.device),
            'rotation': rotation.to(self.device),
            'color': color.to(self.device),
            'object_ids': object_ids.to(self.device),
            'semantics': semantics.to(self.device),
            'num_gaussians': N,
            'num_anchors': len(self.anchors),
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
        print(f"Position:         {params['pos'].shape}")
        print(f"Opacity:          {params['opacity_raw'].shape}")
        print(f"Scale:            {params['scale_raw'].shape}")
        print(f"Rotation:         {params['rotation'].shape}")
        print(f"Color:            {params['color'].shape}")
        print(f"Semantics:        {params['semantics'].shape}")
        print("="*60 + "\n")


def initialize_from_pointcloud(point_cloud, colors, object_ids=None,
                               voxel_size=0.02, k=10):
    """
    Convenience function to create ObjectGS model from point cloud
    
    Args:
        point_cloud: [N, 3] numpy array
        colors: [N, 3] numpy array [0, 1]
        object_ids: [N] numpy array (or None for all 1s)
        voxel_size: float - voxel size in meters
        k: int - Gaussians per anchor
        
    Returns:
        ObjectGSModel instance
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