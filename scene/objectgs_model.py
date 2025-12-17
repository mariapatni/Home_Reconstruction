"""
ObjectGS: Anchor-based Gaussian Splatting with Object Awareness

FIXES APPLIED:
1. Color delta scale: 0.2 → 1.0 (line ~380)
2. Color MLP weight init: 0.01 → 0.1 (line ~200)
3. Opacity init: sigmoid(-2.2) ≈ 0.1 (line ~190)
4. Scale init: exp(-4) ≈ 0.018 (line ~185)

LOGGING: Comprehensive logging of all parameter statistics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
from datetime import datetime


# =============================================================================
# LOGGING SETUP
# =============================================================================

_LOGGER = None
_LOG_DIR = '/workspace/Home_Reconstruction/outputs'


def get_logger():
    """Get or create the shared logger"""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = setup_logger()
    return _LOGGER


def setup_logger(log_dir=None):
    """Setup logger for model and training diagnostics"""
    global _LOG_DIR, _LOGGER
    if log_dir:
        _LOG_DIR = log_dir
    
    os.makedirs(_LOG_DIR, exist_ok=True)
    
    logger = logging.getLogger('ObjectGS')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler - captures everything
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(_LOG_DIR, f'objectgs_training_{timestamp}.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler - info and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', 
                                  datefmt='%H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info("="*70)
    logger.info(f"LOG FILE: {log_file}")
    logger.info("="*70)
    
    _LOGGER = logger
    return logger


# =============================================================================
# ANCHOR CLASS
# =============================================================================

class Anchor(nn.Module):
    """Anchor point that generates k neural Gaussians"""
    
    def __init__(self, position, color, object_id, k=10, feature_dim=32):
        super().__init__()
        
        self.register_buffer('position', torch.tensor(position, dtype=torch.float32))
        self.register_buffer('color', torch.tensor(color, dtype=torch.float32))
        
        self.object_id = object_id
        self.k = k
        
        # Learnable parameters
        feature_init = torch.randn(feature_dim) * 0.1
        feature_init[:3] = torch.tensor(color, dtype=torch.float32)
        self.feature = nn.Parameter(feature_init)
        
        self.scaling = nn.Parameter(torch.tensor(1.0))
        self.offsets = nn.Parameter(torch.randn(k, 3) * 0.01)
        
    def get_gaussian_positions(self):
        return self.position.unsqueeze(0) + self.offsets * self.scaling


# =============================================================================
# ATTRIBUTE MLP
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
    
    def forward(self, anchor_feature):
        """
        Args:
            anchor_feature: [N, F] batched anchor features
        Returns:
            opacity_raw, scale_raw, rotation, color_delta
        """
        batch_size = anchor_feature.shape[0]
        
        opacity_raw = self.opacity_mlp(anchor_feature)
        
        scale_raw = self.scale_mlp(anchor_feature)
        scale_raw = scale_raw.reshape(batch_size, self.k, 3)
        
        rotation = self.rotation_mlp(anchor_feature)
        rotation = rotation.reshape(batch_size, self.k, 4)
        rotation = rotation / (rotation.norm(dim=-1, keepdim=True) + 1e-9)
        
        color = self.color_mlp(anchor_feature)
        color = color.reshape(batch_size, self.k, 3)
        color = torch.tanh(color) * 0.5 + 0.5  # Map to [0, 1]
        
        return opacity_raw, scale_raw, rotation, color
    
    def get_weight_stats(self):
        """Return dictionary of weight statistics for logging"""
        stats = {}
        for name, mlp in [('opacity', self.opacity_mlp), 
                          ('scale', self.scale_mlp),
                          ('rotation', self.rotation_mlp),
                          ('color', self.color_mlp)]:
            for i, layer in enumerate(mlp):
                if hasattr(layer, 'weight'):
                    w = layer.weight.data
                    stats[f'{name}_L{i}_w_mean'] = w.mean().item()
                    stats[f'{name}_L{i}_w_std'] = w.std().item()
                    stats[f'{name}_L{i}_w_absmax'] = w.abs().max().item()
                if hasattr(layer, 'bias') and layer.bias is not None:
                    b = layer.bias.data
                    stats[f'{name}_L{i}_b_mean'] = b.mean().item()
                    stats[f'{name}_L{i}_b_std'] = b.std().item()
        return stats
    
    def get_gradient_stats(self):
        """Return gradient statistics for logging"""
        stats = {}
        for name, mlp in [('opacity', self.opacity_mlp), 
                          ('scale', self.scale_mlp),
                          ('rotation', self.rotation_mlp),
                          ('color', self.color_mlp)]:
            for i, layer in enumerate(mlp):
                if hasattr(layer, 'weight') and layer.weight.grad is not None:
                    g = layer.weight.grad
                    stats[f'{name}_L{i}_w_grad_mean'] = g.mean().item()
                    stats[f'{name}_L{i}_w_grad_std'] = g.std().item()
                    stats[f'{name}_L{i}_w_grad_absmax'] = g.abs().max().item()
                if hasattr(layer, 'bias') and layer.bias is not None and layer.bias.grad is not None:
                    g = layer.bias.grad
                    stats[f'{name}_L{i}_b_grad_mean'] = g.mean().item()
        return stats


# =============================================================================
# MAIN MODEL
# =============================================================================

class ObjectGSModel(nn.Module):
    """
    ObjectGS Model with ALL FIXES APPLIED
    """
    
    def __init__(self, point_cloud, colors, object_ids=None, 
                 voxel_size=0.02, k=10, feature_dim=32):
        super().__init__()
        
        self.logger = get_logger()
        self.k = k
        self.feature_dim = feature_dim
        self.voxel_size = voxel_size
        self._call_count = 0
        
        if object_ids is None:
            object_ids = np.ones(len(point_cloud), dtype=np.int32)
        
        self.num_objects = int(object_ids.max()) + 1
        
        # Log initialization
        self.logger.info("="*70)
        self.logger.info("INITIALIZING ObjectGSModel")
        self.logger.info("="*70)
        self.logger.info(f"Input points: {len(point_cloud):,}")
        self.logger.info(f"Voxel size: {voxel_size}")
        self.logger.info(f"k (Gaussians/anchor): {k}")
        self.logger.info(f"Feature dim: {feature_dim}")
        
        # Log input statistics
        self.logger.info(f"Point cloud bounds:")
        self.logger.info(f"  X: [{point_cloud[:,0].min():.3f}, {point_cloud[:,0].max():.3f}]")
        self.logger.info(f"  Y: [{point_cloud[:,1].min():.3f}, {point_cloud[:,1].max():.3f}]")
        self.logger.info(f"  Z: [{point_cloud[:,2].min():.3f}, {point_cloud[:,2].max():.3f}]")
        self.logger.info(f"Input colors - mean: {colors.mean():.4f}, std: {colors.std():.4f}")
        self.logger.info(f"Input colors - R: {colors[:,0].mean():.3f}, G: {colors[:,1].mean():.3f}, B: {colors[:,2].mean():.3f}")
        
        # Create anchors
        self.anchors = self._create_anchors(point_cloud, colors, object_ids)
        self.logger.info(f"Created {len(self.anchors)} anchors → {len(self.anchors) * k:,} Gaussians")
        
        # Create MLP
        self.attribute_mlp = AttributeMLP(feature_dim=feature_dim, k=k)
        
        # =====================================================================
        # FIXED MLP INITIALIZATION
        # =====================================================================
        self.logger.info("-"*50)
        self.logger.info("APPLYING FIXED MLP INITIALIZATION")
        self.logger.info("-"*50)
        
        with torch.no_grad():
            # Scale: exp(-4) ≈ 0.018
            scale_out = self.attribute_mlp.scale_mlp[-1]
            scale_out.bias.data.fill_(-4.0)
            scale_out.weight.data *= 0.1
            self.logger.info(f"Scale: bias=-4.0 (exp=->{np.exp(-4):.4f}), weights*=0.1")
            
            # Opacity: sigmoid(-2.2) ≈ 0.10
            opacity_out = self.attribute_mlp.opacity_mlp[-1]
            opacity_out.bias.data.fill_(-2.2)
            opacity_out.weight.data *= 0.1
            self.logger.info(f"Opacity: bias=-2.2 (sig=->{1/(1+np.exp(2.2)):.4f}), weights*=0.1")
            
            # Color: FIXED - weights *= 0.1 (was 0.01)
            color_out = self.attribute_mlp.color_mlp[-1]
            color_out.weight.data *= 2.0  # FIX: was 0.01
            color_out.bias.data.zero_()
            self.logger.info(f"Color: weights*=2.0), bias=0")
            
            # Rotation: identity quaternion
            rot_out = self.attribute_mlp.rotation_mlp[-1]
            rot_out.bias.data.zero_()
            rot_out.bias.data[3::4] = 1.0
            self.logger.info(f"Rotation: identity quaternion [0,0,0,1]")
        
        # Pre-compute anchor data
        self._precompute_anchor_data()
        
        # Log anchor color statistics
        ac = self._anchor_colors.cpu().numpy()
        self.logger.info(f"Anchor colors - mean: {ac.mean():.4f}, std: {ac.std():.4f}")
        self.logger.info(f"Anchor colors - R: {ac[:,0].mean():.3f}±{ac[:,0].std():.3f}")
        self.logger.info(f"Anchor colors - G: {ac[:,1].mean():.3f}±{ac[:,1].std():.3f}")
        self.logger.info(f"Anchor colors - B: {ac[:,2].mean():.3f}±{ac[:,2].std():.3f}")
        
        # Log initial MLP weights
        self._log_mlp_weights("INIT")
        
        self.device = torch.device('cpu')
        self.logger.info("="*70)
    
    def _create_anchors(self, points, colors, object_ids):
        """Voxelize and create anchors"""
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        
        voxel_dict = {}
        for i, vidx in enumerate(voxel_indices):
            key = tuple(vidx)
            if key not in voxel_dict:
                voxel_dict[key] = {'colors': [], 'object_ids': []}
            voxel_dict[key]['colors'].append(colors[i])
            voxel_dict[key]['object_ids'].append(object_ids[i])
        
        # Log voxelization stats
        pts_per_voxel = [len(v['colors']) for v in voxel_dict.values()]
        self.logger.info(f"Voxelization: {len(voxel_dict)} voxels, "
                        f"pts/voxel: {np.mean(pts_per_voxel):.1f}±{np.std(pts_per_voxel):.1f}")
        
        anchors = []
        for vidx, data in voxel_dict.items():
            pos = (np.array(vidx) + 0.5) * self.voxel_size
            obj_id = int(np.bincount(data['object_ids']).argmax())
            avg_color = np.mean(data['colors'], axis=0)
            
            anchors.append(Anchor(
                position=pos, color=avg_color, object_id=obj_id,
                k=self.k, feature_dim=self.feature_dim
            ))
        
        return nn.ModuleList(anchors)
    
    def _precompute_anchor_data(self):
        """Pre-compute anchor colors and object IDs"""
        obj_ids = []
        colors = []
        for anchor in self.anchors:
            obj_ids.extend([anchor.object_id] * self.k)
            colors.extend([anchor.color] * self.k)
        
        self.register_buffer('_anchor_object_ids', torch.tensor(obj_ids, dtype=torch.long))
        self.register_buffer('_anchor_colors', torch.stack(colors))
    
    def to(self, device):
        self.device = device
        return super().to(device)
    
    def _log_mlp_weights(self, prefix):
        """Log MLP weight statistics"""
        stats = self.attribute_mlp.get_weight_stats()
        self.logger.debug(f"[{prefix}] MLP Weight Stats:")
        for key in ['opacity', 'scale', 'color', 'rotation']:
            w_mean = stats.get(f'{key}_L2_w_mean', 0)
            w_std = stats.get(f'{key}_L2_w_std', 0)
            b_mean = stats.get(f'{key}_L2_b_mean', 0)
            self.logger.debug(f"  {key:8s} out: w={w_mean:+.5f}±{w_std:.5f}, b={b_mean:+.4f}")
    
    def _log_mlp_gradients(self, prefix):
        """Log MLP gradient statistics"""
        stats = self.attribute_mlp.get_gradient_stats()
        if stats:
            self.logger.debug(f"[{prefix}] MLP Gradient Stats:")
            for key in ['opacity', 'scale', 'color', 'rotation']:
                g_mean = stats.get(f'{key}_L2_w_grad_mean', 0)
                g_std = stats.get(f'{key}_L2_w_grad_std', 0)
                g_max = stats.get(f'{key}_L2_w_grad_absmax', 0)
                if g_max > 0:
                    self.logger.debug(f"  {key:8s} grad: mean={g_mean:+.6f}, std={g_std:.6f}, max={g_max:.6f}")
    
    def get_parameters_as_tensors(self):
        """
        Get all Gaussian parameters
        
        FIX APPLIED: Color delta scale 0.2 → 1.0
        """
        self._call_count += 1
        
        num_anchors = len(self.anchors)
        
        # Stack anchor parameters
        anchor_features = torch.stack([a.feature for a in self.anchors])
        anchor_positions = torch.stack([a.position for a in self.anchors])
        anchor_offsets = torch.stack([a.offsets for a in self.anchors])
        anchor_scalings = torch.stack([a.scaling for a in self.anchors])
        
        # Positions
        positions = anchor_positions.unsqueeze(1) + anchor_offsets * anchor_scalings.unsqueeze(1).unsqueeze(2)
        positions = positions.reshape(-1, 3)
        
        # MLP forward
        opacity_raw, scale_raw, rotation, color_delta = self.attribute_mlp(anchor_features)
        
        # Flatten
        opacity_raw = opacity_raw.reshape(-1, 1)
        scale_raw = scale_raw.reshape(-1, 3)
        rotation = rotation.reshape(-1, 4)
        color_delta = color_delta.reshape(-1, 3)
        
        # =====================================================================
        # FIX: Color delta scale 0.2 → 1.0
        # =====================================================================
        # OLD: color = self._anchor_colors + (color_delta - 0.5) * 0.2
        # NEW:
        color = self._anchor_colors + (color_delta - 0.5) * 1.0
        color = torch.clamp(color, 0, 1)
        
        # Object IDs
        object_ids = self._anchor_object_ids
        semantics = F.one_hot(object_ids, num_classes=self.num_objects).float()
        
        return {
            'pos': positions,
            'opacity_raw': opacity_raw,
            'scale_raw': scale_raw,
            'rotation': rotation,
            'color': color,
            'color_delta': color_delta,
            'anchor_colors': self._anchor_colors,
            'anchor_features': anchor_features,
            'object_ids': object_ids,
            'semantics': semantics,
            'num_gaussians': positions.shape[0],
            'num_anchors': num_anchors,
            'num_objects': self.num_objects
        }
    
    def get_param_stats(self):
        """Get parameter statistics for logging"""
        with torch.no_grad():
            params = self.get_parameters_as_tensors()
            
            opacity = torch.sigmoid(params['opacity_raw']).squeeze()
            scale = torch.exp(params['scale_raw'])
            color = params['color']
            color_delta = params['color_delta']
            anchor_features = params['anchor_features']
            
            return {
                'opacity_mean': opacity.mean().item(),
                'opacity_std': opacity.std().item(),
                'opacity_min': opacity.min().item(),
                'opacity_max': opacity.max().item(),
                'scale_mean': scale.mean().item(),
                'scale_std': scale.std().item(),
                'scale_min': scale.min().item(),
                'scale_max': scale.max().item(),
                'color_mean': color.mean().item(),
                'color_std': color.std().item(),
                'color_r_mean': color[:,0].mean().item(),
                'color_g_mean': color[:,1].mean().item(),
                'color_b_mean': color[:,2].mean().item(),
                'color_delta_mean': color_delta.mean().item(),
                'color_delta_std': color_delta.std().item(),
                'anchor_feat_mean': anchor_features.mean().item(),
                'anchor_feat_std': anchor_features.std().item(),
            }
    
    def log_state(self, prefix="", level="info"):
        """Log current model state"""
        stats = self.get_param_stats()
        log_fn = self.logger.info if level == "info" else self.logger.debug
        log_fn(f"{prefix}Opacity: {stats['opacity_mean']:.4f}±{stats['opacity_std']:.4f} "
               f"[{stats['opacity_min']:.4f}, {stats['opacity_max']:.4f}]")
        log_fn(f"{prefix}Scale:   {stats['scale_mean']:.4f}±{stats['scale_std']:.4f} "
               f"[{stats['scale_min']:.4f}, {stats['scale_max']:.4f}]")
        log_fn(f"{prefix}Color:   {stats['color_mean']:.4f}±{stats['color_std']:.4f} "
               f"RGB=({stats['color_r_mean']:.3f},{stats['color_g_mean']:.3f},{stats['color_b_mean']:.3f})")
        log_fn(f"{prefix}ColorΔ:  {stats['color_delta_mean']:.4f}±{stats['color_delta_std']:.4f}")
        log_fn(f"{prefix}AnchorF: {stats['anchor_feat_mean']:.4f}±{stats['anchor_feat_std']:.4f}")
        return stats
    
    def save(self, path):
        """Save model"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'voxel_size': self.voxel_size,
                'k': self.k,
                'feature_dim': self.feature_dim,
                'num_objects': self.num_objects
            }
        }, path)
        self.logger.info(f"Saved model to {path}")