"""
Training script for ObjectGS

FIXES APPLIED:
1. NO color regularization (lambda_color forced to 0)
2. NO hard clamping on opacity/scale
3. CORRECT pruning logic (remove LOW opacity, not high)
4. SCALE REGULARIZATION with mean + max (catches outliers!)
5. Comprehensive logging
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import gsplat
from torchvision.utils import save_image
from pytorch_msssim import ssim as ssim_func
import json
import logging
import os
from datetime import datetime


# =============================================================================
# LOGGING SETUP
# =============================================================================

_TRAINER_LOGGER = None


def get_trainer_logger(log_dir='/workspace/outputs'):
    global _TRAINER_LOGGER
    if _TRAINER_LOGGER is None:
        _TRAINER_LOGGER = setup_trainer_logger(log_dir)
    return _TRAINER_LOGGER


def setup_trainer_logger(log_dir='/workspace/outputs'):
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('ObjectGS_Trainer')
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', 
                                  datefmt='%H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info("="*70)
    logger.info(f"LOG FILE: {log_file}")
    logger.info("="*70)
    
    return logger


# =============================================================================
# TRAINER CLASS
# =============================================================================

class GaussianTrainer:
    """
    Trainer for ObjectGS with all fixes applied.
    """
    
    def __init__(self, model, scene, config=None):
        self.model = model
        self.scene = scene
        
        # Default config
        self.config = {
            'lr': 0.001,
            'lr_position': 0.00016,
            'lr_feature': 0.005,
            'lr_opacity': 0.05,
            'lr_scaling': 0.005,
            'num_iterations': 5000,
            'save_interval': 1000,
            'test_interval': 100,
            'log_interval': 50,
            'detailed_log_interval': 200,
            'checkpoint_dir': '/workspace/outputs/checkpoints',
            'output_dir': '/workspace/outputs/renders',
            'prune_interval': 1000,
            'prune_opacity_threshold': 0.005,
            'prune_scale_threshold': 0.5,
            'lambda_ssim': 0.2,
            'lambda_vol': 0.00001,
            'lambda_scale': 10.0,
            'scale_threshold': 0.03,
        }
        
        if config:
            self.config.update(config)
        
        # Setup logger
        self.logger = get_trainer_logger(self.config.get('output_dir', '/workspace/outputs'))
        
        # FORCE disable color regularization
        if 'lambda_color' in self.config:
            self.logger.warning(f"lambda_color={self.config['lambda_color']} FORCED to 0!")
            self.config['lambda_color'] = 0.0
        
        self.logger.info("="*70)
        self.logger.info("INITIALIZING GaussianTrainer")
        self.logger.info("="*70)
        self.logger.info("FIXES APPLIED:")
        self.logger.info("  1. NO color regularization")
        self.logger.info("  2. NO hard clamping")
        self.logger.info("  3. CORRECT pruning (remove LOW opacity)")
        self.logger.info("  4. SCALE REG with mean+max (catches outliers)")
        self.logger.info("-"*50)
        self.logger.info("Config:")
        for k, v in self.config.items():
            self.logger.info(f"  {k}: {v}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Device: {self.device}")
        self.model.to(self.device)
        
        self.train_cameras = scene.getTrainCameras()
        self.test_cameras = scene.getTestCameras()
        self.logger.info(f"Train cameras: {len(self.train_cameras)}")
        self.logger.info(f"Test cameras: {len(self.test_cameras)}")
        
        self._prepare_cameras()
        self._setup_optimizer()
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.995
        )
        
        Path(self.config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)
        
        self.current_iteration = 0
        self.losses_history = []
        self.param_history = []
        
        self.logger.info("-"*50)
        self.logger.info("INITIAL MODEL STATE:")
        self._log_model_state("  ")
        self.logger.info("="*70)
    
    def _setup_optimizer(self):
        all_features = []
        all_scalings = []
        all_offsets = []
        
        for anchor in self.model.anchors:
            all_features.append(anchor.feature)
            all_scalings.append(anchor.scaling)
            all_offsets.append(anchor.offsets)
        
        self.optimizer = torch.optim.Adam([
            {'params': all_features, 'lr': self.config['lr_feature'], 'name': 'features'},
            {'params': all_scalings, 'lr': self.config['lr_scaling'], 'name': 'scalings'},
            {'params': all_offsets, 'lr': self.config['lr_position'], 'name': 'offsets'},
            {'params': self.model.attribute_mlp.parameters(), 'lr': self.config['lr'], 'name': 'mlp'}
        ])
        
        self.logger.info(f"Optimizer: Adam with {len(self.optimizer.param_groups)} param groups")
        for pg in self.optimizer.param_groups:
            self.logger.info(f"  {pg['name']}: lr={pg['lr']}")
    
    def _prepare_cameras(self):
        self.logger.info("Caching camera matrices...")
        
        for cam in self.train_cameras + self.test_cameras:
            if not hasattr(cam, '_viewmat_gpu'):
                cam._viewmat_gpu = cam.get_opencv_viewmat().to(self.device).contiguous()
            if not hasattr(cam, '_K_gpu'):
                cam._K_gpu = torch.tensor([
                    [cam.fx, 0, cam.cx],
                    [0, cam.fy, cam.cy],
                    [0, 0, 1]
                ], device=self.device, dtype=torch.float32).contiguous()
            if not hasattr(cam, '_gt_image_gpu'):
                cam._gt_image_gpu = cam.original_image.to(self.device).contiguous()
        
        sample_gt = self.train_cameras[0]._gt_image_gpu
        self.logger.info(f"GT images: shape={list(sample_gt.shape)}, "
                        f"range=[{sample_gt.min():.3f}, {sample_gt.max():.3f}], "
                        f"mean={sample_gt.mean():.3f}")
    
    def _get_param_stats(self):
        with torch.no_grad():
            params = self.model.get_parameters_as_tensors()
            
            opacity = torch.sigmoid(params['opacity_raw']).squeeze()
            scale = torch.exp(params['scale_raw'])
            color = params['color']
            color_delta = params.get('color_delta', None)
            
            stats = {
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
                'num_gaussians': params['num_gaussians'],
            }
            
            if color_delta is not None:
                stats['color_delta_mean'] = color_delta.mean().item()
                stats['color_delta_std'] = color_delta.std().item()
            
            return stats
    
    def _log_model_state(self, prefix=""):
        stats = self._get_param_stats()
        self.logger.info(f"{prefix}Opacity: {stats['opacity_mean']:.4f}±{stats['opacity_std']:.4f} "
                        f"[{stats['opacity_min']:.4f}, {stats['opacity_max']:.4f}]")
        self.logger.info(f"{prefix}Scale:   {stats['scale_mean']:.5f}±{stats['scale_std']:.5f} "
                        f"[{stats['scale_min']:.5f}, {stats['scale_max']:.5f}]")
        self.logger.info(f"{prefix}Color:   mean={stats['color_mean']:.4f}, std={stats['color_std']:.4f}")
        self.logger.info(f"{prefix}Color RGB: ({stats['color_r_mean']:.3f}, {stats['color_g_mean']:.3f}, {stats['color_b_mean']:.3f})")
        if 'color_delta_mean' in stats:
            self.logger.info(f"{prefix}ColorΔ:  {stats['color_delta_mean']:.4f}±{stats['color_delta_std']:.4f}")
        return stats
    
    def _get_mlp_stats(self):
        stats = {}
        mlp = self.model.attribute_mlp
        
        for name, submlp in [('opacity', mlp.opacity_mlp), 
                             ('scale', mlp.scale_mlp),
                             ('rotation', mlp.rotation_mlp),
                             ('color', mlp.color_mlp)]:
            for i, layer in enumerate(submlp):
                if hasattr(layer, 'weight'):
                    w = layer.weight.data
                    stats[f'{name}_L{i}_w_mean'] = w.mean().item()
                    stats[f'{name}_L{i}_w_std'] = w.std().item()
                    stats[f'{name}_L{i}_w_absmax'] = w.abs().max().item()
                    
                    if layer.weight.grad is not None:
                        g = layer.weight.grad
                        stats[f'{name}_L{i}_grad_mean'] = g.mean().item()
                        stats[f'{name}_L{i}_grad_std'] = g.std().item()
                        stats[f'{name}_L{i}_grad_absmax'] = g.abs().max().item()
        
        return stats
    
    def render(self, camera, params=None):
        """Render using gsplat - NO CLAMPING"""
        if params is None:
            params = self.model.get_parameters_as_tensors()
        
        means = params['pos']
        opacities = torch.sigmoid(params['opacity_raw']).squeeze(-1)
        scales = torch.exp(params['scale_raw'])
        quats = params['rotation']
        colors = params['color']
        
        viewmat = camera._viewmat_gpu
        K = camera._K_gpu
        
        renders, alphas, info = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K[None],
            width=camera.image_width,
            height=camera.image_height,
            packed=False,
        )
        
        return renders[0].permute(2, 0, 1), alphas[0], info
    
    def prune_gaussians(self):
        """CORRECT pruning: remove LOW opacity and LARGE scales"""
        with torch.no_grad():
            params = self.model.get_parameters_as_tensors()
            
            opacities = torch.sigmoid(params['opacity_raw']).squeeze()
            scales = torch.exp(params['scale_raw'])
            max_scales = scales.max(dim=-1)[0]
            
            valid_mask = (opacities > self.config['prune_opacity_threshold']) & \
                        (max_scales < self.config['prune_scale_threshold'])
            
            num_invalid = (~valid_mask).sum().item()
            
            self.logger.info(f"Pruning: {num_invalid} invalid Gaussians "
                           f"(op<{self.config['prune_opacity_threshold']} or "
                           f"sc>{self.config['prune_scale_threshold']})")
            
            if num_invalid == 0:
                return 0
            
            k = self.model.k
            num_anchors = len(self.model.anchors)
            
            anchors_to_remove = []
            for i in range(num_anchors):
                start_idx = i * k
                end_idx = start_idx + k
                anchor_valid = valid_mask[start_idx:end_idx].sum()
                
                if anchor_valid < k * 0.5:
                    anchors_to_remove.append(i)
            
            if len(anchors_to_remove) > 0:
                kept = [a for i, a in enumerate(self.model.anchors) if i not in anchors_to_remove]
                self.model.anchors = torch.nn.ModuleList(kept)
                self.model._precompute_anchor_data()
                
                self._setup_optimizer()
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=0.995
                )
                
                self.logger.info(f"Pruned {len(anchors_to_remove)} anchors "
                               f"({len(self.model.anchors)} remaining)")
                return len(anchors_to_remove)
            
            return 0
    
    def train_step(self):
        """Single training step with scale regularization"""
        cam_idx = torch.randint(0, len(self.train_cameras), (1,)).item()
        camera = self.train_cameras[cam_idx]
        gt_image = camera._gt_image_gpu
        
        params = self.model.get_parameters_as_tensors()
        rendered, alpha, info = self.render(camera, params)
        
        # =====================================================================
        # LOSSES
        # =====================================================================
        
        # L1 loss
        l1_loss = F.l1_loss(rendered, gt_image)
        
        # SSIM loss
        ssim_val = ssim_func(
            rendered.unsqueeze(0), gt_image.unsqueeze(0),
            data_range=1.0, size_average=True
        )
        ssim_loss = 1.0 - ssim_val
        
        # Volume regularization
        opacities = torch.sigmoid(params['opacity_raw']).squeeze(-1)
        scales = torch.exp(params['scale_raw'])
        volumes = scales.prod(dim=-1) * opacities
        vol_loss = volumes.mean()
        
        # =====================================================================
        # SCALE REGULARIZATION - KEY FIX!
        # Uses BOTH mean AND max to catch outliers
        # =====================================================================
        scale_threshold = self.config.get('scale_threshold', 0.03)
        lambda_scale = self.config.get('lambda_scale', 10.0)
        
        scale_excess = torch.relu(scales - scale_threshold)
        scale_reg_mean = (scale_excess ** 2).mean()
        scale_reg_max = (scale_excess ** 2).max()
        scale_reg = scale_reg_mean + scale_reg_max  # Max catches outliers!
        
        # =====================================================================
        # TOTAL LOSS (no color regularization!)
        # =====================================================================
        loss = (l1_loss + 
                self.config['lambda_ssim'] * ssim_loss + 
                self.config['lambda_vol'] * vol_loss + 
                lambda_scale * scale_reg)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Record losses
        losses = {
            'loss': loss.item(),
            'l1': l1_loss.item(),
            'ssim': ssim_loss.item(),
            'ssim_val': ssim_val.item(),
            'vol': vol_loss.item(),
            'scale_reg': scale_reg.item(),
            'scale_reg_max': scale_reg_max.item(),
        }
        
        with torch.no_grad():
            losses['render_mean'] = rendered.mean().item()
            losses['render_std'] = rendered.std().item()
            losses['render_min'] = rendered.min().item()
            losses['render_max'] = rendered.max().item()
            losses['alpha_mean'] = alpha.mean().item()
        
        self.losses_history.append(losses)
        return losses
    
    def log_detailed_state(self, iteration):
        self.logger.info(f"[Iter {iteration}] DETAILED STATE:")
        
        stats = self._get_param_stats()
        self.param_history.append({'iteration': iteration, **stats})
        
        self.logger.info(f"  Parameters:")
        self.logger.info(f"    Opacity: {stats['opacity_mean']:.4f}±{stats['opacity_std']:.4f} "
                        f"[{stats['opacity_min']:.4f}, {stats['opacity_max']:.4f}]")
        self.logger.info(f"    Scale:   {stats['scale_mean']:.5f}±{stats['scale_std']:.5f} "
                        f"[{stats['scale_min']:.5f}, {stats['scale_max']:.5f}]")
        self.logger.info(f"    Color:   mean={stats['color_mean']:.4f}, std={stats['color_std']:.4f}")
        if 'color_delta_mean' in stats:
            self.logger.info(f"    ColorΔ:  {stats['color_delta_mean']:.4f}±{stats['color_delta_std']:.4f}")
        
        mlp_stats = self._get_mlp_stats()
        self.logger.info(f"  MLP Output Weights:")
        for key in ['opacity', 'scale', 'color', 'rotation']:
            w_mean = mlp_stats.get(f'{key}_L2_w_mean', 0)
            w_std = mlp_stats.get(f'{key}_L2_w_std', 0)
            self.logger.info(f"    {key:8s}: w={w_mean:+.5f}±{w_std:.5f}")
        
        has_grads = any('grad' in k for k in mlp_stats.keys())
        if has_grads:
            self.logger.info(f"  MLP Gradients:")
            for key in ['opacity', 'scale', 'color', 'rotation']:
                g_mean = mlp_stats.get(f'{key}_L2_grad_mean', 0)
                g_max = mlp_stats.get(f'{key}_L2_grad_absmax', 0)
                if g_max > 0:
                    self.logger.info(f"    {key:8s}: mean={g_mean:+.6f}, max={g_max:.6f}")
        
        self._check_for_problems(stats, iteration)
    
    def _check_for_problems(self, stats, iteration):
        problems = []
        
        if stats['color_std'] < 0.05:
            problems.append(f"COLOR COLLAPSE: std={stats['color_std']:.4f}")
        if stats['opacity_mean'] < 0.02:
            problems.append(f"OPACITY COLLAPSE: mean={stats['opacity_mean']:.4f}")
        if stats['opacity_max'] < 0.1:
            problems.append(f"ALL OPACITY LOW: max={stats['opacity_max']:.4f}")
        if stats['scale_max'] > 0.1:
            problems.append(f"SCALE EXPLOSION: max={stats['scale_max']:.4f}")
        if stats['scale_mean'] > 0.05:
            problems.append(f"SCALES GROWING: mean={stats['scale_mean']:.4f}")
        if 'color_delta_mean' in stats:
            if abs(stats['color_delta_mean'] - 0.5) < 0.01 and stats['color_delta_std'] < 0.05:
                problems.append(f"COLOR DELTA STUCK")
        
        for p in problems:
            self.logger.warning(f"  ⚠️  {p}")
        
        return problems
    
    def train(self):
        self.logger.info("="*70)
        self.logger.info(f"STARTING TRAINING: {self.config['num_iterations']} iterations")
        self.logger.info("="*70)
        
        self.log_detailed_state(0)
        
        pbar = tqdm(range(1, self.config['num_iterations'] + 1))
        
        for i in pbar:
            self.current_iteration = i
            
            try:
                losses = self.train_step()
            except Exception as e:
                self.logger.error(f"Error at iteration {i}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
            
            pbar.set_description(f"L:{losses['loss']:.4f} L1:{losses['l1']:.4f}")
            
            if i % self.config['prune_interval'] == 0 and i > 500:
                self.prune_gaussians()
            
            if i % 100 == 0:
                self.scheduler.step()
            
            # Regular logging - NOW WITH ScReg!
            if i % self.config['log_interval'] == 0:
                stats = self._get_param_stats()
                
                self.logger.info(
                    f"[{i:5d}] Loss={losses['loss']:.4f} L1={losses['l1']:.4f} "
                    f"SSIM={losses['ssim_val']:.3f} ScReg={losses['scale_reg']:.4f} | "
                    f"Gauss={stats['num_gaussians']:,} | "
                    f"Op={stats['opacity_mean']:.3f} Sc={stats['scale_mean']:.4f}/{stats['scale_max']:.4f} "
                    f"Col={stats['color_std']:.3f} | "
                    f"Rend={losses['render_mean']:.3f}±{losses['render_std']:.3f}"
                )
            
            if i % self.config['detailed_log_interval'] == 0:
                self.log_detailed_state(i)
            
            if i % self.config['test_interval'] == 0:
                self.evaluate(i)
            
            if i % self.config['save_interval'] == 0:
                self.save_checkpoint(i)
        
        pbar.close()
        
        self.logger.info("="*70)
        self.logger.info("TRAINING COMPLETE")
        self.log_detailed_state(self.config['num_iterations'])
        self.evaluate(self.config['num_iterations'])
        self.save_checkpoint(self.config['num_iterations'])
        self.save_training_history()
        self.logger.info("="*70)
    
    def evaluate(self, iteration):
        self.logger.info(f"[{iteration}] Evaluating...")
        
        self.model.eval()
        test_losses = []
        
        with torch.no_grad():
            for cam_idx, camera in enumerate(self.test_cameras[:5]):
                gt = camera._gt_image_gpu
                rendered, _, _ = self.render(camera)
                
                l1 = F.l1_loss(rendered, gt).item()
                test_losses.append(l1)
                
                if cam_idx == 0:
                    save_path = Path(self.config['output_dir']) / f'iter_{iteration:06d}.png'
                    save_image(rendered, save_path)
                    
                    self.logger.info(f"  Render: mean={rendered.mean():.3f}, "
                                   f"std={rendered.std():.3f}, "
                                   f"range=[{rendered.min():.3f}, {rendered.max():.3f}]")
                    self.logger.info(f"  GT:     mean={gt.mean():.3f}, std={gt.std():.3f}")
        
        avg_loss = sum(test_losses) / len(test_losses)
        self.logger.info(f"  Test L1: {avg_loss:.4f}")
        
        self.model.train()
    
    def save_checkpoint(self, iteration):
        path = Path(self.config['checkpoint_dir']) / f'checkpoint_{iteration:06d}.pth'
        
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }, path)
        
        self.logger.info(f"Saved checkpoint: {path}")
    
    def save_training_history(self):
        history = {
            'losses': self.losses_history,
            'params': self.param_history,
            'config': self.config,
        }
        
        path = Path(self.config['output_dir']) / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Saved history: {path}")