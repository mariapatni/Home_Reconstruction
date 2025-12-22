"""
Training script for ObjectGS

COMPLETE IMPLEMENTATION with:
- Semantic rendering and cross-entropy loss
- Integration with view-dependent rendering
- Anchor grow/prune scheduling
- Full export with opacity filtering
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
import pytz
import numpy as np
from typing import Optional, Dict, List, Tuple


# =============================================================================
# OUTPUT DIRECTORY MANAGER
# =============================================================================

class RunManager:
    """Manages output directories for training runs."""
    
    def __init__(self, base_dir: str, scene_name: str):
        self.base_dir = Path(base_dir)
        self.scene_name = scene_name.replace(" ", "_")  # Sanitize
        self.scene_dir = self.base_dir / self.scene_name
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_number = self._get_next_run_number()
        self.run_name = f"training_run_{self.run_number}"
        self.run_dir = self.scene_dir / self.run_name
        
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.final_outputs_dir = self.run_dir / "final_outputs"
        self.progress_renders_dir = self.run_dir / "progress_renders"
        self.logs_dir = self.run_dir / "logs"
        
        for d in [self.checkpoints_dir, self.final_outputs_dir, 
                  self.progress_renders_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.logs_dir / f"{self.run_name}.log"
    
    def _get_next_run_number(self) -> int:
        existing_runs = list(self.scene_dir.glob("training_run_*"))
        if not existing_runs:
            return 1
        numbers = []
        for run_path in existing_runs:
            try:
                num = int(run_path.name.split("_")[-1])
                numbers.append(num)
            except ValueError:
                continue
        return max(numbers, default=0) + 1
    
    def get_pst_timestamp(self) -> str:
        pst = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pst)
        date_str = now.strftime("%m/%d/%Y")
        time_str = now.strftime("%I:%M %p").lstrip("0").lower()
        return f"{date_str}        {time_str} PST"


# =============================================================================
# LOGGER SETUP
# =============================================================================

def setup_training_logger(run_manager: RunManager) -> logging.Logger:
    logger = logging.getLogger(f'ObjectGS_{run_manager.run_name}')
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
    
    fh = logging.FileHandler(run_manager.log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info("=" * 70)
    logger.info(f"Training Run Start: {run_manager.get_pst_timestamp()}")
    logger.info("=" * 70)
    logger.info(f"Scene: {run_manager.scene_name}")
    logger.info(f"Run: {run_manager.run_name}")
    logger.info(f"Output directory: {run_manager.run_dir}")
    logger.info("=" * 70)
    
    return logger


# =============================================================================
# GAUSSIAN TRAINER WITH SEMANTIC LOSS
# =============================================================================

class GaussianTrainer:
    """
    Trainer for ObjectGS with:
    - Semantic rendering and cross-entropy loss
    - View-dependent color rendering
    - Anchor grow/prune scheduling
    """
    
    def __init__(self, model, scene, scene_name: str, config: Optional[Dict] = None,
                 base_output_dir: str = "/workspace/Home_Reconstruction/outputs",
                 object_names: Optional[List[str]] = None):
        """
        Args:
            model: ObjectGSModel instance
            scene: Record3DScene instance
            scene_name: Name of the scene
            config: Training configuration dict
            base_output_dir: Base directory for outputs
            object_names: List of object names
        """
        # Setup run manager first
        self.run_manager = RunManager(base_output_dir, scene_name)
        self.logger = setup_training_logger(self.run_manager)
        
        self.model = model
        self.scene = scene
        
        # Store object names
        if object_names is not None:
            self.object_names = object_names
        elif hasattr(model, 'object_names'):
            self.object_names = model.object_names
        else:
            self.object_names = [f"object_{i}" for i in range(model.num_objects)]
            self.object_names[0] = "background"
        
        # Default config with all options
        self.config = {
            # Learning rates
            'lr': 0.001,
            'lr_position': 0.00016,
            'lr_feature': 0.0025,
            'lr_scaling': 0.005,
            
            # Training schedule
            'num_iterations': 30000,
            'save_interval': 5000,
            'test_interval': 500,
            'log_interval': 100,
            
            # Loss weights
            'lambda_ssim': 0.2,
            'lambda_vol': 0.00005,
            'lambda_scale': 1.0,
            'scale_threshold': 0.03,
            
            # === SEMANTIC LOSS (NEW) ===
            'use_semantic_loss': True,
            'lambda_semantic': 0.1,
            'semantic_loss_start': 500,  # Start semantic loss after N iterations
            
            # === DENSIFICATION (NEW) ===
            'use_densification': True,
            'densify_start': 1000,        # Start densification after N iterations
            'densify_interval': 1000,     # Densify every N iterations
            'densify_until': 25000,       # Stop densification after N iterations
            'densify_grad_threshold': 0.0002,
            'prune_opacity_threshold': 0.005,
            
            # Constraints
            'min_opacity': 0.001,
            'max_scale': 0.1,
        }
        
        if config:
            self.config.update(config)
        
        # Log config
        self.logger.info("Configuration:")
        for k, v in sorted(self.config.items()):
            self.logger.info(f"  {k}: {v}")
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Device: {self.device}")
        
        if self.device.type == 'cuda':
            self.logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"CUDA Memory: {mem_gb:.1f} GB")
        
        self.model.to(self.device)
        
        # Log semantic info
        self._log_semantic_info()
        
        # Setup cameras
        self.train_cameras = scene.getTrainCameras()
        self.test_cameras = scene.getTestCameras()
        self.logger.info(f"Train cameras: {len(self.train_cameras)}")
        self.logger.info(f"Test cameras: {len(self.test_cameras)}")
        
        # Check for semantic masks in training cameras
        self._check_semantic_masks()
        
        self._prepare_cameras()
        self._setup_optimizer()
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.995
        )
        
        self.current_iteration = 0
        self.losses_history = []
        
        self.logger.info("-" * 50)
        self.logger.info(f"Model: {self.model.num_anchors:,} anchors, "
                        f"{self.model.num_gaussians:,} Gaussians")
        self.logger.info("=" * 70)
    
    def _log_semantic_info(self):
        self.logger.info("-" * 50)
        self.logger.info("SEMANTIC INFORMATION:")
        self.logger.info(f"  Number of objects: {self.model.num_objects}")
        self.logger.info(f"  Semantic loss enabled: {self.config['use_semantic_loss']}")
        
        if self.config['use_semantic_loss']:
            self.logger.info(f"  Semantic loss weight: {self.config['lambda_semantic']}")
            self.logger.info(f"  Semantic loss starts at: {self.config['semantic_loss_start']}")
        
        for i, name in enumerate(self.object_names[:10]):
            count = (self.model.anchor_object_ids == i).sum().item()
            self.logger.info(f"    ID {i}: {name} ({count:,} anchors)")
        
        if len(self.object_names) > 10:
            self.logger.info(f"    ... and {len(self.object_names) - 10} more")
        self.logger.info("-" * 50)
    
    def _check_semantic_masks(self):
        """Check how many training cameras have semantic masks"""
        n_with_masks = sum(
            1 for cam in self.train_cameras 
            if hasattr(cam, 'object_mask') and cam.object_mask is not None
        )
        
        self.logger.info(f"Cameras with semantic masks: {n_with_masks}/{len(self.train_cameras)}")
        
        if self.config['use_semantic_loss'] and n_with_masks == 0:
            self.logger.warning("⚠️  Semantic loss enabled but no cameras have masks!")
            self.logger.warning("   Semantic loss will be skipped.")
    
    def _setup_optimizer(self):
        param_groups = self.model.get_optimizer_param_groups(self.config)
        self.optimizer = torch.optim.Adam(param_groups)
        
        self.logger.info(f"Optimizer: Adam with {len(param_groups)} param groups")
        for pg in param_groups:
            self.logger.info(f"  {pg['name']}: lr={pg['lr']}")
    
    def _prepare_cameras(self):
        """Cache camera data on GPU"""
        self.logger.info("Caching camera data on GPU...")
        
        for cam in self.train_cameras + self.test_cameras:
            # View matrix
            if not hasattr(cam, '_viewmat_gpu'):
                cam._viewmat_gpu = cam.get_opencv_viewmat().to(self.device).contiguous()
            
            # Intrinsics
            if not hasattr(cam, '_K_gpu'):
                cam._K_gpu = torch.tensor([
                    [cam.fx, 0, cam.cx],
                    [0, cam.fy, cam.cy],
                    [0, 0, 1]
                ], device=self.device, dtype=torch.float32).contiguous()
            
            # Ground truth image
            if not hasattr(cam, '_gt_image_gpu'):
                cam._gt_image_gpu = cam.original_image.to(self.device).contiguous()
            
            # Camera center (for view-dependent rendering)
            if not hasattr(cam, '_camera_center_gpu'):
                cam._camera_center_gpu = cam.camera_center.to(self.device).contiguous()
            
            # Semantic mask (if available)
            if hasattr(cam, 'object_mask') and cam.object_mask is not None:
                if not hasattr(cam, '_object_mask_gpu'):
                    cam._object_mask_gpu = cam.object_mask.to(self.device).long().contiguous()
        
        sample_gt = self.train_cameras[0]._gt_image_gpu
        self.logger.info(f"GT images: shape={list(sample_gt.shape)}, "
                        f"range=[{sample_gt.min():.3f}, {sample_gt.max():.3f}]")
    
    # =========================================================================
    # RENDERING
    # =========================================================================
    
    def render(self, camera, params: Optional[Dict] = None,
               return_info: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Render RGB image using gsplat CUDA rasterizer.
        
        Args:
            camera: Camera object
            params: Pre-computed Gaussian parameters (optional)
            return_info: Whether to return rasterization info
        
        Returns:
            rendered image [3, H, W], alpha [H, W], info dict (optional)
        """
        if params is None:
            # Use view-dependent rendering
            params = self.model.get_parameters_as_tensors(
                camera_center=camera._camera_center_gpu
            )
        
        means = params['pos']
        opacities = torch.sigmoid(params['opacity_raw']).squeeze(-1)
        scales = torch.exp(params['scale_raw'])
        quats = params['rotation']
        colors = params['color']
        
        renders, alphas, info = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=camera._viewmat_gpu.unsqueeze(0),
            Ks=camera._K_gpu[None],
            width=camera.image_width,
            height=camera.image_height,
            packed=False,
        )
        
        if return_info:
            return renders[0].permute(2, 0, 1), alphas[0], info
        return renders[0].permute(2, 0, 1), alphas[0], None
    
    def render_semantics(self, camera, params: Optional[Dict] = None) -> torch.Tensor:
        """
        Render semantic probability map via alpha blending of one-hot encodings.
        
        This implements Equation 5 from the paper:
        P(x) = Σ_k α_k · T_k · E_{i_k}
        
        Args:
            camera: Camera object
            params: Pre-computed Gaussian parameters (optional)
        
        Returns:
            semantic_probs: [H, W, num_objects] probability map
        """
        if params is None:
            params = self.model.get_parameters_as_tensors(
                camera_center=camera._camera_center_gpu
            )
        
        means = params['pos']
        opacities = torch.sigmoid(params['opacity_raw']).squeeze(-1)
        scales = torch.exp(params['scale_raw'])
        quats = params['rotation']
        semantics = params['semantics']  # [N, num_objects] one-hot
        
        # Render semantics as "colors" (N channels instead of 3)
        renders, _, _ = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=semantics,  # One-hot encodings as colors
            viewmats=camera._viewmat_gpu.unsqueeze(0),
            Ks=camera._K_gpu[None],
            width=camera.image_width,
            height=camera.image_height,
            packed=False,
        )
        
        # renders shape: [1, H, W, num_objects]
        return renders[0]  # [H, W, num_objects]
    
    # =========================================================================
    # LOSS FUNCTIONS
    # =========================================================================
    
    def compute_semantic_loss(self, semantic_render: torch.Tensor,
                               gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for semantic supervision.
        
        Implements Equation 7 from the paper:
        L_semantic = -Σ_x Σ_i 𝟙(ID'(x)=i) · log(P_i(x))
        
        Args:
            semantic_render: [H, W, num_objects] from render_semantics
            gt_mask: [H, W] integer object IDs
        
        Returns:
            Cross-entropy loss scalar
        """
        H, W, C = semantic_render.shape
        
        # Reshape for cross-entropy: [H*W, num_objects]
        pred = semantic_render.reshape(-1, C)
        
        # Add small epsilon for numerical stability
        pred = pred + 1e-8
        
        # Normalize to get probabilities (should already be ~normalized from alpha blending)
        pred = pred / (pred.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Log probabilities
        log_pred = torch.log(pred + 1e-8)
        
        # Target: [H*W]
        target = gt_mask.reshape(-1).long()
        
        # Clamp target to valid range
        target = target.clamp(0, C - 1)
        
        # Cross-entropy loss (NLL with log-softmax already applied conceptually)
        loss = F.nll_loss(log_pred, target, reduction='mean')
        
        return loss
    
    def compute_losses(self, camera, params: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for a single training step.
        
        Returns:
            Dictionary of individual loss terms
        """
        gt_image = camera._gt_image_gpu
        
        # Render RGB
        rendered, alpha, info = self.render(camera, params, return_info=True)
        
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
        
        # Scale regularization
        scale_threshold = self.config.get('scale_threshold', 0.03)
        scale_excess = torch.relu(scales - scale_threshold)
        scale_reg = (scale_excess ** 2).mean()
        
        losses = {
            'l1': l1_loss,
            'ssim': ssim_loss,
            'ssim_val': ssim_val,
            'vol': vol_loss,
            'scale_reg': scale_reg,
            'rendered': rendered,
            'alpha': alpha,
            'info': info,
        }
        
        # Semantic loss (if enabled and mask available)
        if (self.config['use_semantic_loss'] and 
            self.current_iteration >= self.config['semantic_loss_start'] and
            hasattr(camera, '_object_mask_gpu')):
            
            semantic_render = self.render_semantics(camera, params)
            semantic_loss = self.compute_semantic_loss(
                semantic_render, camera._object_mask_gpu
            )
            losses['semantic'] = semantic_loss
        else:
            losses['semantic'] = torch.tensor(0.0, device=self.device)
        
        return losses
    
    def compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine individual losses into total loss"""
        total = (
            losses['l1'] +
            self.config['lambda_ssim'] * losses['ssim'] +
            self.config['lambda_vol'] * losses['vol'] +
            self.config['lambda_scale'] * losses['scale_reg']
        )
        
        # Add semantic loss if present and non-zero
        if losses['semantic'].item() > 0:
            total = total + self.config['lambda_semantic'] * losses['semantic']
        
        return total
    
    # =========================================================================
    # TRAINING STEP
    # =========================================================================
    
    def train_step(self) -> Dict[str, float]:
        """Single training step"""
        # Sample random camera
        cam_idx = torch.randint(0, len(self.train_cameras), (1,)).item()
        camera = self.train_cameras[cam_idx]
        
        # Get parameters with view dependence
        params = self.model.get_parameters_as_tensors(
            camera_center=camera._camera_center_gpu
        )
        
        # Compute losses
        losses = self.compute_losses(camera, params)
        
        # Total loss
        total_loss = self.compute_total_loss(losses)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update gradient stats for densification (if enabled)
        if (self.config['use_densification'] and 
            self.current_iteration >= self.config['densify_start'] and
            self.current_iteration < self.config['densify_until']):
            
            # Get position gradients
            if self.model.anchor_offsets.grad is not None:
                # Approximate viewspace gradients from offset gradients
                offset_grads = self.model.anchor_offsets.grad
                grad_norms = offset_grads.reshape(-1, 3).norm(dim=-1)
                
                # Create visibility mask (all visible for simplicity)
                visibility = torch.ones(self.model.num_gaussians, 
                                        dtype=torch.bool, device=self.device)
                
                self.model.update_gradient_stats(
                    offset_grads.reshape(-1, 3),
                    visibility
                )
        
        # Optimizer step
        self.optimizer.step()
        
        # Return loss values
        return {
            'loss': total_loss.item(),
            'l1': losses['l1'].item(),
            'ssim': losses['ssim'].item(),
            'ssim_val': losses['ssim_val'].item(),
            'vol': losses['vol'].item(),
            'scale_reg': losses['scale_reg'].item(),
            'semantic': losses['semantic'].item(),
            'render_mean': losses['rendered'].mean().item(),
            'render_std': losses['rendered'].std().item(),
        }
    
    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================
    
    def train(self):
        """Main training loop"""
        self.logger.info("=" * 70)
        self.logger.info(f"STARTING TRAINING: {self.config['num_iterations']} iterations")
        self.logger.info("=" * 70)
        
        # Timing
        self._run_warmup()
        
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
            
            self.losses_history.append(losses)
            
            # Update progress bar
            desc = f"L1:{losses['l1']:.4f} SSIM:{losses['ssim_val']:.3f}"
            if losses['semantic'] > 0:
                desc += f" Sem:{losses['semantic']:.4f}"
            pbar.set_description(desc)
            
            # Learning rate scheduling
            if i % 100 == 0:
                self.scheduler.step()
            
            # Logging
            if i % self.config['log_interval'] == 0:
                self._log_iteration(i, losses)
            
            # Densification
            if (self.config['use_densification'] and
                i >= self.config['densify_start'] and
                i < self.config['densify_until'] and
                i % self.config['densify_interval'] == 0):
                
                self._run_densification()
            
            # Evaluation
            if i % self.config['test_interval'] == 0:
                self.evaluate(i, scale=0.5)
            
            # Checkpointing
            if i % self.config['save_interval'] == 0:
                self.save_checkpoint(i)
        
        pbar.close()
        
        # Final outputs
        self.logger.info("=" * 70)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("=" * 70)
        
        self.evaluate(self.config['num_iterations'], final=True, scale=0.5)
        self.save_checkpoint(self.config['num_iterations'], final=True)
        self.save_training_history()
        
        self.export_final_outputs()
        
        
        self.logger.info(f"All outputs saved to: {self.run_manager.run_dir}")
    
    def _run_warmup(self):
        """Run warmup iterations for timing"""
        self.logger.info("Running warm-up for timing...")
        import time
        
        torch.cuda.synchronize()
        start = time.time()
        _ = self.train_step()
        torch.cuda.synchronize()
        warmup_time = time.time() - start
        self.logger.info(f"Warm-up iteration: {warmup_time:.3f}s")
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = self.train_step()
        torch.cuda.synchronize()
        batch_time = (time.time() - start) / 10
        self.logger.info(f"Average iteration time: {batch_time:.4f}s ({1/batch_time:.1f} it/s)")
        self.logger.info("-" * 50)
    
    def _log_iteration(self, iteration: int, losses: Dict):
        """Log training progress"""
        msg = (f"[{iteration:5d}] Loss={losses['loss']:.4f} "
               f"L1={losses['l1']:.4f} SSIM={losses['ssim_val']:.3f} "
               f"Vol={losses['vol']:.6f} ScReg={losses['scale_reg']:.4f}")
        
        if losses['semantic'] > 0:
            msg += f" Sem={losses['semantic']:.4f}"
        
        self.logger.info(msg)
    
    def _run_densification(self):
        """Run anchor grow/prune"""
        self.logger.info(f"[{self.current_iteration}] Running densification...")

        stats = self.model.densify_and_prune(
            grad_threshold=self.config['densify_grad_threshold'],
            min_opacity=self.config['prune_opacity_threshold']
        )
        
        # ADD THIS: Log the results explicitly
        self.logger.info(
            f"  Anchors: {stats['anchors_before']:,} → {stats['anchors_after']:,} "
            f"(pruned {stats['pruned']:,}, grew {stats['grown']:,})"
        )
        
        
        
        # Recreate optimizer with new parameters
        if stats['pruned'] > 0 or stats['grown'] > 0:
            self.optimizer = self.model.get_optimizer_param_groups_after_densify(
                self.optimizer, self.config
            )
            
            # Recreate scheduler
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.995
            )
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    
    def evaluate(self, iteration: int, final: bool = False, scale: float = 0.5):
        """Evaluate on test set"""
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
                    comparison = torch.cat([rendered, gt], dim=2)
                    
                    if scale != 1.0:
                        comparison = F.interpolate(
                            comparison.unsqueeze(0),
                            scale_factor=scale,
                            mode='bilinear',
                            align_corners=False
                        )[0]
                    
                    if final:
                        save_path = self.run_manager.final_outputs_dir / 'final_comparison.png'
                    else:
                        save_path = self.run_manager.progress_renders_dir / f'iter_{iteration:06d}.png'
                    
                    save_image(comparison, save_path)
        
        avg_loss = sum(test_losses) / len(test_losses)
        self.logger.info(f"  Test L1: {avg_loss:.4f}")
        
        self.model.train()
        return avg_loss
    
    # =========================================================================
    # CHECKPOINTING
    # =========================================================================
    
    def save_checkpoint(self, iteration: int, final: bool = False):
        """Save model checkpoint"""
        if final:
            path = self.run_manager.final_outputs_dir / 'model_final.pth'
        else:
            path = self.run_manager.checkpoints_dir / f'checkpoint_{iteration:06d}.pth'
        
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'scene_name': self.run_manager.scene_name,
            'run_name': self.run_manager.run_name,
            'object_names': self.object_names,
            'num_objects': self.model.num_objects,
            'num_anchors': self.model.num_anchors,
            'num_gaussians': self.model.num_gaussians,
        }, path)
        
        self.logger.info(f"Saved checkpoint: {path}")
    
    def save_training_history(self):
        """Save training history to JSON"""
        history = {
            'losses': self.losses_history,
            'config': self.config,
            'scene_name': self.run_manager.scene_name,
            'run_name': self.run_manager.run_name,
            'object_names': self.object_names,
            'num_objects': self.model.num_objects,
            'final_num_anchors': self.model.num_anchors,
            'final_num_gaussians': self.model.num_gaussians,
        }
        
        path = self.run_manager.final_outputs_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Saved history: {path}")
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def export_final_outputs(self):
        """Export all final outputs"""
        self.logger.info("=" * 70)
        self.logger.info("EXPORTING FINAL OUTPUTS")
        self.logger.info("=" * 70)
        
        # 1. PLY with opacity filtering
        self.save_splat_ply()
        
        # 2. Semantic PLY
        self.save_semantic_ply()
        self.export_for_supersplat()
        
        # 3. Per-object PLYs
        self.export_all_objects_ply()
        
        # 4. Test renders
        render_dir = self.run_manager.final_outputs_dir / 'test_renders'
        render_dir.mkdir(exist_ok=True)
        self.render_test_sequence(output_dir=render_dir)
        
        # 5. Comparison grid
        self.create_comparison_grid(render_dir)
        
        # 6. Object decomposition render
        self.render_object_comparison()
        
        self.logger.info("=" * 70)
    
    def save_splat_ply(self, min_opacity: float = 0.01):
        """Export Gaussian splat to PLY with opacity filtering"""
        save_path = self.run_manager.final_outputs_dir / f'splat_iter{self.current_iteration:06d}.ply'
        
        self.logger.info(f"Exporting splat to PLY: {save_path}")
        self.model.eval()
        
        with torch.no_grad():
            params = self.model.get_parameters_as_tensors()
            
            pos = params['pos']
            opacity = torch.sigmoid(params['opacity_raw']).squeeze(-1)
            scales = torch.exp(params['scale_raw'])
            rotation = params['rotation']
            colors = params['color']
            
            # Filter by opacity
            mask = opacity > min_opacity
            
            pos = pos[mask].cpu().numpy()
            opacity = opacity[mask].cpu().numpy()
            scales = scales[mask].cpu().numpy()
            rotation = rotation[mask].cpu().numpy()
            colors = colors[mask].cpu().numpy()
            
            colors = np.clip(colors, 0, 1)
            colors_uint8 = (colors * 255).astype(np.uint8)
            
            num_gaussians = pos.shape[0]
            self.logger.info(f"  Exporting {num_gaussians:,} Gaussians (opacity > {min_opacity})")
        
        with open(save_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_gaussians}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property float opacity\n")
            f.write("property float scale_0\n")
            f.write("property float scale_1\n")
            f.write("property float scale_2\n")
            f.write("property float rot_0\n")
            f.write("property float rot_1\n")
            f.write("property float rot_2\n")
            f.write("property float rot_3\n")
            f.write("end_header\n")
            
            for i in range(num_gaussians):
                f.write(f"{pos[i,0]:.6f} {pos[i,1]:.6f} {pos[i,2]:.6f} ")
                f.write(f"{int(colors_uint8[i,0])} {int(colors_uint8[i,1])} {int(colors_uint8[i,2])} ")
                f.write(f"{opacity[i]:.6f} ")
                f.write(f"{scales[i,0]:.6f} {scales[i,1]:.6f} {scales[i,2]:.6f} ")
                f.write(f"{rotation[i,0]:.6f} {rotation[i,1]:.6f} {rotation[i,2]:.6f} {rotation[i,3]:.6f}\n")
        
        self.logger.info(f"Saved PLY: {save_path}")
        self.model.train()
    
    def save_semantic_ply(self, min_opacity: float = 0.01):
        """Export PLY with object IDs"""
        save_path = self.run_manager.final_outputs_dir / f'splat_semantic_iter{self.current_iteration:06d}.ply'
        
        self.model.eval()
        
        with torch.no_grad():
            params = self.model.get_parameters_as_tensors()
            
            pos = params['pos']
            opacity = torch.sigmoid(params['opacity_raw']).squeeze(-1)
            colors = params['color']
            object_ids = params['object_ids']
            
            # Filter by opacity
            mask = opacity > min_opacity
            
            pos = pos[mask].cpu().numpy()
            colors = colors[mask].cpu().numpy()
            object_ids = object_ids[mask].cpu().numpy()
            
            colors = np.clip(colors, 0, 1)
        
        # Save using open3d
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(save_path), pcd)
        
        # Save object IDs separately
        ids_path = save_path.parent / 'object_ids.npy'
        np.save(ids_path, object_ids.astype(np.int32))
        
        # Save mapping
        mapping = {
            'object_names': self.object_names,
            'num_objects': self.model.num_objects,
            'id_to_name': {i: name for i, name in enumerate(self.object_names)},
        }
        mapping_path = save_path.parent / 'object_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        self.logger.info(f"Saved semantic PLY: {save_path}")
        self.logger.info(f"Saved object IDs: {ids_path}")
        
        self.model.train()
    
    def export_all_objects_ply(self, min_opacity: float = 0.1):
        """Export each object as a separate PLY"""
        output_dir = self.run_manager.final_outputs_dir / 'objects'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting {self.model.num_objects} objects to {output_dir}")
        
        for obj_id in range(self.model.num_objects):
            # Check if any anchors have this ID
            count = (self.model.anchor_object_ids == obj_id).sum().item()
            if count == 0:
                continue
            
            name = self.object_names[obj_id] if obj_id < len(self.object_names) else f"object_{obj_id}"
            save_path = output_dir / f'{obj_id:02d}_{name.replace(" ", "_")}.ply'
            
            self.model.export_object_ply(obj_id, str(save_path), min_opacity=min_opacity)

    def export_for_supersplat(self, save_path=None, min_opacity=0.01):
        """
        Export Gaussian splat in format compatible with SuperSplat viewer.
        https://playcanvas.com/supersplat/editor
        """
        import struct
        
        if save_path is None:
            save_path = self.run_manager.final_outputs_dir / 'splat_supersplat.ply'
        else:
            save_path = Path(save_path)
        
        self.logger.info(f"Exporting SuperSplat-compatible PLY: {save_path}")
        self.model.eval()
        
        with torch.no_grad():
            params = self.model.get_parameters_as_tensors()
            
            pos = params['pos']
            opacity = torch.sigmoid(params['opacity_raw']).squeeze(-1)
            scales = torch.exp(params['scale_raw'])
            rotation = params['rotation']
            colors = params['color']
            
            # Filter low opacity
            mask = opacity > min_opacity
            
            pos = pos[mask].cpu().numpy().astype(np.float32)
            opacity = opacity[mask].cpu().numpy().astype(np.float32)
            scales = scales[mask].cpu().numpy().astype(np.float32)
            rotation = rotation[mask].cpu().numpy().astype(np.float32)
            colors = colors[mask].cpu().numpy().astype(np.float32)
            
            num_gaussians = len(pos)
            self.logger.info(f"  Exporting {num_gaussians:,} Gaussians (opacity > {min_opacity})")
        
        # Convert to 3DGS format
        SH_C0 = 0.28209479177387814
        sh_dc = ((colors - 0.5) / SH_C0).astype(np.float32)
        
        # Log-space scales
        log_scales = np.log(scales + 1e-8).astype(np.float32)
        
        # Logit-space opacity
        opacity_clamped = np.clip(opacity, 1e-6, 1 - 1e-6)
        opacity_logit = np.log(opacity_clamped / (1 - opacity_clamped)).astype(np.float32)
        
        # Normals (unused but required)
        normals = np.zeros((num_gaussians, 3), dtype=np.float32)
        
        # Build header with explicit line endings
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {num_gaussians}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]
        header = "\n".join(header_lines) + "\n"
        
        # Write file
        with open(save_path, 'wb') as f:
            # Write header as ASCII
            f.write(header.encode('ascii'))
            
            # Write binary data - all floats packed together per vertex
            for i in range(num_gaussians):
                # Pack all 17 floats for this vertex
                vertex_data = struct.pack(
                    '17f',
                    pos[i, 0], pos[i, 1], pos[i, 2],           # x, y, z
                    normals[i, 0], normals[i, 1], normals[i, 2],  # nx, ny, nz
                    sh_dc[i, 0], sh_dc[i, 1], sh_dc[i, 2],     # f_dc_0, f_dc_1, f_dc_2
                    opacity_logit[i],                           # opacity
                    log_scales[i, 0], log_scales[i, 1], log_scales[i, 2],  # scale_0, scale_1, scale_2
                    rotation[i, 0], rotation[i, 1], rotation[i, 2], rotation[i, 3],  # rot_0-3
                )
                f.write(vertex_data)
        
        self.logger.info(f"  Saved SuperSplat-compatible PLY: {save_path}")
        self.model.train()
        
        return save_path
    
    def render_test_sequence(self, output_dir: Optional[Path] = None, scale: float = 0.5):
        """Render all test views"""
        if output_dir is None:
            output_dir = self.run_manager.final_outputs_dir / 'test_renders'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        
        with torch.no_grad():
            for cam_idx, camera in enumerate(tqdm(self.test_cameras, desc="Rendering")):
                rendered, _, _ = self.render(camera)
                gt = camera._gt_image_gpu
                
                comparison = torch.cat([rendered, gt], dim=2)
                
                if scale != 1.0:
                    comparison = F.interpolate(
                        comparison.unsqueeze(0),
                        scale_factor=scale,
                        mode='bilinear',
                        align_corners=False
                    )[0]
                
                save_image(comparison, output_dir / f'test_{cam_idx:04d}_comparison.png')
        
        self.model.train()
        self.logger.info(f"Saved {len(self.test_cameras)} renders to {output_dir}")
    
    def create_comparison_grid(self, render_dir: Path):
        """Create grid of all test comparisons"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        import math
        
        image_files = sorted(render_dir.glob('test_*_comparison.png'))
        if len(image_files) == 0:
            return
        
        n = len(image_files)
        cols = min(n, 4)
        rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]
        
        for i, img_path in enumerate(image_files):
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f'Test {i}', fontsize=10)
            axes[i].axis('off')
        
        for i in range(len(image_files), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{self.run_manager.scene_name} - Rendered | GT', fontsize=12)
        plt.tight_layout()
        
        save_path = self.run_manager.final_outputs_dir / 'all_comparisons.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved comparison grid: {save_path}")
    
    def render_object_comparison(self, camera_idx: int = 0):
        """Render each object separately in a grid"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        camera = self.test_cameras[camera_idx]
        self.model.eval()
        
        renders = []
        labels = []
        
        with torch.no_grad():
            # Full scene
            full_render, _, _ = self.render(camera)
            renders.append(full_render.cpu())
            labels.append("Full Scene")
            
            # Each object
            for obj_id in range(min(self.model.num_objects, 12)):  # Limit to 12
                count = (self.model.anchor_object_ids == obj_id).sum().item()
                if count == 0:
                    continue
                
                params = self.model.get_parameters_as_tensors(
                    camera_center=camera._camera_center_gpu,
                    object_mask=[obj_id]
                )
                obj_render, _, _ = self.render(camera, params)
                renders.append(obj_render.cpu())
                
                name = self.object_names[obj_id] if obj_id < len(self.object_names) else f"obj_{obj_id}"
                labels.append(f"{name}\n({count * self.model.k:,})")
        
        # Create grid
        n = len(renders)
        cols = min(n, 5)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]
        
        for i, (img, label) in enumerate(zip(renders, labels)):
            img_np = img.permute(1, 2, 0).numpy()
            axes[i].imshow(np.clip(img_np, 0, 1))
            axes[i].set_title(label, fontsize=10)
            axes[i].axis('off')
        
        for i in range(len(renders), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"Object Decomposition - Camera {camera_idx}", fontsize=12)
        plt.tight_layout()
        
        save_path = self.run_manager.final_outputs_dir / f'object_decomposition_cam{camera_idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved object decomposition: {save_path}")
        self.model.train()