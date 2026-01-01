"""
Training script for ObjectGS - Instance Segmentation Version

Features:
- Uses ObjectGSModel with instance-aware voxelization
- Semantic rendering via gsplat (RGB + one-hot concatenated)
- Cross-entropy semantic loss per the ObjectGS paper
- Organized output structure: outputs/{scene}/training_run_{n}/
- PST timestamps in logs
- gsplat CUDA rasterizer
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


# =============================================================================
# OUTPUT DIRECTORY MANAGER
# =============================================================================

class RunManager:
    """
    Manages output directories for training runs.
    
    Structure:
        {base_dir}/{scene_name}/
            training_run_1/
                checkpoints/
                final_outputs/
                progress_renders/
                logs/
                    training_run_1.log
            training_run_2/
                ...
    """
    
    def __init__(self, base_dir: str, scene_name: str):
        self.base_dir = Path(base_dir)
        self.scene_name = scene_name
        self.scene_dir = self.base_dir / scene_name
        
        # Create scene directory if needed
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine run number
        self.run_number = self._get_next_run_number()
        self.run_name = f"training_run_{self.run_number}"
        self.run_dir = self.scene_dir / self.run_name
        
        # Create subdirectories
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.final_outputs_dir = self.run_dir / "final_outputs"
        self.progress_renders_dir = self.run_dir / "progress_renders"
        self.logs_dir = self.run_dir / "logs"
        
        for d in [self.checkpoints_dir, self.final_outputs_dir, 
                  self.progress_renders_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Log file path
        self.log_file = self.logs_dir / f"{self.run_name}.log"
    
    def _get_next_run_number(self) -> int:
        """Count existing runs and return next number"""
        existing_runs = list(self.scene_dir.glob("training_run_*"))
        if not existing_runs:
            return 1
        
        # Extract numbers from existing run names
        numbers = []
        for run_path in existing_runs:
            try:
                num = int(run_path.name.split("_")[-1])
                numbers.append(num)
            except ValueError:
                continue
        
        return max(numbers, default=0) + 1
    
    def get_pst_timestamp(self) -> str:
        """Get formatted PST timestamp"""
        pst = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pst)
        
        # Format: "12/10/2025        3:21 pm PST"
        date_str = now.strftime("%m/%d/%Y")
        time_str = now.strftime("%I:%M %p").lstrip("0").lower()
        
        # Add spacing between date and time
        return f"{date_str}        {time_str} PST"
    
    def __repr__(self):
        return f"RunManager(scene={self.scene_name}, run={self.run_number})"


# =============================================================================
# LOGGER SETUP
# =============================================================================

def setup_training_logger(run_manager: RunManager) -> logging.Logger:
    """Setup logger with PST timestamp header"""
    
    logger = logging.getLogger(f'ObjectGS_{run_manager.run_name}')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(run_manager.log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # Write header with PST timestamp
    logger.info("=" * 70)
    logger.info(f"Training Run Start: {run_manager.get_pst_timestamp()}")
    logger.info("=" * 70)
    logger.info(f"Scene: {run_manager.scene_name}")
    logger.info(f"Run: {run_manager.run_name}")
    logger.info(f"Output directory: {run_manager.run_dir}")
    logger.info("=" * 70)
    
    return logger


# =============================================================================
# TRAINER CLASS
# =============================================================================

class GaussianTrainer:
    """
    Trainer for ObjectGS with instance segmentation support.
    
    Key features:
    - Renders RGB + one-hot semantics together via gsplat
    - Computes cross-entropy semantic loss against GT masks
    - Supports variable number of instances per scene
    """
    
    def __init__(self, model, scene, scene_name: str, config=None, 
                 base_output_dir: str = "/workspace/Home_Reconstruction/outputs"):
        
        # Setup run manager first
        self.run_manager = RunManager(base_output_dir, scene_name)
        
        # Setup logger
        self.logger = setup_training_logger(self.run_manager)
        
        self.model = model
        self.scene = scene
        
        # Default config with semantic loss
        self.config = {
            'lr': 0.001,
            'lr_position': 0.00016,
            'lr_feature': 0.0025,
            'lr_opacity': 0.1,
            'lr_scaling': 0.005,
            'num_iterations': 5000,
            'save_interval': 1000,
            'test_interval': 100,
            'log_interval': 100,
            'detailed_log_interval': 500,
            'prune_interval': 500,
            'prune_opacity_threshold': 0.005,
            'prune_scale_threshold': 0.1,
            'lambda_ssim': 0.2,
            'lambda_vol': 1e-05,
            'lambda_scale': 10.0,
            'lambda_semantic': 0.1,  # Semantic loss weight (paper uses 0.01-0.1)
            'scale_threshold': 0.03,
            'min_opacity': 0.001,
            'max_opacity': 0.999,
            'max_scale': 0.1,
        }
        
        if config:
            self.config.update(config)
        
        # Log config
        self.logger.info("Configuration:")
        for k, v in self.config.items():
            self.logger.info(f"  {k}: {v}")
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Device: {self.device}")
        
        # Check CUDA availability
        if self.device.type == 'cuda':
            self.logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.model.to(self.device)
        
        # Log semantic info
        self.logger.info(f"Number of instances: {self.model.num_objects}")
        self.logger.info(f"One-hot semantic dimension: {self.model.num_objects}")
        
        # Setup cameras
        self.train_cameras = scene.getTrainCameras()
        self.test_cameras = scene.getTestCameras()
        self.logger.info(f"Train cameras: {len(self.train_cameras)}")
        self.logger.info(f"Test cameras: {len(self.test_cameras)}")
        
        self._prepare_cameras()
        self._setup_optimizer()
        
        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.995
        )
        
        # History tracking
        self.current_iteration = 0
        self.losses_history = []
        
        self.logger.info("-" * 50)
        self.logger.info(f"Model: {self.model.num_anchors:,} anchors, {self.model.num_gaussians:,} Gaussians")
        self.logger.info("=" * 70)
    
    def _setup_optimizer(self):
        """Setup optimizer using model's parameter groups"""
        param_groups = self.model.get_optimizer_param_groups(self.config)
        self.optimizer = torch.optim.Adam(param_groups)
        
        self.logger.info(f"Optimizer: Adam with {len(param_groups)} param groups")
        for pg in param_groups:
            self.logger.info(f"  {pg['name']}: lr={pg['lr']}")
    
    def _prepare_cameras(self):
        """Cache camera data on GPU, including GT masks for semantic supervision"""
        self.logger.info("Caching camera data on GPU...")
        
        cameras_with_masks = 0
        cameras_without_masks = 0
        
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
            
            # Cache GT semantic masks for semantic loss
            if not hasattr(cam, '_gt_mask_gpu'):
                if hasattr(cam, 'object_mask') and cam.object_mask is not None:
                    cam._gt_mask_gpu = cam.object_mask.to(self.device).long().contiguous()
                else:
                    cam._gt_mask_gpu = None
            
            # Count masks (separate from caching logic)
            if cam._gt_mask_gpu is not None:
                cameras_with_masks += 1
            else:
                cameras_without_masks += 1
        
        sample_gt = self.train_cameras[0]._gt_image_gpu
        self.logger.info(f"GT images: shape={list(sample_gt.shape)}, "
                        f"range=[{sample_gt.min():.3f}, {sample_gt.max():.3f}]")
        self.logger.info(f"Cameras with semantic masks: {cameras_with_masks}")
        if cameras_without_masks > 0:
            self.logger.warning(f"Cameras WITHOUT semantic masks: {cameras_without_masks} (semantic loss will be skipped)")
        
    def render(self, camera, params=None):
        """
        Render using gsplat CUDA rasterizer.
        
        Returns RGB image only (use render_with_semantics for full output).
        """
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
    
    def render_with_semantics(self, camera, params=None):
        """
        Render RGB and semantics together via gsplat.
        
        This concatenates RGB (3 channels) with one-hot semantics (num_objects channels)
        and renders them all in a single pass. gsplat handles arbitrary feature dimensions.
        
        The alpha blending produces:
        - RGB: blended colors
        - Semantics: probability distribution over instances (per the ObjectGS paper)
        
        Returns:
            rgb: [3, H, W] rendered RGB image
            semantics: [num_objects, H, W] rendered semantic probabilities
            alpha: [H, W] alpha channel
            info: gsplat info dict
        """
        if params is None:
            params = self.model.get_parameters_as_tensors()
        
        means = params['pos']
        opacities = torch.sigmoid(params['opacity_raw']).squeeze(-1)
        scales = torch.exp(params['scale_raw'])
        quats = params['rotation']
        colors = params['color']  # [N, 3]
        semantics = params['semantics']  # [N, num_objects] one-hot
        
        # Concatenate RGB + semantics for joint rendering
        # gsplat will alpha-blend all channels together
        features = torch.cat([colors, semantics], dim=-1)  # [N, 3 + num_objects]
        
        viewmat = camera._viewmat_gpu
        K = camera._K_gpu
        
        renders, alphas, info = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=features,  # [N, 3 + num_objects]
            viewmats=viewmat.unsqueeze(0),
            Ks=K[None],
            width=camera.image_width,
            height=camera.image_height,
            packed=False,
        )
        
        # renders shape: [1, H, W, 3 + num_objects]
        output = renders[0]  # [H, W, 3 + num_objects]
        
        # Split back into RGB and semantics
        rgb = output[..., :3].permute(2, 0, 1)  # [3, H, W]
        semantics_rendered = output[..., 3:].permute(2, 0, 1)  # [num_objects, H, W]
        
        return rgb, semantics_rendered, alphas[0], info
    
    def compute_semantic_loss(self, rendered_semantics, gt_mask):
        """
        Compute cross-entropy semantic loss per the ObjectGS paper (Equation 7).
        
        The rendered semantics are probability distributions over instances
        (result of alpha-blending one-hot vectors). We use cross-entropy to
        push these probabilities toward the ground truth instance labels.
        
        Args:
            rendered_semantics: [num_objects, H, W] probability distribution per pixel
            gt_mask: [H, W] integer instance IDs (ground truth)
        
        Returns:
            semantic_loss: scalar tensor
        """
        if gt_mask is None:
            return torch.tensor(0.0, device=rendered_semantics.device)
        
        # Cross-entropy expects [N, C, H, W] input and [N, H, W] target
        # rendered_semantics is [C, H, W], gt_mask is [H, W]
        pred = rendered_semantics.unsqueeze(0)  # [1, num_objects, H, W]
        target = gt_mask.unsqueeze(0)  # [1, H, W]
        
        # Clamp target to valid range (in case of any label issues)
        num_classes = rendered_semantics.shape[0]
        target = target.clamp(0, num_classes - 1)
        
        # Cross-entropy loss
        # Note: rendered_semantics are already "probabilities" from alpha blending,
        # but cross_entropy expects logits. We can either:
        # 1. Add small epsilon and take log (treating as probabilities)
        # 2. Use the raw values as logits (works if they're in reasonable range)
        # 
        # The paper uses classification loss on the blended one-hot vectors.
        # Since alpha-blended one-hots sum to ~1 and are non-negative, we treat
        # them as unnormalized logits. Cross-entropy will apply softmax internally.
        
        loss = F.cross_entropy(pred, target, reduction='mean')
        
        return loss
    
    def train_step(self):
        """Single training step with RGB and semantic losses"""
        cam_idx = torch.randint(0, len(self.train_cameras), (1,)).item()
        camera = self.train_cameras[cam_idx]
        gt_image = camera._gt_image_gpu
        gt_mask = camera._gt_mask_gpu  # May be None
        
        params = self.model.get_parameters_as_tensors()
        
        # Render RGB + semantics together
        rendered_rgb, rendered_semantics, alpha, info = self.render_with_semantics(camera, params)
        
        # ==================== RGB Losses ====================
        l1_loss = F.l1_loss(rendered_rgb, gt_image)
        
        ssim_val = ssim_func(
            rendered_rgb.unsqueeze(0), gt_image.unsqueeze(0),
            data_range=1.0, size_average=True
        )
        ssim_loss = 1.0 - ssim_val
        
        # ==================== Regularization Losses ====================
        opacities = torch.sigmoid(params['opacity_raw']).squeeze(-1)
        scales = torch.exp(params['scale_raw'])
        
        # Volume regularization
        volumes = scales.prod(dim=-1) * opacities
        vol_loss = volumes.mean()
        
        # Scale regularization
        scale_threshold = self.config.get('scale_threshold', 0.03)
        lambda_scale = self.config.get('lambda_scale', 10.0)
        scale_excess = torch.relu(scales - scale_threshold)
        scale_reg = (scale_excess ** 2).mean() + (scale_excess ** 2).max()
        
        # ==================== Semantic Loss ====================
        if gt_mask is not None:
            semantic_loss = self.compute_semantic_loss(rendered_semantics, gt_mask)
        else:
            semantic_loss = torch.tensor(0.0, device=self.device)
        
        # ==================== Total Loss ====================
        loss = (l1_loss + 
                self.config['lambda_ssim'] * ssim_loss + 
                self.config['lambda_vol'] * vol_loss + 
                lambda_scale * scale_reg +
                self.config['lambda_semantic'] * semantic_loss)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        losses = {
            'loss': loss.item(),
            'l1': l1_loss.item(),
            'ssim': ssim_loss.item(),
            'ssim_val': ssim_val.item(),
            'vol': vol_loss.item(),
            'scale_reg': scale_reg.item(),
            'semantic': semantic_loss.item(),
            'render_mean': rendered_rgb.mean().item(),
            'render_std': rendered_rgb.std().item(),
        }
        
        self.losses_history.append(losses)
        return losses
    
    def train(self):
        """Main training loop"""
        self.logger.info("=" * 70)
        self.logger.info(f"STARTING TRAINING: {self.config['num_iterations']} iterations")
        self.logger.info(f"Semantic loss weight: {self.config['lambda_semantic']}")
        self.logger.info("=" * 70)
        
        # Warm-up timing
        self.logger.info("Running warm-up iteration for timing...")
        import time
        torch.cuda.synchronize()
        start = time.time()
        _ = self.train_step()
        torch.cuda.synchronize()
        warmup_time = time.time() - start
        self.logger.info(f"Warm-up iteration: {warmup_time:.3f}s")
        
        # Time a batch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = self.train_step()
        torch.cuda.synchronize()
        batch_time = (time.time() - start) / 10
        self.logger.info(f"Average iteration time: {batch_time:.4f}s ({1/batch_time:.1f} it/s)")
        self.logger.info("-" * 50)
        
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
            
            pbar.set_description(f"L1:{losses['l1']:.4f} SSIM:{losses['ssim_val']:.3f} Sem:{losses['semantic']:.4f}")
            
            # LR step
            if i % 100 == 0:
                self.scheduler.step()
            
            # Logging
            if i % self.config['log_interval'] == 0:
                self.logger.info(
                    f"[{i:5d}] Loss={losses['loss']:.4f} L1={losses['l1']:.4f} "
                    f"SSIM={losses['ssim_val']:.3f} Sem={losses['semantic']:.4f}"
                )
            
            # Test evaluation with 0.5 scale
            if i % self.config['test_interval'] == 0:
                self.evaluate(i, scale=0.5)
            
            # Checkpoint
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
        
        # Export everything automatically (includes PLY + grid)
        self.export_final_outputs(
            include_ply=True,
            include_comparison_grid=True,
            render_scale=0.5
        )
        
        self.logger.info(f"All outputs saved to: {self.run_manager.run_dir}")
    
    def evaluate(self, iteration, final=False, scale=0.5):
        """
        Evaluate on test set (RGB and semantic accuracy)
        
        Args:
            iteration: Current iteration number
            final: Whether this is final evaluation
            scale: Scale factor for saved images (0.5 = half resolution)
        """
        self.logger.info(f"[{iteration}] Evaluating...")
        
        self.model.eval()
        test_losses = []
        semantic_accuracies = []
        
        with torch.no_grad():
            for cam_idx, camera in enumerate(self.test_cameras[:5]):
                gt = camera._gt_image_gpu
                gt_mask = camera._gt_mask_gpu
                
                rendered_rgb, rendered_semantics, _, _ = self.render_with_semantics(camera)
                
                l1 = F.l1_loss(rendered_rgb, gt).item()
                test_losses.append(l1)
                
                # Compute semantic accuracy if mask available
                if gt_mask is not None:
                    pred_ids = rendered_semantics.argmax(dim=0)  # [H, W]
                    accuracy = (pred_ids == gt_mask).float().mean().item()
                    semantic_accuracies.append(accuracy)
                
                # Save first comparison
                if cam_idx == 0:
                    comparison = torch.cat([rendered_rgb, gt], dim=2)
                    
                    # Apply scaling
                    if scale != 1.0:
                        comparison = F.interpolate(
                            comparison.unsqueeze(0),
                            scale_factor=scale,
                            mode='bilinear',
                            align_corners=False
                        )[0]
                    
                    if final:
                        save_path = self.run_manager.final_outputs_dir / f'final_comparison.png'
                    else:
                        save_path = self.run_manager.progress_renders_dir / f'iter_{iteration:06d}.png'
                    save_image(comparison, save_path)
                    
                    # Also save semantic visualization if mask available
                    if gt_mask is not None and final:
                        self._save_semantic_visualization(
                            rendered_semantics, gt_mask, 
                            self.run_manager.final_outputs_dir / 'final_semantics.png',
                            scale
                        )
                    
                    self.logger.info(f"  Render: mean={rendered_rgb.mean():.3f}, "
                                   f"std={rendered_rgb.std():.3f}, "
                                   f"range=[{rendered_rgb.min():.3f}, {rendered_rgb.max():.3f}]")
                    self.logger.info(f"  GT:     mean={gt.mean():.3f}, std={gt.std():.3f}")
        
        avg_loss = sum(test_losses) / len(test_losses)
        self.logger.info(f"  Test L1: {avg_loss:.4f}")
        
        if semantic_accuracies:
            avg_accuracy = sum(semantic_accuracies) / len(semantic_accuracies)
            self.logger.info(f"  Semantic Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.1f}%)")
        
        self.model.train()
        return avg_loss
    
    def _save_semantic_visualization(self, rendered_semantics, gt_mask, save_path, scale=1.0):
        """Save a visualization comparing predicted vs GT semantics"""
        # Get predicted instance IDs
        pred_ids = rendered_semantics.argmax(dim=0)  # [H, W]
        
        # Create color maps (random but consistent colors per ID)
        num_objects = self.model.num_objects
        np.random.seed(42)
        colors = np.random.rand(num_objects, 3)
        colors[0] = [0.2, 0.2, 0.2]  # Background is dark gray
        colors = torch.tensor(colors, dtype=torch.float32, device=pred_ids.device)
        
        # Map IDs to colors
        pred_vis = colors[pred_ids]  # [H, W, 3]
        gt_vis = colors[gt_mask]  # [H, W, 3]
        
        # Stack side by side: pred | gt
        comparison = torch.cat([pred_vis, gt_vis], dim=1)  # [H, 2*W, 3]
        comparison = comparison.permute(2, 0, 1)  # [3, H, 2*W]
        
        if scale != 1.0:
            comparison = F.interpolate(
                comparison.unsqueeze(0),
                scale_factor=scale,
                mode='nearest'
            )[0]
        
        save_image(comparison, save_path)
        self.logger.info(f"  Saved semantic visualization: {save_path}")
    
    def save_checkpoint(self, iteration, final=False):
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
            'num_objects': self.model.num_objects,
        }, path)
        
        self.logger.info(f"Saved checkpoint: {path}")
    
    def save_training_history(self):
        """Save training history to JSON"""
        history = {
            'losses': self.losses_history,
            'config': self.config,
            'scene_name': self.run_manager.scene_name,
            'run_name': self.run_manager.run_name,
            'num_objects': self.model.num_objects,
        }
        
        path = self.run_manager.final_outputs_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Saved history: {path}")
    
    # =============================================================================
    # CHECKPOINT LOADING AND EXPORT METHODS
    # =============================================================================
    
    @staticmethod
    def find_latest_checkpoint(checkpoint_dir):
        """
        Find the most recent checkpoint file in a directory.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
            
        Returns:
            Path to latest checkpoint file
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob('checkpoint_*.pth'))
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
        # Sort by iteration number
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        return checkpoints[-1]
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, scene, scene_name=None, 
                        base_output_dir="/workspace/Home_Reconstruction/outputs"):
        """
        Create a trainer instance from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model instance (will be loaded with checkpoint weights)
            scene: Scene instance
            scene_name: Optional scene name (extracted from checkpoint if not provided)
            base_output_dir: Base output directory
            
        Returns:
            GaussianTrainer instance with loaded state
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract info
        iteration = checkpoint['iteration']
        config = checkpoint['config']
        if scene_name is None:
            scene_name = checkpoint.get('scene_name', 'unknown_scene')
        
        print(f"Loading checkpoint: {checkpoint_path}")
        print(f"  Iteration: {iteration}")
        print(f"  Scene: {scene_name}")
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create trainer
        trainer = cls(model, scene, scene_name, config, base_output_dir)
        
        # Load optimizer and scheduler state
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        trainer.current_iteration = iteration
        
        trainer.logger.info(f"Loaded checkpoint from iteration {iteration}")
        
        return trainer

    @classmethod  
    def from_latest_checkpoint(cls, checkpoint_dir, model, scene, scene_name=None,
                               base_output_dir="/workspace/Home_Reconstruction/outputs"):
        """
        Create a trainer instance from the latest checkpoint in a directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            model: Model instance
            scene: Scene instance  
            scene_name: Optional scene name
            base_output_dir: Base output directory
            
        Returns:
            GaussianTrainer instance with loaded state
        """
        latest_checkpoint = cls.find_latest_checkpoint(checkpoint_dir)
        return cls.from_checkpoint(latest_checkpoint, model, scene, scene_name, base_output_dir)
    
    def save_splat_ply(self, save_path=None):
        """
        Export Gaussian splat to .ply format for viewing in external tools.
        
        Args:
            save_path: Path to save .ply file (defaults to final_outputs/splat_iter{N}.ply)
            
        Returns:
            Path to saved file
        """
        # Use default path in final_outputs if not provided
        if save_path is None:
            save_path = self.run_manager.final_outputs_dir / f'splat_iter{self.current_iteration:06d}.ply'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting splat to PLY: {save_path}")
        
        self.model.eval()
        
        with torch.no_grad():
            params = self.model.get_parameters_as_tensors()
            
            pos = params['pos'].cpu().numpy()
            opacity = torch.sigmoid(params['opacity_raw']).cpu().numpy()
            scales = torch.exp(params['scale_raw']).cpu().numpy()
            rotation = params['rotation'].cpu().numpy()
            colors = params['color'].cpu().numpy()
            object_ids = params['object_ids'].cpu().numpy()
            
            # Validate and clip colors to [0, 1] range
            colors = np.clip(colors, 0, 1)
            
            # Convert colors to 0-255 uchar range for Open3D compatibility
            colors_uint8 = (colors * 255).astype(np.uint8)
            
            num_gaussians = pos.shape[0]
            
            # Log color statistics for debugging
            self.logger.info(f"  Color stats: min={colors.min():.3f}, max={colors.max():.3f}, mean={colors.mean():.3f}")
            self.logger.info(f"  Exporting {num_gaussians:,} Gaussians with {self.model.num_objects} instance classes")
        
        # Write PLY file with instance IDs included
        with open(save_path, 'w') as f:
            # Header
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
            f.write("property int instance_id\n")  # Added instance ID
            f.write("end_header\n")
            
            # Data
            for i in range(num_gaussians):
                f.write(f"{pos[i,0]:.6f} {pos[i,1]:.6f} {pos[i,2]:.6f} ")
                f.write(f"{int(colors_uint8[i,0])} {int(colors_uint8[i,1])} {int(colors_uint8[i,2])} ")
                op_val = opacity[i,0] if opacity.ndim > 1 else opacity[i]
                f.write(f"{op_val:.6f} ")
                f.write(f"{scales[i,0]:.6f} {scales[i,1]:.6f} {scales[i,2]:.6f} ")
                f.write(f"{rotation[i,0]:.6f} {rotation[i,1]:.6f} {rotation[i,2]:.6f} {rotation[i,3]:.6f} ")
                f.write(f"{int(object_ids[i])}\n")
        
        self.logger.info(f"Saved {num_gaussians:,} Gaussians to {save_path}")
        self.model.train()
        
        return save_path
    
    def export_final_outputs(self, include_ply=True, include_comparison_grid=True, 
                         render_scale=0.5):
        """
        Export all final outputs: PLY, test renders, comparison grid, per-object PLYs.
        
        Args:
            include_ply: Whether to export .ply file
            include_comparison_grid: Whether to generate all_comparisons.png
            render_scale: Scale factor for test renders (0.5 = half size)
            
        Returns:
            dict with paths to exported files
        """
        self.logger.info("=" * 70)
        self.logger.info("EXPORTING FINAL OUTPUTS")
        self.logger.info("=" * 70)
        
        exports = {}
        
        # 1. Save PLY
        if include_ply:
            exports['ply'] = self.save_splat_ply()
        
        # 2. Test renders
        render_dir = self.run_manager.final_outputs_dir / 'test_renders'
        render_dir.mkdir(exist_ok=True)
        self.render_test_sequence(output_dir=render_dir, save_comparison=True, scale=render_scale)
        exports['renders'] = render_dir
        
        # 3. Comparison grid
        if include_comparison_grid:
            grid_path = self.run_manager.final_outputs_dir / 'all_comparisons.png'
            exports['comparison_grid'] = self.show_test_grid(
                output_dir=render_dir, 
                save_path=grid_path
            )
        
        # 4. Export per-object PLYs
        objects_dir = self.run_manager.final_outputs_dir / 'objects'
        self.model.export_all_objects_ply(str(objects_dir), min_opacity=0.1)
        exports['objects'] = objects_dir
        
        # 5. Clean model checkpoint
        clean_model_path = self.run_manager.final_outputs_dir / f'model_iter{self.current_iteration:06d}.pth'
        torch.save({
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scene_name': self.run_manager.scene_name,
            'run_name': self.run_manager.run_name,
            'num_objects': self.model.num_objects,
            'object_names': self.model.object_names,
        }, clean_model_path)
        exports['model'] = clean_model_path
        
        self.logger.info("=" * 70)
        self.logger.info("EXPORT COMPLETE")
        self.logger.info("=" * 70)
        for key, path in exports.items():
            self.logger.info(f"  {key}: {path}")
        self.logger.info("=" * 70)
        
        return exports
        
    def render_all_test_views(self, scale: float = 0.5):
        """Render all test views to final_outputs"""
        self.render_test_sequence(
            output_dir=self.run_manager.final_outputs_dir,
            save_comparison=True,
            scale=scale
        )
    
    def render_test_sequence(self, output_dir=None, save_comparison=True, scale: float = 0.5):
        """
        Render all test views to a directory.
        
        Args:
            output_dir: Directory to save renders (defaults to final_outputs)
            save_comparison: If True, saves side-by-side render|GT
            scale: Resize factor for output images (0.5 = half size)
        """
        if output_dir is None:
            output_dir = self.run_manager.final_outputs_dir
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Rendering test sequence to: {output_dir}")
        self.logger.info(f"Output scale: {scale}x")
        
        self.model.eval()
        
        with torch.no_grad():
            for cam_idx, camera in enumerate(tqdm(self.test_cameras, desc="Rendering")):
                rendered, alpha, _ = self.render(camera)
                
                if save_comparison:
                    gt = camera._gt_image_gpu
                    output = torch.cat([rendered, gt], dim=2)
                    filename = f'test_{cam_idx:04d}_comparison.png'
                else:
                    output = rendered
                    filename = f'test_{cam_idx:04d}.png'
                
                # Resize if scale != 1.0
                if scale != 1.0:
                    output = F.interpolate(
                        output.unsqueeze(0),
                        scale_factor=scale,
                        mode='bilinear',
                        align_corners=False
                    )[0]
                
                save_image(output, output_dir / filename)
        
        self.model.train()
        self.logger.info(f"Saved {len(self.test_cameras)} renders to {output_dir}")
    
    def show_test_grid(self, num_images: int = None, cols: int = 4, figsize_per_image: float = 4.0,
                   output_dir=None, save_path=None):
        """
        Display test comparison images in a grid (for Jupyter notebooks).
        
        Args:
            num_images: Number of images to show (None = all)
            cols: Number of columns in grid
            figsize_per_image: Figure size multiplier per image
            output_dir: Directory containing test images (defaults to final_outputs)
            save_path: If provided, save to this path instead of displaying
        
        Returns:
            matplotlib figure (or path if saved)
        """
        import matplotlib
        if save_path:
            matplotlib.use('Agg')  # Non-interactive backend for saving
        import matplotlib.pyplot as plt
        from PIL import Image
        import math
        
        if output_dir is None:
            output_dir = self.run_manager.final_outputs_dir
        else:
            output_dir = Path(output_dir)
        
        # Find all comparison images
        image_files = sorted(output_dir.glob('test_*_comparison.png'))
        
        if len(image_files) == 0:
            print(f"No comparison images found in {output_dir}")
            print("Run trainer.render_test_sequence() first.")
            return None
        
        if num_images is not None:
            image_files = image_files[:num_images]
        
        n_images = len(image_files)
        rows = math.ceil(n_images / cols)
        
        # Calculate figure size
        sample_img = Image.open(image_files[0])
        aspect = sample_img.width / sample_img.height
        
        fig_width = cols * figsize_per_image * aspect
        fig_height = rows * figsize_per_image
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        # Handle single row/col case
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        for idx, img_path in enumerate(image_files):
            row = idx // cols
            col = idx % cols
            
            img = Image.open(img_path)
            axes[row][col].imshow(img)
            axes[row][col].set_title(f'Test {idx}', fontsize=10)
            axes[row][col].axis('off')
        
        # Hide empty subplots
        for idx in range(n_images, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row][col].axis('off')
        
        plt.suptitle(f'Test Comparisons: Rendered (left) vs Ground Truth (right)\n'
                     f'{self.run_manager.scene_name} - {self.run_manager.run_name}', 
                     fontsize=12, y=1.02)
        plt.tight_layout()
        
        # Save or return
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"Saved comparison grid: {save_path}")
            return save_path
        else:
            return fig
    
    def show_progress_grid(self, cols: int = 5, figsize_per_image: float = 3.0):
        """
        Display training progress renders in a grid (for Jupyter notebooks).
        
        Args:
            cols: Number of columns in grid
            figsize_per_image: Figure size multiplier per image
        
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        from PIL import Image
        import math
        
        output_dir = self.run_manager.progress_renders_dir
        
        # Find all progress images
        image_files = sorted(output_dir.glob('iter_*.png'))
        
        if len(image_files) == 0:
            print(f"No progress images found in {output_dir}")
            return None
        
        n_images = len(image_files)
        rows = math.ceil(n_images / cols)
        
        sample_img = Image.open(image_files[0])
        aspect = sample_img.width / sample_img.height
        
        fig_width = cols * figsize_per_image * aspect
        fig_height = rows * figsize_per_image
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        for idx, img_path in enumerate(image_files):
            row = idx // cols
            col = idx % cols
            
            # Extract iteration number from filename
            iter_num = img_path.stem.replace('iter_', '')
            
            img = Image.open(img_path)
            axes[row][col].imshow(img)
            axes[row][col].set_title(f'Iter {iter_num}', fontsize=10)
            axes[row][col].axis('off')
        
        for idx in range(n_images, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row][col].axis('off')
        
        plt.suptitle(f'Training Progress\n{output_dir}', fontsize=12, y=1.02)
        plt.tight_layout()
        
        return fig
    
    def render_single_object(self, object_id: int, camera=None):
        """
        Render a single object by filtering Gaussians.
        
        Args:
            object_id: Instance ID to render
            camera: Camera to use (defaults to first test camera)
            
        Returns:
            rendered_rgb: [3, H, W] RGB image of just that object
        """
        if camera is None:
            camera = self.test_cameras[0]
        
        self.model.eval()
        
        with torch.no_grad():
            params = self.model.get_parameters_as_tensors(object_mask=[object_id])
            
            if params['num_gaussians'] == 0:
                self.logger.warning(f"No Gaussians found for object {object_id}")
                return None
            
            rendered, _, _ = self.render(camera, params)
        
        self.model.train()
        return rendered