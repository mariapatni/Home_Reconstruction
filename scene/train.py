"""
Training script for ObjectGS - FAST VERSION

Features:
- Uses ObjectGSModelFast with batched parameters (no Python loops)
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
    fh = logging.FileHandler(run_manager.log_file)
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
    Fast trainer for ObjectGS using batched model and gsplat CUDA rasterizer.
    """
    
    def __init__(self, model, scene, scene_name: str, config=None, 
                 base_output_dir: str = "/workspace/Home_Reconstruction/outputs"):
        
        # Setup run manager first
        self.run_manager = RunManager(base_output_dir, scene_name)
        
        # Setup logger
        self.logger = setup_training_logger(self.run_manager)
        
        self.model = model
        self.scene = scene
        
        # Default config
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
        """Cache camera data on GPU"""
        self.logger.info("Caching camera data on GPU...")
        
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
                        f"range=[{sample_gt.min():.3f}, {sample_gt.max():.3f}]")
    
    def render(self, camera, params=None):
        """Render using gsplat CUDA rasterizer"""
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
    
    def train_step(self):
        """Single training step"""
        cam_idx = torch.randint(0, len(self.train_cameras), (1,)).item()
        camera = self.train_cameras[cam_idx]
        gt_image = camera._gt_image_gpu
        
        params = self.model.get_parameters_as_tensors()
        rendered, alpha, info = self.render(camera, params)
        
        # Losses
        l1_loss = F.l1_loss(rendered, gt_image)
        
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
        lambda_scale = self.config.get('lambda_scale', 10.0)
        scale_excess = torch.relu(scales - scale_threshold)
        scale_reg = (scale_excess ** 2).mean() + (scale_excess ** 2).max()
        
        # Total loss
        loss = (l1_loss + 
                self.config['lambda_ssim'] * ssim_loss + 
                self.config['lambda_vol'] * vol_loss + 
                lambda_scale * scale_reg)
        
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
            'render_mean': rendered.mean().item(),
            'render_std': rendered.std().item(),
        }
        
        self.losses_history.append(losses)
        return losses
    
    def train(self):
        """Main training loop"""
        self.logger.info("=" * 70)
        self.logger.info(f"STARTING TRAINING: {self.config['num_iterations']} iterations")
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
            
            pbar.set_description(f"L1:{losses['l1']:.4f} SSIM:{losses['ssim_val']:.3f}")
            
            # LR step
            if i % 100 == 0:
                self.scheduler.step()
            
            # Logging
            if i % self.config['log_interval'] == 0:
                self.logger.info(
                    f"[{i:5d}] Loss={losses['loss']:.4f} L1={losses['l1']:.4f} "
                    f"SSIM={losses['ssim_val']:.3f} ScReg={losses['scale_reg']:.4f}"
                )
            
            # Test evaluation
            if i % self.config['test_interval'] == 0:
                self.evaluate(i)
            
            # Checkpoint
            if i % self.config['save_interval'] == 0:
                self.save_checkpoint(i)
        
        pbar.close()
        
        # Final outputs
        self.logger.info("=" * 70)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("=" * 70)
        
        self.evaluate(self.config['num_iterations'], final=True)
        self.save_checkpoint(self.config['num_iterations'], final=True)
        self.save_training_history()
        self.render_all_test_views()
        
        self.logger.info(f"All outputs saved to: {self.run_manager.run_dir}")
    
    def evaluate(self, iteration, final=False):
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
                
                # Save first comparison
                if cam_idx == 0:
                    comparison = torch.cat([rendered, gt], dim=2)
                    if final:
                        save_path = self.run_manager.final_outputs_dir / f'final_comparison.png'
                    else:
                        save_path = self.run_manager.progress_renders_dir / f'iter_{iteration:06d}.png'
                    save_image(comparison, save_path)
                    
                    self.logger.info(f"  Render: mean={rendered.mean():.3f}, "
                                   f"std={rendered.std():.3f}, "
                                   f"range=[{rendered.min():.3f}, {rendered.max():.3f}]")
                    self.logger.info(f"  GT:     mean={gt.mean():.3f}, std={gt.std():.3f}")
        
        avg_loss = sum(test_losses) / len(test_losses)
        self.logger.info(f"  Test L1: {avg_loss:.4f}")
        
        self.model.train()
        return avg_loss
    
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
        }, path)
        
        self.logger.info(f"Saved checkpoint: {path}")
    
    def save_training_history(self):
        """Save training history to JSON"""
        history = {
            'losses': self.losses_history,
            'config': self.config,
            'scene_name': self.run_manager.scene_name,
            'run_name': self.run_manager.run_name,
        }
        
        path = self.run_manager.final_outputs_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Saved history: {path}")
    
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
                       output_dir=None):
        """
        Display test comparison images in a grid (for Jupyter notebooks).
        
        Args:
            num_images: Number of images to show (None = all)
            cols: Number of columns in grid
            figsize_per_image: Figure size multiplier per image
            output_dir: Directory containing test images (defaults to final_outputs)
        
        Returns:
            matplotlib figure
        """
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
        # Load one image to get aspect ratio
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
        
        plt.suptitle(f'Test Comparisons: Rendered (left) vs Ground Truth (right)\n{output_dir}', 
                     fontsize=12, y=1.02)
        plt.tight_layout()
        
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