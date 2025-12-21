"""
Training script for ObjectGS with SEMANTIC SUPPORT

Features:
- Full semantic object support from SAM3 segmentation
- Object-aware rendering (render specific objects only)
- Semantic PLY export with object IDs
- Optional semantic consistency loss
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
# OUTPUT DIRECTORY MANAGER (unchanged)
# =============================================================================

class RunManager:
    """Manages output directories for training runs."""
    
    def __init__(self, base_dir: str, scene_name: str):
        self.base_dir = Path(base_dir)
        self.scene_name = scene_name
        self.scene_dir = self.base_dir / scene_name
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
# LOGGER SETUP (unchanged)
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
# TRAINER CLASS WITH SEMANTIC SUPPORT
# =============================================================================

class GaussianTrainer:
    """
    Trainer for ObjectGS with full semantic support.
    
    New features:
    - Object-aware rendering
    - Semantic PLY export
    - Per-object visualization
    """
    
    def __init__(self, model, scene, scene_name: str, config=None, 
                 base_output_dir: str = "/workspace/Home_Reconstruction/outputs",
                 object_names: list = None):
        """
        Args:
            model: ObjectGSModel instance
            scene: Record3DScene instance
            scene_name: Name of the scene
            config: Training configuration dict
            base_output_dir: Base directory for outputs
            object_names: List of object names (e.g., ["background", "bed", "dresser"])
        """
        # Setup run manager first
        self.run_manager = RunManager(base_output_dir, scene_name)
        self.logger = setup_training_logger(self.run_manager)
        
        self.model = model
        self.scene = scene
        
        # Store object names (get from model if not provided)
        if object_names is not None:
            self.object_names = object_names
        elif hasattr(model, 'object_names'):
            self.object_names = model.object_names
        else:
            self.object_names = [f"object_{i}" for i in range(model.num_objects)]
            self.object_names[0] = "background"
        
        # Default config with semantic options
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
            'lambda_ssim': 0.2,
            'lambda_vol': 1e-05,
            'lambda_scale': 10.0,
            'scale_threshold': 0.03,
            'min_opacity': 0.001,
            'max_opacity': 0.999,
            'max_scale': 0.1,
            # Semantic options
            'use_semantic_loss': False,  # Enable semantic consistency loss
            'lambda_semantic': 0.1,      # Weight for semantic loss
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
        
        if self.device.type == 'cuda':
            self.logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.model.to(self.device)
        
        # Log semantic info
        self.logger.info("-" * 50)
        self.logger.info("SEMANTIC INFORMATION:")
        self.logger.info(f"  Number of objects: {self.model.num_objects}")
        for i, name in enumerate(self.object_names[:10]):
            count = (self.model.gaussian_object_ids == i).sum().item()
            self.logger.info(f"    ID {i}: {name} ({count:,} Gaussians)")
        if len(self.object_names) > 10:
            self.logger.info(f"    ... and {len(self.object_names) - 10} more")
        self.logger.info("-" * 50)
        
        # Setup cameras
        self.train_cameras = scene.getTrainCameras()
        self.test_cameras = scene.getTestCameras()
        self.logger.info(f"Train cameras: {len(self.train_cameras)}")
        self.logger.info(f"Test cameras: {len(self.test_cameras)}")
        
        self._prepare_cameras()
        self._setup_optimizer()
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.995
        )
        
        self.current_iteration = 0
        self.losses_history = []
        
        self.logger.info("-" * 50)
        self.logger.info(f"Model: {self.model.num_anchors:,} anchors, {self.model.num_gaussians:,} Gaussians")
        self.logger.info("=" * 70)
    
    def _setup_optimizer(self):
        param_groups = self.model.get_optimizer_param_groups(self.config)
        self.optimizer = torch.optim.Adam(param_groups)
        
        self.logger.info(f"Optimizer: Adam with {len(param_groups)} param groups")
        for pg in param_groups:
            self.logger.info(f"  {pg['name']}: lr={pg['lr']}")
    
    def _prepare_cameras(self):
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
            # Cache semantic mask if available
            if hasattr(cam, 'object_mask') and cam.object_mask is not None:
                if not hasattr(cam, '_object_mask_gpu'):
                    cam._object_mask_gpu = cam.object_mask.to(self.device).contiguous()
        
        sample_gt = self.train_cameras[0]._gt_image_gpu
        self.logger.info(f"GT images: shape={list(sample_gt.shape)}, "
                        f"range=[{sample_gt.min():.3f}, {sample_gt.max():.3f}]")
    
    def render(self, camera, params=None, object_mask=None):
        """
        Render using gsplat CUDA rasterizer.
        
        Args:
            camera: Camera object
            params: Pre-computed parameters (optional)
            object_mask: List of object IDs to render (None = all)
        
        Returns:
            rendered image, alpha, info dict
        """
        if params is None:
            params = self.model.get_parameters_as_tensors(object_mask=object_mask)
        
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
    
    def render_object(self, camera, object_id):
        """Render a single object."""
        return self.render(camera, object_mask=[object_id])
    
    def render_objects(self, camera, object_ids):
        """Render multiple specific objects."""
        return self.render(camera, object_mask=object_ids)
    
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
            
            if i % 100 == 0:
                self.scheduler.step()
            
            if i % self.config['log_interval'] == 0:
                self.logger.info(
                    f"[{i:5d}] Loss={losses['loss']:.4f} L1={losses['l1']:.4f} "
                    f"SSIM={losses['ssim_val']:.3f} ScReg={losses['scale_reg']:.4f}"
                )
            
            if i % self.config['test_interval'] == 0:
                self.evaluate(i, scale=0.5)
            
            if i % self.config['save_interval'] == 0:
                self.save_checkpoint(i)
        
        pbar.close()
        
        self.logger.info("=" * 70)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("=" * 70)
        
        self.evaluate(self.config['num_iterations'], final=True, scale=0.5)
        self.save_checkpoint(self.config['num_iterations'], final=True)
        self.save_training_history()
        
        self.export_final_outputs(
            include_ply=True,
            include_semantic_ply=True,  # NEW: Export with object IDs
            include_comparison_grid=True,
            render_scale=0.5
        )
        
        self.logger.info(f"All outputs saved to: {self.run_manager.run_dir}")
    
    def evaluate(self, iteration, final=False, scale=0.5):
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
                        save_path = self.run_manager.final_outputs_dir / f'final_comparison.png'
                    else:
                        save_path = self.run_manager.progress_renders_dir / f'iter_{iteration:06d}.png'
                    save_image(comparison, save_path)
                    
                    self.logger.info(f"  Render: mean={rendered.mean():.3f}, "
                                   f"std={rendered.std():.3f}, "
                                   f"range=[{rendered.min():.3f}, {rendered.max():.3f}]")
        
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
            'object_names': self.object_names,  # Save object names
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
            'object_names': self.object_names,
            'num_objects': self.model.num_objects,
        }
        
        path = self.run_manager.final_outputs_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Saved history: {path}")
    
    # =========================================================================
    # SEMANTIC PLY EXPORT
    # =========================================================================
    
    def save_splat_ply(self, save_path=None, include_object_ids=False):
        """
        Export Gaussian splat to .ply format.
        
        Args:
            save_path: Path to save .ply file
            include_object_ids: Include object_id property in PLY
            
        Returns:
            Path to saved file
        """
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
            
            colors = np.clip(colors, 0, 1)
            colors_uint8 = (colors * 255).astype(np.uint8)
            
            num_gaussians = pos.shape[0]
            
            self.logger.info(f"  Gaussians: {num_gaussians:,}")
            self.logger.info(f"  Color stats: min={colors.min():.3f}, max={colors.max():.3f}")
        
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
            if include_object_ids:
                f.write("property int object_id\n")
            f.write("end_header\n")
            
            for i in range(num_gaussians):
                f.write(f"{pos[i,0]:.6f} {pos[i,1]:.6f} {pos[i,2]:.6f} ")
                f.write(f"{int(colors_uint8[i,0])} {int(colors_uint8[i,1])} {int(colors_uint8[i,2])} ")
                op_val = opacity[i,0] if opacity.ndim > 1 else opacity[i]
                f.write(f"{op_val:.6f} ")
                f.write(f"{scales[i,0]:.6f} {scales[i,1]:.6f} {scales[i,2]:.6f} ")
                f.write(f"{rotation[i,0]:.6f} {rotation[i,1]:.6f} {rotation[i,2]:.6f} {rotation[i,3]:.6f}")
                if include_object_ids:
                    f.write(f" {int(object_ids[i])}")
                f.write("\n")
        
        self.logger.info(f"Saved {num_gaussians:,} Gaussians to {save_path}")
        self.model.train()
        
        return save_path
    
    def save_semantic_ply(self, save_path=None):
        """
        Export PLY with object IDs and save object_ids.npy separately.
        
        This creates files compatible with the visualize_pointclouds semantic viewer.
        """
        if save_path is None:
            save_path = self.run_manager.final_outputs_dir / f'splat_semantic_iter{self.current_iteration:06d}.ply'
        else:
            save_path = Path(save_path)
        
        # Save PLY with object IDs
        self.save_splat_ply(save_path, include_object_ids=True)
        
        # Also save object_ids.npy for the viewer
        with torch.no_grad():
            params = self.model.get_parameters_as_tensors()
            object_ids = params['object_ids'].cpu().numpy()
        
        ids_path = save_path.parent / 'object_ids.npy'
        np.save(ids_path, object_ids.astype(np.int32))
        self.logger.info(f"Saved object IDs to {ids_path}")
        
        # Save object name mapping
        mapping = {
            'object_names': self.object_names,
            'num_objects': self.model.num_objects,
            'id_to_name': {i: name for i, name in enumerate(self.object_names)},
        }
        mapping_path = save_path.parent / 'object_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        self.logger.info(f"Saved object mapping to {mapping_path}")
        
        return save_path
    
    def export_object_ply(self, object_id, save_path=None):
        """Export a single object as a separate PLY file."""
        name = self.object_names[object_id] if object_id < len(self.object_names) else f"object_{object_id}"
        
        if save_path is None:
            save_path = self.run_manager.final_outputs_dir / f'object_{object_id}_{name}.ply'
        else:
            save_path = Path(save_path)
        
        self.logger.info(f"Exporting object {object_id} ({name}) to {save_path}")
        
        self.model.eval()
        
        with torch.no_grad():
            params = self.model.get_parameters_as_tensors(object_mask=[object_id])
            
            pos = params['pos'].cpu().numpy()
            colors = params['color'].cpu().numpy()
            colors = np.clip(colors, 0, 1)
            
            num_pts = len(pos)
        
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(save_path), pcd)
        
        self.logger.info(f"Exported {num_pts:,} Gaussians for {name}")
        self.model.train()
        
        return save_path
    
    def export_all_objects_ply(self, output_dir=None):
        """Export each object as a separate PLY file."""
        if output_dir is None:
            output_dir = self.run_manager.final_outputs_dir / 'objects'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting all {self.model.num_objects} objects to {output_dir}")
        
        paths = {}
        for obj_id in range(self.model.num_objects):
            # Skip if no Gaussians for this object
            count = (self.model.gaussian_object_ids == obj_id).sum().item()
            if count == 0:
                continue
            
            name = self.object_names[obj_id] if obj_id < len(self.object_names) else f"object_{obj_id}"
            save_path = output_dir / f'{obj_id:02d}_{name}.ply'
            self.export_object_ply(obj_id, save_path)
            paths[obj_id] = save_path
        
        return paths
    
    # =========================================================================
    # SEMANTIC VISUALIZATION
    # =========================================================================
    
    def render_object_comparison(self, camera_idx=0, save_path=None, scale=0.5):
        """
        Render comparison showing each object separately.
        
        Creates a grid: [Full Scene] [Obj 1] [Obj 2] ...
        """
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
            for obj_id in range(self.model.num_objects):
                count = (self.model.gaussian_object_ids == obj_id).sum().item()
                if count == 0:
                    continue
                
                obj_render, _, _ = self.render(camera, object_mask=[obj_id])
                renders.append(obj_render.cpu())
                name = self.object_names[obj_id] if obj_id < len(self.object_names) else f"obj_{obj_id}"
                labels.append(f"{name}\n({count:,})")
        
        # Create grid
        n = len(renders)
        cols = min(n, 5)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        axes = np.array(axes).flatten()
        
        for i, (img, label) in enumerate(zip(renders, labels)):
            img_np = img.permute(1, 2, 0).numpy()
            axes[i].imshow(img_np)
            axes[i].set_title(label, fontsize=10)
            axes[i].axis('off')
        
        for i in range(len(renders), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"Object Decomposition - Camera {camera_idx}", fontsize=12)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.run_manager.final_outputs_dir / f'object_decomposition_cam{camera_idx}.png'
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved object decomposition to {save_path}")
        self.model.train()
        
        return save_path
    
    # =========================================================================
    # EXPORT ALL FINAL OUTPUTS
    # =========================================================================
    
    def export_final_outputs(self, include_ply=True, include_semantic_ply=True,
                            include_comparison_grid=True, include_object_decomposition=True,
                            render_scale=0.5):
        """
        Export all final outputs including semantic information.
        """
        self.logger.info("=" * 70)
        self.logger.info("EXPORTING FINAL OUTPUTS")
        self.logger.info("=" * 70)
        
        exports = {}
        
        # 1. Standard PLY
        if include_ply:
            exports['ply'] = self.save_splat_ply()
        
        # 2. Semantic PLY with object IDs
        if include_semantic_ply:
            exports['semantic_ply'] = self.save_semantic_ply()
            exports['per_object_plys'] = self.export_all_objects_ply()
        
        # 3. Test renders
        render_dir = self.run_manager.final_outputs_dir / 'test_renders'
        render_dir.mkdir(exist_ok=True)
        self.render_test_sequence(output_dir=render_dir, save_comparison=True, scale=render_scale)
        exports['renders'] = render_dir
        
        # 4. Comparison grid
        if include_comparison_grid:
            grid_path = self.run_manager.final_outputs_dir / 'all_comparisons.png'
            exports['comparison_grid'] = self.show_test_grid(
                output_dir=render_dir, 
                save_path=grid_path
            )
        
        # 5. Object decomposition render
        if include_object_decomposition:
            exports['object_decomposition'] = self.render_object_comparison()
        
        # 6. Clean model checkpoint
        clean_model_path = self.run_manager.final_outputs_dir / f'model_iter{self.current_iteration:06d}.pth'
        torch.save({
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scene_name': self.run_manager.scene_name,
            'run_name': self.run_manager.run_name,
            'object_names': self.object_names,
            'num_objects': self.model.num_objects,
        }, clean_model_path)
        exports['model'] = clean_model_path
        
        self.logger.info("=" * 70)
        self.logger.info("EXPORT COMPLETE")
        self.logger.info("=" * 70)
        for key, path in exports.items():
            if isinstance(path, dict):
                self.logger.info(f"  {key}: {len(path)} files")
            else:
                self.logger.info(f"  {key}: {path}")
        self.logger.info("=" * 70)
        
        return exports
    
    # =========================================================================
    # HELPER METHODS (from original)
    # =========================================================================
    
    def render_test_sequence(self, output_dir=None, save_comparison=True, scale=0.5):
        """Render all test views to a directory."""
        if output_dir is None:
            output_dir = self.run_manager.final_outputs_dir
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Rendering test sequence to: {output_dir}")
        
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
    
    def show_test_grid(self, num_images=None, cols=4, figsize_per_image=4.0,
                      output_dir=None, save_path=None):
        """Display test comparison images in a grid."""
        import matplotlib
        if save_path:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        import math
        
        if output_dir is None:
            output_dir = self.run_manager.final_outputs_dir
        else:
            output_dir = Path(output_dir)
        
        image_files = sorted(output_dir.glob('test_*_comparison.png'))
        
        if len(image_files) == 0:
            self.logger.warning(f"No comparison images found in {output_dir}")
            return None
        
        if num_images is not None:
            image_files = image_files[:num_images]
        
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
            
            img = Image.open(img_path)
            axes[row][col].imshow(img)
            axes[row][col].set_title(f'Test {idx}', fontsize=10)
            axes[row][col].axis('off')
        
        for idx in range(n_images, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row][col].axis('off')
        
        plt.suptitle(f'Test Comparisons: Rendered | Ground Truth\n'
                    f'{self.run_manager.scene_name} - {self.run_manager.run_name}', 
                    fontsize=12, y=1.02)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"Saved comparison grid: {save_path}")
            return save_path
        else:
            return fig