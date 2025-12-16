"""
Training script for ObjectGS - COMPLETE FIXED VERSION

All optimizations applied:
1. Consolidated optimizer (4 groups instead of 24,486) - FIX FOR 2.3s OPTIMIZER STEP
2. Parameter caching (fetch once per iteration)
3. Lower initial opacity (0.1) and smaller scales
4. Gaussian pruning to prevent blob formation
5. Proper loss weights for indoor scenes
6. Controlled scale growth
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import gsplat
from torchvision.utils import save_image
from pytorch_msssim import ssim as ssim_func
import json
import time


class GaussianTrainer:
    """Trainer for ObjectGS model - FULLY OPTIMIZED"""
    
    def __init__(self, model, scene, config=None):
        self.model = model
        self.scene = scene
        
        # Default config - optimized for indoor scenes
        self.config = {
            # Learning rates
            'lr': 0.001,
            'lr_position': 0.00016,
            'lr_feature': 0.0025,
            'lr_opacity': 0.05,
            'lr_scaling': 0.001,      # Reduced to prevent scale growth
            
            # Training schedule
            'num_iterations': 5000,
            'save_interval': 1000,
            'test_interval': 500,
            'log_interval': 100,
            'checkpoint_dir': 'checkpoints',
            'output_dir': 'outputs',
            
            # Density control (CRITICAL for preventing blobs!)
            'prune_interval': 100,
            'prune_opacity_threshold': 0.005,
            'prune_scale_threshold': 0.05,  # Stricter than before
            
            # Loss weights (from paper - indoor scenes)
            'lambda_ssim': 0.2,
            'lambda_vol': 0.0001,  # Stronger than paper's 0.00002
        }
        
        if config is not None:
            self.config.update(config)
        
        # Setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Get cameras
        self.train_cameras = scene.getTrainCameras()
        self.test_cameras = scene.getTestCameras()
        
        # Pre-cache camera matrices on GPU
        self._prepare_cameras()
        
        # Setup CONSOLIDATED optimizer (KEY FIX!)
        self._setup_optimizer()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, 
            gamma=0.99
        )
        
        # Create directories
        Path(self.config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_iteration = 0
        self.losses = []
        
        print(f"\n{'='*60}")
        print("Trainer initialized (FULLY OPTIMIZED)")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Train cameras: {len(self.train_cameras)}")
        print(f"Test cameras: {len(self.test_cameras)}")
        print(f"Optimizer: CONSOLIDATED (4 groups)")
        print(f"Lambda vol: {self.config['lambda_vol']}")
        print(f"Pruning: Every {self.config['prune_interval']} iters")
        print(f"{'='*60}\n")
    
    def _setup_optimizer(self):
        """
        Setup CONSOLIDATED optimizer
        
        KEY FIX: Instead of 24,486 parameter groups (3 per anchor × 8,162 anchors),
        we now have just 4 groups total. This fixes the 2.3s optimizer step!
        """
        # Collect all parameters by type
        all_features = []
        all_scalings = []
        all_offsets = []
        
        for anchor in self.model.anchors:
            all_features.append(anchor.feature)
            all_scalings.append(anchor.scaling)
            all_offsets.append(anchor.offsets)
        
        # Single optimizer with 4 parameter groups
        self.optimizer = torch.optim.Adam([
            {
                'params': all_features,
                'lr': self.config['lr_feature'],
                'name': 'features'
            },
            {
                'params': all_scalings,
                'lr': self.config['lr_scaling'],
                'name': 'scalings'
            },
            {
                'params': all_offsets,
                'lr': self.config['lr_position'],
                'name': 'offsets'
            },
            {
                'params': self.model.attribute_mlp.parameters(),
                'lr': self.config['lr'],
                'name': 'mlp'
            }
        ])
        
        print(f"✓ Optimizer created with 4 parameter groups")
    
    def _prepare_cameras(self):
        """Pre-compute camera matrices and move to GPU"""
        print("Pre-computing camera matrices...")
        
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
        
        print("✓ Camera data cached on GPU")
    
    def render_with_params(self, camera, params):
        """
        Render using pre-fetched parameters
        
        This avoids calling get_parameters_as_tensors() multiple times
        """
        means = params['pos']
        opacities = params['opacities']  # Already sigmoid'd
        scales = params['scales']        # Already exp'd
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
            Ks=K.unsqueeze(0),
            width=camera.image_width,
            height=camera.image_height,
            packed=False
        )
        
        return renders[0].permute(2, 0, 1), info
    
    def prune_gaussians(self):
        """
        Prune low-opacity and oversized Gaussians
        
        This is CRITICAL for preventing blob formation!
        """
        with torch.no_grad():
            params = self.model.get_parameters_as_tensors()
            
            opacities = torch.sigmoid(params['opacity_raw']).squeeze()
            scales = torch.exp(params['scale_raw'])
            max_scales = scales.max(dim=-1)[0]
            
            # Compute validity mask
            valid_mask = (opacities > self.config['prune_opacity_threshold']) & \
                        (max_scales < self.config['prune_scale_threshold'])
            
            num_pruned = (~valid_mask).sum().item()
            
            if num_pruned == 0:
                return
            
            # Prune entire anchors if most of their Gaussians are invalid
            k = self.model.k
            num_anchors = len(self.model.anchors)
            
            anchors_to_remove = []
            for i in range(num_anchors):
                start_idx = i * k
                end_idx = start_idx + k
                anchor_mask = valid_mask[start_idx:end_idx]
                
                # Remove anchor if <30% of its Gaussians are valid
                if anchor_mask.sum() < k * 0.3:
                    anchors_to_remove.append(i)
            
            if len(anchors_to_remove) > 0:
                # Remove anchors
                kept_anchors = [anchor for i, anchor in enumerate(self.model.anchors) 
                              if i not in anchors_to_remove]
                
                self.model.anchors = torch.nn.ModuleList(kept_anchors)
                self.model._precompute_anchor_data()
                
                # Rebuild optimizer with new anchor count
                self._setup_optimizer()
                
                # Rebuild scheduler
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, 
                    gamma=0.99
                )
                
                print(f"  Pruned {len(anchors_to_remove)} anchors ({num_pruned} Gaussians)")
    
    def train_step(self):
        """
        Single training iteration - FULLY OPTIMIZED
        
        Key optimizations:
        1. Fetch parameters ONCE per iteration
        2. Pre-compute sigmoid/exp ONCE
        3. Reuse in both rendering and loss computation
        """
        # Sample random camera
        cam_idx = torch.randint(0, len(self.train_cameras), (1,)).item()
        camera = self.train_cameras[cam_idx]
        gt_image = camera._gt_image_gpu
        
        # ============================================================
        # OPTIMIZATION: Fetch parameters ONCE, reuse everywhere
        # ============================================================
        params_raw = self.model.get_parameters_as_tensors()
        
        # Pre-compute sigmoid/exp ONCE (reuse in render AND loss)
        params = {
            'pos': params_raw['pos'],
            'rotation': params_raw['rotation'],
            'color': params_raw['color'],
            'opacities': torch.sigmoid(params_raw['opacity_raw']).squeeze(-1),
            'scales': torch.exp(params_raw['scale_raw']),
        }
        
        # ============================================================
        # Render using pre-computed parameters
        # ============================================================
        rendered_image, info = self.render_with_params(camera, params)
        
        # ============================================================
        # Compute all losses (reusing pre-computed tensors)
        # ============================================================
        
        # 1. L1 Loss
        l1_loss = F.l1_loss(rendered_image, gt_image)
        
        # 2. SSIM Loss
        ssim_val = ssim_func(
            rendered_image.unsqueeze(0), 
            gt_image.unsqueeze(0), 
            data_range=1.0, 
            size_average=True
        )
        ssim_loss = 1.0 - ssim_val
        
        # 3. Volume Regularization (reuse params['scales'] and params['opacities'])
        volumes = params['scales'].prod(dim=-1) * params['opacities']
        volume_loss = volumes.mean()
        
        # Total loss
        loss = l1_loss + \
               self.config['lambda_ssim'] * ssim_loss + \
               self.config['lambda_vol'] * volume_loss
        
        # ============================================================
        # Backward and optimize (FAST with consolidated optimizer!)
        # ============================================================
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'l1': l1_loss.item(),
            'ssim': ssim_loss.item(),
            'vol': volume_loss.item()
        }
    
    def train(self):
        """Main training loop"""
        
        print("Starting training...\n")
        
        pbar = tqdm(range(1, self.config['num_iterations'] + 1))
        
        for iteration in pbar:
            self.current_iteration = iteration
            
            # Train step
            try:
                losses = self.train_step()
                self.losses.append(losses['loss'])
            except Exception as e:
                print(f"\nError at iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Update progress bar
            pbar.set_description(f"Loss: {losses['loss']:.4f} | L1: {losses['l1']:.4f}")
            
            # Prune Gaussians periodically (CRITICAL!)
            if iteration % self.config['prune_interval'] == 0 and iteration > 100:
                self.prune_gaussians()
            
            # Learning rate decay
            if iteration % 100 == 0:
                self.scheduler.step()
            
            # Logging with Gaussian statistics
            if iteration % self.config['log_interval'] == 0:
                params = self.model.get_parameters_as_tensors()
                num_gaussians = params['num_gaussians']
                opacity_mean = torch.sigmoid(params['opacity_raw']).mean().item()
                scale_mean = torch.exp(params['scale_raw']).mean().item()
                scale_max = torch.exp(params['scale_raw']).max().item()
                
                tqdm.write(f"Iter {iteration}: Loss={losses['loss']:.4f}, "
                          f"L1={losses['l1']:.4f}, Vol={losses['vol']:.6f} | "
                          f"Gaussians={num_gaussians:,}, "
                          f"Opacity={opacity_mean:.3f}, "
                          f"Scale(mean={scale_mean:.4f}, max={scale_max:.4f})")
                
                # Alert if scales getting too large
                if scale_mean > 0.05:
                    tqdm.write(f"  ⚠ WARNING: Scales too large! Consider increasing lambda_vol")
            
            # Test evaluation
            if iteration % self.config['test_interval'] == 0:
                self.evaluate(iteration)
            
            # Save checkpoint
            if iteration % self.config['save_interval'] == 0:
                self.save_checkpoint(iteration)
        
        print("\n✓ Training complete!")
        
        # Final evaluation and save
        self.evaluate(self.config['num_iterations'])
        self.save_checkpoint(self.config['num_iterations'])
        self.save_training_stats()
    
    @torch.no_grad()
    def evaluate(self, iteration):
        """Evaluate on test cameras"""
        self.model.eval()
        
        print(f"\n[Iter {iteration}] Evaluating...")
        
        test_losses = []
        
        # Fetch parameters once for all test renders
        params_raw = self.model.get_parameters_as_tensors()
        params = {
            'pos': params_raw['pos'],
            'rotation': params_raw['rotation'],
            'color': params_raw['color'],
            'opacities': torch.sigmoid(params_raw['opacity_raw']).squeeze(-1),
            'scales': torch.exp(params_raw['scale_raw']),
        }
        
        for i, camera in enumerate(self.test_cameras[:5]):
            gt_image = camera._gt_image_gpu
            rendered_image, _ = self.render_with_params(camera, params)
            
            loss = F.l1_loss(rendered_image, gt_image).item()
            test_losses.append(loss)
            
            if i == 0:
                save_path = Path(self.config['output_dir']) / f"test_iter_{iteration}.png"
                save_image(rendered_image, save_path)
        
        avg_test_loss = sum(test_losses) / len(test_losses)
        print(f"Test loss: {avg_test_loss:.4f}")
        print(f"Saved: {save_path}\n")
        
        self.model.train()
    
    def save_checkpoint(self, iteration):
        """Save model checkpoint"""
        checkpoint_path = Path(self.config['checkpoint_dir']) / f"model_iter_{iteration}.pth"
        
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': self.losses,
            'config': self.config
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_iteration = checkpoint['iteration']
        self.losses = checkpoint['losses']
        
        print(f"Loaded checkpoint from iteration {self.current_iteration}")
    
    def save_training_stats(self):
        """Save training statistics"""
        stats = {
            'iterations': self.config['num_iterations'],
            'final_loss': self.losses[-1] if self.losses else 0,
            'avg_loss_last_100': sum(self.losses[-100:]) / min(100, len(self.losses)),
            'config': self.config,
            'losses': self.losses
        }
        
        stats_path = Path(self.config['output_dir']) / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved training stats: {stats_path}")
    
    @torch.no_grad()
    def render_test_sequence(self, output_dir='test_renders'):
        """Render all test cameras"""
        self.model.eval()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nRendering {len(self.test_cameras)} test views...")
        
        # Fetch parameters once for all test renders
        params_raw = self.model.get_parameters_as_tensors()
        params = {
            'pos': params_raw['pos'],
            'rotation': params_raw['rotation'],
            'color': params_raw['color'],
            'opacities': torch.sigmoid(params_raw['opacity_raw']).squeeze(-1),
            'scales': torch.exp(params_raw['scale_raw']),
        }
        
        for i, camera in enumerate(tqdm(self.test_cameras)):
            rendered, _ = self.render_with_params(camera, params)
            save_image(rendered, output_path / f"test_{i:04d}.png")
        
        print(f"✓ Saved to {output_path}/")
        
        self.model.train()
    
    @torch.no_grad()
    def render_single(self, camera, save_path=None):
        """Render a single camera view"""
        self.model.eval()
        
        params_raw = self.model.get_parameters_as_tensors()
        params = {
            'pos': params_raw['pos'],
            'rotation': params_raw['rotation'],
            'color': params_raw['color'],
            'opacities': torch.sigmoid(params_raw['opacity_raw']).squeeze(-1),
            'scales': torch.exp(params_raw['scale_raw']),
        }
        
        rendered, _ = self.render_with_params(camera, params)
        
        if save_path:
            save_image(rendered, save_path)
            print(f"Saved: {save_path}")
        
        self.model.train()
        
        return rendered