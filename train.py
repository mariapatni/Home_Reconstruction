"""
Training script for ObjectGS
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import gsplat
from torchvision.utils import save_image
import json


class GaussianTrainer:
    """Trainer for ObjectGS model"""
    
    def __init__(self, model, scene, config=None):
        """
        Args:
            model: ObjectGSModel instance
            scene: Record3DScene instance
            config: Dict with training config (optional)
        """
        self.model = model
        self.scene = scene
        
        # Default config
        self.config = {
            'lr': 0.0001,
            'num_iterations': 5000,
            'save_interval': 1000,
            'test_interval': 500,
            'log_interval': 100,
            'checkpoint_dir': 'checkpoints',
            'output_dir': 'outputs',
            'gamma': 0.99  # Learning rate decay
        }
        
        # Update with user config
        if config is not None:
            self.config.update(config)
        
        # Setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Get cameras
        self.train_cameras = scene.getTrainCameras()
        self.test_cameras = scene.getTestCameras()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['lr']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, 
            gamma=self.config['gamma']
        )
        
        # Create directories
        Path(self.config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_iteration = 0
        self.losses = []
        
        print(f"\n{'='*60}")
        print("Trainer initialized")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Train cameras: {len(self.train_cameras)}")
        print(f"Test cameras: {len(self.test_cameras)}")
        print(f"Learning rate: {self.config['lr']}")
        print(f"Total iterations: {self.config['num_iterations']}")
        print(f"{'='*60}\n")
    
    def render(self, camera):
        """
        Render Gaussians for a given camera
        
        Args:
            camera: Record3DCamera instance
            
        Returns:
            rendered_image: [3, H, W] tensor
        """
        # Get Gaussian parameters
        params = self.model.get_parameters_as_tensors()
        
        means = params['pos']
        opacities = torch.sigmoid(params['opacity_raw']).squeeze(-1)
        scales = torch.exp(params['scale_raw'])
        quats = params['rotation']
        colors = params['color']
        
        # Camera matrices
        viewmat = camera.world_view_transform.T.to(self.device)
        K = torch.tensor([
            [camera.fx, 0, camera.cx],
            [0, camera.fy, camera.cy],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        # Render using gsplat
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
        
        return renders[0].permute(2, 0, 1)  # [3, H, W]
    
    def train_step(self):
        """Single training step"""
        
        # Random camera
        cam_idx = torch.randint(0, len(self.train_cameras), (1,)).item()
        camera = self.train_cameras[cam_idx]
        
        # Ground truth
        gt_image = camera.original_image.to(self.device)
        
        # Render
        rendered_image = self.render(camera)
        
        # Loss
        loss = F.l1_loss(rendered_image, gt_image)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        """Main training loop"""
        
        print("Starting training...\n")
        
        pbar = tqdm(range(1, self.config['num_iterations'] + 1))
        
        for iteration in pbar:
            self.current_iteration = iteration
            
            # Train step
            try:
                loss = self.train_step()
                self.losses.append(loss)
            except Exception as e:
                print(f"\nError at iteration {iteration}: {e}")
                continue
            
            # Update progress bar
            pbar.set_description(f"Loss: {loss:.4f}")
            
            # Learning rate schedule
            if iteration % 100 == 0:
                self.scheduler.step()
            
            # Logging
            if iteration % self.config['log_interval'] == 0:
                avg_loss = sum(self.losses[-100:]) / min(100, len(self.losses))
                tqdm.write(f"Iter {iteration}: Loss={loss:.4f}, Avg={avg_loss:.4f}")
            
            # Test evaluation
            if iteration % self.config['test_interval'] == 0:
                self.evaluate(iteration)
            
            # Save checkpoint
            if iteration % self.config['save_interval'] == 0:
                self.save_checkpoint(iteration)
        
        print("\n✓ Training complete!")
        
        # Final evaluation
        self.evaluate(self.config['num_iterations'])
        self.save_checkpoint(self.config['num_iterations'])
        self.save_training_stats()
    
    @torch.no_grad()
    def evaluate(self, iteration):
        """Evaluate on test cameras"""
        
        self.model.eval()
        
        print(f"\n[Iter {iteration}] Evaluating...")
        
        test_losses = []
        
        for i, camera in enumerate(self.test_cameras[:5]):  # Test on first 5
            gt_image = camera.original_image.to(self.device)
            rendered_image = self.render(camera)
            
            loss = F.l1_loss(rendered_image, gt_image).item()
            test_losses.append(loss)
            
            # Save first test image
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
        
        for i, camera in enumerate(tqdm(self.test_cameras)):
            rendered = self.render(camera)
            save_image(rendered, output_path / f"test_{i:04d}.png")
        
        print(f"✓ Saved to {output_path}/")
        
        self.model.train()
    
    @torch.no_grad()
    def render_single(self, camera, save_path=None):
        """Render a single camera view"""
        
        self.model.eval()
        rendered = self.render(camera)
        
        if save_path:
            save_image(rendered, save_path)
            print(f"Saved: {save_path}")
        
        self.model.train()
        
        return rendered

