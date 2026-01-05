"""
Training script for ObjectGS with FastGS-inspired VCD
Enhanced with comprehensive logging for understanding training dynamics
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

import gsplat

try:
    from pytorch_msssim import ssim as ssim_func
except Exception:
    def ssim_func(x, y, data_range=1.0, size_average=True):
        return torch.zeros((), device=x.device, dtype=x.dtype)


def _save_image_tensor(img_chw: torch.Tensor, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img_chw.detach().cpu().clamp(0, 1)
    if img.shape[0] == 1:
        arr = (img[0].numpy() * 255.0).astype("uint8")
        im = Image.fromarray(arr, mode="L")
    else:
        arr = (img.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        im = Image.fromarray(arr, mode="RGB")
    im.save(str(path))


class RunManager:
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
        self.vcd_logs_dir = self.run_dir / "vcd_logs"

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.final_outputs_dir.mkdir(parents=True, exist_ok=True)
        self.progress_renders_dir.mkdir(parents=True, exist_ok=True)
        self.vcd_logs_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_run_number(self) -> int:
        existing = []
        if self.scene_dir.exists():
            for p in self.scene_dir.glob("training_run_*"):
                try:
                    existing.append(int(p.name.split("_")[-1]))
                except Exception:
                    pass
        return (max(existing) + 1) if existing else 1


def setup_training_logger(run_manager: RunManager, log_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(f"ObjectGS_VCD_{run_manager.scene_name}_{run_manager.run_name}")
    logger.setLevel(log_level)
    logger.propagate = False

    if len(logger.handlers) > 0:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_path = run_manager.run_dir / f"{run_manager.run_name}.log"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info(f"LOG FILE: {log_path}")
    logger.info("=" * 70)
    return logger


@dataclass
class CKPTRecord:
    metric: float
    iteration: int
    path: Path


class TopKCheckpointManager:
    def __init__(self, k: int = 3, lower_is_better: bool = True):
        self.k = int(k)
        self.lower_is_better = bool(lower_is_better)
        self.records: List[CKPTRecord] = []

    def _sort_key(self, rec: CKPTRecord):
        return rec.metric if self.lower_is_better else -rec.metric

    def consider(self, metric: float, iteration: int, path: Path) -> None:
        rec = CKPTRecord(float(metric), int(iteration), Path(path))
        self.records.append(rec)
        self.records = sorted(self.records, key=self._sort_key)

        while len(self.records) > self.k:
            worst = self.records.pop(-1)
            try:
                if worst.path.exists():
                    worst.path.unlink()
            except Exception:
                pass

    def get_best(self) -> Optional[CKPTRecord]:
        return self.records[0] if len(self.records) > 0 else None

    def state_dict(self) -> Dict:
        return {
            "k": self.k,
            "lower_is_better": self.lower_is_better,
            "records": [
                {"metric": r.metric, "iteration": r.iteration, "path": str(r.path)}
                for r in self.records
            ],
        }


class GaussianTrainerVCD:
    """Trainer with FastGS VCD integration."""
    
    def __init__(
        self,
        model,
        scene,
        scene_name: str,
        config: Optional[Dict] = None,
        base_output_dir: str = "/workspace/Home_Reconstruction/outputs",
    ):
        self.run_manager = RunManager(base_output_dir, scene_name)
        self.logger = setup_training_logger(self.run_manager)

        self.model = model
        self.scene = scene

        defaults: Dict = {
            "num_iterations": 40000,
            "lr": 0.001,
            "lr_feature": 0.0025,
            "lr_position": 0.00016,
            "lr_scaling": 0.005,
            "eval_interval": 1000,
            "progress_render_scale": 0.5,
            "checkpoint_topk": 3,
            "checkpoint_metric": "test_l1",
            "lower_is_better": True,
            "use_progressive_resolution": False,
            "progressive_resolution_schedule": [(0.05, 4), (0.15, 2), (0.80, 1)],
            "use_semantic_loss": True,
            "lambda_semantic": 0.05,
            "semantic_loss_start": 2000,
            "semantic_warmup_iters": 0,
            "semantic_ignore_index": -1,
            "semantic_min_valid_fraction": 0.005,
            "semantic_exclude_class_zero": True,
            "lambda_ssim": 0.3,
            "lambda_volume": 0.0001,
            "lambda_scale_reg": 1.0,
            "scale_threshold": 0.03,
            "max_scale": 0.1,
            "lambda_offset_leash": 0.5,
            "adaptive_offset_cap_multiplier": 2.0,
            "nn_recompute_interval": 1000,
            "use_densification": True,
            "densify_start": 1000,
            "densify_until": 25000,
            "densify_interval": 1000,
            "densify_grad_threshold": 5e-5,
            "prune_opacity_threshold": 0.02,
            "min_opacity": 0.01,
            "prune_warmup_iters": 0,
            "early_stop_patience_evals": 20,
            # VCD-specific
            "vcd_mode": "hybrid",
            "vcd_num_views": 10,
            "vcd_error_threshold": 0.5,
            "vcd_densify_threshold": 10.0,
            "vcd_prune_threshold": 0.1,
            "vcd_gradient_weight": 0.5,
        }

        self.config = dict(defaults)
        if config is not None:
            self.config.update(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Update model VCD settings
        if hasattr(self.model, 'vcd_mode'):
            self.model.vcd_mode = self.config.get("vcd_mode", "hybrid")
            self.model.vcd_num_views = int(self.config.get("vcd_num_views", 10))
            self.model.vcd_error_threshold = float(self.config.get("vcd_error_threshold", 0.5))
            self.model.vcd_gradient_weight = float(self.config.get("vcd_gradient_weight", 0.5))

        self.train_cameras = self.scene.getTrainCameras()
        self.test_cameras = self.scene.getTestCameras()

        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.config["num_iterations"]),
            eta_min=float(self.config.get("lr", 0.001)) * 0.1,
        )

        self.ckpt_mgr = TopKCheckpointManager(
            k=int(self.config.get("checkpoint_topk", 3)),
            lower_is_better=bool(self.config.get("lower_is_better", True)),
        )

        self.current_iteration = 0
        self.losses_history: List[Dict] = []
        self.vcd_history: List[Dict] = []
        self.last_semantic_eval_loss: float = 0.0

        self._schedule_bounds = self._compute_progressive_bounds(
            int(self.config["num_iterations"]),
            self.config.get("progressive_resolution_schedule", []),
        )

        self._anchor_nn_distances: Optional[torch.Tensor] = None
        self._last_nn_compute_iter: int = -1
        self._last_nn_anchor_count: int = -1
        
        if float(self.config.get("adaptive_offset_cap_multiplier", 0.0)) > 0.0:
            self._compute_anchor_nn_distances()

        self._log_initialization()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        lr_base = float(self.config.get("lr", 0.001))
        lr_feature = float(self.config.get("lr_feature", 0.0025))
        lr_position = float(self.config.get("lr_position", 0.00016))
        lr_scaling = float(self.config.get("lr_scaling", 0.005))

        param_groups = []

        if hasattr(self.model, 'anchor_features') and self.model.anchor_features is not None:
            param_groups.append({'params': [self.model.anchor_features], 'lr': lr_feature, 'name': 'features'})

        if hasattr(self.model, 'anchor_scalings') and self.model.anchor_scalings is not None:
            param_groups.append({'params': [self.model.anchor_scalings], 'lr': lr_scaling, 'name': 'scalings'})

        if hasattr(self.model, 'anchor_offsets') and self.model.anchor_offsets is not None:
            param_groups.append({'params': [self.model.anchor_offsets], 'lr': lr_position, 'name': 'offsets'})

        mlp_params = []
        if hasattr(self.model, 'attribute_mlp') and self.model.attribute_mlp is not None:
            mlp_params = list(self.model.attribute_mlp.parameters())

        if mlp_params:
            param_groups.append({'params': mlp_params, 'lr': lr_base, 'name': 'mlp'})

        if not param_groups:
            param_groups = [{'params': self.model.parameters(), 'lr': lr_base, 'name': 'all'}]

        optimizer = torch.optim.Adam(param_groups)

        self.logger.info("Optimizer: Adam with %d param groups", len(param_groups))
        for pg in param_groups:
            name = pg.get('name', 'unnamed')
            lr = pg.get('lr', lr_base)
            n_params = sum(p.numel() for p in pg['params'])
            self.logger.info("  %s: lr=%.6f (%d params)", name, lr, n_params)

        return optimizer

    def _log_initialization(self):
        self.logger.info("=" * 70)
        self.logger.info("TRAINING CONFIGURATION (VCD-ENHANCED)")
        self.logger.info("=" * 70)
        for k, v in sorted(self.config.items()):
            self.logger.info(f"  {k}: {v}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Train cameras: {len(self.train_cameras)}")
        self.logger.info(f"Test cameras: {len(self.test_cameras)}")
        self.logger.info(f"Model anchors: {self.model.num_anchors:,}")
        self.logger.info(f"Model Gaussians: {self.model.num_gaussians:,}")
        
        vcd_mode = self.config.get("vcd_mode", "hybrid")
        self.logger.info("=" * 70)
        self.logger.info("VCD CONFIGURATION")
        self.logger.info("=" * 70)
        self.logger.info(f"  VCD mode: {vcd_mode}")
        self.logger.info(f"  VCD num_views: {self.config.get('vcd_num_views', 10)}")
        self.logger.info(f"  VCD error_threshold: {self.config.get('vcd_error_threshold', 0.5)}")
        self.logger.info(f"  VCD densify_threshold: {self.config.get('vcd_densify_threshold', 10.0)}")
        self.logger.info(f"  VCD prune_threshold: {self.config.get('vcd_prune_threshold', 0.1)}")
        self.logger.info(f"  VCD gradient_weight (hybrid): {self.config.get('vcd_gradient_weight', 0.5)}")
        self.logger.info("=" * 70)

    @torch.no_grad()
    def _compute_anchor_nn_distances(self, force: bool = False) -> None:
        cap_mult = float(self.config.get("adaptive_offset_cap_multiplier", 0.0))
        if cap_mult <= 0.0:
            return
            
        current_anchor_count = self.model.num_anchors
        recompute_interval = int(self.config.get("nn_recompute_interval", 1000))
        
        need_recompute = (
            force or
            self._anchor_nn_distances is None or
            self._last_nn_anchor_count != current_anchor_count or
            (recompute_interval > 0 and self.current_iteration - self._last_nn_compute_iter >= recompute_interval)
        )
        
        if not need_recompute:
            return
            
        positions = self.model.anchor_positions.detach().cpu().numpy()
        N = positions.shape[0]
        
        if N <= 1:
            self._anchor_nn_distances = torch.ones(N, device=self.device) * 0.1
            return
        
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        distances, _ = tree.query(positions, k=2, workers=-1)
        nn_distances = distances[:, 1]
        
        voxel_size = float(getattr(self.model, 'voxel_size', 0.02))
        nn_distances = torch.from_numpy(nn_distances).float().clamp(min=voxel_size * 0.5)
        
        self._anchor_nn_distances = nn_distances.to(self.device)
        self._last_nn_compute_iter = self.current_iteration
        self._last_nn_anchor_count = current_anchor_count

    @torch.no_grad()
    def _apply_adaptive_offset_cap(self) -> int:
        cap_mult = float(self.config.get("adaptive_offset_cap_multiplier", 0.0))
        if cap_mult <= 0.0:
            return 0
            
        if self._anchor_nn_distances is None:
            self._compute_anchor_nn_distances(force=True)
            
        if self._anchor_nn_distances is None:
            return 0
        
        max_radius_per_anchor = self._anchor_nn_distances * cap_mult
        
        off = self.model.anchor_offsets.data
        scl = self.model.anchor_scalings.data.view(-1, 1, 1)
        
        world_off = off * scl
        dist = world_off.norm(dim=-1)
        
        max_radius = max_radius_per_anchor.view(-1, 1).expand_as(dist)
        
        exceed_mask = dist > max_radius
        n_clamped = int(exceed_mask.sum().item())
        
        if n_clamped > 0:
            safe_dist = dist.clamp(min=1e-8)
            scale_factor = torch.where(exceed_mask, max_radius / safe_dist, torch.ones_like(dist))
            clamped_world_off = world_off * scale_factor.unsqueeze(-1)
            safe_scl = scl.clamp(min=1e-8)
            clamped_off = clamped_world_off / safe_scl
            self.model.anchor_offsets.data.copy_(clamped_off)
        
        return n_clamped

    @staticmethod
    def _compute_progressive_bounds(n_iters: int, schedule: List) -> List[Tuple[int, int, int]]:
        if not schedule:
            return [(1, 1, n_iters)]
        bounds = []
        cur = 1
        for frac, ds in schedule:
            length = max(1, int(frac * n_iters))
            bounds.append((ds, cur, cur + length - 1))
            cur += length
        return bounds

    def _current_downscale(self, iteration: int) -> int:
        if not self.config.get("use_progressive_resolution", False):
            return 1
        for ds, lo, hi in self._schedule_bounds:
            if lo <= iteration <= hi:
                return int(ds)
        return 1

    def _get_downscaled_gt(self, camera, downscale: int):
        img = camera._gt_image_gpu
        mask = getattr(camera, "_gt_mask_gpu", None)
        K = camera._K_gpu.clone()

        if downscale == 1:
            H, W = int(img.shape[1]), int(img.shape[2])
            return img, mask, K, W, H

        cache_key = f"_ds_{downscale}"
        if hasattr(camera, cache_key):
            cached = getattr(camera, cache_key)
            return cached["img"], cached.get("mask", None), cached["K"], cached["W"], cached["H"]

        H = int(img.shape[1]) // downscale
        W = int(img.shape[2]) // downscale

        img_ds = F.interpolate(img[None], size=(H, W), mode="bilinear", align_corners=False).squeeze(0).contiguous()

        mask_ds = None
        if mask is not None:
            mask_ds = F.interpolate(mask[None, None].float(), size=(H, W), mode="nearest")[0, 0].long().contiguous()

        K[0, 0] = K[0, 0] / downscale
        K[1, 1] = K[1, 1] / downscale
        K[0, 2] = K[0, 2] / downscale
        K[1, 2] = K[1, 2] / downscale

        cached = {"img": img_ds, "K": K, "W": W, "H": H}
        if mask_ds is not None:
            cached["mask"] = mask_ds
        setattr(camera, cache_key, cached)
        return img_ds, mask_ds, K, W, H

    def render_with_semantics(self, camera, params=None, K_override=None, width_override=None,
                               height_override=None, packed: bool = False, model_override=None,
                               exclude_class_zero_from_semantics: bool = False):
        src_model = model_override if model_override is not None else self.model
        if params is None:
            params = src_model.get_parameters_as_tensors()

        means = params["pos"]
        opacities = torch.sigmoid(params["opacity_raw"]).squeeze(-1)
        scales = torch.exp(params["scale_raw"])
        max_scale = float(self.config.get("max_scale", 0.03))
        scales = torch.clamp(scales, min=1e-4, max=max_scale)

        quats = params["rotation"]
        colors = params["color"]
        semantics = params["semantics"]
        
        if exclude_class_zero_from_semantics:
            object_ids = params["object_ids"]
            semantic_mask = (object_ids != 0).float().unsqueeze(-1)
            semantics = semantics * semantic_mask

        features = torch.cat([colors, semantics], dim=-1)

        viewmat = camera._viewmat_gpu
        K = K_override if K_override is not None else camera._K_gpu
        W = int(width_override) if width_override is not None else int(camera.image_width)
        H = int(height_override) if height_override is not None else int(camera.image_height)

        renders, alphas, info = gsplat.rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=features,
            viewmats=viewmat.unsqueeze(0), Ks=K[None], width=W, height=H, packed=bool(packed),
        )

        out = renders[0]
        rgb = out[..., :3].permute(2, 0, 1).contiguous()
        sem = out[..., 3:].permute(2, 0, 1).contiguous()
        alpha = alphas[0].contiguous()
        return rgb, sem, alpha, info

    def _vcd_render_fn(self, camera, params):
        """Render function for VCD score computation."""
        gt_image = camera._gt_image_gpu
        rgb, sem, alpha, info = self.render_with_semantics(camera, params, packed=False)
        return rgb, gt_image, info

    def _semantic_weight(self, iteration: int) -> float:
        start = int(self.config.get("semantic_loss_start", 2000))
        warm = int(self.config.get("semantic_warmup_iters", 0))
        lam_max = float(self.config.get("lambda_semantic", 0.05))
        if not self.config.get("use_semantic_loss", True) or lam_max <= 0:
            return 0.0
        if iteration < start:
            return 0.0
        if warm <= 0:
            return lam_max
        t = (iteration - start) / float(warm)
        t = float(np.clip(t, 0.0, 1.0))
        return lam_max * t

    def compute_semantic_loss(self, rendered_sem: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        ignore = int(self.config.get("semantic_ignore_index", -1))
        C, H, W = rendered_sem.shape

        valid = (gt_mask >= 0) & (gt_mask < C)
        valid_frac = float(valid.float().mean().item())
        if valid_frac < float(self.config.get("semantic_min_valid_fraction", 0.0)):
            return torch.zeros((), device=rendered_sem.device)

        probs = rendered_sem.clamp(min=0)
        probs = probs / (probs.sum(dim=0, keepdim=True).clamp(min=1e-8))

        flat_probs = probs.permute(1, 2, 0).reshape(-1, C)
        flat_gt = gt_mask.reshape(-1)

        flat_gt = torch.where((flat_gt >= 0) & (flat_gt < C), flat_gt, torch.full_like(flat_gt, ignore))
        logp = torch.log(flat_probs.clamp(min=1e-8))
        loss = F.nll_loss(logp, flat_gt, ignore_index=ignore, reduction="mean")
        return loss

    def compute_offset_leash_loss(self) -> torch.Tensor:
        off = self.model.anchor_offsets
        scl = self.model.anchor_scalings.view(-1, 1, 1)
        world_off = off * scl
        leash_loss = (world_off.norm(dim=-1) ** 2).mean()
        return leash_loss

    @staticmethod
    def _extract_visibility_mask(info: Dict, n_gaussians: int, device) -> torch.Tensor:
        if isinstance(info, dict):
            for k in ("visibility_filter", "visibility_mask", "visible", "vis_mask"):
                if k in info and isinstance(info[k], torch.Tensor):
                    vis = info[k].to(device)
                    if vis.numel() == n_gaussians:
                        return vis.reshape(-1).bool()
        return torch.ones((n_gaussians,), device=device, dtype=torch.bool)

    def _rebuild_optimizer_after_densify(self, old_optimizer, old_scheduler, old_anchor_params, mapping):
        old_sched_state = old_scheduler.state_dict()
        new_optimizer = self._build_optimizer()

        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            new_optimizer,
            T_max=int(self.config["num_iterations"]),
            eta_min=float(self.config.get("lr", 0.001)) * 0.1,
        )
        try:
            new_scheduler.load_state_dict(old_sched_state)
        except Exception as e:
            self.logger.warning(f"Failed to restore scheduler state: {e}")

        self.optimizer = new_optimizer
        self.scheduler = new_scheduler

    def train_step(self, iteration: int) -> Dict:
        cam_idx = torch.randint(0, len(self.train_cameras), (1,), device=self.device).item()
        camera = self.train_cameras[cam_idx]

        downscale = self._current_downscale(iteration)
        gt_image, gt_mask, K, W, H = self._get_downscaled_gt(camera, downscale)

        params = self.model.get_parameters_as_tensors(camera_center=getattr(camera, "_campos_gpu", None))

        if self.config.get("use_densification", True):
            try:
                params["pos"].retain_grad()
            except Exception:
                pass

        exclude_class_zero = self.config.get("semantic_exclude_class_zero", True)
        
        rendered_rgb, rendered_sem, alpha, info = self.render_with_semantics(
            camera, params, K_override=K, width_override=W, height_override=H, packed=True,
            exclude_class_zero_from_semantics=exclude_class_zero,
        )

        l1 = F.l1_loss(rendered_rgb, gt_image)
        ssim_val = ssim_func(rendered_rgb.unsqueeze(0), gt_image.unsqueeze(0), data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val

        sc = torch.exp(params["scale_raw"])
        vol_loss = (sc.prod(dim=-1).mean() if sc.numel() > 0 else torch.zeros((), device=self.device))

        max_scale = float(self.config.get("max_scale", 0.1))
        scale_reg = torch.relu(sc.max(dim=-1).values - max_scale).mean()

        sem_w = self._semantic_weight(iteration)
        if (gt_mask is not None) and (sem_w > 0.0) and self.config.get("use_semantic_loss", True):
            sem_loss = self.compute_semantic_loss(rendered_sem, gt_mask)
        else:
            sem_loss = torch.zeros((), device=self.device)

        leash_w = float(self.config.get("lambda_offset_leash", 0.0))
        if leash_w > 0.0:
            leash_loss = self.compute_offset_leash_loss()
        else:
            leash_loss = torch.zeros((), device=self.device)

        loss = (
            l1
            + float(self.config.get("lambda_ssim", 0.3)) * ssim_loss
            + float(self.config.get("lambda_volume", 0.0001)) * vol_loss
            + float(self.config.get("lambda_scale_reg", 1.0)) * scale_reg
            + float(sem_w) * sem_loss
            + leash_w * leash_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        n_clamped = self._apply_adaptive_offset_cap()

        if self.config.get("use_densification", True):
            pos = params.get("pos", None)
            if pos is not None and getattr(pos, "grad", None) is not None:
                n_g = int(pos.shape[0])
                vis = self._extract_visibility_mask(info, n_g, device=pos.device)
                self.model.update_gradient_stats(pos.grad.detach(), vis)

        with torch.no_grad():
            world_off = self.model.anchor_offsets * self.model.anchor_scalings.view(-1, 1, 1)
            offset_norms = world_off.norm(dim=-1)
            offset_max = float(offset_norms.max().item())
            offset_mean = float(offset_norms.mean().item())

        out = {
            "iter": int(iteration),
            "loss": float(loss.item()),
            "l1": float(l1.item()),
            "ssim": float(ssim_loss.item()),
            "vol": float(vol_loss.item()),
            "scale_reg": float(scale_reg.item()),
            "semantic": float(sem_loss.item()) if sem_loss is not None else 0.0,
            "semantic_w": float(sem_w),
            "leash": float(leash_loss.item()),
            "offset_max": offset_max,
            "offset_mean": offset_mean,
            "offset_clamped": n_clamped,
            "train_res": f"{W}x{H}",
            "downscale": int(downscale),
            "num_anchors": self.model.num_anchors,
            "num_gaussians": self.model.num_gaussians,
        }
        self.losses_history.append(out)
        return out

    @torch.no_grad()
    def evaluate(self, iteration: int, save_preview: bool = True, max_views: int = 5) -> Dict[str, float]:
        self.model.eval()

        l1s: List[float] = []
        sems: List[float] = []
        saw_mask = False

        exclude_class_zero = self.config.get("semantic_exclude_class_zero", True)

        views = self.test_cameras[:max_views]
        for vi, cam in enumerate(views):
            gt_image = cam._gt_image_gpu
            gt_mask = getattr(cam, "_gt_mask_gpu", None)
            K = cam._K_gpu
            W = int(cam.image_width)
            H = int(cam.image_height)

            params = self.model.get_parameters_as_tensors(camera_center=getattr(cam, "_campos_gpu", None))
            rgb, sem, alpha, info = self.render_with_semantics(
                cam, params, K_override=K, width_override=W, height_override=H, packed=False,
                exclude_class_zero_from_semantics=exclude_class_zero,
            )

            l1s.append(float(F.l1_loss(rgb, gt_image).item()))

            if gt_mask is not None:
                saw_mask = True
                sem_loss = float(self.compute_semantic_loss(sem, gt_mask).item())
                sems.append(sem_loss)

            if save_preview and vi == 0:
                preview_path = self.run_manager.progress_renders_dir / f"iter_{iteration:06d}.png"
                _save_image_tensor(rgb, preview_path)

        mean_l1 = float(np.mean(l1s)) if l1s else 0.0

        if saw_mask and sems:
            mean_sem = float(np.mean(sems))
            self.last_semantic_eval_loss = mean_sem
        else:
            mean_sem = self.last_semantic_eval_loss

        self.model.train()

        return {"test_l1": mean_l1, "test_semantic": mean_sem, "overall": mean_l1 + 0.1 * mean_sem}

    def maybe_save_topk_checkpoint(self, iteration: int, eval_metrics: Dict[str, float]) -> Optional[Path]:
        metric_name = str(self.config.get("checkpoint_metric", "test_l1"))
        metric_val = float(eval_metrics.get(metric_name, eval_metrics["test_l1"]))

        ckpt_path = self.run_manager.checkpoints_dir / f"ckpt_iter_{iteration:06d}.pt"

        ckpt = {
            "iteration": int(iteration),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "eval_metrics": eval_metrics,
            "model_meta": self.model.state_metadata() if hasattr(self.model, "state_metadata") else {},
        }
        torch.save(ckpt, ckpt_path)

        self.ckpt_mgr.consider(metric_val, iteration, ckpt_path)
        return ckpt_path

    def train(self) -> Dict[str, Path]:
        self.logger.info("=" * 70)
        self.logger.info("STARTING TRAINING (VCD-ENHANCED)")
        self.logger.info("=" * 70)

        n_iters = int(self.config["num_iterations"])
        eval_interval = int(self.config.get("eval_interval", 1000))
        densify_until = int(self.config.get("densify_until", n_iters))

        best_metric = float("inf") if bool(self.config.get("lower_is_better", True)) else -float("inf")
        no_improve_evals = 0
        best_ckpt_path: Optional[Path] = None

        pbar = tqdm(range(1, n_iters + 1))
        for i in pbar:
            self.current_iteration = i

            # Densification with VCD
            if (
                self.config.get("use_densification", True)
                and i >= int(self.config.get("densify_start", 1000))
                and i <= densify_until
                and i % int(self.config.get("densify_interval", 1000)) == 0
            ):
                old_optimizer = self.optimizer
                old_scheduler = self.scheduler
                old_anchor_params = {
                    "anchor_features": self.model.anchor_features,
                    "anchor_scalings": self.model.anchor_scalings,
                    "anchor_offsets": self.model.anchor_offsets,
                }
                
                self.logger.info("=" * 70)
                self.logger.info(f"[DENSIFY/PRUNE CYCLE] iter={i}")
                self.logger.info("=" * 70)
                
                stats = self.model.densify_and_prune(
                    iteration=i,
                    grad_threshold=float(self.config.get("densify_grad_threshold", 5e-5)),
                    min_opacity=float(self.config.get("prune_opacity_threshold", 0.02)),
                    prune_warmup_iters=int(self.config.get("prune_warmup_iters", 0)),
                    prune_grad_factor=float(self.config.get("prune_grad_factor", 0.25)),
                    min_cycles=int(self.config.get("min_cycles", 5)),
                    vcd_densify_threshold=float(self.config.get("vcd_densify_threshold", 10.0)),
                    vcd_prune_threshold=float(self.config.get("vcd_prune_threshold", 0.1)),
                    train_cameras=self.train_cameras,
                    render_fn=self._vcd_render_fn,
                )
                
                self.vcd_history.append({"iteration": i, "stats": stats})
                self.logger.info(f"[DENSIFY/PRUNE] Complete: {stats}")

                self._rebuild_optimizer_after_densify(
                    old_optimizer, old_scheduler, old_anchor_params, stats,
                )
                
                if float(self.config.get("adaptive_offset_cap_multiplier", 0.0)) > 0.0:
                    self._compute_anchor_nn_distances(force=True)

            losses = self.train_step(i)

            desc = f"iter {i} | loss {losses['loss']:.4f} | L1 {losses['l1']:.4f} | anchors {losses['num_anchors']:,}"
            pbar.set_description(desc)

            if (i % eval_interval) == 0:
                eval_metrics = self.evaluate(i, save_preview=True)
                self.maybe_save_topk_checkpoint(i, eval_metrics)
                best_rec = self.ckpt_mgr.get_best()
                best_ckpt_path = best_rec.path if best_rec is not None else best_ckpt_path

                metric_name = str(self.config.get("checkpoint_metric", "test_l1"))
                metric_val = float(eval_metrics.get(metric_name, eval_metrics["test_l1"]))

                improved = (metric_val < best_metric) if self.config.get("lower_is_better", True) else (metric_val > best_metric)
                if improved:
                    best_metric = metric_val
                    no_improve_evals = 0
                else:
                    no_improve_evals += 1

                self.logger.info(
                    f"[EVAL] iter={i} L1={eval_metrics['test_l1']:.4f} "
                    f"| anchors={self.model.num_anchors:,} | best={best_metric:.4f} | no_improve={no_improve_evals}"
                )

                if no_improve_evals >= int(self.config.get("early_stop_patience_evals", 20)):
                    self.logger.info(f"[EARLY STOP] No improvement for {no_improve_evals} evals. Stopping.")
                    break

        # Save final
        final_iter = int(self.current_iteration)
        final_metrics = self.evaluate(final_iter, save_preview=True)
        self.maybe_save_topk_checkpoint(final_iter, final_metrics)

        hist_path = self.run_manager.final_outputs_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump({
                "losses": self.losses_history, 
                "config": self.config, 
                "topk": self.ckpt_mgr.state_dict(),
                "vcd_history": self.vcd_history,
            }, f, indent=2)
        self.logger.info(f"Saved history: {hist_path}")

        return {"history": hist_path}

    def save_splat_ply_marble(
        self,
        save_path: Optional[Union[str, Path]] = None,
        iteration: Optional[int] = None,
        include_f_rest: bool = False,
        include_instance_id: bool = False,
        params_override: Optional[Dict] = None,
        model_override=None,
    ) -> Path:
        """
        FAST Marble exporter with EXACT same binary layout as your original loop:
    
        Per-vertex order:
          x,y,z,
          f_dc_0,f_dc_1,f_dc_2,
          opacity,
          scale_0,scale_1,scale_2,
          rot_0,rot_1,rot_2,rot_3,
          [optional f_rest_0..44 floats],
          [optional instance_id int32]
    
        All floats are little-endian float32. instance_id is little-endian int32.
        """
        import numpy as np
        from pathlib import Path
    
        if iteration is None:
            iteration = int(self.current_iteration)
    
        if save_path is None:
            save_path = self.run_manager.final_outputs_dir / f"marble_iter_{iteration:06d}.ply"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
        src_model = model_override if model_override is not None else self.model
        params = params_override if params_override is not None else src_model.get_parameters_as_tensors()
    
        # --- CPU numpy buffers (same sources as your original) ---
        pos = params["pos"].detach().cpu().to(torch.float32).numpy()                # [N,3]
        op_raw = params["opacity_raw"].detach().cpu().to(torch.float32).numpy()     # [N,1] or [N]
        if op_raw.ndim == 2:
            op_raw = op_raw[:, 0]
        scl_raw = params["scale_raw"].detach().cpu().to(torch.float32).numpy()      # [N,3]
        quat = params["rotation"].detach().cpu().to(torch.float32).numpy()          # [N,4]
        color = params["color"].detach().cpu().to(torch.float32).numpy()            # [N,3]
    
        instance_id = None
        if include_instance_id and "object_ids" in params:
            instance_id = params["object_ids"].detach().cpu().to(torch.int32).numpy()  # [N]
    
        N = int(pos.shape[0])
    
        # --- Header EXACTLY like your earlier version ---
        header_lines = [
            "ply", "format binary_little_endian 1.0", f"element vertex {N}",
            "property float x", "property float y", "property float z",
            "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
            "property float opacity",
            "property float scale_0", "property float scale_1", "property float scale_2",
            "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3",
        ]
        if include_f_rest:
            for i in range(45):
                header_lines.append(f"property float f_rest_{i}")
        if include_instance_id and instance_id is not None:
            header_lines.append("property int instance_id")
        header_lines.append("end_header")
        header = "\n".join(header_lines) + "\n"
    
        # --- Same DC conversion as your loop ---
        dc = (color - 0.5) / 0.2821  # float32
    
        # --- Build PACKED dtype with NO ALIGNMENT/PADDING ---
        fields = [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
            ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
        ]
        if include_f_rest:
            fields += [(f"f_rest_{i}", "<f4") for i in range(45)]
        if include_instance_id and instance_id is not None:
            fields += [("instance_id", "<i4")]
    
        dtype = np.dtype(fields, align=False)
    
        # Hard guarantee: identical per-vertex byte size to your original pack
        n_float = 14 + (45 if include_f_rest else 0)
        expected_itemsize = (n_float * 4) + (4 if (include_instance_id and instance_id is not None) else 0)
        assert dtype.itemsize == expected_itemsize, (
            f"Marble dtype has padding! itemsize={dtype.itemsize}, expected={expected_itemsize}. "
            "This would NOT match your original exporter."
        )
    
        data = np.empty(N, dtype=dtype)
    
        # Fill (vectorized)
        data["x"] = pos[:, 0]
        data["y"] = pos[:, 1]
        data["z"] = pos[:, 2]
    
        data["f_dc_0"] = dc[:, 0]
        data["f_dc_1"] = dc[:, 1]
        data["f_dc_2"] = dc[:, 2]
    
        data["opacity"] = op_raw
    
        data["scale_0"] = scl_raw[:, 0]
        data["scale_1"] = scl_raw[:, 1]
        data["scale_2"] = scl_raw[:, 2]
    
        data["rot_0"] = quat[:, 0]
        data["rot_1"] = quat[:, 1]
        data["rot_2"] = quat[:, 2]
        data["rot_3"] = quat[:, 3]
    
        if include_f_rest:
            for i in range(45):
                data[f"f_rest_{i}"] = 0.0
    
        if include_instance_id and instance_id is not None:
            data["instance_id"] = instance_id
    
        # Write header + payload (payload is interleaved per-vertex EXACTLY like loop)
        with open(save_path, "wb") as f:
            f.write(header.encode("utf-8"))
            data.tofile(f)
    
        self.logger.info(f"[EXPORT] Marble PLY saved: {save_path}")
        return save_path

    
    
    def export_best_marble(self, best_ckpt_path: Optional[Union[str, Path]]) -> Dict[str, Path]:
        self.logger.info("=" * 70)
        self.logger.info("EXPORTING BEST (MARBLE ONLY)")
        self.logger.info("=" * 70)
    
        exports: Dict[str, Path] = {}
        best_iter = int(self.current_iteration)
    
        export_model = self.model
        if best_ckpt_path is not None and Path(best_ckpt_path).exists():
            ckpt = torch.load(Path(best_ckpt_path), map_location=self.device)
            best_iter = int(ckpt.get("iteration", best_iter))
    
            try:
                # IMPORTANT: fresh model instance from checkpoint
                export_model, _ = self.model.__class__.from_checkpoint(ckpt, device=self.device)
            except Exception as e:
                self.logger.exception(f"[EXPORT] Failed to rebuild model from ckpt; exporting current model. err={e}")
                export_model = self.model
    
            self.logger.info(f"[EXPORT] Using best checkpoint: iter={best_iter} path={best_ckpt_path}")
        else:
            self.logger.warning("[EXPORT] No best checkpoint found; exporting current model state.")
    
        # Scene export
        scene_ply = self.save_splat_ply_marble(
            iteration=best_iter,
            include_f_rest=False,
            include_instance_id=False,
            model_override=export_model,
        )
        exports["marble_scene_ply"] = scene_ply
    
        # Per-object exports (Marble format)
        obj_dir = self.run_manager.final_outputs_dir / "marble_objects"
        obj_dir.mkdir(parents=True, exist_ok=True)
    
        num_objects = int(getattr(export_model, "num_objects", 0))
        object_names = getattr(export_model, "object_names", None) or []
    
        for obj_id in range(num_objects):
            params_obj = export_model.get_parameters_as_tensors(object_mask=[int(obj_id)])
            if int(params_obj["pos"].shape[0]) == 0:
                continue
    
            name = object_names[obj_id] if obj_id < len(object_names) else f"object_{obj_id}"
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
            out_path = obj_dir / f"{obj_id:03d}_{safe}.ply"
    
            self.save_splat_ply_marble(
                save_path=out_path,
                iteration=best_iter,
                include_instance_id=False,
                params_override=params_obj,
            )
    
        exports["marble_objects_dir"] = obj_dir
        return exports
