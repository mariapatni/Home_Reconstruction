# =========================
# file: scene/train.py
# =========================
"""
Training script for ObjectGS - Instance Segmentation Version (FIXED)

Includes:
1) Per-parameter learning rates (features, scalings, offsets, mlp)
2) Progressive resolution schedule (DISABLED by default to match Run 3)
3) Export:
   - ONLY Marble exporter kept (save_splat_ply_marble)
   - best checkpoint is automatically exported at end of training
   - per-object exports also use Marble format
   - IMPORTANT FIX: export loads best checkpoint into a FRESH model instance
4) Semantic-loss caching for evaluation:
   - semantic eval loss cached; recomputed only if eval view has a mask
5) Early stopping:
   - stop if no improvement for `early_stop_patience_evals` evaluation runs
6) Densification + pruning wiring:
   - training render with packed=True
   - retain pos grads and accumulate visible gradient magnitude stats
   - every densify_interval: densify above densify_grad_threshold and prune below prune_opacity_threshold
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
    """
    Minimal replacement for torchvision.utils.save_image to avoid torchvision dependency.
    Expects img in [C,H,W], float in [0,1].
    """
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


# =============================================================================
# Output directory manager
# =============================================================================

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

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.final_outputs_dir.mkdir(parents=True, exist_ok=True)
        self.progress_renders_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_run_number(self) -> int:
        existing = []
        if self.scene_dir.exists():
            for p in self.scene_dir.glob("training_run_*"):
                try:
                    existing.append(int(p.name.split("_")[-1]))
                except Exception:
                    pass
        return (max(existing) + 1) if existing else 1


def setup_training_logger(run_manager: RunManager) -> logging.Logger:
    logger = logging.getLogger(f"ObjectGS_{run_manager.scene_name}_{run_manager.run_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # avoid duplicated handlers if reloaded
    if len(logger.handlers) > 0:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_path = run_manager.run_dir / f"{run_manager.run_name}.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info(f"LOG FILE: {log_path}")
    logger.info("=" * 70)
    return logger


# =============================================================================
# Top-K checkpoint manager
# =============================================================================

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

    def load_state_dict(self, state: Dict) -> None:
        self.k = int(state.get("k", self.k))
        self.lower_is_better = bool(state.get("lower_is_better", self.lower_is_better))
        self.records = []
        for r in state.get("records", []):
            self.records.append(CKPTRecord(float(r["metric"]), int(r["iteration"]), Path(r["path"])))
        self.records = sorted(self.records, key=self._sort_key)


# =============================================================================
# Trainer
# =============================================================================

class GaussianTrainer:
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

        # =================================================================
        # DEFAULT CONFIG - MATCHED TO SUCCESSFUL RUN 3
        # =================================================================
        defaults: Dict = {
            # training iterations
            "num_iterations": 40000,  # was 20000, Run 3 used 40000
            
            # per-parameter learning rates (CRITICAL - from Run 3)
            "lr": 0.001,              # base/mlp LR - was 0.01, Run 3 used 0.001
            "lr_feature": 0.0025,     # anchor features LR
            "lr_position": 0.00016,   # anchor offsets (position) LR
            "lr_scaling": 0.005,      # anchor scalings LR
            
            # evaluation and checkpointing
            "eval_interval": 1000,    # was 500, Run 3 used 1000
            "progress_render_scale": 0.5,
            "checkpoint_topk": 3,
            "checkpoint_metric": "test_l1",
            "lower_is_better": True,

            # progressive resolution schedule - DISABLED by default (Run 3 didn't use it)
            "use_progressive_resolution": False,
            "progressive_resolution_schedule": [
                (0.05, 4),   # first 5%: 4x downscale
                (0.15, 2),   # next 15%: 2x downscale  
                (0.80, 1),   # remaining 80%: full res
            ],

            # semantic loss (matched to Run 3)
            "use_semantic_loss": True,
            "lambda_semantic": 0.05,  # was 1.0, Run 3 used 0.05
            "semantic_loss_start": 2000,  # was 0, Run 3 used 2000
            "semantic_warmup_iters": 0,   # Run 3 didn't use warmup
            "semantic_ignore_index": -1,
            "semantic_min_valid_fraction": 0.005,

            # rgb losses (matched to Run 3)
            "lambda_ssim": 0.3,  # was 0.2, Run 3 used 0.3

            # regularization (matched to Run 3)
            "lambda_volume": 0.0001,  # Run 3 used 0.0001
            "lambda_scale_reg": 1.0,  # Run 3 used lambda_scale: 1.0
            "scale_threshold": 0.03,
            "max_scale": 0.1,

            # densify/prune (matched to Run 3)
            "use_densification": True,
            "densify_start": 1000,
            "densify_until": 25000,  # Run 3 had this
            "densify_interval": 1000,
            "densify_grad_threshold": 5e-5,
            "prune_opacity_threshold": 0.01,
            "min_opacity": 0.001,  # Run 3 had this

            # early stopping (based on checkpoint_metric)
            "early_stop_patience_evals": 20,  # increased patience
        }

        self.config = dict(defaults)
        if config is not None:
            self.config.update(config)

        # Backward-compatible mapping from old lowres settings
        if ("lowres_scale" in self.config) or ("lowres_until" in self.config):
            ls = float(self.config.get("lowres_scale", 1.0))
            lu = int(self.config.get("lowres_until", 0))
            if ls in (1.0, 0.5, 0.25) and lu > 0:
                factor = int(round(1.0 / ls))
                n = int(self.config["num_iterations"])
                p = min(1.0, lu / max(1, n))
                self.config["use_progressive_resolution"] = True
                self.config["progressive_resolution_schedule"] = [(p, factor), (1.0 - p, 1)]
            self.config.pop("lowres_scale", None)
            self.config.pop("lowres_until", None)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.train_cameras = self.scene.getTrainCameras()
        self.test_cameras = self.scene.getTestCameras()

        # =================================================================
        # PER-PARAMETER LEARNING RATES (CRITICAL FIX)
        # =================================================================
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

        # semantic eval cache
        self.last_semantic_eval_loss: float = 0.0

        self._schedule_bounds: List[Tuple[int, int]] = self._compute_progressive_bounds(
            int(self.config["num_iterations"]),
            self.config.get("progressive_resolution_schedule", []),
        )

        self._log_initialization()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build optimizer with per-parameter learning rates.
        This matches Run 3's successful configuration.
        """
        lr_base = float(self.config.get("lr", 0.001))
        lr_feature = float(self.config.get("lr_feature", 0.0025))
        lr_position = float(self.config.get("lr_position", 0.00016))
        lr_scaling = float(self.config.get("lr_scaling", 0.005))
        
        # Collect parameters into groups
        param_groups = []
        
        # Group 1: anchor_features
        if hasattr(self.model, 'anchor_features') and self.model.anchor_features is not None:
            param_groups.append({
                'params': [self.model.anchor_features],
                'lr': lr_feature,
                'name': 'features'
            })
        
        # Group 2: anchor_scalings
        if hasattr(self.model, 'anchor_scalings') and self.model.anchor_scalings is not None:
            param_groups.append({
                'params': [self.model.anchor_scalings],
                'lr': lr_scaling,
                'name': 'scalings'
            })
        
        # Group 3: anchor_offsets (positions)
        if hasattr(self.model, 'anchor_offsets') and self.model.anchor_offsets is not None:
            param_groups.append({
                'params': [self.model.anchor_offsets],
                'lr': lr_position,
                'name': 'offsets'
            })
        
        # Group 4: MLP parameters (attribute_mlp)
        mlp_params = []
        if hasattr(self.model, 'attribute_mlp') and self.model.attribute_mlp is not None:
            mlp_params = list(self.model.attribute_mlp.parameters())
        
        if mlp_params:
            param_groups.append({
                'params': mlp_params,
                'lr': lr_base,
                'name': 'mlp'
            })
        
        # Fallback: if no specific groups found, use all parameters
        if not param_groups:
            self.logger.warning("Could not identify parameter groups, using single LR for all params")
            param_groups = [{'params': self.model.parameters(), 'lr': lr_base, 'name': 'all'}]
        
        optimizer = torch.optim.Adam(param_groups)
        
        # Log the parameter groups
        self.logger.info("Optimizer: Adam with %d param groups", len(param_groups))
        for pg in param_groups:
            name = pg.get('name', 'unnamed')
            lr = pg.get('lr', lr_base)
            n_params = sum(p.numel() for p in pg['params'])
            self.logger.info("  %s: lr=%.6f (%d params)", name, lr, n_params)
        
        return optimizer

    def _rebuild_optimizer_after_densify(self):
        """
        Rebuild optimizer after densification changes parameter shapes.
        Preserves learning rate schedule state.
        """
        old_state = self.scheduler.state_dict()
        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.config["num_iterations"]),
            eta_min=float(self.config.get("lr", 0.001)) * 0.1,
        )
        # Restore scheduler state (current step)
        try:
            self.scheduler.load_state_dict(old_state)
        except Exception:
            # If state doesn't match, just set the step manually
            self.scheduler.last_epoch = self.current_iteration

    def _log_initialization(self):
        """Log initialization info."""
        self.logger.info("INITIALIZING GaussianTrainer")
        self.logger.info("-" * 70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Train cams: {len(self.train_cameras)} | Test cams: {len(self.test_cameras)}")
        self.logger.info(f"Iterations: {self.config['num_iterations']}")
        
        if self.config.get("use_progressive_resolution", False):
            self.logger.info(f"Progressive schedule (end_iter, downscale): {self._schedule_bounds}")
        else:
            self.logger.info("Progressive resolution: DISABLED (training at full resolution)")
        
        self.logger.info("-" * 70)
        self.logger.info("Configuration:")
        for key in sorted(self.config.keys()):
            self.logger.info(f"  {key}: {self.config[key]}")
        self.logger.info("-" * 70)

    # -------------------------------------------------------------------------
    # Progressive resolution helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_progressive_bounds(num_iters: int, schedule: List[Tuple[float, int]]) -> List[Tuple[int, int]]:
        bounds: List[Tuple[int, int]] = []
        if not schedule:
            return [(num_iters, 1)]
        fracs = [float(f) for f, _ in schedule]
        s = sum(fracs)
        if s <= 0:
            return [(num_iters, 1)]
        fracs = [f / s for f in fracs]
        cum = 0.0
        for frac, (_, ds) in zip(fracs, schedule):
            cum += frac
            end_iter = int(round(cum * num_iters))
            bounds.append((max(1, end_iter), int(ds)))
        if bounds[-1][0] != num_iters:
            bounds[-1] = (num_iters, bounds[-1][1])
        return bounds

    def _current_downscale(self, iteration: int) -> int:
        if not self.config.get("use_progressive_resolution", False):
            return 1
        it = int(iteration)
        for end_it, ds in self._schedule_bounds:
            if it <= end_it:
                return int(ds)
        return 1

    def _get_downscaled_gt(self, camera, downscale: int):
        """
        Returns (gt_image[C,H,W], gt_mask[H,W] or None, K[3,3], W, H)
        Caches downscaled tensors on the camera object.
        """
        downscale = int(max(1, downscale))
        if downscale == 1:
            gt_image = camera._gt_image_gpu
            gt_mask = getattr(camera, "_gt_mask_gpu", None)
            K = camera._K_gpu
            W = int(camera.image_width)
            H = int(camera.image_height)
            return gt_image, gt_mask, K, W, H

        cache_key = f"_ds_{downscale}"
        if hasattr(camera, cache_key):
            cached = getattr(camera, cache_key)
            return cached["img"], cached.get("mask", None), cached["K"], cached["W"], cached["H"]

        img = camera._gt_image_gpu  # [3,H,W]
        mask = getattr(camera, "_gt_mask_gpu", None)  # [H,W] or None
        K = camera._K_gpu.clone()

        H0, W0 = img.shape[-2:]
        H = int(H0 // downscale)
        W = int(W0 // downscale)

        img_ds = F.interpolate(
            img.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).contiguous()

        mask_ds = None
        if mask is not None:
            mask_ds = F.interpolate(
                mask[None, None].float(),
                size=(H, W),
                mode="nearest",
            )[0, 0].long().contiguous()

        # scale intrinsics
        K[0, 0] = K[0, 0] / downscale
        K[1, 1] = K[1, 1] / downscale
        K[0, 2] = K[0, 2] / downscale
        K[1, 2] = K[1, 2] / downscale

        cached = {"img": img_ds, "K": K, "W": W, "H": H}
        if mask_ds is not None:
            cached["mask"] = mask_ds
        setattr(camera, cache_key, cached)
        return img_ds, mask_ds, K, W, H

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def render_with_semantics(
        self,
        camera,
        params=None,
        K_override=None,
        width_override=None,
        height_override=None,
        packed: bool = False,
        model_override=None,
    ):
        src_model = model_override if model_override is not None else self.model
        if params is None:
            params = src_model.get_parameters_as_tensors()

        means = params["pos"]
        opacities = torch.sigmoid(params["opacity_raw"]).squeeze(-1)
        scales = torch.exp(params["scale_raw"])
        quats = params["rotation"]
        colors = params["color"]
        semantics = params["semantics"]

        features = torch.cat([colors, semantics], dim=-1)

        viewmat = camera._viewmat_gpu
        K = K_override if K_override is not None else camera._K_gpu
        W = int(width_override) if width_override is not None else int(camera.image_width)
        H = int(height_override) if height_override is not None else int(camera.image_height)

        renders, alphas, info = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=features,
            viewmats=viewmat.unsqueeze(0),
            Ks=K[None],
            width=W,
            height=H,
            packed=bool(packed),
        )

        out = renders[0]  # [H,W,3+C]
        rgb = out[..., :3].permute(2, 0, 1).contiguous()
        sem = out[..., 3:].permute(2, 0, 1).contiguous()
        alpha = alphas[0].contiguous()
        return rgb, sem, alpha, info

    # -------------------------------------------------------------------------
    # Losses
    # -------------------------------------------------------------------------

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
        """
        rendered_sem: [C,H,W] nonnegative (alpha-blended one-hot), not guaranteed to sum to 1
        gt_mask: [H,W] int labels (may contain invalid indices)
        """
        ignore = int(self.config.get("semantic_ignore_index", -1))
        C, H, W = rendered_sem.shape

        valid = (gt_mask >= 0) & (gt_mask < C)
        valid_frac = float(valid.float().mean().item())
        if valid_frac < float(self.config.get("semantic_min_valid_fraction", 0.0)):
            return torch.zeros((), device=rendered_sem.device)

        probs = rendered_sem.clamp(min=0)
        probs = probs / (probs.sum(dim=0, keepdim=True).clamp(min=1e-8))  # [C,H,W]

        flat_probs = probs.permute(1, 2, 0).reshape(-1, C)
        flat_gt = gt_mask.reshape(-1)

        flat_gt = torch.where((flat_gt >= 0) & (flat_gt < C), flat_gt, torch.full_like(flat_gt, ignore))
        logp = torch.log(flat_probs.clamp(min=1e-8))
        loss = F.nll_loss(logp, flat_gt, ignore_index=ignore, reduction="mean")
        return loss

    # -------------------------------------------------------------------------
    # Densification helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_visibility_mask(info: Dict, n_gaussians: int, device) -> torch.Tensor:
        if isinstance(info, dict):
            for k in ("visibility_filter", "visibility_mask", "visible", "vis_mask"):
                if k in info and isinstance(info[k], torch.Tensor):
                    vis = info[k].to(device)
                    if vis.numel() == n_gaussians:
                        return vis.reshape(-1).bool()
        return torch.ones((n_gaussians,), device=device, dtype=torch.bool)

    # -------------------------------------------------------------------------
    # Train step
    # -------------------------------------------------------------------------

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

        rendered_rgb, rendered_sem, alpha, info = self.render_with_semantics(
            camera,
            params,
            K_override=K,
            width_override=W,
            height_override=H,
            packed=True,
        )

        l1 = F.l1_loss(rendered_rgb, gt_image)
        ssim_val = ssim_func(
            rendered_rgb.unsqueeze(0),
            gt_image.unsqueeze(0),
            data_range=1.0,
            size_average=True,
        )
        ssim_loss = 1.0 - ssim_val

        sc = torch.exp(params["scale_raw"])
        vol_loss = (sc.prod(dim=-1).mean() if sc.numel() > 0 else torch.zeros((), device=self.device))
        
        # Scale regularization with max_scale threshold
        max_scale = float(self.config.get("max_scale", 0.1))
        scale_reg = torch.relu(sc.max(dim=-1).values - max_scale).mean()

        sem_w = self._semantic_weight(iteration)
        if (gt_mask is not None) and (sem_w > 0.0) and self.config.get("use_semantic_loss", True):
            sem_loss = self.compute_semantic_loss(rendered_sem, gt_mask)
        else:
            sem_loss = torch.zeros((), device=self.device)

        loss = (
            l1
            + float(self.config.get("lambda_ssim", 0.3)) * ssim_loss
            + float(self.config.get("lambda_volume", 0.0001)) * vol_loss
            + float(self.config.get("lambda_scale_reg", 1.0)) * scale_reg
            + float(sem_w) * sem_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Gradient stats -> densify
        if self.config.get("use_densification", True):
            pos = params.get("pos", None)
            if pos is not None and getattr(pos, "grad", None) is not None:
                n_g = int(pos.shape[0])
                vis = self._extract_visibility_mask(info, n_g, device=pos.device)
                self.model.update_gradient_stats(pos.grad.detach(), vis)

        out = {
            "iter": int(iteration),
            "loss": float(loss.item()),
            "l1": float(l1.item()),
            "ssim": float(ssim_loss.item()),
            "vol": float(vol_loss.item()),
            "scale_reg": float(scale_reg.item()),
            "semantic": float(sem_loss.item()) if sem_loss is not None else 0.0,
            "semantic_w": float(sem_w),
            "train_res": f"{W}x{H}",
            "downscale": int(downscale),
            "has_mask": int(gt_mask is not None),
        }
        self.losses_history.append(out)
        return out

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, iteration: int, save_preview: bool = True, preview_scale: float = 0.5, max_views: int = 5) -> Dict[str, float]:
        """
        Returns dict with:
          test_l1, test_semantic (cached), overall

        Semantic caching rule:
          - if we see at least one masked view during this eval, recompute semantic mean over masked views
          - otherwise keep previous semantic value
        """
        self.model.eval()

        l1s: List[float] = []
        sems: List[float] = []
        saw_mask = False

        views = self.test_cameras[: max_views]
        for vi, cam in enumerate(views):
            gt_image = cam._gt_image_gpu
            gt_mask = getattr(cam, "_gt_mask_gpu", None)
            K = cam._K_gpu
            W = int(cam.image_width)
            H = int(cam.image_height)

            params = self.model.get_parameters_as_tensors(camera_center=getattr(cam, "_campos_gpu", None))
            rgb, sem, alpha, info = self.render_with_semantics(cam, params, K_override=K, width_override=W, height_override=H, packed=False)

            l1 = F.l1_loss(rgb, gt_image).item()
            l1s.append(float(l1))

            if gt_mask is not None and self.config.get("use_semantic_loss", True):
                sem_loss = self.compute_semantic_loss(sem, gt_mask).item()
                sems.append(float(sem_loss))
                saw_mask = True

            if save_preview and vi == 0:
                if preview_scale != 1.0:
                    h2 = int(H * preview_scale)
                    w2 = int(W * preview_scale)
                    rgb_small = F.interpolate(rgb.unsqueeze(0), size=(h2, w2), mode="bilinear", align_corners=False)[0]
                    gt_small = F.interpolate(gt_image.unsqueeze(0), size=(h2, w2), mode="bilinear", align_corners=False)[0]
                else:
                    rgb_small, gt_small = rgb, gt_image

                out = torch.cat([gt_small, rgb_small], dim=2).clamp(0, 1)
                fn = f"iter_{iteration:06d}.png"
                _save_image_tensor(out, self.run_manager.progress_renders_dir / fn)

        test_l1 = float(np.mean(l1s)) if l1s else float("inf")

        if saw_mask and len(sems) > 0:
            self.last_semantic_eval_loss = float(np.mean(sems))

        sem_cached = float(self.last_semantic_eval_loss)
        sem_w = self._semantic_weight(iteration)
        overall = float(test_l1 + sem_w * sem_cached)

        self.model.train()
        return {"test_l1": test_l1, "test_semantic": sem_cached, "overall": overall}

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def _save_checkpoint_file(self, iteration: int, metric: float) -> Path:
        path = self.run_manager.checkpoints_dir / f"checkpoint_{iteration:06d}.pth"

        model_meta = None
        if hasattr(self.model, "state_metadata"):
            try:
                model_meta = self.model.state_metadata()
            except Exception:
                model_meta = None

        torch.save(
            {
                "iteration": int(iteration),
                "metric": float(metric),
                "metric_name": str(self.config.get("checkpoint_metric", "test_l1")),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": dict(self.config),
                "scene_name": self.run_manager.scene_name,
                "run_name": self.run_manager.run_name,
                "num_objects": int(getattr(self.model, "num_objects", -1)),
                "object_names": getattr(self.model, "object_names", None),
                "model_meta": model_meta,
            },
            path,
        )
        return path

    def load_checkpoint(self, ckpt_path: Union[str, Path], load_optimizer: bool = True) -> Dict:
        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)

        if load_optimizer and "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                self.logger.warning("Could not load optimizer state, rebuilding optimizer")
                self._rebuild_optimizer_after_densify()
                
        if load_optimizer and "scheduler_state_dict" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                self.logger.warning("Could not load scheduler state")

        self.current_iteration = int(ckpt.get("iteration", 0))
        return ckpt

    def maybe_save_topk_checkpoint(self, iteration: int, metrics: Dict[str, float]) -> Optional[Path]:
        metric_name = str(self.config.get("checkpoint_metric", "test_l1"))
        metric_val = float(metrics.get(metric_name, metrics.get("test_l1", float("inf"))))
        ckpt_path = self._save_checkpoint_file(iteration, metric_val)
        self.ckpt_mgr.consider(metric_val, iteration, ckpt_path)
        return ckpt_path

    # -------------------------------------------------------------------------
    # Marble export (ONLY export path kept)
    # -------------------------------------------------------------------------

    def save_splat_ply_marble(
        self,
        save_path: Optional[Path] = None,
        iteration: Optional[int] = None,
        include_f_rest: bool = False,
        include_instance_id: bool = True,
        params_override: Optional[Dict[str, torch.Tensor]] = None,
        model_override=None,
    ) -> Path:
        """
        Marble/Spark-friendly 3DGS PLY:
          - binary_little_endian
          - opacity is LOGIT (no sigmoid)
          - scale_0..2 are LOG-SCALES (no exp)
          - color stored as f_dc_0..2 (SH DC), not RGB
          - optional f_rest_0..44 (zeros)
          - optional instance_id appended at end
        """
        import struct

        if iteration is None:
            iteration = self.current_iteration

        if save_path is None:
            save_path = self.run_manager.final_outputs_dir / f"splats_marble_{int(iteration):06d}.ply"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            if params_override is not None:
                params = params_override
            else:
                src = model_override if model_override is not None else self.model
                params = src.get_parameters_as_tensors()

            pos = params["pos"].detach().cpu().numpy().astype(np.float32)
            rgb = params["color"].detach().cpu().numpy().astype(np.float32)
            op_raw = params["opacity_raw"].detach().cpu().numpy().astype(np.float32).reshape(-1)
            scl_raw = params["scale_raw"].detach().cpu().numpy().astype(np.float32)
            quat = params["rotation"].detach().cpu().numpy().astype(np.float32)

            qn = np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12
            quat = quat / qn

            SH_C0 = 0.28209479177387814
            f_dc = (rgb - 0.5) / SH_C0

            N = pos.shape[0]

            instance_id = None
            if include_instance_id and "object_ids" in params:
                instance_id = params["object_ids"].detach().cpu().numpy().astype(np.int32)

        props = [
            ("float", "x"), ("float", "y"), ("float", "z"),
            ("float", "f_dc_0"), ("float", "f_dc_1"), ("float", "f_dc_2"),
            ("float", "opacity"),
            ("float", "scale_0"), ("float", "scale_1"), ("float", "scale_2"),
            ("float", "rot_0"), ("float", "rot_1"), ("float", "rot_2"), ("float", "rot_3"),
        ]
        if include_f_rest:
            for i in range(45):
                props.append(("float", f"f_rest_{i}"))
        if instance_id is not None:
            props.append(("int", "instance_id"))

        header = ["ply", "format binary_little_endian 1.0", f"element vertex {N}"]
        header += [f"property {t} {n}" for t, n in props]
        header.append("end_header\n")
        header_bytes = ("\n".join(header)).encode("ascii")

        with open(save_path, "wb") as f:
            f.write(header_bytes)
            for i in range(N):
                row = [
                    float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]),
                    float(f_dc[i, 0]), float(f_dc[i, 1]), float(f_dc[i, 2]),
                    float(op_raw[i]),
                    float(scl_raw[i, 0]), float(scl_raw[i, 1]), float(scl_raw[i, 2]),
                    float(quat[i, 0]), float(quat[i, 1]), float(quat[i, 2]), float(quat[i, 3]),
                ]
                if include_f_rest:
                    row += [0.0] * 45

                f.write(struct.pack("<" + "f" * (len(row)), *row))
                if instance_id is not None:
                    f.write(struct.pack("<i", int(instance_id[i])))

        self.logger.info(f"[EXPORT] Marble PLY saved: {save_path}")
        return save_path

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    def train(self) -> Dict[str, Path]:
        self.logger.info("=" * 70)
        self.logger.info("STARTING TRAINING")
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

            # Densification (only up to densify_until)
            if (
                self.config.get("use_densification", True)
                and i >= int(self.config.get("densify_start", 1000))
                and i <= densify_until
                and i % int(self.config.get("densify_interval", 1000)) == 0
            ):
                stats = self.model.densify_and_prune(
                    grad_threshold=float(self.config.get("densify_grad_threshold", 5e-5)),
                    min_opacity=float(self.config.get("prune_opacity_threshold", 0.01)),
                )
                self.model.reset_gradient_stats()
                self.logger.info(f"[DENSIFY/PRUNE] iter={i} stats={stats}")
                
                # Rebuild optimizer after densification changes parameter shapes
                self._rebuild_optimizer_after_densify()

            losses = self.train_step(i)
            
            # Progress bar description
            desc = f"iter {i} | loss {losses['loss']:.4f} | L1 {losses['l1']:.4f}"
            if losses['downscale'] > 1:
                desc += f" | ds {losses['downscale']}"
            pbar.set_description(desc)

            if (i % eval_interval) == 0:
                eval_metrics = self.evaluate(
                    i,
                    save_preview=True,
                    preview_scale=float(self.config.get("progress_render_scale", 0.5)),
                )
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

                # Log with semantic info if applicable
                sem_str = ""
                if eval_metrics['test_semantic'] > 0:
                    sem_str = f" Sem={eval_metrics['test_semantic']:.4f}"
                
                self.logger.info(
                    f"[EVAL] iter={i} L1={eval_metrics['test_l1']:.4f}{sem_str} "
                    f"| best={best_metric:.4f} | no_improve={no_improve_evals}"
                )

                if no_improve_evals >= int(self.config.get("early_stop_patience_evals", 20)):
                    self.logger.info(
                        f"[EARLY STOP] No improvement for {no_improve_evals} evals "
                        f"(patience={self.config.get('early_stop_patience_evals', 20)}). Stopping at iter {i}."
                    )
                    break

        final_iter = int(self.current_iteration)
        final_metrics = self.evaluate(final_iter, save_preview=True, preview_scale=float(self.config.get("progress_render_scale", 0.5)))
        self.maybe_save_topk_checkpoint(final_iter, final_metrics)

        best_rec = self.ckpt_mgr.get_best()
        best_ckpt_path = best_rec.path if best_rec is not None else best_ckpt_path

        hist_path = self.run_manager.final_outputs_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump({"losses": self.losses_history, "config": self.config, "topk": self.ckpt_mgr.state_dict()}, f, indent=2)
        self.logger.info(f"Saved history: {hist_path}")

        exports = self.export_best_marble(best_ckpt_path)
        exports["history"] = hist_path

        self.logger.info(f"All outputs saved to: {self.run_manager.run_dir}")
        return exports

    # -------------------------------------------------------------------------
    # Export best checkpoint (Marble ONLY)
    # -------------------------------------------------------------------------

    def export_best_marble(self, best_ckpt_path: Optional[Union[str, Path]]) -> Dict[str, Path]:
        """
        CRITICAL FIX:
        - NEVER load the best checkpoint into self.model (topology mismatch after densify/prune).
        - Instead: load checkpoint -> rebuild a FRESH model instance -> export from that.
        """
        self.logger.info("=" * 70)
        self.logger.info("EXPORTING BEST (MARBLE ONLY)")
        self.logger.info("=" * 70)

        exports: Dict[str, Path] = {}
        best_iter = int(self.current_iteration)

        export_model = self.model  # fallback
        if best_ckpt_path is not None and Path(best_ckpt_path).exists():
            ckpt = torch.load(Path(best_ckpt_path), map_location=self.device)
            best_iter = int(ckpt.get("iteration", best_iter))

            try:
                # Requires ObjectGSModel.from_checkpoint()
                export_model, _ = self.model.__class__.from_checkpoint(ckpt, device=self.device)
            except Exception as e:
                self.logger.exception(f"[EXPORT] Failed to rebuild model from ckpt; exporting current model. err={e}")
                export_model = self.model

            self.logger.info(f"[EXPORT] Using best checkpoint: iter={best_iter} path={best_ckpt_path}")
        else:
            self.logger.warning("[EXPORT] No best checkpoint found; exporting current model state.")

        scene_ply = self.save_splat_ply_marble(
            iteration=best_iter,
            include_f_rest=False,
            include_instance_id=False,
            model_override=export_model,
        )
        exports["marble_scene_ply"] = scene_ply

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