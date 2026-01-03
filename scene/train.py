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
7) FLOATER PREVENTION:
   - Offset-distance penalty (soft leash) to prevent Gaussians from drifting far from anchors
   - Adaptive hard cap based on local anchor density (nearest neighbor distance)
   - Controlled by `lambda_offset_leash` and `adaptive_offset_cap_multiplier` config parameters
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
            "lambda_ssim": 0.3,
            "lambda_volume": 0.0001,
            "lambda_scale_reg": 1.0,
            "scale_threshold": 0.03,
            "max_scale": 0.1,
            # FLOATER PREVENTION
            "lambda_offset_leash": 0.5,
            "adaptive_offset_cap_multiplier": 2.0,
            "nn_recompute_interval": 1000,
            # Densification
            "use_densification": True,
            "densify_start": 1000,
            "densify_until": 25000,
            "densify_interval": 1000,
            "densify_grad_threshold": 5e-5,
            "prune_opacity_threshold": 0.02,
            "min_opacity": 0.01,
            "prune_warmup_iters": 0,
            "early_stop_patience_evals": 20,
        }

        self.config = dict(defaults)
        if config is not None:
            self.config.update(config)

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
        self.last_semantic_eval_loss: float = 0.0

        self._schedule_bounds: List[Tuple[int, int]] = self._compute_progressive_bounds(
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
            self.logger.warning("Could not identify parameter groups, using single LR for all params")
            param_groups = [{'params': self.model.parameters(), 'lr': lr_base, 'name': 'all'}]

        optimizer = torch.optim.Adam(param_groups)

        self.logger.info("Optimizer: Adam with %d param groups", len(param_groups))
        for pg in param_groups:
            name = pg.get('name', 'unnamed')
            lr = pg.get('lr', lr_base)
            n_params = sum(p.numel() for p in pg['params'])
            self.logger.info("  %s: lr=%.6f (%d params)", name, lr, n_params)

        return optimizer

    def _rebuild_optimizer_after_densify_preserve_state(
        self,
        old_optimizer: torch.optim.Optimizer,
        old_scheduler: torch.optim.lr_scheduler._LRScheduler,
        old_anchor_params: Dict[str, torch.Tensor],
        mapping: Dict,
    ):
        old_sched_state = old_scheduler.state_dict()
        new_optimizer = self._build_optimizer()

        for group in new_optimizer.param_groups:
            for p in group["params"]:
                if p in old_optimizer.state:
                    new_optimizer.state[p] = old_optimizer.state[p]

        keep_old = mapping.get("keep_indices_old", None)
        grow_old = mapping.get("grow_parent_old", None)
        if isinstance(keep_old, torch.Tensor):
            keep_old = keep_old.long()
        if isinstance(grow_old, torch.Tensor):
            grow_old = grow_old.long()

        def _remap_adam_state(old_p: torch.Tensor, new_p: torch.Tensor):
            if old_p not in old_optimizer.state:
                return
            st = old_optimizer.state[old_p]
            if not isinstance(st, dict):
                return
            if "exp_avg" not in st or "exp_avg_sq" not in st:
                return

            exp_avg_old = st["exp_avg"]
            exp_avg_sq_old = st["exp_avg_sq"]
            step_old = st.get("step", 0)

            exp_avg_new = torch.zeros_like(new_p.data)
            exp_avg_sq_new = torch.zeros_like(new_p.data)

            n_keep = int(keep_old.numel()) if keep_old is not None else 0
            n_grow = int(grow_old.numel()) if grow_old is not None else 0

            if keep_old is not None and n_keep > 0:
                exp_avg_new[:n_keep].copy_(exp_avg_old[keep_old.to(exp_avg_old.device)])
                exp_avg_sq_new[:n_keep].copy_(exp_avg_sq_old[keep_old.to(exp_avg_sq_old.device)])

            if grow_old is not None and n_grow > 0:
                exp_avg_new[n_keep:n_keep + n_grow].copy_(exp_avg_old[grow_old.to(exp_avg_old.device)])
                exp_avg_sq_new[n_keep:n_keep + n_grow].copy_(exp_avg_sq_old[grow_old.to(exp_avg_sq_old.device)])

            new_optimizer.state[new_p] = {"step": step_old, "exp_avg": exp_avg_new, "exp_avg_sq": exp_avg_sq_new}

        try:
            _remap_adam_state(old_anchor_params["anchor_features"], self.model.anchor_features)
            _remap_adam_state(old_anchor_params["anchor_scalings"], self.model.anchor_scalings)
            _remap_adam_state(old_anchor_params["anchor_offsets"], self.model.anchor_offsets)
        except Exception as e:
            self.logger.warning(f"Failed to remap Adam state for anchors: {e}")

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

    def _log_initialization(self):
        self.logger.info("=" * 70)
        self.logger.info("TRAINING CONFIGURATION")
        self.logger.info("=" * 70)
        for k, v in sorted(self.config.items()):
            self.logger.info(f"  {k}: {v}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Train cameras: {len(self.train_cameras)}")
        self.logger.info(f"Test cameras: {len(self.test_cameras)}")
        self.logger.info(f"Model anchors: {self.model.num_anchors:,}")
        self.logger.info(f"Model Gaussians: {self.model.num_gaussians:,}")
        
        leash_w = float(self.config.get("lambda_offset_leash", 0.0))
        cap_mult = float(self.config.get("adaptive_offset_cap_multiplier", 0.0))
        
        if leash_w > 0:
            self.logger.info(f"Offset leash enabled: lambda={leash_w}")
        else:
            self.logger.info("Offset leash disabled (lambda_offset_leash=0)")
            
        if cap_mult > 0:
            self.logger.info(f"Adaptive offset cap enabled: multiplier={cap_mult}")
            if self._anchor_nn_distances is not None:
                nn_min = float(self._anchor_nn_distances.min().item())
                nn_max = float(self._anchor_nn_distances.max().item())
                nn_mean = float(self._anchor_nn_distances.mean().item())
                self.logger.info(f"  NN distances: min={nn_min:.4f}m, max={nn_max:.4f}m, mean={nn_mean:.4f}m")
                self.logger.info(f"  Effective cap range: [{nn_min*cap_mult:.4f}m, {nn_max*cap_mult:.4f}m]")
        else:
            self.logger.info("Adaptive offset cap disabled (adaptive_offset_cap_multiplier=0)")
        
        self.logger.info("=" * 70)

    @torch.no_grad()
    def _compute_anchor_nn_distances(self, force: bool = False) -> None:
        """Compute nearest neighbor distance for each anchor for adaptive offset cap.
        
        Uses CPU computation with double-chunking to avoid GPU OOM on large anchor sets.
        For 640k anchors, full pairwise distances would need ~1.5TB, so we use a 
        chunked approach that processes small batches at a time.
        """
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
        
        self.logger.info(f"[NN] Computing nearest neighbor distances for {current_anchor_count:,} anchors...")
            
        # Move to CPU for memory efficiency
        positions = self.model.anchor_positions.detach().cpu().float()  # [N, 3]
        N = positions.shape[0]
        
        if N <= 1:
            self._anchor_nn_distances = torch.ones(N, device=self.device) * 0.1
            return
        
        # For very large N, we can't compute full pairwise distances
        # Instead, use a chunked approach on CPU
        # Each anchor only needs to find its nearest neighbor, not all distances
        
        nn_distances = torch.full((N,), float('inf'), dtype=torch.float32)
        
        # Chunk size for query points (rows)
        query_chunk = 5000
        # Chunk size for reference points (cols) - keep small to limit memory
        ref_chunk = 50000
        
        for q_start in range(0, N, query_chunk):
            q_end = min(q_start + query_chunk, N)
            query_pos = positions[q_start:q_end]  # [query_chunk, 3]
            
            # Track minimum distance for this query chunk
            chunk_min_dist = torch.full((q_end - q_start,), float('inf'), dtype=torch.float32)
            
            for r_start in range(0, N, ref_chunk):
                r_end = min(r_start + ref_chunk, N)
                ref_pos = positions[r_start:r_end]  # [ref_chunk, 3]
                
                # Compute pairwise squared distances for this chunk pair
                # query_pos: [Q, 3], ref_pos: [R, 3]
                # diff: [Q, R, 3] -> sum over last dim -> [Q, R]
                diff = query_pos.unsqueeze(1) - ref_pos.unsqueeze(0)  # [Q, R, 3]
                dists_sq = (diff ** 2).sum(dim=-1)  # [Q, R]
                
                # Mask out self-distances (where query index == ref index)
                for i in range(q_end - q_start):
                    global_q_idx = q_start + i
                    if r_start <= global_q_idx < r_end:
                        local_r_idx = global_q_idx - r_start
                        dists_sq[i, local_r_idx] = float('inf')
                
                # Update minimum distances
                chunk_dists, _ = dists_sq.min(dim=1)  # [Q]
                chunk_min_dist = torch.minimum(chunk_min_dist, chunk_dists)
            
            # Store results (take sqrt at the end)
            nn_distances[q_start:q_end] = chunk_min_dist
        
        # Convert squared distances to distances
        nn_distances = torch.sqrt(nn_distances)
        
        # Clamp to reasonable range
        voxel_size = float(getattr(self.model, 'voxel_size', 0.02))
        nn_distances = nn_distances.clamp(min=voxel_size * 0.5)
        
        # Move back to GPU
        self._anchor_nn_distances = nn_distances.to(self.device)
        self._last_nn_compute_iter = self.current_iteration
        self._last_nn_anchor_count = current_anchor_count
        
        nn_min = float(nn_distances.min().item())
        nn_max = float(nn_distances.max().item())
        nn_mean = float(nn_distances.mean().item())
        self.logger.info(f"[NN] Done. min={nn_min:.4f}m, max={nn_max:.4f}m, mean={nn_mean:.4f}m")

    @torch.no_grad()
    def _apply_adaptive_offset_cap(self) -> int:
        """Apply adaptive hard cap on offsets based on local anchor density."""
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
    def _compute_progressive_bounds(n_iters: int, schedule: List) -> List[Tuple[int, int]]:
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

    def _get_downscaled_gt(self, camera, downscale: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, int, int]:
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

    def render_with_semantics(self, camera, params=None, K_override=None, width_override=None, height_override=None, packed: bool = False, model_override=None):
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
            camera, params, K_override=K, width_override=W, height_override=H, packed=True,
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
            "leash_w": float(leash_w),
            "offset_max": offset_max,
            "offset_mean": offset_mean,
            "offset_clamped": n_clamped,
            "train_res": f"{W}x{H}",
            "downscale": int(downscale),
            "has_mask": int(gt_mask is not None),
        }
        self.losses_history.append(out)
        return out

    @torch.no_grad()
    def evaluate(self, iteration: int, save_preview: bool = True, preview_scale: float = 0.5, max_views: int = 5) -> Dict[str, float]:
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

    def save_splat_ply_marble(self, save_path: Optional[Union[str, Path]] = None, iteration: Optional[int] = None,
                              include_f_rest: bool = False, include_instance_id: bool = False,
                              params_override: Optional[Dict] = None, model_override=None) -> Path:
        import struct

        if iteration is None:
            iteration = int(self.current_iteration)

        if save_path is None:
            save_path = self.run_manager.final_outputs_dir / f"marble_iter_{iteration:06d}.ply"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        src_model = model_override if model_override is not None else self.model
        if params_override is not None:
            params = params_override
        else:
            params = src_model.get_parameters_as_tensors()

        pos = params["pos"].detach().cpu()
        op_raw = params["opacity_raw"].detach().cpu().squeeze(-1)
        scl_raw = params["scale_raw"].detach().cpu()
        quat = params["rotation"].detach().cpu()
        color = params["color"].detach().cpu()

        instance_id = None
        if include_instance_id and "object_ids" in params:
            instance_id = params["object_ids"].detach().cpu().int()

        N = int(pos.shape[0])

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

        with open(save_path, "wb") as f:
            f.write(header.encode("utf-8"))
            for i in range(N):
                c0 = (float(color[i, 0]) - 0.5) / 0.2821
                c1 = (float(color[i, 1]) - 0.5) / 0.2821
                c2 = (float(color[i, 2]) - 0.5) / 0.2821

                row = [
                    float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]),
                    c0, c1, c2, float(op_raw[i]),
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
                stats = self.model.densify_and_prune(
                    iteration=i,
                    grad_threshold=float(self.config.get("densify_grad_threshold", 5e-5)),
                    min_opacity=float(self.config.get("prune_opacity_threshold", 0.02)),
                    prune_warmup_iters=int(self.config.get("prune_warmup_iters", 0)),
                    prune_grad_factor=float(self.config.get("prune_grad_factor", 0.25)),
                    min_cycles=int(self.config.get("min_cycles", 5)),
                )
                self.logger.info(f"[DENSIFY/PRUNE] iter={i} stats={stats}")

                self._rebuild_optimizer_after_densify_preserve_state(
                    old_optimizer=old_optimizer, old_scheduler=old_scheduler,
                    old_anchor_params=old_anchor_params, mapping=stats,
                )
                
                if float(self.config.get("adaptive_offset_cap_multiplier", 0.0)) > 0.0:
                    self._compute_anchor_nn_distances(force=True)

            losses = self.train_step(i)

            desc = f"iter {i} | loss {losses['loss']:.4f} | L1 {losses['l1']:.4f}"
            if losses['leash_w'] > 0:
                desc += f" | leash {losses['leash']:.4f}"
            if losses['offset_clamped'] > 0:
                desc += f" | clamped {losses['offset_clamped']}"
            if losses['downscale'] > 1:
                desc += f" | ds {losses['downscale']}"
            pbar.set_description(desc)

            if (i % eval_interval) == 0:
                eval_metrics = self.evaluate(i, save_preview=True, preview_scale=float(self.config.get("progress_render_scale", 0.5)))
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

                sem_str = ""
                if eval_metrics['test_semantic'] > 0:
                    sem_str = f" Sem={eval_metrics['test_semantic']:.4f}"

                offset_str = f" offset_max={losses['offset_max']:.4f}m offset_mean={losses['offset_mean']:.4f}m"
                if losses['offset_clamped'] > 0:
                    offset_str += f" clamped={losses['offset_clamped']}"

                self.logger.info(
                    f"[EVAL] iter={i} L1={eval_metrics['test_l1']:.4f}{sem_str}{offset_str} "
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
                export_model, _ = self.model.__class__.from_checkpoint(ckpt, device=self.device)
            except Exception as e:
                self.logger.exception(f"[EXPORT] Failed to rebuild model from ckpt; exporting current model. err={e}")
                export_model = self.model

            self.logger.info(f"[EXPORT] Using best checkpoint: iter={best_iter} path={best_ckpt_path}")
        else:
            self.logger.warning("[EXPORT] No best checkpoint found; exporting current model state.")

        scene_ply = self.save_splat_ply_marble(iteration=best_iter, include_f_rest=False, include_instance_id=False, model_override=export_model)
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
            self.save_splat_ply_marble(save_path=out_path, iteration=best_iter, include_instance_id=False, params_override=params_obj)

        exports["marble_objects_dir"] = obj_dir
        return exports