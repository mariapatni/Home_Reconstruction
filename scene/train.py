# train.py
# ============================================================
# ObjectGS Training Script
# ============================================================

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from objectgs_model import ObjectGSModel

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

class Trainer:
    def __init__(self, config, run_manager):
        self.config = config
        self.run_manager = run_manager

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ObjectGSModel(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        self.best_loss = float("inf")
        self.best_iter = -1

    # --------------------------------------------------------
    # Semantic warm-in
    # --------------------------------------------------------

    def semantic_weight(self, iteration):
        start = self.config.get("semantic_start", 0)
        warmup = self.config.get("semantic_warmup_iters", 1000)

        if iteration < start:
            return 0.0
        return min(1.0, (iteration - start) / warmup)

    # --------------------------------------------------------
    # Training step
    # --------------------------------------------------------

    def train_step(self, iteration):
        loss_rgb = torch.rand(1, device=self.device)
        loss_sem = torch.rand(1, device=self.device)

        sem_w = self.semantic_weight(iteration)
        loss = loss_rgb + sem_w * self.config["lambda_semantic"] * loss_sem

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, iteration):
        loss = np.random.rand()
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_iter = iteration
            self.save_checkpoint(iteration, best=True)
        return loss

    # --------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------

    def save_checkpoint(self, iteration, best=False):
        name = "model_best.pth" if best else f"checkpoint_{iteration:06d}.pth"
        path = self.run_manager.checkpoints_dir / name

        torch.save({
            "iteration": iteration,
            "model": self.model.state_dict(),
            "best_loss": self.best_loss
        }, path)

    def load_best_checkpoint(self):
        path = self.run_manager.checkpoints_dir / "model_best.pth"
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    def export_all(self):
        self.load_best_checkpoint()

        raw_dir = self.run_manager.final_outputs_dir / "exports_raw"
        raw_dir.mkdir(exist_ok=True)

        raw = self.model.export_raw_gaussians()
        np.save(raw_dir / "raw_gaussians.npy", raw)

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    def train(self):
        n_iters = self.config["num_iters"]

        for i in tqdm(range(n_iters)):
            self.train_step(i)
            if i % self.config["eval_interval"] == 0:
                self.evaluate(i)

        self.export_all()
