# objectgs_model.py
# ============================================================
# ObjectGS: Anchor-based Gaussian Splatting with Object Awareness
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

# ------------------------------------------------------------
# View-dependent attribute MLP
# ------------------------------------------------------------

class ViewDependentAttributeMLP(nn.Module):
    def __init__(self, feature_dim=32, k=10):
        super().__init__()
        self.k = k

        self.opacity_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, k)
        )

        self.color_mlp = nn.Sequential(
            nn.Linear(feature_dim + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * k)
        )

        self.scale_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * k)
        )

        self.rot_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4 * k)
        )

    def forward(self, features, view_dir, view_dist):
        B = features.shape[0]
        view = torch.cat([view_dist[:, None], view_dir], dim=-1)
        view = view[:, None, :].expand(B, self.k, 4)

        feat = features[:, None, :].expand(B, self.k, features.shape[-1])
        view_feat = torch.cat([feat, view], dim=-1)

        opacity = self.opacity_mlp(features).view(B, self.k)
        color = self.color_mlp(view_feat.reshape(B * self.k, -1)).view(B, self.k, 3)
        scale = self.scale_mlp(features).view(B, self.k, 3)
        rot = self.rot_mlp(features).view(B, self.k, 4)

        return color, scale, rot, opacity


# ------------------------------------------------------------
# ObjectGS Main Model
# ------------------------------------------------------------

class ObjectGSModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.anchor_positions = nn.Parameter(torch.randn(config["num_anchors"], 3))
        self.anchor_features = nn.Parameter(torch.randn(config["num_anchors"], config["feature_dim"]))
        self.anchor_object_ids = nn.Parameter(
            torch.zeros(config["num_anchors"], dtype=torch.long),
            requires_grad=False
        )

        self.mlp = ViewDependentAttributeMLP(
            feature_dim=config["feature_dim"],
            k=config["gaussians_per_anchor"]
        )

    # --------------------------------------------------------
    # RAW Gaussian export (NO CLAMPING, NO SH PADDING)
    # --------------------------------------------------------

    @torch.no_grad()
    def export_raw_gaussians(self) -> dict:
        pos = self.anchor_positions.detach().cpu().numpy()
        feats = self.anchor_features.detach().cpu()
        obj_ids = self.anchor_object_ids.detach().cpu().numpy()

        view_dir = torch.zeros((feats.shape[0], 3), device=feats.device)
        view_dist = torch.ones((feats.shape[0]), device=feats.device)

        color, scale, rot, opacity = self.mlp(feats, view_dir, view_dist)

        return {
            "pos": pos,
            "color": color.cpu().numpy(),
            "scale": scale.cpu().numpy(),
            "rot": rot.cpu().numpy(),
            "opacity": opacity.cpu().numpy(),
            "object_ids": obj_ids
        }
