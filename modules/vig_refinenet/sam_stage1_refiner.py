"""
Stage-1 GNN refiner (PLAN §2 / §0.5).

SAM image embedding initializes first-layer graph nodes only.
Trainable inputs: RGB image, 3 SAM proposal masks, weak-signal spatial map.
Outputs 1 refined mask logit per weak-signal type at mask_size (256).
SAM still provides 3 multimask proposals as encoder input.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeMLP(nn.Module):
    """MLP for edge weights from node pair features and relative position."""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, fi: torch.Tensor, fj: torch.Tensor, delta_pos: torch.Tensor
    ) -> torch.Tensor:
        edge_input = torch.cat([fi, fj, delta_pos], dim=-1)
        return self.mlp(edge_input)


class NodeMLP(nn.Module):
    """MLP for node update after message aggregation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GNNStage(nn.Module):
    """One round of attention-weighted message passing on a spatial graph."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        connectivity: str = "grid",
        k_neighbors: int = 8,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.connectivity = connectivity
        self.k_neighbors = k_neighbors

        self.edge_mlp = EdgeMLP(in_dim, hidden_dim)
        self.node_mlp = NodeMLP(in_dim, hidden_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self._edge_cache: dict = {}

    def _get_positions(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(0, 1, h, device=device)
        x = torch.linspace(0, 1, w, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    def _build_edges_grid(
        self, h: int, w: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_list, dst_list = [], []
        for i in range(h):
            for j in range(w):
                node_idx = i * w + j
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            src_list.append(node_idx)
                            dst_list.append(ni * w + nj)
        src = torch.tensor(src_list, dtype=torch.long, device=device)
        dst = torch.tensor(dst_list, dtype=torch.long, device=device)
        return src, dst

    def _build_edges_local(
        self, h: int, w: int, k: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = h * w
        pos = self._get_positions(h, w, device)
        dist = torch.cdist(pos, pos)
        dist.fill_diagonal_(float("inf"))
        _, indices = dist.topk(k, dim=1, largest=False)
        src = torch.arange(num_nodes, device=device).repeat_interleave(k)
        dst = indices.flatten()
        return src, dst

    def _get_edges(
        self, h: int, w: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (h, w, self.connectivity, self.k_neighbors, str(device))
        if key not in self._edge_cache:
            if self.connectivity == "local":
                edges = self._build_edges_local(h, w, self.k_neighbors, device)
            else:
                edges = self._build_edges_grid(h, w, device)
            self._edge_cache[key] = edges
        return self._edge_cache[key]

    def forward(self, features: torch.Tensor, h: int, w: int) -> torch.Tensor:
        B, N, D = features.shape
        device = features.device
        pos = self._get_positions(h, w, device)
        src, dst = self._get_edges(h, w, device)
        delta_pos = pos[dst] - pos[src]

        out_features = []
        for b in range(B):
            f = features[b]
            fi, fj = f[src], f[dst]
            edge_weights = self.edge_mlp(fi, fj, delta_pos).squeeze(-1)
            edge_weights_exp = torch.exp(edge_weights)

            messages = torch.zeros(N, D, device=device)
            weights_sum = torch.zeros(N, device=device)
            weighted_fj = edge_weights_exp.unsqueeze(-1) * fj
            messages.index_add_(0, src, weighted_fj)
            weights_sum.index_add_(0, src, edge_weights_exp)
            messages = messages / (weights_sum.unsqueeze(-1) + 1e-8)

            updated = self.node_mlp(f + messages)
            updated = updated + self.proj(f)
            out_features.append(updated)

        return torch.stack(out_features, dim=0)


class SamStage1Refiner(nn.Module):
    """
    GNN mask refiner per PLAN §2.

    ``sam_embed`` seeds first-layer node features only (frozen SAM encoder output).
    Learned fusion uses RGB image, 3 SAM proposal masks, and weak-signal maps.
    """

    def __init__(
        self,
        sam_channels: int = 256,
        feat_dim: int = 128,
        hidden_dim: int = 64,
        out_dim: int = 64,
        grid_size: int = 32,
        mask_size: int = 256,
        num_gnn_layers: int = 2,
        connectivity: str = "grid",
        k_neighbors: int = 8,
        num_output_masks: int = 1,
        num_sam_mask_inputs: int = 3,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.mask_size = mask_size
        self.num_output_masks = num_output_masks
        self.num_sam_mask_inputs = num_sam_mask_inputs

        # SAM embed → node initialization (not the sole forward input)
        self.node_init_proj = nn.Sequential(
            nn.Conv2d(sam_channels, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feat_dim, kernel_size=1),
        )

        self.mask_encoder = nn.Sequential(
            nn.Conv2d(num_sam_mask_inputs, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feat_dim, kernel_size=1),
        )

        # single weak channel: point OR box OR scribble (type selected per batch row)
        self.weak_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feat_dim, kernel_size=1),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim * 4, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

        stages: List[GNNStage] = []
        for i in range(num_gnn_layers):
            in_d = feat_dim if i == 0 else out_dim
            stages.append(
                GNNStage(
                    in_dim=in_d,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    connectivity=connectivity,
                    k_neighbors=k_neighbors,
                )
            )
        self.stages = nn.ModuleList(stages)

        self.mask_head = nn.Sequential(
            nn.Conv2d(out_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_output_masks, kernel_size=1),
        )

    def _to_grid(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == (self.grid_size, self.grid_size):
            return x
        return F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))

    def forward(
        self,
        sam_embed: torch.Tensor,
        images: torch.Tensor,
        sam_masks_3: torch.Tensor,
        weak_signal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sam_embed: [B, 256, 64, 64] frozen SAM encoder — node init only.
            images: [B, 3, H, W] RGB in [0, 1].
            sam_masks_3: [B, 3, mask_size, mask_size] SAM decoder proposals.
            weak_signal: [B, 1, mask_size, mask_size] single weak channel.

        Returns:
            mask_logits: [B, 1, mask_size, mask_size]
        """
        B = sam_embed.shape[0]

        node_init = self.node_init_proj(self._to_grid(sam_embed))

        img = F.interpolate(
            images,
            size=(self.mask_size, self.mask_size),
            mode="bilinear",
            align_corners=False,
        )
        image_feat = self._to_grid(self.image_encoder(img))

        mask_feat = self._to_grid(self.mask_encoder(sam_masks_3.float()))
        weak_feat = self._to_grid(self.weak_encoder(weak_signal))

        x = self.fuse(torch.cat([node_init, image_feat, mask_feat, weak_feat], dim=1))

        feats = x.flatten(2).transpose(1, 2)
        h = w = self.grid_size
        for stage in self.stages:
            feats = stage(feats, h, w)

        spatial = feats.transpose(1, 2).reshape(B, -1, h, w)
        logits = self.mask_head(spatial)
        if logits.shape[-2:] != (self.mask_size, self.mask_size):
            logits = F.interpolate(
                logits,
                size=(self.mask_size, self.mask_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits


def build_sam_stage1_refiner(config: Optional[dict] = None) -> SamStage1Refiner:
    """Build refiner from optional config dict."""
    cfg = config or {}
    model_cfg = cfg.get("model", cfg)
    return SamStage1Refiner(
        sam_channels=model_cfg.get("sam_channels", 256),
        feat_dim=model_cfg.get("feat_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        out_dim=model_cfg.get("out_dim", 64),
        grid_size=model_cfg.get("grid_size", 32),
        mask_size=model_cfg.get("mask_size", 256),
        num_gnn_layers=model_cfg.get("num_gnn_layers", 2),
        connectivity=model_cfg.get("connectivity", "grid"),
        k_neighbors=model_cfg.get("k_neighbors", 8),
        num_output_masks=model_cfg.get("num_output_masks", 1),
        num_sam_mask_inputs=model_cfg.get("num_sam_mask_inputs", 3),
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
