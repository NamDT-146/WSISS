"""Frozen / trainable SAM + GNN teacher for Stage-2 pseudo-labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from modules.vig_refinenet.sam_stage1_common import (
    forward_teacher_objects,
    get_sam_pixel_stats,
    load_sam_vit_b,
)
from modules.wssis.pseudo_label_confidence import (
    DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
    resolve_pseudo_confidence_threshold,
)
from modules.vig_refinenet.sam_stage1_refiner import build_sam_stage1_refiner
from modules.wssis.paths import gnn_checkpoint, sam_vit_b_checkpoint
from modules.wssis.weak_prompts import build_image_prompts


class WssisTeacherStack(nn.Module):
    """SAM ViT-B + optional GNN refiner for pseudo instance generation."""

    def __init__(
        self,
        device: torch.device,
        gnn_ckpt_path: Optional[Path] = None,
        use_gnn: bool = True,
        freeze_gnn: bool = False,
        mask_size: int = 256,
        pseudo_confidence_threshold: float | None = None,
    ):
        super().__init__()
        self.device = device
        self.use_gnn = use_gnn
        self.mask_size = mask_size
        self.pseudo_confidence_threshold = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD
        sam_path = sam_vit_b_checkpoint()
        self.sam = load_sam_vit_b(str(sam_path), device)
        self.pixel_mean, self.pixel_std = get_sam_pixel_stats(device)
        self.gnn = None
        if use_gnn:
            ckpt = gnn_ckpt_path or gnn_checkpoint()
            cfg = {
                "model": {
                    "embed_dim": 256,
                    "gnn_hidden": 128,
                    "num_gnn_layers": 3,
                    "mask_size": mask_size,
                }
            }
            self.gnn = build_sam_stage1_refiner(cfg).to(device)
            state = torch.load(ckpt, map_location=device, weights_only=False)
            sd = state.get("state_dict", state)
            self.gnn.load_state_dict(sd, strict=False)
            ckpt_thresh = resolve_pseudo_confidence_threshold(state.get("config"))
            self.pseudo_confidence_threshold = (
                float(pseudo_confidence_threshold)
                if pseudo_confidence_threshold is not None
                else ckpt_thresh
            )
            if freeze_gnn:
                for p in self.gnn.parameters():
                    p.requires_grad = False
            else:
                for p in self.gnn.parameters():
                    p.requires_grad = True
        elif pseudo_confidence_threshold is not None:
            self.pseudo_confidence_threshold = float(pseudo_confidence_threshold)

    @torch.no_grad()
    def generate_pseudo_for_image(
        self,
        image: torch.Tensor,
        object_masks_np: List[np.ndarray],
        meta: dict,
        *,
        prompt_policy: str = "train_online",
        signal_type: str = "mixed",
        use_sam_cache: bool = True,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Returns list of binary pseudo masks (H,W) and category_ids (0 if unknown).
        Uses oracle-jitter prompts from GT geometry on weak split.
        """
        if not object_masks_np:
            return [], []
        masks_t = torch.stack(
            [
                torch.from_numpy(m.astype(np.float32)).to(self.device)
                for m in object_masks_np
            ],
            dim=0,
        )
        meta_ext = {
            "image_id": meta["image_id"],
            "ann_ids": meta.get("ann_ids", list(range(len(object_masks_np)))),
            "split": meta.get("split", "train"),
        }
        pseudo, _, _ = forward_teacher_objects(
            self.sam,
            self.gnn if self.use_gnn else None,
            image.to(self.device),
            masks_t,
            self.pixel_mean,
            self.pixel_std,
            self.mask_size,
            meta_ext,
            prompt_policy=prompt_policy,
            signal_type=signal_type,
            use_gnn=self.use_gnn,
            use_sam_cache=use_sam_cache,
            confidence_threshold=self.pseudo_confidence_threshold,
        )
        out_masks = []
        for i in range(pseudo.shape[0]):
            m = (pseudo[i, 0].detach().cpu().numpy() > 0).astype(np.uint8)
            out_masks.append(m)
        cats = meta.get("category_ids", [0] * len(out_masks))
        return out_masks, cats

    def forward_teacher_on_batch(
        self,
        images: torch.Tensor,
        masks_list: List[torch.Tensor],
        metas: List[dict],
        *,
        prompt_policy: str = "train_online",
        signal_type: str = "mixed",
        use_sam_cache: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run teacher for a batch of images; returns aggregated pseudo tensors."""
        all_pseudo = []
        for i in range(images.shape[0]):
            mask_np = [
                (masks_list[i][j].detach().cpu().numpy() > 0.5).astype(np.uint8)
                for j in range(masks_list[i].shape[0])
            ]
            pseudo_masks, _ = self.generate_pseudo_for_image(
                images[i],
                mask_np,
                metas[i],
                prompt_policy=prompt_policy,
                signal_type=signal_type,
                use_sam_cache=use_sam_cache,
            )
            if pseudo_masks:
                stacked = np.stack(pseudo_masks, axis=0)
                all_pseudo.append(torch.from_numpy(stacked).float())
        if not all_pseudo:
            return {"pseudo_masks": torch.zeros(0)}
        return {"pseudo_masks": all_pseudo}


def masks_to_detectron_instances(
    pseudo_masks: List[np.ndarray],
    category_ids: List[int],
    image_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """Convert pseudo masks to Detectron2 instance dicts (minimal)."""
    from detectron2.structures import BitMasks, Instances, Boxes

    instances = []
    h, w = image_size
    for mask, cat in zip(pseudo_masks, category_ids):
        if mask.sum() == 0:
            continue
        ys, xs = np.where(mask > 0)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        inst = {
            "bbox": [float(x0), float(y0), float(x1 + 1), float(y1 + 1)],
            "bbox_mode": 0,
            "category_id": int(cat) if cat else 1,
            "segmentation": mask.astype(np.uint8),
        }
        instances.append(inst)
    return instances
