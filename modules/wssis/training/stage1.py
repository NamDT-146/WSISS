"""
Stage-1 GNN training loop (embed-only prototype until PLAN §0.5 fix).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from modules.wssis.paths import checkpoints_dir, gnn_checkpoint, sam_vit_b_checkpoint


def _symmetric_loss(logits: torch.Tensor) -> torch.Tensor:
    """MSE between single-mask logits duplicated as 3 channels (placeholder)."""
    p = torch.sigmoid(logits)
    return torch.tensor(0.0, device=logits.device)  # no multi-mask yet


def train_stage1_gnn(
    config: dict,
    device: str = "cuda",
    output_name: str = "gnn_refiner_stage1.pt",
) -> Path:
    import sys
    from pathlib import Path as P

    repo = P(__file__).resolve().parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from modules.vig_refinenet.coco_sam_stage1_dataset import (
        CocoSamStage1Dataset,
        collate_stage1,
    )
    from modules.vig_refinenet.sam_stage1_common import (
        CombinedSegLoss,
        MetricTracker,
        encode_sam_embeddings,
        get_sam_pixel_stats,
        load_sam_vit_b,
        resolve_device,
    )
    from modules.vig_refinenet.sam_stage1_refiner import build_sam_stage1_refiner, count_parameters
    from torch.utils.data import DataLoader

    data_cfg = config["data"]
    paths = {
        "coco_root": P(data_cfg["coco_root"]),
        "train_ann": P(data_cfg["coco_root"]) / "annotations" / "instances_train2017.json",
        "val_ann": P(data_cfg["coco_root"]) / "annotations" / "instances_val2017.json",
        "train_txt": P(data_cfg["train_image_txt"]),
        "val_txt": P(data_cfg["val_image_txt"]),
    }

    common = dict(
        coco_root=paths["coco_root"],
        img_size=data_cfg.get("img_size", 1024),
        mask_size=data_cfg.get("mask_size", 256),
        max_instances=data_cfg.get("max_instances"),
    )
    train_ds = CocoSamStage1Dataset(
        ann_json=paths["train_ann"],
        image_id_txt=paths["train_txt"],
        split="train",
        **common,
    )
    val_ds = CocoSamStage1Dataset(
        ann_json=paths["val_ann"],
        image_id_txt=paths["val_txt"],
        split="val",
        **common,
    )

    training_cfg = config["training"]
    batch_size = training_cfg.get("batch_size", 4)
    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_stage1,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_stage1,
    )

    dev = resolve_device(prefer_cuda=device.startswith("cuda"))
    sam_ckpt = sam_vit_b_checkpoint()
    if not sam_ckpt.exists():
        raise FileNotFoundError(f"Missing SAM weights: {sam_ckpt}")

    sam = load_sam_vit_b(str(sam_ckpt), dev)
    pixel_mean, pixel_std = get_sam_pixel_stats(dev)
    refiner = build_sam_stage1_refiner(config).to(dev)
    print(f"[stage1] GNN params: {count_parameters(refiner):,}")

    criterion = CombinedSegLoss(
        bce_weight=training_cfg.get("bce_weight", 1.0),
        dice_weight=training_cfg.get("dice_weight", 1.0),
    )
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=training_cfg.get("lr", 1e-4),
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )

    sym_w = training_cfg.get("symmetric_weight", 0.0) if config.get("use_symmetric_loss", True) else 0.0
    history = []

    viz_cfg = config.get("visualization", {})
    viz_enabled = viz_cfg.get("enabled", True)
    viz_samples = viz_cfg.get("num_samples", 4)
    viz_run_name = viz_cfg.get("run_name") or output_name.replace(".pt", "")

    if viz_enabled:
        from modules.wssis.training.visualize import visualize_stage1_epoch

    for epoch in range(1, training_cfg.get("epochs", 20) + 1):
        refiner.train()
        train_loss, n = 0.0, 0
        for images, masks, _ in train_loader:
            images, masks = images.to(dev), masks.to(dev)
            with torch.no_grad():
                sam_embed = encode_sam_embeddings(sam, images, pixel_mean, pixel_std)
            logits = refiner(sam_embed)
            loss = criterion(logits, masks)
            if sym_w > 0:
                loss = loss + sym_w * _symmetric_loss(logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n += 1

        refiner.eval()
        val_loss, vn = 0.0, 0
        tracker = MetricTracker()
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(dev), masks.to(dev)
                sam_embed = encode_sam_embeddings(sam, images, pixel_mean, pixel_std)
                logits = refiner(sam_embed)
                val_loss += criterion(logits, masks).item()
                vn += 1
                tracker.update(logits, masks)
        metrics = tracker.compute()
        metrics["loss"] = val_loss / max(vn, 1)
        row = {"epoch": epoch, "train_loss": train_loss / max(n, 1), **metrics}
        history.append(row)
        print(
            f"[stage1] epoch {epoch} train={row['train_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f} iou={metrics.get('iou', 0):.4f}"
        )

        if viz_enabled:
            visualize_stage1_epoch(
                epoch=epoch,
                dataset=val_ds,
                sam_model=sam,
                refiner=refiner,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
                device=dev,
                run_name=viz_run_name,
                num_samples=viz_samples,
                policy=viz_cfg.get("prompt_policy", "val_fixed"),
            )

    checkpoints_dir().mkdir(parents=True, exist_ok=True)
    out = checkpoints_dir() / output_name
    torch.save({"config": config, "state_dict": refiner.state_dict(), "history": history}, out)
    return out
