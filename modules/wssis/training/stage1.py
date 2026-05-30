"""
Stage-1 GNN training with unified logging, checkpoints, early stopping, resume.

Pipeline (PLAN §2):
  SAM embed (node init) + image + SAM 3-mask proposals + weak signal → GNN → 3 refined masks
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch

from modules.wssis.paths import checkpoints_dir, sam_vit_b_checkpoint
from modules.wssis.run_context import EarlyStopping, RunContext, gpu_memory_mb


def _loss_components(criterion, logits: torch.Tensor, masks: torch.Tensor) -> Dict[str, float]:
    """Decompose CombinedSegLoss for metrics.jsonl."""
    bce = criterion.bce(logits, masks).item()
    dice = criterion.dice(logits, masks).item()
    return {
        "bce_loss": bce,
        "dice_loss": dice,
        "total_loss": criterion.bce_weight * bce + criterion.dice_weight * dice,
    }


def _stage1_forward_batch(
    sam,
    refiner,
    images,
    gt_masks,
    pixel_mean,
    pixel_std,
    mask_size,
    prompt_policy,
    signal_type,
    metas=None,
    active_signal=None,
    unified_weak_maps=True,
):
    """Run teacher (SAM decoder) + GNN refiner for one batch."""
    from modules.vig_refinenet.sam_stage1_common import (
        build_batch_prompts_from_masks,
        build_weak_signal_tensor,
        decode_sam_masks_3_batch,
        encode_sam_embeddings,
    )
    from modules.wssis.weak_prompts import sam_prompt_for_signal

    device = images.device
    prompt_signal = "mixed" if unified_weak_maps else signal_type
    prompts = build_batch_prompts_from_masks(
        gt_masks,
        policy=prompt_policy,
        signal_type=prompt_signal,
        metas=metas,
    )
    mask_np_list = [
        (gt_masks[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        for i in range(gt_masks.shape[0])
    ]

    if active_signal:
        sam_prompts = [sam_prompt_for_signal(p, active_signal) for p in prompts]
    else:
        sam_prompts = prompts

    with torch.no_grad():
        sam_embed = encode_sam_embeddings(sam, images, pixel_mean, pixel_std)
        sam_masks_3, sam_scores = decode_sam_masks_3_batch(
            sam,
            images,
            sam_prompts,
            mask_size=mask_size,
            prompt_space=mask_size,
        )
    weak_signal = build_weak_signal_tensor(
        prompts,
        spatial_size=mask_size,
        device=device,
        mask_np_list=mask_np_list,
        active_signal=active_signal if not unified_weak_maps else None,
        policy=prompt_policy,
    )
    refined_logits = refiner(sam_embed, images, sam_masks_3, weak_signal)
    return refined_logits, sam_masks_3, sam_scores, gt_masks.repeat(1, 3, 1, 1)


def train_stage1_gnn(
    config: dict,
    device: str = "cuda",
    output_name: str = "gnn_refiner_stage1.pt",
    run_ctx: Optional[RunContext] = None,
    resume: bool = False,
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
        RefinementMetricTracker,
        get_sam_pixel_stats,
        load_sam_vit_b,
        resolve_device,
        symmetric_loss,
    )
    from modules.vig_refinenet.sam_stage1_refiner import (
        build_sam_stage1_refiner,
        count_parameters,
    )
    from torch.utils.data import DataLoader

    training_cfg = config.get("training", {})
    run_name = config.get("visualization", {}).get("run_name") or output_name.replace(".pt", "")
    mask_size = config.get("data", {}).get("mask_size", 256)
    prompt_policy_train = training_cfg.get("prompt_policy", "train_online")
    prompt_policy_val = config.get("visualization", {}).get("prompt_policy", "val_fixed")
    signal_type = training_cfg.get("weak_signal", "mixed")

    ctx = run_ctx or RunContext(
        run_id=config.get("run_id"),
        run_dir=config.get("run_dir"),
        task="stage1_gnn",
    )
    ctx.save_config(config)
    ctx.update_step("stage1_gnn", {"status": "running", "epoch": 0})
    if config.get("logging", {}).get("tensorboard", True):
        ctx.init_tensorboard()
    if config.get("logging", {}).get("wandb", True):
        ctx.init_wandb(config)

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
        mask_size=mask_size,
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
    ctx.log("GNN params: %s", f"{count_parameters(refiner):,}")

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
    max_epochs = training_cfg.get("epochs", 20)
    start_epoch = 1
    history = []
    best_val_ap = -1.0

    es_cfg = config.get("early_stopping", {})
    early_stop = EarlyStopping(
        patience=es_cfg.get("patience", 0),
        monitor=es_cfg.get("monitor", "val_refined_ap"),
        mode=es_cfg.get("mode", "max"),
    )
    save_every = training_cfg.get("save_every_epochs", 1)

    legacy_ckpt = checkpoints_dir() / output_name

    if resume:
        ckpt = ctx.load_checkpoint()
        if ckpt:
            refiner.load_state_dict(ckpt["state_dict"], strict=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
            history = ckpt.get("history", [])
            best_val_ap = ckpt.get("best_val_refined_ap", ckpt.get("best_val_iou", best_val_ap))
            early_stop.best = ckpt.get("early_stop_best")
            early_stop.counter = ckpt.get("early_stop_counter", 0)
            ctx.log("Resumed at epoch %d (best_val_refined_ap=%.4f)", start_epoch, best_val_ap)

    viz_cfg = config.get("visualization", {})
    viz_enabled = viz_cfg.get("enabled", True)
    viz_samples = viz_cfg.get("num_samples", 4)
    if viz_enabled:
        from modules.wssis.training.visualize import visualize_stage1_epoch

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(dev)

        refiner.train()
        train_totals = {"bce_loss": 0.0, "dice_loss": 0.0, "total_loss": 0.0, "sym_loss": 0.0}
        n_batches = 0

        for images, masks, meta in train_loader:
            images, masks = images.to(dev), masks.to(dev)
            logits, _, _, gt3 = _stage1_forward_batch(
                sam,
                refiner,
                images,
                masks,
                pixel_mean,
                pixel_std,
                mask_size,
                prompt_policy_train,
                signal_type,
                metas=meta,
                unified_weak_maps=True,
            )
            loss = criterion(logits, gt3)
            sym_val = 0.0
            if sym_w > 0:
                sym_l = symmetric_loss(logits)
                loss = loss + sym_w * sym_l
                sym_val = sym_l.item()
            comps = _loss_components(criterion, logits.detach(), gt3)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_totals["bce_loss"] += comps["bce_loss"]
            train_totals["dice_loss"] += comps["dice_loss"]
            train_totals["total_loss"] += comps["total_loss"] + sym_w * sym_val
            train_totals["sym_loss"] += sym_val
            n_batches += 1

        for k in train_totals:
            train_totals[k] /= max(n_batches, 1)

        refiner.eval()
        val_totals = {"bce_loss": 0.0, "dice_loss": 0.0, "total_loss": 0.0}
        vn = 0
        tracker = RefinementMetricTracker()
        with torch.no_grad():
            for images, masks, meta in val_loader:
                images, masks = images.to(dev), masks.to(dev)
                logits, sam_masks_3, sam_scores, gt3 = _stage1_forward_batch(
                    sam,
                    refiner,
                    images,
                    masks,
                    pixel_mean,
                    pixel_std,
                    mask_size,
                    prompt_policy_val,
                    signal_type,
                    metas=meta,
                    unified_weak_maps=True,
                )
                comps = _loss_components(criterion, logits, gt3)
                for k in val_totals:
                    val_totals[k] += comps[k]
                vn += 1
                tracker.update(sam_masks_3, sam_scores, logits, masks)
        for k in val_totals:
            val_totals[k] /= max(vn, 1)

        val_metrics = tracker.compute()
        epoch_time = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "phase": "train",
            "train_loss": train_totals["total_loss"],
            "train_bce_loss": train_totals["bce_loss"],
            "train_dice_loss": train_totals["dice_loss"],
            "train_sym_loss": train_totals["sym_loss"],
            "val_loss": val_totals["total_loss"],
            "val_bce_loss": val_totals["bce_loss"],
            "val_dice_loss": val_totals["dice_loss"],
            "val_iou": val_metrics["refined_iou"],
            "raw_sam_iou": val_metrics["raw_sam_iou"],
            "refined_iou": val_metrics["refined_iou"],
            "delta_iou": val_metrics["delta_iou"],
            "raw_sam_ap": val_metrics["raw_sam_ap"],
            "val_refined_ap": val_metrics["refined_ap"],
            "delta_ap": val_metrics["delta_ap"],
            "raw_sam_ap50": val_metrics["raw_sam_ap50"],
            "refined_ap50": val_metrics["refined_ap50"],
            "delta_ap50": val_metrics["delta_ap50"],
            "epoch_time_s": epoch_time,
            "gpu_mem_mb": gpu_memory_mb(),
            "agreement_rate": 0.0,
        }
        history.append(row)
        ctx.log_metrics(row, step=epoch)
        ctx.log(
            "epoch %d train_loss=%.4f raw_AP=%.4f refined_AP=%.4f delta_AP=%+.4f (primary) time=%.1fs",
            epoch,
            row["train_loss"],
            row["raw_sam_ap"],
            row["val_refined_ap"],
            row["delta_ap"],
            epoch_time,
        )
        ctx.update_step(
            "stage1_gnn",
            {
                "status": "running",
                "epoch": epoch,
                "max_epochs": max_epochs,
                "val_refined_ap": row["val_refined_ap"],
                "delta_ap": row["delta_ap"],
            },
        )

        is_best = row["val_refined_ap"] > best_val_ap
        if is_best:
            best_val_ap = row["val_refined_ap"]

        payload = {
            "epoch": epoch,
            "config": config,
            "state_dict": refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best_val_refined_ap": best_val_ap,
            "early_stop_best": early_stop.best,
            "early_stop_counter": early_stop.counter,
        }
        ctx.save_checkpoint(payload, "last.pt", copy_to_legacy=legacy_ckpt if epoch == max_epochs else None)
        if is_best:
            ctx.save_checkpoint(payload, "best.pt", copy_to_legacy=legacy_ckpt)
        if save_every and epoch % save_every == 0:
            ctx.save_checkpoint(payload, f"epoch_{epoch:03d}.pt")

        if viz_enabled:
            visualize_stage1_epoch(
                epoch=epoch,
                dataset=val_ds,
                sam_model=sam,
                refiner=refiner,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
                device=dev,
                output_dir=ctx.viz_dir,
                num_samples=viz_samples,
                policy=prompt_policy_val,
            )

        improved = early_stop.step(row)
        if early_stop.patience > 0 and improved:
            ctx.log("EarlyStopping: new best %s=%.4f", early_stop.monitor, early_stop.best)
        if early_stop.should_stop:
            ctx.log("Early stopping at epoch %d (patience=%d)", epoch, early_stop.patience)
            break

    history_path = ctx.logs_dir / "metrics_history.json"
    import json

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    final_path = ctx.ckpt_dir / "best.pt"
    if not final_path.exists():
        final_path = ctx.ckpt_dir / "last.pt"
    shutil = __import__("shutil")
    checkpoints_dir().mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_path, legacy_ckpt)

    ctx.update_step("stage1_gnn", {"status": "done", "best_val_refined_ap": best_val_ap})
    ctx.finalize_report_bundle()
    ctx.close()
    ctx.log("Stage-1 done. Run bundle: %s", ctx.root)
    return legacy_ckpt
