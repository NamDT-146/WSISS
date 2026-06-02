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
from tqdm import tqdm
import numpy as np

from modules.wssis.paths import checkpoints_dir, sam_embeddings_dir, sam_vit_b_checkpoint
from modules.wssis.run_context import EarlyStopping, RunContext, gpu_memory_mb


def _loss_components(
    criterion,
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    sym_weight: float = 0.0,
    sym_raw: float = 0.0,
) -> Dict[str, float]:
    """Decompose BCE, Dice, symmetric, and weighted totals for metrics.jsonl."""
    bce_raw = criterion.bce(logits, masks).item()
    dice_raw = criterion.dice(logits, masks).item()
    bce_weighted = criterion.bce_weight * bce_raw
    dice_weighted = criterion.dice_weight * dice_raw
    seg_weighted = bce_weighted + dice_weighted
    sym_weighted = sym_weight * sym_raw
    return {
        "bce_raw": bce_raw,
        "dice_raw": dice_raw,
        "bce_weighted": bce_weighted,
        "dice_weighted": dice_weighted,
        "seg_weighted": seg_weighted,
        "sym_raw": sym_raw,
        "sym_weighted": sym_weighted,
        "total": seg_weighted + sym_weighted,
    }


def _loss_totals_template(*, with_sym: bool) -> Dict[str, float]:
    keys = ["bce_raw", "dice_raw", "bce_weighted", "dice_weighted", "seg_weighted", "total"]
    if with_sym:
        keys.extend(["sym_raw", "sym_weighted"])
    return {k: 0.0 for k in keys}


def _accumulate_losses(totals: Dict[str, float], comps: Dict[str, float]) -> None:
    for k in totals:
        totals[k] += comps[k]


def _mean_losses(totals: Dict[str, float], n: int) -> Dict[str, float]:
    if n <= 0:
        return totals
    return {k: v / n for k, v in totals.items()}


def _prefix_loss_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": metrics[k] for k in metrics}


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
    use_sam_cache: bool = True,
    sam_predictor=None,
):
    """Run teacher (SAM decoder) + GNN refiner for one batch."""
    from modules.vig_refinenet.sam_stage1_common import (
        build_batch_prompts_from_masks,
        build_weak_signal_tensor,
        decode_sam_masks_3_batch,
    )
    from modules.wssis.sam_cache import fetch_sam_embeddings_batch
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
        sam_embed, cache_stats = fetch_sam_embeddings_batch(
            metas or [],
            sam,
            images,
            pixel_mean,
            pixel_std,
            use_cache=use_sam_cache,
        )
        sam_masks_3, sam_scores = decode_sam_masks_3_batch(
            sam,
            images,
            sam_prompts,
            mask_size=mask_size,
            prompt_space=mask_size,
            image_embeddings=sam_embed,
            predictor=sam_predictor,
        )
    weak_signal = build_weak_signal_tensor(
        prompts,
        spatial_size=mask_size,
        device=device,
        mask_np_list=mask_np_list,
        active_signal=active_signal if not unified_weak_maps else None,
        policy=prompt_policy,
    )
    refined_logits = refiner(
        sam_embed.detach(),
        images,
        sam_masks_3.detach(),
        weak_signal,
    )
    return (
        refined_logits,
        sam_masks_3,
        sam_scores,
        gt_masks.repeat(1, 3, 1, 1),
        cache_stats,
    )


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

    from modules.wssis.datasets.coco_image_dataset import (
        CocoImageDataset,
        collate_image_to_instances,
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
    from modules.wssis.pseudo_label_confidence import (
        agreement_rate,
        build_threshold_policy,
        over_threshold_ratio,
    )

    threshold_policy = build_threshold_policy(config)
    _pl_cfg = config.get("pseudo_label") or {}
    if "freematch_time_p" in _pl_cfg:
        threshold_policy._time_p = float(_pl_cfg["freematch_time_p"])

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

    from modules.wssis.eval_splits import resolve_eval_val_split

    data_cfg = config["data"]
    use_labeled_holdout = data_cfg.get("val_use_labeled_holdout", True)
    val_spec = resolve_eval_val_split(use_labeled_5pct_holdout=use_labeled_holdout)
    paths = {
        "coco_root": P(data_cfg["coco_root"]),
        "train_ann": P(data_cfg["coco_root"]) / "annotations" / "instances_train2017.json",
        "val_ann": val_spec["val_ann"],
        "train_txt": P(data_cfg["train_image_txt"]),
        "val_txt": P(val_spec["val_image_txt"]),
        "val_image_split": val_spec["image_split"],
    }

    from modules.wssis.smoke_profile import get_smoke_profile

    smoke = get_smoke_profile()
    max_images = data_cfg.get("max_images")
    max_objects = data_cfg.get("max_objects_per_image")
    if smoke:
        max_images = smoke.max_images
        max_objects = smoke.max_objects_per_image

    common = dict(
        coco_root=paths["coco_root"],
        img_size=data_cfg.get("img_size", 1024),
        mask_size=mask_size,
        max_images=max_images,
        max_objects_per_image=max_objects,
    )
    train_ds = CocoImageDataset(
        ann_json=paths["train_ann"],
        image_id_txt=paths["train_txt"],
        split="train",
        **common,
    )
    val_ds = CocoImageDataset(
        ann_json=paths["val_ann"],
        image_id_txt=paths["val_txt"],
        split=paths["val_image_split"],
        **common,
    )

    if paths["train_txt"].name != "labeled_5pct_train.txt":
        raise ValueError(
            "Stage-1 GNN must train on labeled_5pct_train; "
            f"got train_image_txt={paths['train_txt']}. Run P0.1 generate_splits --force"
        )

    config.setdefault("data", {})["train_split"] = data_cfg.get("train_split", "labeled_5pct_train")
    config["data"]["val_split"] = data_cfg.get("val_split", "labeled_5pct_val")
    config["data"]["val_eval_scope"] = val_spec["scope"]
    config["data"]["n_train_images"] = len(train_ds)
    config["data"]["n_val_images"] = len(val_ds)
    n_train_obj = sum(len(s[1]) for s in train_ds.samples)
    n_val_obj = sum(len(s[1]) for s in val_ds.samples)
    config["data"]["n_train_objects"] = n_train_obj
    config["data"]["n_val_objects"] = n_val_obj
    ctx.log(
        "Stage-1 train: %d images (%d objects) from %s",
        len(train_ds),
        n_train_obj,
        paths["train_txt"],
    )
    ctx.log(
        "Stage-1 val (early stop): %d images (%d objects) from %s (%s)",
        len(val_ds),
        n_val_obj,
        paths["val_txt"],
        val_spec["scope"],
    )
    use_sam_cache = data_cfg.get("use_sam_embedding_cache", True)
    if use_sam_cache and not sam_embeddings_dir().exists():
        ctx.log(
            "WARNING: SAM embedding cache missing (%s). Run P0.2 or set use_sam_embedding_cache=false.",
            sam_embeddings_dir(),
        )
    elif use_sam_cache:
        ctx.log("SAM embedding cache enabled (P0.2) — skips ViT-B encoder when npy exists")
    ctx.save_config(config)

    batch_size = training_cfg.get("batch_size", 4)
    num_workers = data_cfg.get("num_workers", 4)
    stage1_max_steps = training_cfg.get("max_steps")
    max_epochs = training_cfg.get("epochs", 30)
    if smoke:
        batch_size = smoke.batch_size
        num_workers = min(num_workers, 0)
        stage1_max_steps = smoke.stage1_max_steps
        max_epochs = min(max_epochs, smoke.stage1_epochs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_image_to_instances,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_image_to_instances,
    )

    dev = resolve_device(prefer_cuda=device.startswith("cuda"))
    sam_ckpt = sam_vit_b_checkpoint()
    if not sam_ckpt.exists():
        raise FileNotFoundError(f"Missing SAM weights: {sam_ckpt}")

    sam = load_sam_vit_b(str(sam_ckpt), dev)
    sam.eval()
    for p in sam.parameters():
        p.requires_grad = False
    from modules.vig_refinenet.sam_stage1_common import get_sam_predictor, clear_sam_predictor

    sam_predictor = get_sam_predictor(sam)
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
            if ckpt.get("wssis_ckpt_version", 1) < 2:
                raise RuntimeError(
                    "Checkpoint is pre-image-level (per-instance). Re-run P0.4 after True SWSIS pull."
                )
            refiner.load_state_dict(ckpt["state_dict"], strict=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
            history = ckpt.get("history", [])
            best_val_ap = ckpt.get("best_val_refined_ap", ckpt.get("best_val_iou", best_val_ap))
            early_stop.best = ckpt.get("early_stop_best")
            early_stop.counter = ckpt.get("early_stop_counter", 0)
            _pl_cfg = config.get("pseudo_label") or {}
            if "freematch_time_p" in _pl_cfg:
                threshold_policy._time_p = float(_pl_cfg["freematch_time_p"])
            ctx.log("Resumed at epoch %d (best_val_refined_ap=%.4f)", start_epoch, best_val_ap)

    viz_cfg = config.get("visualization", {})
    viz_enabled = viz_cfg.get("enabled", True)
    viz_samples = viz_cfg.get("num_samples", 4)
    if smoke:
        viz_samples = smoke.viz_samples
    if viz_enabled:
        from modules.wssis.training.visualize import visualize_stage1_epoch

    epoch_pbar = tqdm(
        range(start_epoch, max_epochs + 1),
        desc="Stage-1 GNN",
        unit="epoch",
    )
    for epoch in epoch_pbar:
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(dev)

        refiner.train()
        train_totals = _loss_totals_template(with_sym=True)
        n_batches = 0
        train_over_thresh_sum = 0.0
        train_agreement_sum = 0.0
        train_effective_thresh_sum = 0.0
        cache_epoch = {"cache_hits": 0, "cache_misses": 0, "unique_images": 0}

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{max_epochs} train",
            leave=False,
            unit="batch",
        )
        for images, masks, meta in train_pbar:
            images, masks = images.to(dev), masks.to(dev)
            logits, _, _, gt3, cstats = _stage1_forward_batch(
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
                use_sam_cache=use_sam_cache,
                sam_predictor=sam_predictor,
            )
            for k in cache_epoch:
                cache_epoch[k] += cstats.get(k, 0)
            loss = criterion(logits, gt3)
            sym_val = 0.0
            if sym_w > 0:
                sym_l = symmetric_loss(logits)
                loss = loss + sym_w * sym_l
                sym_val = sym_l.item()
            comps = _loss_components(
                criterion,
                logits.detach(),
                gt3,
                sym_weight=sym_w,
                sym_raw=sym_val,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            _accumulate_losses(train_totals, comps)
            n_batches += 1
            with torch.no_grad():
                train_effective_thresh_sum += threshold_policy.effective_threshold(
                    logits, update=True
                )
                train_over_thresh_sum += over_threshold_ratio(
                    logits, threshold_policy=threshold_policy, update=False
                )
                train_agreement_sum += agreement_rate(
                    logits, threshold_policy=threshold_policy, update=False
                )
            train_pbar.set_postfix(
                total=f"{comps['total']:.4f}",
                seg=f"{comps['seg_weighted']:.4f}",
                sym=f"{comps['sym_weighted']:.4f}",
            )
            del loss, logits, gt3, images, masks, meta, comps
            if stage1_max_steps and n_batches >= stage1_max_steps:
                break

        clear_sam_predictor(sam)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_mean = _mean_losses(train_totals, n_batches)
        if n_batches and use_sam_cache:
            ctx.log(
                "epoch %d SAM cache: hits=%d misses=%d unique_images=%d",
                epoch,
                cache_epoch["cache_hits"],
                cache_epoch["cache_misses"],
                cache_epoch["unique_images"],
            )

        refiner.eval()
        val_totals = _loss_totals_template(with_sym=False)
        vn = 0
        val_over_thresh_sum = 0.0
        val_agreement_sum = 0.0
        val_effective_thresh_sum = 0.0
        tracker = RefinementMetricTracker()
        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{max_epochs} val",
            leave=False,
            unit="batch",
        )
        with torch.no_grad():
            for images, masks, meta in val_pbar:
                images, masks = images.to(dev), masks.to(dev)
                logits, sam_masks_3, sam_scores, gt3, _ = _stage1_forward_batch(
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
                    use_sam_cache=use_sam_cache,
                    sam_predictor=sam_predictor,
                )
                comps = _loss_components(criterion, logits, gt3)
                _accumulate_losses(val_totals, comps)
                vn += 1
                tracker.update(sam_masks_3, sam_scores, logits, masks)
                val_effective_thresh_sum += threshold_policy.effective_threshold(
                    logits, update=False
                )
                val_over_thresh_sum += over_threshold_ratio(
                    logits, threshold_policy=threshold_policy, update=False
                )
                val_agreement_sum += agreement_rate(
                    logits, threshold_policy=threshold_policy, update=False
                )
                val_pbar.set_postfix(
                    total=f"{comps['total']:.4f}",
                    bce_w=f"{comps['bce_weighted']:.4f}",
                    dice_w=f"{comps['dice_weighted']:.4f}",
                )
                del logits, sam_masks_3, sam_scores, gt3, images, masks, meta, comps
        clear_sam_predictor(sam)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        val_mean = _mean_losses(val_totals, vn)

        val_metrics = tracker.compute()
        epoch_time = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "phase": "train",
            **_prefix_loss_metrics(train_mean, "train"),
            **_prefix_loss_metrics(val_mean, "val"),
            # Legacy aliases (total + raw BCE/Dice/sym)
            "train_loss": train_mean["total"],
            "train_bce_loss": train_mean["bce_raw"],
            "train_dice_loss": train_mean["dice_raw"],
            "train_sym_loss": train_mean["sym_raw"],
            "val_loss": val_mean["total"],
            "val_bce_loss": val_mean["bce_raw"],
            "val_dice_loss": val_mean["dice_raw"],
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
            "pseudo_threshold_mode": threshold_policy.mode,
            "pseudo_confidence_threshold": threshold_policy.p_cutoff,
            "train_effective_pseudo_threshold": train_effective_thresh_sum / max(
                n_batches, 1
            ),
            "val_effective_pseudo_threshold": val_effective_thresh_sum / max(vn, 1),
            "train_over_threshold_ratio": train_over_thresh_sum / max(n_batches, 1),
            "val_over_threshold_ratio": val_over_thresh_sum / max(vn, 1),
            "train_agreement_rate": train_agreement_sum / max(n_batches, 1),
            "agreement_rate": val_agreement_sum / max(vn, 1),
        }
        history.append(row)
        ctx.log_metrics(row, step=epoch)
        ctx.log(
            "epoch %d train_total=%.4f (bce_w=%.4f dice_w=%.4f sym_w=%.4f) "
            "val_total=%.4f raw_AP=%.4f refined_AP=%.4f delta_AP=%+.4f time=%.1fs",
            epoch,
            row["train_loss"],
            row["train_bce_weighted"],
            row["train_dice_weighted"],
            row["train_sym_weighted"],
            row["val_loss"],
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

        pl_state = threshold_policy.state_dict()
        config.setdefault("pseudo_label", {})["freematch_time_p"] = pl_state.get("time_p")
        payload = {
            "epoch": epoch,
            "config": config,
            "state_dict": refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best_val_refined_ap": best_val_ap,
            "early_stop_best": early_stop.best,
            "early_stop_counter": early_stop.counter,
            "wssis_ckpt_version": 2,
            "sample_unit": "image",
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
                use_sam_cache=use_sam_cache,
                threshold_policy=threshold_policy,
            )

        epoch_pbar.set_postfix(
            train_loss=f"{row['train_loss']:.4f}",
            refined_AP=f"{row['val_refined_ap']:.4f}",
            delta_AP=f"{row['delta_ap']:+.4f}",
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

    if config.get("run_final_eval", True):
        from modules.wssis.training.evaluate_teacher import evaluate_teacher_on_val

        ckpt = final_path if final_path.exists() else legacy_ckpt
        ctx.log("Running final teacher eval on full val_all (per-signal ablation)...")
        evaluate_teacher_on_val(
            gnn_ckpt=ckpt,
            run_ctx=ctx,
            full_val=True,
        )
        ctx.log(
            "Running holdout teacher eval (unified weak maps, matches val metrics.jsonl)..."
        )
        evaluate_teacher_on_val(
            gnn_ckpt=ckpt,
            run_ctx=ctx,
            use_labeled_5pct_holdout=True,
            unified_weak_maps=True,
        )

    ctx.close()
    ctx.log("Stage-1 done. Run bundle: %s", ctx.root)
    return legacy_ckpt
