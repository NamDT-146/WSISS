"""
Stage-1 GNN v2 training: triplet weak-signal rows, single-channel GNN I/O, KL + symmetric aux losses.

Pipeline (PLAN §2 v2):
  Per instance → 3 batch rows (point / scribble / box) → SAM 3 proposals + 1-ch weak → GNN → 1 mask
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


def _expand_gt_to_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Match GT [B,1,H,W] to logits [B,3,H,W] for per-head BCE/Dice logging."""
    if (
        logits.dim() == 4
        and masks.dim() == 4
        and logits.shape[1] > 1
        and masks.shape[1] == 1
    ):
        return masks.expand(-1, logits.shape[1], -1, -1)
    return masks


def _loss_components(
    criterion,
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    sym_weight: float = 0.0,
    sym_raw: float = 0.0,
) -> Dict[str, float]:
    """Decompose BCE, Dice, symmetric, and weighted totals for metrics.jsonl."""
    masks = _expand_gt_to_logits(logits, masks)
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
    metas=None,
    use_sam_cache: bool = True,
    sam_predictor=None,
):
    """Run SAM decoder + GNN for one batch (one weak-signal type per row in ``metas``)."""
    from modules.vig_refinenet.sam_stage1_common import (
        build_batch_prompts_from_masks,
        build_weak_signal_tensor,
        decode_sam_masks_3_batch,
    )
    from modules.wssis.sam_cache import fetch_sam_embeddings_batch
    from modules.wssis.weak_prompts import sam_prompt_for_signal

    device = images.device
    weak_types = [
        m.get("weak_signal_type", "points_only") for m in (metas or [])
    ]
    prompts = []
    sam_prompts = []
    for i in range(gt_masks.shape[0]):
        sig = weak_types[i] if i < len(weak_types) else "points_only"
        p = build_batch_prompts_from_masks(
            gt_masks[i : i + 1],
            policy=prompt_policy,
            signal_type=sig,
            metas=[metas[i]] if metas and i < len(metas) else None,
        )[0]
        prompts.append(p)
        sam_prompts.append(sam_prompt_for_signal(p, sig))

    mask_np_list = [
        (gt_masks[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        for i in range(gt_masks.shape[0])
    ]

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
    weak_chunks = []
    for i, sig in enumerate(weak_types):
        weak_chunks.append(
            build_weak_signal_tensor(
                [prompts[i]],
                spatial_size=mask_size,
                device=device,
                mask_np_list=[mask_np_list[i]],
                active_signal=sig,
                policy=prompt_policy,
            )
        )
    weak_signal = torch.cat(weak_chunks, dim=0)
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
        gt_masks,
        weak_signal,
        cache_stats,
    )


def train_stage1_gnn(
    config: dict,
    device: str = "cuda",
    output_name: str = "gnn_refiner_stage1_v2.pt",
    run_ctx: Optional[RunContext] = None,
    resume: bool = False,
    local_rank: int = 0,
    world_size: int = 1,
) -> Path:
    import sys
    from pathlib import Path as P

    repo = P(__file__).resolve().parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from modules.wssis.datasets.coco_image_dataset import (
        CocoInstanceDataset,
        collate_instance_triplets,
    )
    from modules.vig_refinenet.sam_stage1_common import (
        CombinedSegLoss,
        RefinementMetricTracker,
        get_sam_pixel_stats,
        load_sam_vit_b,
        resolve_device,
    )
    from modules.wssis.training.gnn_losses import Stage1LossWarmup, Stage1V2Loss
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
    from modules.wssis.pseudo_label_confidence import (
        agreement_rate,
        build_threshold_policy,
        over_threshold_ratio,
    )

    threshold_policy = build_threshold_policy(config)
    _pl_cfg = config.get("pseudo_label") or {}
    if "freematch_time_p" in _pl_cfg:
        threshold_policy._time_p = float(_pl_cfg["freematch_time_p"])

    from modules.wssis.stage1_distributed import (
        barrier,
        build_stage1_dataloader,
        stage1_is_main,
        stage1_world_size,
        state_dict_for_save,
        unwrap_refiner,
        wrap_stage1_refiner,
    )

    _world = world_size if world_size > 1 else stage1_world_size()
    _rank = local_rank
    _main = (_rank == 0) if _world > 1 else stage1_is_main()

    if _main:
        ctx = run_ctx or RunContext(
            run_id=config.get("run_id"),
            run_dir=config.get("run_dir"),
            task="stage1_gnn",
        )
    else:
        ctx = None
    ckpt_dir = (
        Path(config["run_dir"]) / "checkpoints"
        if config.get("run_dir")
        else (ctx.ckpt_dir if ctx is not None else checkpoints_dir())
    )

    if _main and ctx is not None:
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

    max_instances = data_cfg.get("max_instances")
    common = dict(
        coco_root=paths["coco_root"],
        img_size=data_cfg.get("img_size", 1024),
        mask_size=mask_size,
        max_images=max_images,
        max_objects_per_image=max_objects,
        max_instances=max_instances,
    )
    train_ds = CocoInstanceDataset(
        ann_json=paths["train_ann"],
        image_id_txt=paths["train_txt"],
        split="train",
        **common,
    )
    val_ds = CocoInstanceDataset(
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
    n_train_inst = len(train_ds)
    n_val_inst = len(val_ds)
    train_image_ids = {img_id for img_id, _ in train_ds.samples}
    val_image_ids = {img_id for img_id, _ in val_ds.samples}
    config["data"]["n_train_instances"] = n_train_inst
    config["data"]["n_val_instances"] = n_val_inst
    config["data"]["n_train_images"] = len(train_image_ids)
    config["data"]["n_val_images"] = len(val_image_ids)
    config["data"]["n_train_objects"] = n_train_inst
    config["data"]["n_val_objects"] = n_val_inst
    config["data"]["batch_unit"] = "instance"
    use_sam_cache = data_cfg.get("use_sam_embedding_cache", True)
    if _main and ctx is not None:
        if use_sam_cache and not sam_embeddings_dir().exists():
            ctx.log(
                "WARNING: SAM embedding cache missing (%s). Run P0.2 or set use_sam_embedding_cache=false.",
                sam_embeddings_dir(),
            )
        elif use_sam_cache:
            ctx.log("SAM embedding cache enabled (P0.2) — skips ViT-B encoder when npy exists")

    batch_size = training_cfg.get("batch_size", 4)  # instances per batch (not images)
    num_workers = data_cfg.get("num_workers", 4)
    stage1_max_steps = training_cfg.get("max_steps")
    max_epochs = training_cfg.get("epochs", 30)
    if smoke:
        batch_size = smoke.batch_size
        num_workers = min(num_workers, 0)
        stage1_max_steps = smoke.stage1_max_steps
        max_epochs = min(max_epochs, smoke.stage1_epochs)
    if _main:
        ctx.log(
            "Stage-1 train: %d instances (%d images) from %s; batch_size=%d instances/GPU "
            "(x3 weak signals = %d forward rows/batch; world_size=%d)",
            n_train_inst,
            len(train_image_ids),
            paths["train_txt"],
            batch_size,
            batch_size * 3,
            _world,
        )
        ctx.log(
            "Stage-1 val (early stop): %d instances (%d images) from %s (%s)",
            n_val_inst,
            len(val_image_ids),
            paths["val_txt"],
            val_spec["scope"],
        )
        ctx.save_config(config)

    train_loader, train_sampler = build_stage1_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_instance_triplets,
    )
    val_loader, _val_sampler = build_stage1_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_instance_triplets,
    )

    if _world > 1 and torch.cuda.is_available():
        dev = torch.device(f"cuda:{_rank}")
    else:
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
    if _main:
        ctx.log("GNN params: %s", f"{count_parameters(refiner):,}")

    seg_criterion = CombinedSegLoss(
        bce_weight=training_cfg.get("bce_weight", 1.0),
        dice_weight=training_cfg.get("dice_weight", 1.0),
    )
    wu_cfg = training_cfg.get("loss_warmup", {})
    loss_warmup = Stage1LossWarmup(
        warmup_epochs=int(wu_cfg.get("warmup_epochs", 5)),
        kl_start=float(wu_cfg.get("kl_start", 0.2)),
        kl_end=float(wu_cfg.get("kl_end", training_cfg.get("kl_weight", 0.1))),
        sym_start=float(wu_cfg.get("sym_start", 0.02)),
        sym_end=float(
            wu_cfg.get(
                "sym_end",
                training_cfg.get(
                    "sym_weight",
                    training_cfg.get("sym_triplet_weight", 0.1),
                ),
            )
        ),
    )
    v2_loss = Stage1V2Loss(
        seg_criterion,
        kl_weight=training_cfg.get("kl_weight", 0.1),
        sym_weight=training_cfg.get(
            "sym_weight", training_cfg.get("sym_triplet_weight", 0.1)
        ),
        loss_warmup=loss_warmup,
    )
    if _main:
        ctx.log(
            "Stage-1 loss: BCE+Dice (3 heads) + 9-proposal KL/sym; warmup %d epochs "
            "(kl %.2f->%.2f, sym %.2f->%.2f)",
            loss_warmup.warmup_epochs,
            loss_warmup.kl_start,
            loss_warmup.kl_end,
            loss_warmup.sym_start,
            loss_warmup.sym_end,
        )

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

    resume_ckpt = None
    if resume:
        ckpt_path = ckpt_dir / "last.pt"
        if ckpt_path.exists():
            resume_ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        elif _main and ctx is not None:
            resume_ckpt = ctx.load_checkpoint()
        if resume_ckpt:
            if resume_ckpt.get("wssis_ckpt_version", 1) < 3:
                raise RuntimeError(
                    "Checkpoint is pre-GNN-v2 (wssis_ckpt_version < 3). Re-run P0.4 for wssis_v2."
                )
            refiner.load_state_dict(resume_ckpt["state_dict"], strict=False)
            start_epoch = resume_ckpt.get("epoch", 0) + 1
            history = resume_ckpt.get("history", [])
            best_val_ap = resume_ckpt.get(
                "best_val_refined_ap", resume_ckpt.get("best_val_iou", best_val_ap)
            )
            early_stop.best = resume_ckpt.get("early_stop_best")
            early_stop.counter = resume_ckpt.get("early_stop_counter", 0)
            _pl_cfg = config.get("pseudo_label") or {}
            if "freematch_time_p" in _pl_cfg:
                threshold_policy._time_p = float(_pl_cfg["freematch_time_p"])
            if _main:
                ctx.log(
                    "Resumed at epoch %d (best_val_refined_ap=%.4f)",
                    start_epoch,
                    best_val_ap,
                )

    refiner = wrap_stage1_refiner(refiner, _rank)
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=training_cfg.get("lr", 1e-4),
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )
    if resume_ckpt and "optimizer" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
    barrier()

    viz_cfg = config.get("visualization", {})
    viz_enabled = viz_cfg.get("enabled", True)
    viz_samples = viz_cfg.get("num_samples", 4)
    if smoke:
        viz_samples = smoke.viz_samples
    if _world > 1 and viz_enabled and not viz_cfg.get("ddp_viz", False):
        viz_enabled = False
        if _main and ctx is not None:
            ctx.log(
                "DDP (%d GPUs): per-epoch viz disabled (set visualization.ddp_viz=true or use --no-viz). "
                "Rank 0-only viz caused other ranks to deadlock without a barrier.",
                _world,
            )
    if viz_enabled:
        from modules.wssis.training.visualize import visualize_stage1_epoch

    epoch_pbar = tqdm(
        range(start_epoch, max_epochs + 1),
        desc="Stage-1 GNN",
        unit="epoch",
        disable=not _main,
    )
    for epoch in epoch_pbar:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if _main and ctx is not None:
            ctx.log(
                "epoch %d/%d: train (all %d ranks) then val on rank 0",
                epoch,
                max_epochs,
                _world,
            )
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
            disable=not _main,
        )
        for images, masks, meta in train_pbar:
            images, masks = images.to(dev), masks.to(dev)
            logits, sam_masks_3, _, gt, weak_sig, cstats = _stage1_forward_batch(
                sam,
                refiner,
                images,
                masks,
                pixel_mean,
                pixel_std,
                mask_size,
                prompt_policy_train,
                metas=meta,
                use_sam_cache=use_sam_cache,
                sam_predictor=sam_predictor,
            )
            for k in cache_epoch:
                cache_epoch[k] += cstats.get(k, 0)
            loss, v2_comps = v2_loss(
                logits, gt, sam_masks_3, meta, epoch=epoch
            )
            comps = _loss_components(
                seg_criterion,
                logits.detach(),
                gt,
                sym_weight=0.0,
                sym_raw=v2_comps.get("sym_nine", 0.0),
            )
            comps["total"] = v2_comps["total"]
            comps["sym_raw"] = v2_comps.get("sym_nine", 0.0)
            comps["sym_weighted"] = v2_comps.get("sym_weighted", 0.0)
            comps["kl_weighted"] = v2_comps.get("kl_weighted", 0.0)
            comps["seg"] = v2_comps.get("seg", comps.get("seg_weighted", 0.0))

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
                total=f"{v2_comps['total']:.4f}",
                seg=f"{v2_comps['seg']:.4f}",
                kl=f"{v2_comps.get('kl_weighted', 0):.4f}",
                sym=f"{v2_comps.get('sym_weighted', 0):.4f}",
            )
            del loss, logits, gt, images, masks, meta, comps, sam_masks_3, weak_sig
            if stage1_max_steps and n_batches >= stage1_max_steps:
                break

        clear_sam_predictor(sam)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_mean = _mean_losses(train_totals, n_batches)
        if _main and ctx is not None and n_batches and use_sam_cache:
            ctx.log(
                "epoch %d SAM cache: hits=%d misses=%d unique_images=%d",
                epoch,
                cache_epoch["cache_hits"],
                cache_epoch["cache_misses"],
                cache_epoch["unique_images"],
            )

        val_mean = _mean_losses(_loss_totals_template(with_sym=False), 0)
        val_metrics = {
            "refined_iou": 0.0,
            "raw_sam_iou": 0.0,
            "delta_iou": 0.0,
            "raw_sam_ap": 0.0,
            "refined_ap": 0.0,
            "delta_ap": 0.0,
            "raw_sam_ap50": 0.0,
            "refined_ap50": 0.0,
            "delta_ap50": 0.0,
        }
        val_over_thresh_sum = 0.0
        val_agreement_sum = 0.0
        val_effective_thresh_sum = 0.0
        vn = 0

        if _main:
            refiner.eval()
            val_totals = _loss_totals_template(with_sym=False)
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
                    logits, sam_masks_3, sam_scores, gt, _, _ = _stage1_forward_batch(
                        sam,
                        refiner,
                        images,
                        masks,
                        pixel_mean,
                        pixel_std,
                        mask_size,
                        prompt_policy_val,
                        metas=meta,
                        use_sam_cache=use_sam_cache,
                        sam_predictor=sam_predictor,
                    )
                    comps = _loss_components(seg_criterion, logits, gt)
                    _accumulate_losses(val_totals, comps)
                    vn += 1
                    tracker.update(sam_masks_3, sam_scores, logits, gt)
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
                    del logits, sam_masks_3, sam_scores, gt, images, masks, meta, comps
            val_mean = _mean_losses(val_totals, vn)
            val_metrics = tracker.compute()

        clear_sam_predictor(sam)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        barrier()

        epoch_time = time.perf_counter() - t0
        stop_flag = False
        if _main:
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
            **{
                f"loss_{k}": v
                for k, v in loss_warmup.weights_for_epoch(epoch).items()
            },
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
            config.setdefault("pseudo_label", {})["freematch_time_p"] = pl_state.get(
                "time_p"
            )
            payload = {
                "epoch": epoch,
                "config": config,
                "state_dict": state_dict_for_save(refiner),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best_val_refined_ap": best_val_ap,
                "early_stop_best": early_stop.best,
                "early_stop_counter": early_stop.counter,
                "wssis_ckpt_version": 3,
                "sample_unit": "instance",
            }
            ctx.save_checkpoint(
                payload,
                "last.pt",
                copy_to_legacy=legacy_ckpt if epoch == max_epochs else None,
            )
            if is_best:
                ctx.save_checkpoint(payload, "best.pt", copy_to_legacy=legacy_ckpt)
            if save_every and epoch % save_every == 0:
                ctx.save_checkpoint(payload, f"epoch_{epoch:03d}.pt")

            if early_stop.patience > 0 and early_stop.step(row):
                ctx.log(
                    "EarlyStopping: new best %s=%.4f",
                    early_stop.monitor,
                    early_stop.best,
                )
            stop_flag = early_stop.should_stop
            if stop_flag:
                ctx.log(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch,
                    early_stop.patience,
                )

        # Rank 0 checkpoint/logging above is slow; non-zero ranks must not enter
        # collectives until rank 0 finishes or NCCL broadcast deadlocks.
        barrier()

        if _world > 1:
            import torch.distributed as dist

            stop_t = torch.tensor([int(stop_flag)], device=dev, dtype=torch.long)
            dist.broadcast(stop_t, src=0)
            stop_flag = bool(stop_t.item())
        if stop_flag:
            break

        if _main and viz_enabled:
            if _world > 1 and ctx is not None:
                ctx.log(
                    "Rank 0: epoch %d visualization (other GPUs synced after barrier)",
                    epoch,
                )
            visualize_stage1_epoch(
                epoch=epoch,
                dataset=val_ds,
                sam_model=sam,
                refiner=unwrap_refiner(refiner),
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
                device=dev,
                output_dir=ctx.viz_dir,
                num_samples=viz_samples,
                policy=prompt_policy_val,
                use_sam_cache=use_sam_cache,
                threshold_policy=threshold_policy,
            )

        # DDP: all ranks must finish rank-0-only work before the next train step.
        barrier()

        if _main:
            epoch_pbar.set_postfix(
                train_loss=f"{row['train_loss']:.4f}",
                refined_AP=f"{row['val_refined_ap']:.4f}",
                delta_AP=f"{row['delta_ap']:+.4f}",
            )

    barrier()

    if not _main:
        return legacy_ckpt

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
        ctx.log("Running holdout teacher eval (per-signal, training-matched)...")
        evaluate_teacher_on_val(
            gnn_ckpt=ckpt,
            run_ctx=ctx,
            use_labeled_5pct_holdout=True,
        )

    ctx.close()
    ctx.log("Stage-1 done. Run bundle: %s", ctx.root)
    return legacy_ckpt
