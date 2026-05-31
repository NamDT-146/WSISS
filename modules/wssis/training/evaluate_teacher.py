"""
Teacher evaluation on COCO val: raw SAM vs GNN-refined (PLAN §2 / EXPERIMENT Phase 3).

Reports instance-seg mask AP per weak-signal type:
  - boxes_only
  - points_only
  - scribbles_only

GNN checkpoint is trained once on labeled_5pct (unified 3-channel weak maps).
Eval sweeps signal type for SAM prompt + active weak-signal channel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.wssis.paths import build_coco_paths, checkpoints_dir, gnn_checkpoint, sam_vit_b_checkpoint
from modules.wssis.run_context import RunContext
from modules.wssis.weak_prompts import WEAK_SIGNAL_TYPES


def _forward_teacher_batch(
    sam,
    refiner,
    images,
    gt_masks,
    pixel_mean,
    pixel_std,
    mask_size,
    active_signal,
    metas,
):
    """SAM decoder (+ optional GNN) for one batch under a single weak-signal type."""
    from modules.vig_refinenet.sam_stage1_common import (
        build_batch_prompts_from_masks,
        build_weak_signal_tensor,
        decode_sam_masks_3_batch,
    )
    from modules.wssis.sam_cache import fetch_sam_embeddings_batch
    from modules.wssis.weak_prompts import sam_prompt_for_signal

    device = images.device
    prompts = build_batch_prompts_from_masks(
        gt_masks,
        policy="val_fixed",
        signal_type=active_signal,
        metas=metas,
    )
    mask_np_list = [
        (gt_masks[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        for i in range(gt_masks.shape[0])
    ]
    sam_prompts = [sam_prompt_for_signal(p, active_signal) for p in prompts]

    with torch.no_grad():
        sam_embed, _ = fetch_sam_embeddings_batch(
            metas,
            sam,
            images,
            pixel_mean,
            pixel_std,
            use_cache=True,
        )
        sam_masks_3, sam_scores = decode_sam_masks_3_batch(
            sam,
            images,
            sam_prompts,
            mask_size=mask_size,
            prompt_space=mask_size,
            image_embeddings=sam_embed,
        )

    refined_logits = None
    if refiner is not None:
        weak_signal = build_weak_signal_tensor(
            prompts,
            spatial_size=mask_size,
            device=device,
            mask_np_list=mask_np_list,
            active_signal=active_signal,
            policy="val_fixed",
        )
        refined_logits = refiner(sam_embed, images, sam_masks_3, weak_signal)

    return sam_masks_3, sam_scores, refined_logits


def evaluate_teacher_on_val(
    gnn_ckpt: Optional[Path] = None,
    device: str = "cuda",
    batch_size: int = 4,
    max_instances: Optional[int] = None,
    run_ctx: Optional[RunContext] = None,
    modes: tuple[str, ...] = ("raw_sam", "gnn_refined"),
    *,
    full_val: bool = False,
    use_labeled_5pct_holdout: bool = False,
    skip_if_done: bool = False,
) -> Dict:
    """
    Run val-set eval for raw SAM and/or GNN-refined teacher across all weak-signal types.

    Returns nested dict: results[mode][signal_type] -> metrics.
    """
    from modules.wssis.datasets.coco_image_dataset import CocoImageDataset, collate_image_to_instances
    from modules.vig_refinenet.sam_stage1_common import (
        RefinementMetricTracker,
        get_sam_pixel_stats,
        load_sam_vit_b,
        resolve_device,
    )
    from modules.vig_refinenet.sam_stage1_refiner import build_sam_stage1_refiner

    from modules.wssis.eval_splits import eval_report_name, resolve_eval_val_split

    paths = build_coco_paths()
    val_spec = resolve_eval_val_split(
        full_val=full_val,
        use_labeled_5pct_holdout=use_labeled_5pct_holdout,
    )
    ctx = run_ctx or RunContext(task="teacher_eval")
    report_name = eval_report_name(val_spec["scope"])
    out_path = ctx.eval_dir / report_name
    step_key = f"teacher_eval_{val_spec['scope']}"

    if skip_if_done and out_path.exists():
        ctx.log("Skipping teacher eval (%s exists)", out_path)
        if not ctx.is_step_done(step_key):
            ctx.update_step(step_key, {"status": "done", "report": str(out_path), "skipped": True})
        return json.loads(out_path.read_text(encoding="utf-8"))

    mask_size = 256
    img_size = 1024

    from modules.wssis.smoke_profile import get_smoke_profile

    smoke = get_smoke_profile()
    max_images = smoke.max_images if smoke else None
    max_objects = smoke.max_objects_per_image if smoke else None
    num_workers = 2
    if smoke:
        batch_size = smoke.batch_size
        num_workers = 0

    val_ds = CocoImageDataset(
        coco_root=paths["coco_root"],
        ann_json=val_spec["val_ann"],
        image_id_txt=val_spec["val_image_txt"],
        split=val_spec["image_split"],
        img_size=img_size,
        mask_size=mask_size,
        max_images=max_images,
        max_objects_per_image=max_objects,
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
    pixel_mean, pixel_std = get_sam_pixel_stats(dev)

    refiner = None
    gnn_cfg = {}
    if "gnn_refined" in modes:
        ckpt_path = gnn_ckpt or gnn_checkpoint()
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"GNN checkpoint not found: {ckpt_path}. Run P0.4 train_stage1_gnn first."
            )
        try:
            payload = torch.load(ckpt_path, map_location=dev, weights_only=False)
        except TypeError:
            payload = torch.load(ckpt_path, map_location=dev)
        if payload.get("wssis_ckpt_version", 1) < 2:
            raise RuntimeError(
                f"GNN checkpoint {ckpt_path} is pre-image-level (version "
                f"{payload.get('wssis_ckpt_version', 1)}). Re-run P0.4: "
                "python -m modules.wssis.prep.train_stage1_gnn --run-id <id>"
            )
        gnn_cfg = payload.get("config", {})
        refiner = build_sam_stage1_refiner(gnn_cfg).to(dev)
        refiner.load_state_dict(payload["state_dict"], strict=False)
        refiner.eval()

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for mode in modes:
        results[mode] = {}
        use_gnn = mode == "gnn_refined"

        for signal_type in WEAK_SIGNAL_TYPES:
            tracker = RefinementMetricTracker()
            eval_pbar = tqdm(
                val_loader,
                desc=f"Teacher eval {mode} | {signal_type}",
                leave=False,
                unit="batch",
            )
            with torch.no_grad():
                for images, masks, meta in eval_pbar:
                    images, masks = images.to(dev), masks.to(dev)
                    sam_masks_3, sam_scores, refined_logits = _forward_teacher_batch(
                        sam,
                        refiner if use_gnn else None,
                        images,
                        masks,
                        pixel_mean,
                        pixel_std,
                        mask_size,
                        signal_type,
                        meta,
                    )
                    if use_gnn and refined_logits is not None:
                        tracker.update(sam_masks_3, sam_scores, refined_logits, masks)
                    else:
                        tracker.update(sam_masks_3, sam_scores, sam_masks_3, masks)

            metrics = tracker.compute()
            n_inst = len(val_ds) if max_instances is None else min(max_instances, len(val_ds))

            if not use_gnn:
                results[mode][signal_type] = {
                    "iou": metrics["raw_sam_iou"],
                    "ap": metrics["raw_sam_ap"],
                    "ap50": metrics["raw_sam_ap50"],
                    "n_instances": n_inst,
                }
            else:
                results[mode][signal_type] = {
                    "raw_sam_iou": metrics["raw_sam_iou"],
                    "refined_iou": metrics["refined_iou"],
                    "delta_iou": metrics["delta_iou"],
                    "raw_sam_ap": metrics["raw_sam_ap"],
                    "refined_ap": metrics["refined_ap"],
                    "delta_ap": metrics["delta_ap"],
                    "raw_sam_ap50": metrics["raw_sam_ap50"],
                    "refined_ap50": metrics["refined_ap50"],
                    "delta_ap50": metrics["delta_ap50"],
                    "n_instances": n_inst,
                }

    report = {
        "dataset": "coco_val",
        "eval_scope": val_spec["scope"],
        "val_list": str(val_spec["val_image_txt"]),
        "gnn_checkpoint": str(gnn_ckpt or gnn_checkpoint()) if "gnn_refined" in modes else None,
        "weak_signal_types": list(WEAK_SIGNAL_TYPES),
        "training_note": "GNN trained on labeled_5pct with unified 3-channel weak maps (point+box+scribble)",
        "results": results,
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ctx.update_step(step_key, {"status": "done", "report": str(out_path)})
    ctx.log("Teacher eval report -> %s", out_path)
    for mode, by_sig in results.items():
        for sig, m in by_sig.items():
            if mode == "raw_sam":
                ctx.log("  [%s] %s  AP=%.4f  AP50=%.4f", mode, sig, m["ap"], m["ap50"])
            else:
                ctx.log(
                    "  [%s] %s  raw_AP=%.4f  refined_AP=%.4f  delta_AP=%+.4f",
                    mode,
                    sig,
                    m["raw_sam_ap"],
                    m["refined_ap"],
                    m["delta_ap"],
                )
    ctx.finalize_report_bundle(extra_files={"teacher_val_report.json": out_path})
    return report


def main(argv: list[str] | None = None) -> None:
    repo = Path(__file__).resolve().parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    parser = argparse.ArgumentParser(description="Evaluate raw SAM and GNN-refined teacher on val")
    parser.add_argument("--gnn-checkpoint", default=None, help="Path to gnn_refiner_stage1.pt")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-instances", type=int, default=None, help="Debug subset")
    parser.add_argument("--raw-only", action="store_true", help="Skip GNN-refined eval")
    parser.add_argument(
        "--full-val",
        action="store_true",
        help="Evaluate on full val_all (default: val_sample_20pct for speed)",
    )
    parser.add_argument(
        "--stage1-holdout",
        action="store_true",
        help="Evaluate on labeled_5pct_val holdout only",
    )
    parser.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Skip when report JSON already exists in the run eval dir",
    )
    args = parser.parse_args(argv)

    ctx = RunContext(run_id=args.run_id, run_dir=args.run_dir, task="teacher_eval")
    modes = ("raw_sam",) if args.raw_only else ("raw_sam", "gnn_refined")
    gnn_path = Path(args.gnn_checkpoint) if args.gnn_checkpoint else None

    evaluate_teacher_on_val(
        gnn_ckpt=gnn_path,
        device=args.device,
        batch_size=args.batch_size,
        max_instances=args.max_instances,
        run_ctx=ctx,
        modes=modes,
        full_val=args.full_val,
        use_labeled_5pct_holdout=args.stage1_holdout,
        skip_if_done=args.skip_if_done,
    )


if __name__ == "__main__":
    main()
