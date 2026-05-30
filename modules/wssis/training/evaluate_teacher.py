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
        encode_sam_embeddings,
    )
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
        sam_embed = encode_sam_embeddings(sam, images, pixel_mean, pixel_std)
        sam_masks_3, sam_scores = decode_sam_masks_3_batch(
            sam,
            images,
            sam_prompts,
            mask_size=mask_size,
            prompt_space=mask_size,
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
) -> Dict:
    """
    Run val-set eval for raw SAM and/or GNN-refined teacher across all weak-signal types.

    Returns nested dict: results[mode][signal_type] -> metrics.
    """
    from modules.vig_refinenet.coco_sam_stage1_dataset import CocoSamStage1Dataset, collate_stage1
    from modules.vig_refinenet.sam_stage1_common import (
        RefinementMetricTracker,
        get_sam_pixel_stats,
        load_sam_vit_b,
        resolve_device,
    )
    from modules.vig_refinenet.sam_stage1_refiner import build_sam_stage1_refiner

    paths = build_coco_paths()
    mask_size = 256
    img_size = 1024

    val_ds = CocoSamStage1Dataset(
        coco_root=paths["coco_root"],
        ann_json=paths["val_ann"],
        image_id_txt=paths["val_all_txt"],
        split="val",
        img_size=img_size,
        mask_size=mask_size,
        max_instances=max_instances,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_stage1,
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
        payload = torch.load(ckpt_path, map_location=dev)
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
        "val_list": str(paths["val_all_txt"]),
        "gnn_checkpoint": str(gnn_ckpt or gnn_checkpoint()) if "gnn_refined" in modes else None,
        "weak_signal_types": list(WEAK_SIGNAL_TYPES),
        "training_note": "GNN trained on labeled_5pct with unified 3-channel weak maps (point+box+scribble)",
        "results": results,
    }

    ctx = run_ctx or RunContext(task="teacher_eval")
    out_path = ctx.eval_dir / "teacher_val_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
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
    )


if __name__ == "__main__":
    main()
