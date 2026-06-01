"""
Eval split selection: fast subset vs full val (PLAN / RUNBOOK policy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from modules.wssis.paths import build_coco_paths


def resolve_eval_val_split(
    *,
    full_val: bool = False,
    use_labeled_5pct_holdout: bool = False,
) -> Dict[str, object]:
    """
    Choose val image list + COCO ann file + image folder split.

    - Stage-1 training loop: use_labeled_5pct_holdout=True → labeled_5pct_val (train2017 images)
    - Fast eval (Stage-2 / routine): val_sample_20pct (~20% of val_all)
    - Final eval: full val_all
    """
    paths = build_coco_paths()

    if full_val:
        return {
            "val_image_txt": paths["val_all_txt"],
            "val_ann": paths["val_ann"],
            "image_split": "val",
            "scope": "full_val",
        }

    if use_labeled_5pct_holdout:
        return {
            "val_image_txt": paths["labeled_5pct_val_txt"],
            "val_ann": paths["train_ann"],
            "image_split": "train",
            "scope": "labeled_5pct_holdout",
        }

    return {
        "val_image_txt": paths["val_sample_20pct_txt"],
        "val_ann": paths["val_ann"],
        "image_split": "val",
        "scope": "val_sample_20pct",
    }


def eval_report_name(scope: str, *, unified_weak_maps: bool = False) -> str:
    suffix = "_unified" if unified_weak_maps else ""
    if scope == "full_val":
        return f"teacher_val_report_full{suffix}.json"
    if scope == "labeled_5pct_holdout":
        return f"teacher_val_report_stage1_holdout{suffix}.json"
    return f"teacher_val_report_subset{suffix}.json"
