"""
P0.1 — Generate fixed train/val splits and val_prompts_fixed.json (seed=42).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from modules.wssis.paths import build_coco_paths, ensure_dirs, splits_dir
from modules.wssis.weak_prompts import build_instance_prompts

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


def _load_image_ids(txt_path: Path) -> list[int]:
    ids = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.search(r"(\d{12})", line)
            if m:
                ids.append(int(m.group(1)))
    return sorted(set(ids))


def _ids_to_txt_lines(ids: list[int], split: str) -> list[str]:
    return [f"./images/{split}2017/{i:012d}.jpg" for i in ids]


def _ann_to_mask(ann: dict, height: int, width: int) -> np.ndarray:
    seg = ann["segmentation"]
    if isinstance(seg, list) and mask_utils is not None:
        rle = mask_utils.frPyObjects(seg, height, width)
        m = mask_utils.decode(mask_utils.merge(rle))
        if m.ndim == 3:
            m = m.max(axis=2)
        return m.astype(np.uint8)
    raise ValueError("Need pycocotools for polygon masks")


def run(
    labeled_fraction: float = 0.05,
    seed: int = 42,
    force: bool = False,
) -> None:
    ensure_dirs()
    paths = build_coco_paths()

    if not paths["minitrain_train_txt"].exists():
        raise FileNotFoundError(
            f"Missing minitrain list: {paths['minitrain_train_txt']}. "
            "Run scripts/setup/01_download_data.py first."
        )

    out_labeled = paths["labeled_5pct_txt"]
    if out_labeled.exists() and not force:
        print(f"[P0.1] Splits already exist at {splits_dir()}; use --force to regenerate.")
        return

    # Copy full minitrain lists
    shutil.copy2(paths["minitrain_train_txt"], paths["train_all_txt"])
    shutil.copy2(paths["minitrain_val_txt"], paths["val_all_txt"])

    train_ids = _load_image_ids(paths["train_all_txt"])
    rng = np.random.RandomState(seed)
    n_labeled = max(1, int(round(len(train_ids) * labeled_fraction)))
    labeled_ids = sorted(rng.choice(train_ids, size=n_labeled, replace=False).tolist())
    labeled_set = set(labeled_ids)
    weak_ids = [i for i in train_ids if i not in labeled_set]

    paths["labeled_5pct_txt"].write_text(
        "\n".join(_ids_to_txt_lines(labeled_ids, "train")) + "\n", encoding="utf-8"
    )
    paths["weak_95pct_txt"].write_text(
        "\n".join(_ids_to_txt_lines(weak_ids, "train")) + "\n", encoding="utf-8"
    )

    # labeled_5pct.json — ann ids per image
    with open(paths["train_ann"], encoding="utf-8") as f:
        coco_train = json.load(f)

    img_by_id = {img["id"]: img for img in coco_train["images"] if img["id"] in labeled_set}
    labeled_anns: dict[str, list[int]] = defaultdict(list)
    for ann in coco_train["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann["image_id"] in labeled_set:
            labeled_anns[str(ann["image_id"])].append(ann["id"])

    paths["labeled_5pct_json"].write_text(
        json.dumps({"image_ids": labeled_ids, "anns_by_image": dict(labeled_anns)}, indent=2),
        encoding="utf-8",
    )

    # val_prompts_fixed.json
    with open(paths["val_ann"], encoding="utf-8") as f:
        coco_val = json.load(f)

    val_ids = set(_load_image_ids(paths["val_all_txt"]))
    val_images = {img["id"]: img for img in coco_val["images"] if img["id"] in val_ids}
    val_prompts: dict = {}

    for ann in coco_val["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        if img_id not in val_images:
            continue
        info = val_images[img_id]
        mask = _ann_to_mask(ann, info["height"], info["width"])
        prompts = build_instance_prompts(mask, policy="val_fixed")
        key = str(img_id)
        if key not in val_prompts:
            val_prompts[key] = {"instances": []}
        val_prompts[key]["instances"].append(
            {
                "ann_id": ann["id"],
                "category_id": ann["category_id"],
                **prompts,
            }
        )

    paths["val_prompts_json"].write_text(json.dumps(val_prompts, indent=2), encoding="utf-8")

    report = {
        "seed": seed,
        "labeled_fraction": labeled_fraction,
        "n_train_all": len(train_ids),
        "n_labeled_5pct": len(labeled_ids),
        "n_weak_95pct": len(weak_ids),
        "n_val": len(val_ids),
        "n_val_prompt_images": len(val_prompts),
    }
    paths["split_report_json"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[P0.1] Split generation complete:")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="P0.1 generate fixed COCO splits")
    parser.add_argument("--labeled-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(labeled_fraction=args.labeled_fraction, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
