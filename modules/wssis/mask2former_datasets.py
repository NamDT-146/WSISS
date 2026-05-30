"""Register WSSIS COCO splits with Detectron2 for Mask2Former training."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Set, Tuple

from detectron2.config import CfgNode as CN
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data.datasets.coco import load_coco_json

from modules.wssis.eval_splits import resolve_eval_val_split
from modules.wssis.paths import build_coco_paths


def load_image_ids_from_txt(txt_path: Path) -> Set[int]:
    ids: Set[int] = set()
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.search(r"(\d{12})", line)
            if match:
                ids.add(int(match.group(1)))
    return ids


def _coco_thing_classes() -> list[str]:
    return [c["name"] for c in COCO_CATEGORIES if int(c.get("isthing", 1)) == 1]


def _register_filtered_coco(
    name: str,
    *,
    json_file: Path,
    image_root: Path,
    image_ids: Optional[Set[int]],
) -> None:
    if name in DatasetCatalog.list():
        return

    json_file = json_file.resolve()
    image_root = image_root.resolve()
    if not json_file.is_file():
        raise FileNotFoundError(
            f"COCO annotation file not found: {json_file}\n"
            "Run: bash scripts/setup/01_download_data.sh"
        )
    if not image_root.is_dir():
        raise FileNotFoundError(
            f"COCO image root not found: {image_root}\n"
            "Run: bash scripts/setup/01_download_data.sh"
        )

    allowed = image_ids

    def loader() -> list[dict]:
        records = load_coco_json(str(json_file), str(image_root), name)
        if allowed is None:
            return records
        return [record for record in records if record["image_id"] in allowed]

    DatasetCatalog.register(name, loader)
    MetadataCatalog.get(name).set(
        json_file=str(json_file),
        image_root=str(image_root),
        evaluator_type="coco",
        thing_classes=_coco_thing_classes(),
    )


def resolve_train_image_ids(cfg: CN) -> Set[int]:
    paths = build_coco_paths()
    ids: Set[int] = set()

    labeled = cfg.WSSIS.LABELED_SPLIT
    if labeled == "train_all":
        ids |= load_image_ids_from_txt(paths["train_all_txt"])
    elif labeled == "labeled_5pct":
        ids |= load_image_ids_from_txt(paths["labeled_5pct_txt"])

    if cfg.WSSIS.WEAK_SPLIT == "weak_95pct":
        ids |= load_image_ids_from_txt(paths["weak_95pct_txt"])

    if not ids:
        raise ValueError(
            "No training images resolved from WSSIS split config "
            f"(labeled_split={labeled!r}, weak_split={cfg.WSSIS.WEAK_SPLIT!r})"
        )
    return ids


def wssis_dataset_names(experiment_id: str) -> Tuple[str, str]:
    exp = experiment_id or "default"
    return f"wssis_train_{exp}", f"wssis_val_{exp}"


def register_wssis_datasets(cfg: CN) -> Tuple[str, str]:
    """Register train/val COCO datasets under data/coco2017 for this experiment."""
    exp_id = cfg.WSSIS.EXPERIMENT_ID
    train_name, val_name = wssis_dataset_names(exp_id)
    paths = build_coco_paths()

    train_ids = resolve_train_image_ids(cfg)
    _register_filtered_coco(
        train_name,
        json_file=paths["train_ann"],
        image_root=paths["coco_root"] / "images" / "train2017",
        image_ids=train_ids,
    )

    val_split = resolve_eval_val_split(full_val=False)
    val_ids = load_image_ids_from_txt(Path(val_split["val_image_txt"]))
    val_split_name = val_split["image_split"]
    _register_filtered_coco(
        val_name,
        json_file=Path(val_split["val_ann"]),
        image_root=paths["coco_root"] / "images" / f"{val_split_name}2017",
        image_ids=val_ids,
    )
    return train_name, val_name


def ensure_wssis_datasets_in_cfg(cfg: CN) -> None:
    """Register datasets and point cfg.DATASETS at WSSIS names when configured."""
    if not cfg.WSSIS.EXPERIMENT_ID:
        return

    train_name, val_name = register_wssis_datasets(cfg)
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
