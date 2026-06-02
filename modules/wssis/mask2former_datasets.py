"""Register WSSIS COCO splits with Detectron2 — same paths/filtering as P0 prep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from detectron2.config import CfgNode as CN
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data.datasets.coco import load_coco_json

from modules.vig_refinenet.coco_sam_stage1_dataset import (
    ann_to_mask,
    filter_coco_json,
    load_image_ids_from_txt,
)
from modules.wssis.eval_splits import resolve_eval_val_split
from modules.wssis.mask2former_dataloader import WssisSemiWeakDataset
from modules.wssis.paths import (
    build_coco_paths,
    load_weak_95pct_signal_map,
    resolve_coco_image_dir,
    resolve_experiment_train_image_txt,
)
from modules.wssis.smoke_profile import get_smoke_profile


def _coco_thing_classes() -> list[str]:
    return [c["name"] for c in COCO_CATEGORIES if int(c.get("isthing", 1)) == 1]


def _ensure_filtered_coco_json(
    source_ann: Path,
    image_id_txt: Path,
    out_path: Path,
    *,
    strip_annotations: bool = False,
) -> Path:
    """Write a filtered COCO json (same filter as CocoSamStage1Dataset / P0)."""
    source_ann = source_ann.resolve()
    image_id_txt = image_id_txt.resolve()
    if not source_ann.is_file():
        raise FileNotFoundError(
            f"COCO annotation file not found: {source_ann}\n"
            "Run: bash scripts/setup/01_download_data.sh"
        )
    if not image_id_txt.is_file():
        raise FileNotFoundError(
            f"Split image list not found: {image_id_txt}\n"
            "Run: python -m modules.wssis.prep.generate_splits"
        )

    out_path = out_path.resolve()
    cache_key = (strip_annotations,)
    if out_path.exists() and out_path.stat().st_mtime >= max(
        source_ann.stat().st_mtime, image_id_txt.stat().st_mtime
    ):
        if not strip_annotations:
            return out_path

    with open(source_ann, encoding="utf-8") as f:
        full_data = json.load(f)
    subset_ids = load_image_ids_from_txt(image_id_txt)
    filtered = filter_coco_json(full_data, subset_ids)
    if strip_annotations:
        filtered["annotations"] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(filtered), encoding="utf-8")
    return out_path


def _register_coco_instances(name: str, json_file: Path, image_root: Path) -> None:
    if name in DatasetCatalog.list():
        return

    json_file = json_file.resolve()
    image_root = image_root.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(
            f"COCO image root not found: {image_root}\n"
            "Run: bash scripts/setup/01_download_data.sh"
        )

    def loader() -> list[dict]:
        return load_coco_json(str(json_file), str(image_root), name)

    DatasetCatalog.register(name, loader)
    MetadataCatalog.get(name).set(
        json_file=str(json_file),
        image_root=str(image_root),
        evaluator_type="coco",
        thing_classes=_coco_thing_classes(),
    )


def _register_semi_weak_train(
    name: str,
    labeled_json: Path,
    weak_teacher_json: Path,
    image_root: Path,
    weak_signal_map: dict[str, str] | None = None,
) -> None:
    if name in DatasetCatalog.list():
        return

    labeled = load_coco_json(str(labeled_json), str(image_root), name + "_lbl")
    weak_full = load_coco_json(str(weak_teacher_json), str(image_root), name + "_w")
    signal_map = weak_signal_map or load_weak_95pct_signal_map()
    for rec in weak_full:
        sig = signal_map.get(str(rec.get("image_id", "")), "points_only")
        rec["wssis_weak_signal_type"] = sig

    smoke = get_smoke_profile()
    if smoke:
        labeled = labeled[: smoke.max_images]
        weak_full = weak_full[: smoke.max_images]

    def loader() -> list[dict]:
        ds = WssisSemiWeakDataset(labeled, weak_full)
        return [ds[i] for i in range(len(ds))]

    DatasetCatalog.register(name, loader)
    MetadataCatalog.get(name).set(
        json_file=str(labeled_json),
        image_root=str(image_root),
        evaluator_type="coco",
        thing_classes=_coco_thing_classes(),
    )


def wssis_dataset_names(experiment_id: str) -> Tuple[str, str]:
    exp = experiment_id or "default"
    return f"wssis_train_{exp}", f"wssis_val_{exp}"


def wssis_labeled_name(experiment_id: str) -> str:
    return f"wssis_labeled_{experiment_id or 'default'}"


def wssis_weak_name(experiment_id: str) -> str:
    return f"wssis_weak_{experiment_id or 'default'}"


def wssis_val_full_name(experiment_id: str) -> str:
    return f"wssis_val_full_{experiment_id or 'default'}"


def _register_val_split(
    name: str,
    val_spec: dict,
    paths,
    cache_dir: Path,
) -> None:
    val_txt = Path(val_spec["val_image_txt"])
    val_json = _ensure_filtered_coco_json(
        Path(val_spec["val_ann"]),
        val_txt,
        cache_dir / f"coco_instances_val_{val_spec['scope']}_{name}.json",
    )
    val_image_root = resolve_coco_image_dir(paths["coco_root"], val_spec["image_split"])
    _register_coco_instances(name, val_json, val_image_root)


def register_wssis_datasets(cfg: CN) -> Tuple[str, str]:
    """Register train/val datasets for experiment."""
    exp_id = cfg.WSSIS.EXPERIMENT_ID
    train_name, val_name = wssis_dataset_names(exp_id)
    paths = build_coco_paths()
    cache_dir = Path(cfg.OUTPUT_DIR).parent
    train_image_root = resolve_coco_image_dir(paths["coco_root"], "train")

    use_semi = getattr(cfg.WSSIS, "USE_SEMI_WEAK", False)

    if use_semi:
        labeled_txt = paths["labeled_5pct_txt"]
        weak_txt = paths["weak_95pct_txt"]
        labeled_json = _ensure_filtered_coco_json(
            paths["train_ann"],
            labeled_txt,
            cache_dir / f"coco_instances_labeled_{exp_id}.json",
        )
        weak_teacher_json = _ensure_filtered_coco_json(
            paths["train_ann"],
            weak_txt,
            cache_dir / f"coco_instances_weak_{exp_id}_teacher.json",
            strip_annotations=False,
        )
        _ensure_filtered_coco_json(
            paths["train_ann"],
            weak_txt,
            cache_dir / f"coco_instances_weak_{exp_id}_images.json",
            strip_annotations=True,
        )
        _register_semi_weak_train(
            train_name,
            labeled_json,
            weak_teacher_json,
            train_image_root,
        )
    else:
        train_txt = resolve_experiment_train_image_txt(
            cfg.WSSIS.LABELED_SPLIT,
            cfg.WSSIS.WEAK_SPLIT,
        )
        train_json = _ensure_filtered_coco_json(
            paths["train_ann"],
            train_txt,
            cache_dir / f"coco_instances_train_{train_txt.stem}.json",
        )
        _register_coco_instances(train_name, train_json, train_image_root)

    val_split = resolve_eval_val_split(full_val=False)
    _register_val_split(val_name, val_split, paths, cache_dir)

    full_val_name = wssis_val_full_name(exp_id)
    full_val_split = resolve_eval_val_split(full_val=True)
    _register_val_split(full_val_name, full_val_split, paths, cache_dir)

    return train_name, val_name


def ensure_wssis_datasets_in_cfg(cfg: CN) -> None:
    """Register datasets and point cfg.DATASETS at WSSIS names when configured."""
    if not cfg.WSSIS.EXPERIMENT_ID:
        return

    train_name, val_name = register_wssis_datasets(cfg)
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)


def coco_anns_to_masks_for_image(
    anns: List[dict],
    height: int,
    width: int,
    max_objects: int | None = None,
) -> Tuple[List, List[int], List[int]]:
    """Extract instance masks and ids from COCO anns."""
    masks, ann_ids, cats = [], [], []
    for ann in anns:
        if ann.get("iscrowd", 0):
            continue
        masks.append(ann_to_mask(ann, height, width))
        ann_ids.append(int(ann["id"]))
        cats.append(int(ann.get("category_id", 1)))
        if max_objects and len(masks) >= max_objects:
            break
    return masks, ann_ids, cats
