"""Register WSSIS COCO splits with Detectron2 — same paths/filtering as P0 prep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from detectron2.config import CfgNode as CN
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data.datasets.coco import load_coco_json

from modules.vig_refinenet.coco_sam_stage1_dataset import (
    filter_coco_json,
    load_image_ids_from_txt,
)
from modules.wssis.eval_splits import resolve_eval_val_split
from modules.wssis.paths import (
    build_coco_paths,
    resolve_coco_image_dir,
    resolve_experiment_train_image_txt,
)


def _coco_thing_classes() -> list[str]:
    return [c["name"] for c in COCO_CATEGORIES if int(c.get("isthing", 1)) == 1]


def _ensure_filtered_coco_json(
    source_ann: Path,
    image_id_txt: Path,
    out_path: Path,
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
    if out_path.exists() and out_path.stat().st_mtime >= max(
        source_ann.stat().st_mtime, image_id_txt.stat().st_mtime
    ):
        return out_path

    with open(source_ann, encoding="utf-8") as f:
        full_data = json.load(f)
    subset_ids = load_image_ids_from_txt(image_id_txt)
    filtered = filter_coco_json(full_data, subset_ids)
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


def wssis_dataset_names(experiment_id: str) -> Tuple[str, str]:
    exp = experiment_id or "default"
    return f"wssis_train_{exp}", f"wssis_val_{exp}"


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
    """
    Register train/val datasets using P0 layout:
    ``data/coco2017`` + ``data/splits/*.txt`` + filtered instances json.
    """
    exp_id = cfg.WSSIS.EXPERIMENT_ID
    train_name, val_name = wssis_dataset_names(exp_id)
    paths = build_coco_paths()
    cache_dir = Path(cfg.OUTPUT_DIR).parent

    train_txt = resolve_experiment_train_image_txt(
        cfg.WSSIS.LABELED_SPLIT,
        cfg.WSSIS.WEAK_SPLIT,
    )
    train_json = _ensure_filtered_coco_json(
        paths["train_ann"],
        train_txt,
        cache_dir / f"coco_instances_train_{train_txt.stem}.json",
    )
    train_image_root = resolve_coco_image_dir(paths["coco_root"], "train")
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
