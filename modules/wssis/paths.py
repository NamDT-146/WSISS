"""
Central path resolution for WSSIS (repo root, data, checkpoints, outputs).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def repo_root() -> Path:
    env = os.environ.get("WSSIS_REPO_ROOT")
    if env:
        return Path(env).resolve()
    # modules/wssis/paths.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def checkpoints_dir() -> Path:
    return repo_root() / "checkpoints"


def outputs_dir() -> Path:
    return repo_root() / "outputs"


def splits_dir() -> Path:
    return data_dir() / "splits"


def cache_dir() -> Path:
    return data_dir() / "cache"


def sam_embeddings_dir() -> Path:
    return cache_dir() / "sam_embeddings"


def coco_root() -> Path:
    return data_dir() / "coco2017"


def minitrain_root() -> Path:
    return data_dir() / "coco_minitrain_10k"


def kaggle_config_dir() -> Path:
    """Directory containing kaggle.json (KAGGLE_CONFIG_DIR)."""
    return data_dir()


def gnn_checkpoint(name: str = "gnn_refiner_stage1.pt") -> Path:
    return checkpoints_dir() / name


def sam_vit_b_checkpoint() -> Path:
    return checkpoints_dir() / "sam_vit_b_01ec64.pth"


def experiment_output_dir(exp_id: str) -> Path:
    d = outputs_dir() / "experiments" / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage1_viz_dir(run_name: str = "default") -> Path:
    """Per-epoch refinement grids for Stage-1 GNN training."""
    d = outputs_dir() / "stage1" / run_name / "visualizations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_coco_paths(coco_root_override: Optional[Path] = None) -> Dict[str, Path]:
    """Paths for COCO 2017 + minitrain-10k (local remote-machine layout)."""
    root = Path(coco_root_override) if coco_root_override else coco_root()
    mini = minitrain_root()
    ann = root / "annotations"
    return {
        "coco_root": root,
        "train_ann": ann / "instances_train2017.json",
        "val_ann": ann / "instances_val2017.json",
        "train_all_txt": splits_dir() / "train_all.txt",
        "val_all_txt": splits_dir() / "val_all.txt",
        "labeled_5pct_txt": splits_dir() / "labeled_5pct.txt",
        "weak_95pct_txt": splits_dir() / "weak_95pct.txt",
        "labeled_5pct_json": splits_dir() / "labeled_5pct.json",
        "val_prompts_json": splits_dir() / "val_prompts_fixed.json",
        "split_report_json": splits_dir() / "split_report.json",
        "minitrain_train_txt": mini / "train2017.txt",
        "minitrain_val_txt": mini / "val2017.txt",
    }


def ensure_dirs() -> None:
    for d in (
        data_dir(),
        splits_dir(),
        sam_embeddings_dir() / "train",
        sam_embeddings_dir() / "val",
        checkpoints_dir(),
        outputs_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)
