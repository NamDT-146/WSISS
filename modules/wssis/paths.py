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


def gnn_checkpoint(name: str = "gnn_refiner_stage1_v2.pt") -> Path:
    """Default Stage-1 GNN v2 weights (override via experiment registry)."""
    p = checkpoints_dir() / name
    if p.exists():
        return p
    legacy = checkpoints_dir() / "gnn_refiner_stage1.pt"
    return legacy if legacy.exists() else p


def load_weak_95pct_signal_map() -> Dict[str, str]:
    """image_id (str) -> points_only | scribbles_only | boxes_only."""
    path = build_coco_paths()["weak_95pct_signal_json"]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m modules.wssis.prep.generate_splits --weak-signal-only"
        )
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def sam_vit_b_checkpoint() -> Path:
    return checkpoints_dir() / "sam_vit_b_01ec64.pth"


def swin_tiny_checkpoint() -> Path:
    return checkpoints_dir() / "swin_tiny_patch4_window7_224.pkl"


def mit_b0_checkpoint() -> Path:
    return checkpoints_dir() / "mit_b0_pretrained.pkl"


def experiment_output_dir(exp_id: str) -> Path:
    d = outputs_dir() / "experiments" / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage1_viz_dir(run_name: str = "default") -> Path:
    """Per-epoch refinement grids for Stage-1 GNN training."""
    d = outputs_dir() / "stage1" / run_name / "visualizations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_coco_image_dir(coco_root: Path, split: str) -> Path:
    """Return ``train2017`` / ``val2017`` image folder (matches P0 / CocoSamStage1Dataset)."""
    root = Path(coco_root)
    tried: list[Path] = []
    for sub in (f"{split}2017", f"images/{split}2017"):
        candidate = root / sub
        tried.append(candidate)
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"COCO image directory not found for split={split!r} under {root}. "
        f"Tried: {[str(p) for p in tried]}. Run: bash scripts/setup/01_download_data.sh"
    )


def resolve_experiment_train_image_txt(labeled_split: str, weak_split: str) -> Path:
    """Train image list for Stage-2 (P0.1 ``data/splits/*.txt``, same as prep)."""
    paths = build_coco_paths()
    if labeled_split == "train_all":
        return paths["train_all_txt"]
    if labeled_split == "labeled_5pct" and weak_split == "weak_95pct":
        return paths["train_all_txt"]
    if labeled_split == "labeled_5pct":
        return paths["labeled_5pct_txt"]
    if weak_split == "weak_95pct":
        return paths["weak_95pct_txt"]
    return paths["train_all_txt"]


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
        "labeled_5pct_train_txt": splits_dir() / "labeled_5pct_train.txt",
        "labeled_5pct_val_txt": splits_dir() / "labeled_5pct_val.txt",
        "val_sample_20pct_txt": splits_dir() / "val_sample_20pct.txt",
        "weak_95pct_txt": splits_dir() / "weak_95pct.txt",
        "weak_95pct_signal_json": splits_dir() / "weak_95pct_signal.json",
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
