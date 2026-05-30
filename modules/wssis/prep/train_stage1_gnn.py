"""
P0.4 — Train Stage-1 GNN refiner on labeled_5pct (current embed-only prototype).

See report/PLAN.md §0.5: full SAM-decoder + 3-mask refinement pipeline is not yet wired.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from modules.wssis.paths import build_coco_paths, checkpoints_dir, ensure_dirs, gnn_checkpoint


def _build_config(epochs: int, batch_size: int, lr: float, max_instances: int | None) -> dict:
    return {
        "data": {
            "img_size": 1024,
            "mask_size": 256,
            "num_workers": 4,
            "max_instances": max_instances,
            "coco_root": str(build_coco_paths()["coco_root"]),
            "train_image_txt": str(build_coco_paths()["labeled_5pct_txt"]),
            "val_image_txt": str(build_coco_paths()["val_all_txt"]),
        },
        "model": {
            "sam_channels": 256,
            "feat_dim": 128,
            "hidden_dim": 64,
            "out_dim": 64,
            "grid_size": 32,
            "mask_size": 256,
            "num_gnn_layers": 2,
            "connectivity": "grid",
            "k_neighbors": 8,
        },
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": 1e-4,
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "symmetric_weight": 0.1,
        },
        "use_symmetric_loss": True,
        "visualization": {
            "enabled": True,
            "num_samples": 4,
            "prompt_policy": "val_fixed",
        },
    }


def run(
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_instances: int | None = None,
    symmetric_weight: float = 0.1,
    output_name: str = "gnn_refiner_stage1.pt",
    device: str = "cuda",
    config_overrides: dict | None = None,
) -> Path:
    ensure_dirs()
    paths = build_coco_paths()
    if not paths["labeled_5pct_txt"].exists():
        raise FileNotFoundError("Run P0.1 generate_splits first.")

    repo = Path(__file__).resolve().parents[2].parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from modules.wssis.training.stage1 import train_stage1_gnn

    cfg = _build_config(epochs, batch_size, lr, max_instances)
    cfg["training"]["symmetric_weight"] = symmetric_weight
    cfg["use_symmetric_loss"] = symmetric_weight > 0
    if config_overrides:
        for key, val in config_overrides.items():
            if key == "visualization" and isinstance(val, dict):
                cfg.setdefault("visualization", {}).update(val)
            else:
                cfg[key] = val

    out = train_stage1_gnn(cfg, device=device, output_name=output_name)
    print(f"[P0.4] Saved checkpoint: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P0.4 train Stage-1 GNN")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--symmetric-weight", type=float, default=0.1)
    parser.add_argument("--output-name", default="gnn_refiner_stage1.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-viz", action="store_true", help="Disable per-epoch visualization")
    parser.add_argument("--viz-samples", type=int, default=4, help="Val samples per epoch grid")
    args = parser.parse_args()

    cfg_extra = {}
    if args.no_viz:
        cfg_extra["visualization"] = {"enabled": False}
    else:
        cfg_extra["visualization"] = {"enabled": True, "num_samples": args.viz_samples}

    run(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_instances=args.max_instances,
        symmetric_weight=args.symmetric_weight,
        output_name=args.output_name,
        device=args.device,
        config_overrides=cfg_extra,
    )


if __name__ == "__main__":
    main()
