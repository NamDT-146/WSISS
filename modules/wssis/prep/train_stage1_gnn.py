"""
P0.4 — Train Stage-1 GNN refiner on labeled_5pct.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modules.wssis.paths import build_coco_paths, ensure_dirs
from modules.wssis.run_context import RunContext


def _build_config(
    epochs: int,
    batch_size: int,
    lr: float,
    max_instances: int | None,
    run_id: str | None = None,
    run_dir: str | None = None,
) -> dict:
    paths = build_coco_paths()
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "data": {
            "img_size": 1024,
            "mask_size": 256,
            "num_workers": 4,
            "max_instances": max_instances,
            "coco_root": str(paths["coco_root"]),
            # Stage-1: train + in-loop val both from 5% pool (P0.1 holdout); final eval uses val_all
            "train_split": "labeled_5pct_train",
            "train_image_txt": str(paths["labeled_5pct_train_txt"]),
            "val_split": "labeled_5pct_val",
            "val_image_txt": str(paths["labeled_5pct_val_txt"]),
            "val_use_labeled_holdout": True,
            "run_final_eval": True,
            # P0.2 npy cache — major speedup vs re-encoding SAM every instance
            "use_sam_embedding_cache": True,
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
            "save_every_epochs": 1,
        },
        "early_stopping": {
            "patience": 3,
            "monitor": "val_refined_ap",
            "mode": "max",
        },
        "logging": {"tensorboard": True, "wandb": True},
        "use_symmetric_loss": True,
        "run_final_eval": True,
        "pseudo_label": {
            "confidence_threshold": 0.5,
        },
        "visualization": {
            "enabled": True,
            "num_samples": 4,
            "prompt_policy": "val_fixed",
        },
    }


def run(
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_instances: int | None = None,
    symmetric_weight: float = 0.1,
    output_name: str = "gnn_refiner_stage1.pt",
    device: str = "cuda",
    config_overrides: dict | None = None,
    run_id: str | None = None,
    run_dir: str | None = None,
    resume: bool = False,
    patience: int = 3,
    run_final_eval: bool = True,
) -> Path:
    ensure_dirs()
    paths = build_coco_paths()
    for req in (paths["labeled_5pct_train_txt"], paths["labeled_5pct_val_txt"]):
        if not req.exists():
            raise FileNotFoundError(f"Missing {req}. Run P0.1 generate_splits (--force to regenerate).")

    repo = Path(__file__).resolve().parents[2].parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from modules.wssis.training.stage1 import train_stage1_gnn

    cfg = _build_config(epochs, batch_size, lr, max_instances, run_id, run_dir)
    cfg["training"]["symmetric_weight"] = symmetric_weight
    cfg["use_symmetric_loss"] = symmetric_weight > 0
    cfg["early_stopping"]["patience"] = patience

    if config_overrides:
        for key, val in config_overrides.items():
            if key == "visualization" and isinstance(val, dict):
                cfg.setdefault("visualization", {}).update(val)
            elif key == "early_stopping" and isinstance(val, dict):
                cfg.setdefault("early_stopping", {}).update(val)
            else:
                cfg[key] = val

    from modules.wssis.smoke_profile import get_smoke_profile

    smoke = get_smoke_profile()
    if smoke:
        cfg["data"]["max_images"] = smoke.max_images
        cfg["data"]["max_objects_per_image"] = smoke.max_objects_per_image
        cfg["training"]["batch_size"] = smoke.batch_size
        cfg["training"]["max_steps"] = smoke.stage1_max_steps
        cfg["training"]["epochs"] = min(cfg["training"]["epochs"], smoke.stage1_epochs)
        cfg["data"]["num_workers"] = 0
        cfg["visualization"]["num_samples"] = smoke.viz_samples
        cfg["early_stopping"]["patience"] = 0

    ctx = RunContext(run_id=cfg.get("run_id"), run_dir=cfg.get("run_dir"), task="stage1_gnn")
    cfg["run_id"] = ctx.run_id
    cfg["run_dir"] = str(ctx.root)

    cfg["run_final_eval"] = run_final_eval

    out = train_stage1_gnn(
        cfg,
        device=device,
        output_name=output_name,
        run_ctx=ctx,
        resume=resume,
    )

    print(f"[P0.4] Saved checkpoint: {out}")
    print(f"[P0.4] Run bundle: {ctx.root}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P0.4 train Stage-1 GNN")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--symmetric-weight", type=float, default=0.1)
    parser.add_argument("--output-name", default="gnn_refiner_stage1.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--viz-samples", type=int, default=4)
    parser.add_argument("--run-id", default=None, help="Run folder name under outputs/runs/")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory path")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt in run dir")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (0=off)")
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument(
        "--no-final-eval",
        action="store_true",
        help="Skip full val_all teacher eval after training",
    )
    parser.add_argument(
        "--pseudo-confidence-threshold",
        type=float,
        default=None,
        help="Min sigmoid prob per GNN head for pseudo-label voting (default: 0.5)",
    )
    args = parser.parse_args()

    overrides = {}
    if args.no_viz:
        overrides["visualization"] = {"enabled": False}
    else:
        overrides["visualization"] = {"enabled": True, "num_samples": args.viz_samples}
    if args.no_early_stop:
        overrides["early_stopping"] = {"patience": 0}
    if args.pseudo_confidence_threshold is not None:
        overrides["pseudo_label"] = {
            "confidence_threshold": float(args.pseudo_confidence_threshold),
        }

    run(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_instances=args.max_instances,
        symmetric_weight=args.symmetric_weight,
        output_name=args.output_name,
        device=args.device,
        config_overrides=overrides,
        run_id=args.run_id,
        run_dir=args.run_dir,
        resume=args.resume,
        patience=0 if args.no_early_stop else args.patience,
        run_final_eval=not args.no_final_eval,
    )


if __name__ == "__main__":
    main()
