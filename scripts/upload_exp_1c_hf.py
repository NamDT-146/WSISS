#!/usr/bin/env python3
"""
Upload Exp 1C best checkpoint + config to Hugging Face Hub for demo.

Requires: pip install huggingface_hub
  huggingface-cli login

Usage:
  python scripts/upload_exp_1c_hf.py --repo-id YOUR_USER/wssis-1c-demo
  python scripts/upload_exp_1c_hf.py --repo-id YOUR_USER/wssis-1c-demo --run-id 20260529_120000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Exp 1C best weights to Hugging Face")
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. user/wssis-1c-demo")
    parser.add_argument("--run-id", default=None, help="WSSIS run id under outputs/runs/")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    from modules.wssis.paths import outputs_dir
    from modules.wssis.run_context import resolve_run_dir

    run_root = resolve_run_dir(args.run_id, args.run_dir)

    candidates = [
        run_root / "checkpoints" / "best.pt",
        run_root / "experiments" / "1C" / "mask2former" / "model_final.pth",
        run_root / "report" / "best_checkpoint.pt",
        REPO / "checkpoints" / "gnn_refiner_stage1.pt",
    ]
    ckpt = next((p for p in candidates if p.exists()), None)
    if ckpt is None:
        print("No checkpoint found. Tried:")
        for p in candidates:
            print(f"  {p}")
        sys.exit(1)

    config = run_root / "config.json"
    if not config.exists():
        config = run_root / "report" / "config.json"

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Install: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()
    create_repo(args.repo_id, private=args.private, exist_ok=True)

    print(f"Uploading {ckpt} -> {args.repo_id}")
    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo="pytorch_model.bin",
        repo_id=args.repo_id,
        repo_type="model",
    )
    if config.exists():
        api.upload_file(
            path_or_fileobj=str(config),
            path_in_repo="config.json",
            repo_id=args.repo_id,
            repo_type="model",
        )

    readme = run_root / "report" / "README.txt"
    if readme.exists():
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.txt",
            repo_id=args.repo_id,
            repo_type="model",
        )

    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
