"""
Run full P0 preparation pipeline.
"""

from __future__ import annotations

import argparse

from modules.wssis.prep.generate_splits import run as run_splits
from modules.wssis.prep.precompute_sam_embeddings import run as run_embeddings
from modules.wssis.prep.train_stage1_gnn import run as run_stage1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P0 prep (splits → embeddings → stage1)")
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--embedding-limit", type=int, default=None, help="Debug cap for P0.2")
    parser.add_argument("--force-splits", action="store_true")
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=20)
    args = parser.parse_args()

    if not args.skip_splits:
        run_splits(force=args.force_splits)
    if not args.skip_embeddings:
        run_embeddings(limit=args.embedding_limit, force=args.force_embeddings)
    if not args.skip_stage1:
        run_stage1(epochs=args.stage1_epochs)


if __name__ == "__main__":
    main()
