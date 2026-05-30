"""
Run full P0 preparation pipeline with global progress tracking.
"""

from __future__ import annotations

import argparse
import os

from modules.wssis.prep.generate_splits import run as run_splits
from modules.wssis.prep.precompute_sam_embeddings import run as run_embeddings
from modules.wssis.prep.train_stage1_gnn import run as run_stage1
from modules.wssis.run_context import RunContext


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P0 prep (splits → embeddings → stage1)")
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--embedding-limit", type=int, default=None)
    parser.add_argument("--force-splits", action="store_true")
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Stage-1 GNN DataLoader batch size (default: 4)",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume stage1 from checkpoint in run dir")
    args = parser.parse_args()

    ctx = RunContext(run_id=args.run_id, run_dir=args.run_dir, task="p0_prep")
    ctx.log("P0 pipeline run_id=%s root=%s", ctx.run_id, ctx.root)

    if not args.skip_splits:
        if ctx.is_step_done("p0_splits") and not args.force_splits:
            ctx.log("Skipping p0_splits (already done)")
        else:
            ctx.update_step("p0_splits", {"status": "running"})
            run_splits(force=args.force_splits)
            ctx.update_step("p0_splits", "done")

    if not args.skip_embeddings:
        emb_state = ctx.step_status("p0_embeddings")
        if emb_state == "done" and not args.force_embeddings:
            ctx.log("Skipping p0_embeddings (already done)")
        else:
            ctx.update_step("p0_embeddings", {"status": "running"})
            run_embeddings(
                limit=args.embedding_limit,
                force=args.force_embeddings,
                run_ctx=ctx,
            )
            ctx.update_step("p0_embeddings", "done")

    if not args.skip_stage1:
        st = ctx.step_status("p0_stage1")
        if st == "done" and not args.resume:
            ctx.log("Skipping p0_stage1 (already done; use --resume to continue training)")
        else:
            ctx.update_step("p0_stage1", {"status": "running"})
            run_stage1(
                epochs=args.stage1_epochs,
                batch_size=args.batch_size,
                run_id=ctx.run_id,
                run_dir=str(ctx.root),
                resume=args.resume,
            )
            ctx.update_step("p0_stage1", "done")

    ctx.finalize_report_bundle()
    ctx.log("P0 complete. Bundle: %s", ctx.report_dir)


if __name__ == "__main__":
    main()
