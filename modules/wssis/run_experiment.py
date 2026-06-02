"""
CLI entry: run preparation or a single experiment.

  python -m modules.wssis.run_experiment --exp 1C --stage train --run-id my_run
  python -m modules.wssis.run_experiment --exp p0 --stage prep --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def main(argv: list[str] | None = None) -> None:
    _ensure_repo_on_path()

    from tqdm import tqdm

    from modules.wssis.experiments.registry import DEFAULT_RUN_ORDER, EXPERIMENTS, get_experiment
    from modules.wssis.prep import run_p0
    from modules.wssis.run_context import RunContext
    from modules.wssis.training.stage2 import evaluate_experiment, train_experiment

    parser = argparse.ArgumentParser(description="WSSIS experiment runner")
    parser.add_argument("--exp", default=None, help="Experiment ID (1A, 1C, ...) or 'all' / 'p0'")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="With --exp all: continue remaining experiments after a failure",
    )
    parser.add_argument(
        "--stage",
        choices=["prep", "train", "eval", "all"],
        default="train",
        help="prep=P0, train=stage2, eval=post-train metrics",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--skip-p0-check", action="store_true")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    parser.add_argument("--run-id", default=None, help="Run folder under outputs/runs/")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory path")
    parser.add_argument("--resume", action="store_true", help="Resume from progress/checkpoints")
    parser.add_argument(
        "--full-val",
        action="store_true",
        help="Eval stage: use full val_all (default: 20%% val subset for speed)",
    )
    parser.add_argument(
        "--with-teacher-eval",
        action="store_true",
        help="Eval stage: also run teacher AP (default off; run once after P0 via run_teacher_eval.sh)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Super-quick smoke profile (<10 min, 1 GPU, minimal data)",
    )
    args = parser.parse_args(argv)

    import os

    # Avoid accidental smoke behavior from stale shell environment variables.
    os.environ["WSSIS_SMOKE"] = "1" if args.smoke else "0"
    if args.smoke:
        os.environ.setdefault("WSSIS_NUM_GPUS", "1")
        if not args.run_id:
            args.run_id = "smoke_quick"

    if args.list:
        for eid, spec in EXPERIMENTS.items():
            print(f"{eid}: {spec.name} [{spec.phase}]")
        return

    if not args.exp:
        parser.error("--exp is required (or use --list)")

    if args.exp.lower() in ("p0", "prep"):
        if args.stage in ("prep", "all"):
            sys.argv = [
                "run_p0",
                *(["--run-id", args.run_id] if args.run_id else []),
                *(["--run-dir", args.run_dir] if args.run_dir else []),
                *(["--resume"] if args.resume else []),
            ]
            run_p0.main()
        return

    ctx = RunContext(
        run_id=args.run_id,
        run_dir=args.run_dir,
        task="experiments",
    )
    ctx.log("Experiment run bundle: %s", ctx.root)

    if args.exp.lower() == "all":
        ids = DEFAULT_RUN_ORDER
    else:
        ids = [args.exp.upper().replace("EXP", "")]

    show_pbar = len(ids) > 1
    exp_iter = (
        tqdm(ids, desc="WSSIS experiments", unit="exp", dynamic_ncols=True)
        if show_pbar
        else ids
    )
    failures: list[str] = []

    for eid in exp_iter:
        if show_pbar:
            exp_iter.set_postfix_str(f"exp={eid} stage={args.stage}", refresh=False)

        spec = get_experiment(eid)
        step_key = f"exp_{eid}"
        eval_key = f"eval_{eid}"

        train_done = ctx.is_step_done(step_key)
        eval_done = ctx.is_step_done(eval_key)

        if args.resume:
            if args.stage == "train" and train_done:
                ctx.log("Skipping train for %s (done)", eid)
                continue
            if args.stage == "eval" and eval_done:
                ctx.log("Skipping eval for %s (done)", eid)
                continue
            if args.stage == "all" and train_done and eval_done:
                ctx.log("Skipping %s (train + eval done)", eid)
                continue

        try:
            if args.stage in ("train", "all"):
                if args.resume and train_done:
                    ctx.log("Skipping train for %s (done)", eid)
                else:
                    train_experiment(
                        spec,
                        dry_run=args.dry_run,
                        skip_p0_check=args.skip_p0_check,
                        run_ctx=RunContext(
                            run_id=ctx.run_id,
                            run_dir=ctx.root,
                            task=f"exp_{eid}",
                            experiment_id=eid,
                        ),
                    )
            if args.stage in ("eval", "all"):
                if args.resume and eval_done:
                    ctx.log("Skipping eval for %s (done)", eid)
                else:
                    evaluate_experiment(
                        spec,
                        dry_run=args.dry_run,
                        run_ctx=RunContext(
                            run_id=ctx.run_id,
                            run_dir=ctx.root,
                            task=eval_key,
                            experiment_id=eid,
                        ),
                        full_val=args.full_val,
                        with_teacher_eval=args.with_teacher_eval,
                    )
                    if not args.dry_run:
                        ctx.update_step(eval_key, {"status": "done", "experiment_id": eid})
        except Exception as exc:
            ctx.log("Experiment %s failed: %s", eid, exc)
            failures.append(eid)
            if not args.continue_on_error:
                raise

    if failures:
        ctx.log("Finished with %d failure(s): %s", len(failures), ", ".join(failures))

    ctx.finalize_report_bundle()


if __name__ == "__main__":
    main()
