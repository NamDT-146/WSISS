"""
CLI entry: run preparation or a single experiment.

  python -m modules.wssis.run_experiment --exp 1C --stage train
  python -m modules.wssis.run_experiment --exp all --stage prep
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

    from modules.wssis.experiments.registry import DEFAULT_RUN_ORDER, EXPERIMENTS, get_experiment
    from modules.wssis.prep import run_p0
    from modules.wssis.training.stage2 import evaluate_experiment, train_experiment

    parser = argparse.ArgumentParser(description="WSSIS experiment runner")
    parser.add_argument("--exp", default=None, help="Experiment ID (1A, 1C, ...) or 'all' / 'p0'")
    parser.add_argument(
        "--stage",
        choices=["prep", "train", "eval", "all"],
        default="train",
        help="prep=P0, train=stage2, eval=post-train metrics",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--skip-p0-check", action="store_true")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    args = parser.parse_args(argv)

    if args.list:
        for eid, spec in EXPERIMENTS.items():
            print(f"{eid}: {spec.name} [{spec.phase}]")
        return

    if not args.exp:
        parser.error("--exp is required (or use --list)")

    if args.exp.lower() in ("p0", "prep"):
        if args.stage in ("prep", "all"):
            import sys as _sys
            _sys.argv = ["run_p0"]
            run_p0.main()
        return

    if args.exp.lower() == "all":
        ids = DEFAULT_RUN_ORDER
    else:
        ids = [args.exp.upper().replace("EXP", "")]

    for eid in ids:
        spec = get_experiment(eid)
        if args.stage in ("train", "all"):
            train_experiment(spec, dry_run=args.dry_run, skip_p0_check=args.skip_p0_check)
        if args.stage in ("eval", "all"):
            evaluate_experiment(spec, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
