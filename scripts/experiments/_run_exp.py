#!/usr/bin/env python3
"""
Helper for experiment scripts.

  python scripts/experiments/_run_exp.py 1C [--stage train|eval|all] [--run-id ID] [--dry-run] [--resume]

Uses WSSIS_RUN_ID from the environment when --run-id is omitted.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from modules.wssis.run_experiment import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        exp_id = "1C"
    else:
        exp_id = args[0]
        args = args[1:]

    if "--stage" not in args:
        args = ["--stage", "train", *args]

    if "--run-id" not in args and os.environ.get("WSSIS_RUN_ID"):
        args = ["--run-id", os.environ["WSSIS_RUN_ID"], *args]

    main(["--exp", exp_id, *args])
