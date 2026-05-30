#!/usr/bin/env python3
"""Helper: python scripts/experiments/_run_exp.py 1C [--stage train] [--dry-run]"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from modules.wssis.run_experiment import main

if __name__ == "__main__":
    exp_id = sys.argv[1] if len(sys.argv) > 1 else "1C"
    extra = sys.argv[2:]
    main(["--exp", exp_id, "--stage", "all", *extra])
