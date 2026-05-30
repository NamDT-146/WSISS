#!/usr/bin/env bash
# Run P0 (optional) then all experiments in PLAN order.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

RUN_P0=false
DRY_RUN=""
for arg in "$@"; do
  case "$arg" in
    --with-p0) RUN_P0=true ;;
    --dry-run) DRY_RUN="--dry-run" ;;
  esac
done

if $RUN_P0; then
  bash scripts/prep/run_p0.sh
fi

EXPS=(1C 1A 1B 1D 2A 2B 2C 3A 3B 3C 4A)
for exp in "${EXPS[@]}"; do
  echo "========== Experiment $exp =========="
  python -m modules.wssis.run_experiment --exp "$exp" --stage all $DRY_RUN || {
    echo "WARNING: Experiment $exp failed (continuing)"
  }
done

echo "All experiments finished. Results under outputs/experiments/"
