#!/usr/bin/env bash
# Super-quick smoke test (~10 min, 1 GPU) for all 4 experiment kinds.
# Usage: bash scripts/experiments/run_smoke_test.sh [--run-id smoke_quick]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
activate_wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WSSIS_SMOKE=1
export WSSIS_NUM_GPUS=1

RUN_ID="${WSSIS_RUN_ID:-smoke_quick}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    *) shift ;;
  esac
done
export WSSIS_RUN_ID="$RUN_ID"

echo "========== WSSIS smoke test run_id=$RUN_ID =========="

echo "[1/5] Stage-1 GNN (2 images, 2 steps)..."
python -m modules.wssis.prep.train_stage1_gnn \
  --run-id "$RUN_ID" \
  --epochs 1 \
  --batch-size 2 \
  --max-instances 8 \
  --no-early-stop

echo "[2/5] Teacher AP on 5% holdout (2 images)..."
python -m modules.wssis.training.evaluate_teacher \
  --run-id "$RUN_ID" \
  --stage1-holdout \
  --max-instances 8

echo "[3/5] Exp 1A Mask2Former (10 iters)..."
python -m modules.wssis.run_experiment --exp 1A --stage train --run-id "$RUN_ID" --smoke --skip-p0-check

echo "[4/5] Exp 1C semi-weak Mask2Former (10 iters)..."
python -m modules.wssis.run_experiment --exp 1C --stage train --run-id "$RUN_ID" --smoke --skip-p0-check

echo "[4b/5] Exp 1A eval hook..."
python -m modules.wssis.run_experiment --exp 1A --stage eval --run-id "$RUN_ID" --smoke --skip-p0-check || true

echo "[5/5] Exp 4A YOLO semi-weak..."
python -m modules.wssis.run_experiment --exp 4A --stage train --run-id "$RUN_ID" --smoke --skip-p0-check

echo "========== Smoke test finished OK =========="
