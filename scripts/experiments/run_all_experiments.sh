#!/usr/bin/env bash
# Run P0 (optional) then Stage-2 experiments in PLAN order.
#
# Default: train only (no per-experiment eval). Teacher AP runs once during P0.4.
# After all training, run student eval in one batch:
#   bash scripts/eval/run_all_experiment_eval.sh --run-id $WSSIS_RUN_ID
#
# Usage:
#   bash scripts/experiments/run_all_experiments.sh --with-p0 --run-id wssis_main
#   bash scripts/experiments/run_all_experiments.sh --run-id wssis_main --resume
#   bash scripts/experiments/run_all_experiments.sh --run-id wssis_main --parallel 5
#   bash scripts/experiments/run_all_experiments.sh --run-id wssis_main --with-eval
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
activate_wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

RUN_P0=false
DRY_RUN=""
RESUME=""
WITH_EVAL=false
STAGE="train"
PARALLEL=0
RUN_ID="${WSSIS_RUN_ID:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-p0) RUN_P0=true; shift ;;
    --dry-run) DRY_RUN="--dry-run"; shift ;;
    --resume) RESUME="--resume"; shift ;;
    --with-eval) WITH_EVAL=true; shift ;;
    --stage=*) STAGE="${1#*=}"; shift ;;
    --stage) STAGE="${2:-train}"; shift 2 ;;
    --parallel=*) PARALLEL="${1#*=}"; shift ;;
    --parallel) PARALLEL="${2:-5}"; shift 2 ;;
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

RUN_FLAGS=(--stage "$STAGE")
[[ -n "$RUN_ID" ]] && RUN_FLAGS+=(--run-id "$RUN_ID")
[[ -n "$RESUME" ]] && RUN_FLAGS+=($RESUME)

P0_FLAGS=()
[[ -n "$RUN_ID" ]] && P0_FLAGS+=(--run-id "$RUN_ID")
[[ -n "$RESUME" ]] && P0_FLAGS+=($RESUME)

if $RUN_P0; then
  bash scripts/prep/run_p0.sh "${P0_FLAGS[@]}" "${EXTRA_ARGS[@]}"
fi

if (( PARALLEL > 0 )); then
  PAR_ARGS=(--jobs "$PARALLEL")
  [[ -n "$RUN_ID" ]] && PAR_ARGS+=(--run-id "$RUN_ID")
  [[ -n "$RESUME" ]] && PAR_ARGS+=($RESUME)
  [[ -n "$DRY_RUN" ]] && PAR_ARGS+=($DRY_RUN)
  bash scripts/experiments/run_experiments_parallel.sh "${PAR_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
  EXPS=(1C 1A 1B 1D 2A 2B 2C 3A 3B 3C 4A)
  for exp in "${EXPS[@]}"; do
    echo "========== Experiment $exp (stage=$STAGE) =========="
    python -m modules.wssis.run_experiment --exp "$exp" "${RUN_FLAGS[@]}" $DRY_RUN "${EXTRA_ARGS[@]}" || {
      echo "WARNING: Experiment $exp failed (continuing)"
    }
  done
fi

if $WITH_EVAL; then
  echo "========== Batch student eval (all experiments) =========="
  EVAL_ARGS=()
  [[ -n "$RUN_ID" ]] && EVAL_ARGS+=(--run-id "$RUN_ID")
  [[ -n "$RESUME" ]] && EVAL_ARGS+=($RESUME)
  bash scripts/eval/run_all_experiment_eval.sh "${EVAL_ARGS[@]}" || {
    echo "WARNING: batch eval failed"
  }
fi

echo "All experiment training finished."
echo "Run bundle: outputs/runs/${RUN_ID:-<latest>}/"
echo "Teacher AP (once after P0): outputs/runs/<run_id>/eval/teacher_val_report_full.json"
echo "Student eval batch: bash scripts/eval/run_all_experiment_eval.sh --run-id ${RUN_ID:-<id>}"
