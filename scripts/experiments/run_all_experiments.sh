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
#   bash scripts/experiments/run_all_experiments.sh --run-id wssis_main --parallel
#     → auto-detect GPUs: 1C on all, then 10 others at N-wide parallel
#   bash scripts/experiments/run_all_experiments.sh --run-id wssis_main --parallel 4
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
# shellcheck source=scripts/lib/detect_gpus.sh
source "$REPO_ROOT/scripts/lib/detect_gpus.sh"
# shellcheck source=scripts/lib/cleanup_gpu_workers.sh
source "$REPO_ROOT/scripts/lib/cleanup_gpu_workers.sh"

_wssis_cleanup_on_signal() {
  echo "[cleanup] Stopping training workers..." >&2
  wssis_kill_training_workers || true
}
trap _wssis_cleanup_on_signal INT TERM

activate_wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

RUN_P0=false
DRY_RUN=""
RESUME=""
WITH_EVAL=false
STAGE="train"
PARALLEL=0
PARALLEL_JOBS=""
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
    --parallel=*) PARALLEL=1; PARALLEL_JOBS="${1#*=}"; shift ;;
    --parallel)
      PARALLEL=1
      if [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
        PARALLEL_JOBS="$2"
        shift 2
      else
        PARALLEL_JOBS=""
        shift
      fi
      ;;
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
  PAR_ARGS=()
  [[ -n "$PARALLEL_JOBS" ]] && PAR_ARGS+=(--jobs "$PARALLEL_JOBS")
  [[ -n "$RUN_ID" ]] && PAR_ARGS+=(--run-id "$RUN_ID")
  [[ -n "$RESUME" ]] && PAR_ARGS+=($RESUME)
  [[ -n "$DRY_RUN" ]] && PAR_ARGS+=($DRY_RUN)
  bash scripts/experiments/run_experiments_parallel.sh "${PAR_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
  echo "========== Sequential sweep (stage=$STAGE, progress bar enabled) =========="
  python -u -m modules.wssis.run_experiment --exp all --continue-on-error \
    "${RUN_FLAGS[@]}" $DRY_RUN "${EXTRA_ARGS[@]}" || {
    echo "WARNING: One or more experiments failed (see log above)"
  }
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
