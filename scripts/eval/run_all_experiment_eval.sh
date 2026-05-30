#!/usr/bin/env bash
# Batch student eval for all Stage-2 experiments (no teacher AP — see run_teacher_eval.sh).
#
# Usage:
#   bash scripts/eval/run_all_experiment_eval.sh --run-id wssis_main
#   bash scripts/eval/run_all_experiment_eval.sh --run-id wssis_main --resume
#   bash scripts/eval/run_all_experiment_eval.sh --run-id wssis_main --full-val
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
activate_wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

RUN_ID="${WSSIS_RUN_ID:-}"
RESUME=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --resume) RESUME="--resume"; shift ;;
    --full-val) EXTRA+=(--full-val); shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

EVAL_ARGS=()
[[ -n "$RUN_ID" ]] && EVAL_ARGS+=(--run-id "$RUN_ID")
[[ -n "$RESUME" ]] && EVAL_ARGS+=($RESUME)

echo "========== Batch student eval (progress bar enabled) =========="
python -u -m modules.wssis.run_experiment --exp all --stage eval --continue-on-error \
  "${EVAL_ARGS[@]}" "${EXTRA[@]}" || {
  echo "WARNING: One or more evals failed (see log above)"
}

echo "[eval] Batch student eval finished."
