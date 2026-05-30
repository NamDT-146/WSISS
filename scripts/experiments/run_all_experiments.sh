#!/usr/bin/env bash
# Run P0 (optional) then all experiments in PLAN order.
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
RUN_ID="${WSSIS_RUN_ID:-}"
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --with-p0) RUN_P0=true ;;
    --dry-run) DRY_RUN="--dry-run" ;;
    --resume) RESUME="--resume" ;;
    --run-id=*) RUN_ID="${arg#*=}" ;;
    --run-id) shift; RUN_ID="${1:-}" ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

RUN_FLAGS=()
[[ -n "$RUN_ID" ]] && RUN_FLAGS+=(--run-id "$RUN_ID")
[[ -n "$RESUME" ]] && RUN_FLAGS+=($RESUME)

if $RUN_P0; then
  bash scripts/prep/run_p0.sh "${RUN_FLAGS[@]}" "${EXTRA_ARGS[@]}"
fi

EXPS=(1C 1A 1B 1D 2A 2B 2C 3A 3B 3C 4A)
for exp in "${EXPS[@]}"; do
  echo "========== Experiment $exp =========="
  python -m modules.wssis.run_experiment --exp "$exp" --stage all $DRY_RUN "${RUN_FLAGS[@]}" || {
    echo "WARNING: Experiment $exp failed (continuing)"
  }
done

echo "========== Teacher AP eval (val, all signal types) =========="
bash scripts/eval/run_teacher_eval.sh "${RUN_FLAGS[@]}" || echo "WARNING: teacher eval failed"

echo "All experiments finished."
echo "Run bundle: outputs/runs/${RUN_ID:-<latest>}/"
echo "Report upload folder: outputs/runs/<run_id>/report/"
