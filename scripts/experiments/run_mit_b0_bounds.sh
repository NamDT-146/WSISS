#!/usr/bin/env bash
# Train MiT-B0 experiments: 5A (5% lower), 5C (semi-weak main), 5D (100% upper).
# Same Stage-2 recipe as 1A/1C/1D; only the student backbone changes.
#
# Usage (respects WSSIS_RUN_ID, e.g. export WSSIS_RUN_ID=wssis_v2):
#   bash scripts/experiments/run_mit_b0_bounds.sh
#   bash scripts/experiments/run_mit_b0_bounds.sh --run-id wssis_v2 --resume
#   bash scripts/experiments/run_mit_b0_bounds.sh --with-eval
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
activate_wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -f "$REPO_ROOT/checkpoints/mit_b0_pretrained.pkl" ]]; then
  echo "==> MiT-B0 weights missing; downloading..."
  bash scripts/setup/05_download_mit_b0_weights.sh
fi

RUN_ID="${WSSIS_RUN_ID:-wssis_main}"
RESUME=""
WITH_EVAL=false
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --resume) RESUME="--resume"; shift ;;
    --with-eval) WITH_EVAL=true; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

echo "Using run-id: ${RUN_ID} (WSSIS_RUN_ID or --run-id)"

RUN_FLAGS=(--exp mit_bounds --stage train --continue-on-error)
[[ -n "$RUN_ID" ]] && RUN_FLAGS+=(--run-id "$RUN_ID")
[[ -n "$RESUME" ]] && RUN_FLAGS+=($RESUME)

echo "========== MiT-B0: 5A (5%% GT) + 5C (semi-weak) + 5D (100%% GT) =========="
python -u -m modules.wssis.run_experiment "${RUN_FLAGS[@]}" "${EXTRA[@]}" || {
  echo "WARNING: One or more MiT-B0 bound experiments failed"
}

if $WITH_EVAL; then
  echo "========== Eval 5A + 5C + 5D =========="
  EVAL_ARGS=(--exp mit_bounds --stage eval --continue-on-error)
  [[ -n "$RUN_ID" ]] && EVAL_ARGS+=(--run-id "$RUN_ID")
  [[ -n "$RESUME" ]] && EVAL_ARGS+=($RESUME)
  python -u -m modules.wssis.run_experiment "${EVAL_ARGS[@]}" --full-val "${EXTRA[@]}" || {
    echo "WARNING: MiT-B0 bound eval failed"
  }
fi

echo "MiT-B0 bound experiments finished. Run bundle: outputs/runs/${RUN_ID}/"
