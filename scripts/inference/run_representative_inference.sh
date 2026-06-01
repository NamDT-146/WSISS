#!/usr/bin/env bash
# Representative val inference — 5 report settings on 20 curated val images.
#
# Usage:
#   bash scripts/inference/run_representative_inference.sh --build-list
#   bash scripts/inference/run_representative_inference.sh --run-id wssis_main
#   bash scripts/inference/run_representative_inference.sh --run-id wssis_main --teacher-only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
activate_wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

RUN_ID="${WSSIS_RUN_ID:-wssis_main}"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --build-list) EXTRA+=(--build-list); shift ;;
    --teacher-only) EXTRA+=(--skip-students); shift ;;
    --max-images=*) EXTRA+=("${1}"); shift ;;
    --max-images) EXTRA+=("$1" "$2"); shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

CMD=(python -m modules.wssis.inference.run_representative)
if [[ " ${EXTRA[*]} " != *" --build-list "* ]]; then
  CMD+=(--run-id "$RUN_ID")
fi
CMD+=("${EXTRA[@]}")

echo "[inference] ${CMD[*]}"
"${CMD[@]}"

if [[ " ${EXTRA[*]} " != *" --build-list "* ]]; then
  echo "[inference] Grids -> outputs/runs/${RUN_ID}/visualizations/representative_inference/"
fi
