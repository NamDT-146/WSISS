#!/usr/bin/env bash
# Post-train YOLOv8-seg COCO-style val metrics (experiment 4A).
#
# Usage:
#   bash scripts/eval/run_yolo_eval.sh 4A --run-id wssis_v2
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
activate_wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

EXP="${1:-4A}"
shift || true

RUN_ID="${WSSIS_RUN_ID:-}"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

CMD=(python -m modules.wssis.training.evaluate_yolo --exp "$EXP")
[[ -n "$RUN_ID" ]] && CMD+=(--run-id "$RUN_ID")
CMD+=("${EXTRA[@]}")

echo "[eval] YOLO: ${CMD[*]}"
"${CMD[@]}"
