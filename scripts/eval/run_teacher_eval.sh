#!/usr/bin/env bash
# Evaluate raw SAM + GNN-refined teacher on COCO val (AP primary metric).
# Reports all weak-signal types: boxes_only, points_only, scribbles_only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
activate_wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

RUN_ID="${WSSIS_RUN_ID:-}"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --raw-only) EXTRA+=(--raw-only); shift ;;
    --full-val) EXTRA+=(--full-val); shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

CMD=(python -m modules.wssis.training.evaluate_teacher)
[[ -n "$RUN_ID" ]] && CMD+=(--run-id "$RUN_ID")
CMD+=("${EXTRA[@]}")

echo "[eval] Teacher AP report: ${CMD[*]}"
"${CMD[@]}"
echo "[eval] Reports: outputs/runs/${RUN_ID:-<run>}/eval/teacher_val_report_*.json"
echo "  (subset default; use --full-val for teacher_val_report_full.json)"
