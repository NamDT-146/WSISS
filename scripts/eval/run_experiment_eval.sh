#!/usr/bin/env bash
# Post-train student eval for one experiment (teacher AP is separate — run_teacher_eval.sh).
#
# Usage:
#   bash scripts/eval/run_experiment_eval.sh 1C --run-id wssis_main
#   bash scripts/eval/run_experiment_eval.sh 1C --run-id wssis_main --full-val
#   bash scripts/eval/run_experiment_eval.sh 1C --with-teacher-eval   # rarely needed
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
activate_wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

EXP="${1:-1C}"
shift || true

RUN_ID="${WSSIS_RUN_ID:-}"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --resume) EXTRA+=(--resume); shift ;;
    --full-val) EXTRA+=(--full-val); shift ;;
    --with-teacher-eval) EXTRA+=(--with-teacher-eval); shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

CMD=(python -m modules.wssis.run_experiment --exp "$EXP" --stage eval)
[[ -n "$RUN_ID" ]] && CMD+=(--run-id "$RUN_ID")
CMD+=("${EXTRA[@]}")

echo "[eval] Experiment $EXP (student): ${CMD[*]}"
"${CMD[@]}"
