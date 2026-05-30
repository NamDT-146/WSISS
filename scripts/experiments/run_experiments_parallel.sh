#!/usr/bin/env bash
# Train Stage-2 experiments on a GPU worker pool (default 5 jobs = 5 GPUs).
# Starts the next queued experiment when a GPU slot frees.
#
# Usage:
#   bash scripts/experiments/run_experiments_parallel.sh --jobs 5 --run-id wssis_main
#   bash scripts/experiments/run_experiments_parallel.sh --jobs 5 --run-id wssis_main --resume
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
activate_wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

JOBS=5
DRY_RUN=""
RESUME=""
RUN_ID="${WSSIS_RUN_ID:-}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs=*) JOBS="${1#*=}"; shift ;;
    --jobs) JOBS="${2:-5}"; shift 2 ;;
    --dry-run) DRY_RUN="--dry-run"; shift ;;
    --resume) RESUME="--resume"; shift ;;
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if (( JOBS < 1 )); then
  echo "ERROR: --jobs must be >= 1" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-parallel}"
LOG_DIR="outputs/runs/${RUN_ID}/logs/parallel"
mkdir -p "$LOG_DIR"

RUN_ARGS=(--stage train --run-id "$RUN_ID")
[[ -n "$RESUME" ]] && RUN_ARGS+=($RESUME)
[[ -n "$DRY_RUN" ]] && RUN_ARGS+=($DRY_RUN)

QUEUE=(1C 1A 1B 1D 2A 2B 2C 3A 3B 3C 4A)

if command -v parallel >/dev/null 2>&1; then
  echo "[parallel] GNU parallel: $JOBS workers, queue=${QUEUE[*]}"
  export WSSIS_REPO_ROOT PYTHONPATH
  printf '%s\n' "${QUEUE[@]}" | \
    parallel -j "$JOBS" --line-buffer --joblog "$LOG_DIR/joblog.tsv" \
      "CUDA_VISIBLE_DEVICES=\$(( {%} - 1 )) WSSIS_NUM_GPUS=1 WSSIS_REPO_ROOT='$WSSIS_REPO_ROOT' PYTHONPATH='$PYTHONPATH' \
       python -m modules.wssis.run_experiment --exp {} ${RUN_ARGS[*]}"
else
  echo "[parallel] GNU parallel not found; using bash worker pool ($JOBS slots)"
  declare -A PID_GPU=()
  FREE=()
  for (( g=0; g<JOBS; g++ )); do FREE+=("$g"); done
  qi=0
  failed=0

  reap_one() {
    local pid gpu rc=0
    for pid in "${!PID_GPU[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || rc=$?
        gpu=${PID_GPU[$pid]}
        unset 'PID_GPU[$pid]'
        FREE+=("$gpu")
        if (( rc != 0 )); then
          echo "WARNING: job on GPU $gpu failed (exit $rc)"
          failed=$((failed + 1))
        fi
        return 0
      fi
    done
    return 1
  }

  while (( qi < ${#QUEUE[@]} )) || (( ${#PID_GPU[@]} > 0 )); do
    while (( qi < ${#QUEUE[@]} )) && (( ${#FREE[@]} > 0 )); do
      exp=${QUEUE[qi++]}
      gpu=${FREE[0]}
      FREE=("${FREE[@]:1}")
      (
        export CUDA_VISIBLE_DEVICES=$gpu WSSIS_NUM_GPUS=1
        python -m modules.wssis.run_experiment --exp "$exp" "${RUN_ARGS[@]}" "${EXTRA[@]}"
      ) >"$LOG_DIR/exp_${exp}.gpu${gpu}.log" 2>&1 &
      PID_GPU[$!]=$gpu
      echo "START $exp on GPU $gpu (pid $!)"
    done
    sleep 3
    while reap_one; do :; done
  done

  if (( failed > 0 )); then
    echo "WARNING: $failed parallel job(s) failed; check $LOG_DIR"
  fi
fi

echo "[parallel] Training queue finished. Logs: $LOG_DIR"
