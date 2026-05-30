#!/usr/bin/env bash
# Hybrid Stage-2 training schedule:
#   Phase 1 — Exp 1C on ALL visible GPUs (main result, fastest)
#   Phase 2 — Remaining experiments on a 1-GPU-per-job worker pool
#
# GPU count: auto-detected (nvidia-smi / torch). Override with --jobs N or WSSIS_GPU_COUNT=4.
#
# Usage:
#   bash scripts/experiments/run_experiments_parallel.sh --run-id wssis_main
#   bash scripts/experiments/run_experiments_parallel.sh --jobs 4 --run-id wssis_main --resume
#   bash scripts/experiments/run_experiments_parallel.sh --no-main-first --run-id wssis_main
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
# shellcheck source=scripts/lib/detect_gpus.sh
source "$REPO_ROOT/scripts/lib/detect_gpus.sh"
activate_wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

JOBS=""
MAIN_EXP=1C
MAIN_FIRST=true
DRY_RUN=""
RESUME=""
RUN_ID="${WSSIS_RUN_ID:-}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs=*) JOBS="${1#*=}"; shift ;;
    --jobs) JOBS="${2:-}"; shift 2 ;;
    --main-exp=*) MAIN_EXP="${1#*=}"; shift ;;
    --main-exp) MAIN_EXP="${2:-1C}"; shift 2 ;;
    --no-main-first) MAIN_FIRST=false; shift ;;
    --dry-run) DRY_RUN="--dry-run"; shift ;;
    --resume) RESUME="--resume"; shift ;;
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -z "$JOBS" ]]; then
  JOBS="$(wssis_detect_gpu_count)"
  echo "[parallel] Auto-detected $JOBS GPU(s)"
fi
JOBS="$(wssis_clamp_jobs "$JOBS")"

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

if [[ -z "$DRY_RUN" ]]; then
  if ! python - <<'PY'
from modules.wssis.mask2former_ops import verify_msda_import
verify_msda_import()
print("MultiScaleDeformableAttention OK")
PY
  then
    echo "ERROR: Mask2Former ops import failed (see above)." >&2
    exit 1
  fi
fi

gpu_list() {
  local n=$1 out="" i
  for (( i=0; i<n; i++ )); do
    [[ -n "$out" ]] && out+=","
    out+="$i"
  done
  printf '%s' "$out"
}

ALL_EXPS=(1C 1A 1B 1D 2A 2B 2C 3A 3B 3C 4A)
PARALLEL_QUEUE=()
if $MAIN_FIRST; then
  PARALLEL_QUEUE=(1A 1B 1D 2A 2B 2C 3A 3B 3C 4A)
else
  PARALLEL_QUEUE=("${ALL_EXPS[@]}")
fi

run_parallel_queue() {
  local -n queue_ref=$1
  if (( ${#queue_ref[@]} == 0 )); then
    return 0
  fi

  if command -v parallel >/dev/null 2>&1; then
    echo "[parallel] GNU parallel: $JOBS workers (1 GPU each), queue=${queue_ref[*]}"
    export WSSIS_REPO_ROOT PYTHONPATH
    printf '%s\n' "${queue_ref[@]}" | \
      parallel -j "$JOBS" --line-buffer --joblog "$LOG_DIR/joblog.tsv" \
        "CUDA_VISIBLE_DEVICES=\$(( {%} - 1 )) WSSIS_NUM_GPUS=1 WSSIS_REPO_ROOT='$WSSIS_REPO_ROOT' PYTHONPATH='$PYTHONPATH' \
         python -m modules.wssis.run_experiment --exp {} ${RUN_ARGS[*]}"
  else
    echo "[parallel] GNU parallel not found; using bash worker pool ($JOBS slots, 1 GPU each)"
    declare -A PID_GPU=()
    FREE=()
    local qi=0 failed=0 exp gpu pid rc

    for (( g=0; g<JOBS; g++ )); do FREE+=("$g"); done

    reap_one() {
      for pid in "${!PID_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
          rc=0
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

    while (( qi < ${#queue_ref[@]} )) || (( ${#PID_GPU[@]} > 0 )); do
      while (( qi < ${#queue_ref[@]} )) && (( ${#FREE[@]} > 0 )); do
        exp=${queue_ref[qi++]}
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
}

if $MAIN_FIRST; then
  ALL_GPUS="$(gpu_list "$JOBS")"
  echo "========== Phase 1: Exp $MAIN_EXP on all $JOBS GPUs ($ALL_GPUS) =========="
  set +e
  (
    export CUDA_VISIBLE_DEVICES="$ALL_GPUS"
    export WSSIS_NUM_GPUS="$JOBS"
    python -m modules.wssis.run_experiment --exp "$MAIN_EXP" "${RUN_ARGS[@]}" "${EXTRA[@]}"
  ) 2>&1 | tee "$LOG_DIR/exp_${MAIN_EXP}.all_gpus.log"
  phase1_rc=${PIPESTATUS[0]}
  set -e
  if (( phase1_rc != 0 )); then
    echo "ERROR: Phase 1 ($MAIN_EXP) failed (exit $phase1_rc). Fix before phase 2." >&2
    exit "$phase1_rc"
  fi

  echo "========== Phase 2: ${#PARALLEL_QUEUE[@]} experiments, $JOBS parallel (1 GPU each) =========="
  run_parallel_queue PARALLEL_QUEUE
else
  echo "========== Parallel pool: all experiments, $JOBS workers (1 GPU each) =========="
  run_parallel_queue PARALLEL_QUEUE
fi

echo "[parallel] Training schedule finished. Logs: $LOG_DIR"
