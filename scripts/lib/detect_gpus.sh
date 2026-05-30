#!/usr/bin/env bash
# Detect visible NVIDIA GPU count for experiment scheduling.

wssis_detect_gpu_count() {
  if [[ -n "${WSSIS_GPU_COUNT:-}" ]]; then
    printf '%s' "$WSSIS_GPU_COUNT"
    return 0
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local csv="${CUDA_VISIBLE_DEVICES// /}"
    if [[ "$csv" == *","* ]]; then
      awk -F, '{print NF}' <<<"$csv"
      return 0
    fi
    if [[ "$csv" =~ ^[0-9]+$ ]]; then
      printf '1'
      return 0
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
    return 0
  fi
  python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 0))
PY
}

wssis_clamp_jobs() {
  local requested=$1
  local available
  available="$(wssis_detect_gpu_count)"
  if (( available < 1 )); then
    echo "ERROR: no CUDA GPUs detected (set WSSIS_GPU_COUNT to override)" >&2
    return 1
  fi
  if (( requested > available )); then
    echo "WARNING: requested $requested GPU(s) but only $available visible; using $available" >&2
    printf '%s' "$available"
    return 0
  fi
  printf '%s' "$requested"
}
