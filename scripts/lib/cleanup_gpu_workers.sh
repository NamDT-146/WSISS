#!/usr/bin/env bash
# Shared helpers to stop orphaned GPU training workers after Ctrl+C / crash.

wssis_kill_training_workers() {
  local repo_root="${WSSIS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
  if [[ -f "$repo_root/scripts/lib/activate_wssis.sh" ]]; then
    # shellcheck source=scripts/lib/activate_wssis.sh
    source "$repo_root/scripts/lib/activate_wssis.sh"
    activate_wssis 2>/dev/null || true
  fi
  export WSSIS_REPO_ROOT="$repo_root"
  export PYTHONPATH="$repo_root:${PYTHONPATH:-}"
  python -m modules.wssis.proc_utils 2>/dev/null || {
    pkill -f "train_net.py" 2>/dev/null || true
    pkill -f "modules.wssis.run_experiment" 2>/dev/null || true
  }
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  wssis_kill_training_workers
fi
