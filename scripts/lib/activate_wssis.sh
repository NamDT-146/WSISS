#!/usr/bin/env bash
# Shared conda activation for scripts run with `set -euo pipefail`.
# gcc_linux-64 deactivate hooks reference optional vars; nounset breaks `conda activate`.

activate_wssis() {
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  set +u
  conda activate wssis
  set -u
}
