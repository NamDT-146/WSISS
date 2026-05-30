#!/usr/bin/env bash
# Download COCO 2017 + coco-minitrain-10k via Kaggle API into data/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate wssis 2>/dev/null || true

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export KAGGLE_CONFIG_DIR="$REPO_ROOT/data"

if [[ ! -f "$REPO_ROOT/data/kaggle.json" ]]; then
  echo "ERROR: Copy your Kaggle API token to: $REPO_ROOT/data/kaggle.json"
  echo "  cp ~/.kaggle/kaggle.json $REPO_ROOT/data/kaggle.json"
  exit 1
fi

python -m modules.wssis.prep.download_kaggle "$@"
