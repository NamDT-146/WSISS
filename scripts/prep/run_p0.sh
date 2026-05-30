#!/usr/bin/env bash
# P0: splits → SAM embeddings → Stage-1 GNN
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python -m modules.wssis.prep.run_p0 "$@"
# Resume stage1:  bash scripts/prep/run_p0.sh --run-id MY_RUN --resume --skip-splits --skip-embeddings
