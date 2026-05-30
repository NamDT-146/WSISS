#!/usr/bin/env bash
# Create conda env wssis, install pip deps, Detectron2, compile Mask2Former ops.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "==> Repo root: $REPO_ROOT"

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

# Create or update env
if conda env list | grep -qE '^wssis\s'; then
  echo "==> Updating existing env 'wssis'"
  conda env update -f environment.yml --prune
else
  echo "==> Creating env 'wssis'"
  conda env create -f environment.yml
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "==> Installing pip requirements"
pip install -r requirements.txt

echo "==> Installing Detectron2 (CUDA 11.8 + torch 2.1)"
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html \
  || pip install 'git+https://github.com/facebookresearch/detectron2.git'

echo "==> Compiling Mask2Former MSDeformAttn ops"
OPS_DIR="$REPO_ROOT/modules/mask2former/mask2former/modeling/pixel_decoder/ops"
if [[ -d "$OPS_DIR" ]]; then
  cd "$OPS_DIR"
  if [[ -f make.sh ]]; then
    bash make.sh || python setup.py build install
  fi
  cd "$REPO_ROOT"
fi

echo "==> Download SAM ViT-B weights"
bash "$REPO_ROOT/scripts/setup/02_download_sam_weights.sh"

echo ""
echo "Setup complete. Activate with:  conda activate wssis"
echo "Place Kaggle credentials:       data/kaggle.json"
echo "Then download data:             bash scripts/setup/01_download_data.sh"
