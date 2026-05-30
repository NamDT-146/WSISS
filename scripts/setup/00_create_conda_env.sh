#!/usr/bin/env bash
# Create conda env wssis, install pip deps, Detectron2, compile Mask2Former ops.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTORCH_INDEX="${WSSIS_PYTORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
# Match tested stack (pip list on msra GPU nodes); override if needed.
TORCH_VERSION="${WSSIS_TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${WSSIS_TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${WSSIS_TORCHAUDIO_VERSION:-2.6.0}"

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

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
activate_wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

need_pytorch() {
  python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit(1)
v = torch.__version__.split("+")[0]
major, minor = (int(x) for x in v.split(".")[:2])
if major < 2 or (major == 2 and minor < 4):
    sys.exit(1)
if not torch.cuda.is_available():
    print("WARNING: torch installed but CUDA not available", file=sys.stderr)
sys.exit(0)
PY
}

install_pytorch() {
  if ! need_pytorch; then
    echo "==> Installing PyTorch ${TORCH_VERSION} (${PYTORCH_INDEX})"
    pip install \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" \
      --index-url "$PYTORCH_INDEX"
  else
    python -c "import torch; print('==> PyTorch OK:', torch.__version__, 'CUDA', torch.version.cuda)"
  fi
}

install_pytorch

echo "==> Installing pip requirements"
pip install -r requirements.txt

# ultralytics pulls opencv-python; keep headless-only to avoid duplicate cv2 builds
if python -c "import importlib.util; exit(0 if importlib.util.find_spec('cv2') else 1)" 2>/dev/null; then
  if pip show opencv-python &>/dev/null && pip show opencv-python-headless &>/dev/null; then
    echo "==> Removing opencv-python (keeping opencv-python-headless for headless servers)"
    pip uninstall -y opencv-python || true
  fi
fi

install_detectron2() {
  if python -c "import detectron2" 2>/dev/null; then
    python -c "import detectron2; print('==> Detectron2 OK:', detectron2.__version__)"
    return 0
  fi

  local torch_mm
  torch_mm="$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")"
  local cuda_tag="cpu"
  if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    cuda_tag="cu$(python -c "import torch; v=torch.version.cuda or ''; print(v.replace('.','')[:3])")"
  fi

  echo "==> Installing Detectron2 (torch ${torch_mm}, ${cuda_tag})"
  local wheel_index="https://dl.fbaipublicfiles.com/detectron2/wheels/${cuda_tag}/torch${torch_mm}/index.html"
  if pip install detectron2 -f "$wheel_index"; then
    python -c "import detectron2; print('==> Detectron2 wheel OK:', detectron2.__version__)"
    return 0
  fi

  echo "==> No matching Detectron2 wheel; building vendored modules/detectron2"
  pip install --no-build-isolation -e "$REPO_ROOT/modules/detectron2"
}

install_detectron2

echo "==> Compiling Mask2Former MSDeformAttn ops"
bash "$REPO_ROOT/scripts/setup/03_compile_mask2former_ops.sh"

echo "==> Verifying imports"
python - <<'PY'
import importlib
for mod in ("torch", "detectron2", "segment_anything", "ultralytics", "modules.wssis"):
    importlib.import_module(mod)
import MultiScaleDeformableAttention as MSDA
print("MultiScaleDeformableAttention OK")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
import detectron2
print("detectron2", detectron2.__version__)
PY

echo "==> Download SAM ViT-B weights"
bash "$REPO_ROOT/scripts/setup/02_download_sam_weights.sh"

echo ""
echo "Setup complete. Activate with:  conda activate wssis"
echo "Place Kaggle credentials:       data/kaggle.json"
echo "Then download data:             bash scripts/setup/01_download_data.sh"
