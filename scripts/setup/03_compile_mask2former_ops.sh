#!/usr/bin/env bash
# Compile Mask2Former MultiScaleDeformableAttention CUDA extension (required for Stage-2).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
activate_wssis

export WSSIS_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

OPS_DIR="$REPO_ROOT/modules/mask2former/mask2former/modeling/pixel_decoder/ops"
if [[ ! -d "$OPS_DIR" ]]; then
  echo "ERROR: Mask2Former ops dir not found: $OPS_DIR" >&2
  exit 1
fi

echo "==> Compiling MultiScaleDeformableAttention in $OPS_DIR"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

cd "$OPS_DIR"
if [[ -f make.sh ]]; then
  bash make.sh
else
  python setup.py build install
fi

cd "$REPO_ROOT"
echo "==> Verifying import"
python - <<'PY'
from modules.wssis.mask2former_ops import verify_msda_import
verify_msda_import()
print("MultiScaleDeformableAttention OK")
PY

echo "Mask2Former ops ready."
