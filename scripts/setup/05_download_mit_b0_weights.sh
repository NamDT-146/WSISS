#!/usr/bin/env bash
# Download SegFormer MiT-B0 ImageNet weights and convert to Detectron2 format.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="$REPO_ROOT/checkpoints"
M2F_ROOT="$REPO_ROOT/modules/mask2former"
PTH="$CKPT_DIR/mit_b0.pth"
PKL="$CKPT_DIR/mit_b0_pretrained.pkl"

mkdir -p "$CKPT_DIR"

if [[ -f "$PKL" ]]; then
  echo "MiT-B0 D2 weights already at $PKL"
  exit 0
fi

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
activate_wssis

echo "Fetching / converting MiT-B0 weights -> $PKL"
python "$M2F_ROOT/tools/convert_mit_b0_to_d2.py" --fetch "$PTH" "$PKL"
echo "Done: $PKL"
