#!/usr/bin/env bash
# Download SegFormer MiT-B0 ImageNet weights and convert to Detectron2 format.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="$REPO_ROOT/checkpoints"
M2F_ROOT="$REPO_ROOT/modules/mask2former"
PTH="$CKPT_DIR/mit_b0.pth"
PKL="$CKPT_DIR/mit_b0_pretrained.pkl"
URL="https://github.com/NVlabs/SegFormer/releases/download/v1.0/mit_b0.pth"

mkdir -p "$CKPT_DIR"

if [[ -f "$PKL" ]]; then
  echo "MiT-B0 D2 weights already at $PKL"
  exit 0
fi

if [[ ! -f "$PTH" ]]; then
  echo "Downloading SegFormer MiT-B0 weights -> $PTH"
  if command -v wget &>/dev/null; then
    wget -q --show-progress -O "$PTH" "$URL"
  elif command -v curl &>/dev/null; then
    curl -L -o "$PTH" "$URL"
  else
    python -c "import urllib.request; urllib.request.urlretrieve('$URL', '$PTH')"
  fi
fi

# shellcheck source=scripts/lib/activate_wssis.sh
source "$REPO_ROOT/scripts/lib/activate_wssis.sh"
activate_wssis

echo "Converting MiT-B0 weights to Detectron2 format -> $PKL"
python "$M2F_ROOT/tools/convert_mit_b0_to_d2.py" "$PTH" "$PKL"
echo "Done: $PKL"
