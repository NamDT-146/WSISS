#!/usr/bin/env bash
# Download Swin-T ImageNet weights and convert to Detectron2 format for Mask2Former.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="$REPO_ROOT/checkpoints"
M2F_ROOT="$REPO_ROOT/modules/mask2former"
PTH="$CKPT_DIR/swin_tiny_patch4_window7_224.pth"
PKL="$CKPT_DIR/swin_tiny_patch4_window7_224.pkl"
URL="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"

mkdir -p "$CKPT_DIR"

if [[ -f "$PKL" ]]; then
  echo "Swin-T D2 weights already at $PKL"
  exit 0
fi

if [[ ! -f "$PTH" ]]; then
  echo "Downloading Swin-T ImageNet weights -> $PTH"
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

echo "Converting Swin-T weights to Detectron2 format -> $PKL"
python "$M2F_ROOT/tools/convert-pretrained-swin-model-to-d2.py" "$PTH" "$PKL"
echo "Done: $PKL"
