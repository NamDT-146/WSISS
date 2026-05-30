#!/usr/bin/env bash
# Download SAM ViT-B checkpoint to checkpoints/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="$REPO_ROOT/checkpoints"
mkdir -p "$CKPT_DIR"
OUT="$CKPT_DIR/sam_vit_b_01ec64.pth"

if [[ -f "$OUT" ]]; then
  echo "SAM weights already at $OUT"
  exit 0
fi

URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
echo "Downloading SAM ViT-B -> $OUT"
if command -v wget &>/dev/null; then
  wget -q --show-progress -O "$OUT" "$URL"
elif command -v curl &>/dev/null; then
  curl -L -o "$OUT" "$URL"
else
  python -c "import urllib.request; urllib.request.urlretrieve('$URL', '$OUT')"
fi
echo "Done: $OUT"
