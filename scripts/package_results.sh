#!/usr/bin/env bash
# Package essential run artifacts for submission — not the full outputs/ tree.
#
# Includes: teacher + student best weights, metrics/logs, eval JSON, visualizations.
# Excludes: tensorboard, yolo_export datasets, periodic M2F checkpoints (model_0*.pth).
#
# Usage:
#   bash scripts/package_results.sh
#   bash scripts/package_results.sh --run-id wssis_main -o result.zip
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${WSSIS_RUN_ID:-wssis_main}"
OUT_ZIP="result.zip"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    -o|--output) OUT_ZIP="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      OUT_ZIP="$1"
      shift
      ;;
  esac
done

RUN_DIR="outputs/runs/${RUN_ID}"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "[package] Run directory not found: $RUN_DIR" >&2
  exit 1
fi

LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

_add_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    echo "${f#"$REPO_ROOT"/}" >> "$LIST"
  fi
}

_add_tree() {
  local d="$1"
  if [[ -d "$d" ]]; then
    find "$d" -type f >> "$LIST"
  fi
}

# Run metadata + teacher checkpoint
for rel in config.json progress.json; do
  _add_file "$RUN_DIR/$rel"
done
_add_file "$RUN_DIR/checkpoints/best.pt"

# Metrics / training logs (skip tensorboard — large, not needed for report)
for rel in logs/metrics.jsonl logs/train.log logs/metrics_history.json; do
  _add_file "$RUN_DIR/$rel"
done
if [[ -d "$RUN_DIR/logs/parallel" ]]; then
  find "$RUN_DIR/logs/parallel" -type f -name '*.log' >> "$LIST"
fi

# Eval reports + all visualizations (incl. representative_inference grids)
_add_tree "$RUN_DIR/eval"
_add_tree "$RUN_DIR/visualizations"

# Curated report bundle (if finalize_report_bundle was run)
_add_tree "$RUN_DIR/report"

# Per-experiment: config + best weights + eval metrics (no full training dumps)
shopt -s nullglob
for exp_dir in "$RUN_DIR/experiments"/*; do
  [[ -d "$exp_dir" ]] || continue

  for rel in experiment_config.json mask2former_override.yaml; do
    _add_file "$exp_dir/$rel"
  done

  m2f="$exp_dir/mask2former"
  if [[ -d "$m2f" ]]; then
    for ckpt in model_best.pth model_final.pth; do
      _add_file "$m2f/$ckpt"
    done
    for rel in metrics.json coco_instances_results.json log.txt log.txt.rank0; do
      _add_file "$m2f/$rel"
    done
  fi

  yolo_weights="$exp_dir/yolov8_seg/weights"
  _add_file "$yolo_weights/best.pt"
  for rel in yolov8_seg/results.csv yolov8_seg/args.yaml; do
    _add_file "$exp_dir/$rel"
  done
done
shopt -u nullglob

# De-dupe and make paths repo-relative
sort -u "$LIST" | sed "s|^$REPO_ROOT/||" > "${LIST}.clean"
mv "${LIST}.clean" "$LIST"

if [[ ! -s "$LIST" ]]; then
  echo "[package] No files matched under $RUN_DIR" >&2
  exit 1
fi

rm -f "$OUT_ZIP"
zip -q "$OUT_ZIP" -@ < "$LIST"

n_files="$(wc -l < "$LIST" | tr -d ' ')"
size="$(du -h "$OUT_ZIP" | cut -f1)"
echo "[package] Wrote $OUT_ZIP ($size, $n_files files) from outputs/runs/$RUN_ID"
