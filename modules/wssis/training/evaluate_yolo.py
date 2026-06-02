"""
Post-train YOLOv8-seg eval with COCO-style metrics (parity with Mask2Former SUMMARY tables).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from modules.wssis.experiments.registry import ExperimentSpec, get_experiment
from modules.wssis.run_context import RunContext


def evaluate_yolo_experiment(
    spec: ExperimentSpec,
    run_ctx: Optional[RunContext] = None,
) -> Dict:
    """Run Ultralytics val on exported dataset; write JSON report under run eval/."""
    ctx = run_ctx or RunContext(task=f"eval_{spec.id}", experiment_id=spec.id)
    exp_dir = ctx.exp_dir
    weights = exp_dir / "yolov8_seg" / "weights" / "best.pt"
    if not weights.exists():
        weights = exp_dir / "yolov8_seg" / "weights" / "last.pt"
    data_yaml = exp_dir / "yolo_export" / "data.yaml"
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found under {exp_dir / 'yolov8_seg'}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing YOLO data.yaml: {data_yaml}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split="val", verbose=False)

    def _safe(obj, key, default=0.0):
        try:
            v = obj[key] if hasattr(obj, "__getitem__") else getattr(obj, key, default)
            return float(v) if v is not None else default
        except (KeyError, TypeError, AttributeError):
            return default

    seg = getattr(metrics, "seg", None) or metrics
    report = {
        "experiment_id": spec.id,
        "student": "yolov8_seg",
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "metrics": {
            "AP": _safe(seg, "map"),
            "AP50": _safe(seg, "map50"),
            "AP75": _safe(seg, "map75"),
            "AP_small": _safe(seg, "maps"),
            "AP_medium": _safe(seg, "mapm"),
            "AP_large": _safe(seg, "mapl"),
        },
    }
    out_path = ctx.eval_dir / f"yolo_val_report_{spec.id}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ctx.log("YOLO eval -> %s", out_path)
    ctx.log(
        "  segm AP=%.4f AP50=%.4f AP75=%.4f",
        report["metrics"]["AP"],
        report["metrics"]["AP50"],
        report["metrics"]["AP75"],
    )
    ctx.finalize_report_bundle(extra_files={out_path.name: out_path})
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8-seg experiment on val split")
    parser.add_argument("--exp", required=True, help="Experiment id (e.g. 4A)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)

    spec = get_experiment(args.exp)
    if spec.student != "yolov8":
        raise ValueError(f"Experiment {spec.id} is not YOLO (student={spec.student})")

    ctx = RunContext(run_id=args.run_id, run_dir=args.run_dir, task=f"eval_{spec.id}", experiment_id=spec.id)
    evaluate_yolo_experiment(spec, run_ctx=ctx)


if __name__ == "__main__":
    main()
