"""
Stage-2 unified training orchestration for all experiments.

Mask2Former: generates Detectron2 config and invokes train_net.py.
YOLOv8 (4A): invokes ultralytics training with exported pseudo-label config.

Full SWSIS teacher loop (SAM decoder + GNN + distillation) is integrated incrementally;
experiment flags are persisted to outputs/experiments/<ID>/experiment_config.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from modules.wssis.experiments.registry import ExperimentSpec
from modules.wssis.paths import (
    build_coco_paths,
    experiment_output_dir,
    gnn_checkpoint,
    repo_root,
    resolve_experiment_train_image_txt,
    swin_tiny_checkpoint,
)
from modules.wssis.run_context import RunContext
from modules.wssis.proc_utils import run_subprocess

# Base Mask2Former config uses IMS_PER_BATCH=16 / BASE_LR=0.0001; doubled for WSSIS Stage-2.
STAGE2_IMS_PER_BATCH = 32
STAGE2_BASE_LR = 0.0002
STAGE2_ITERS_PER_EPOCH = 1000


def _check_p0_artifacts(spec: ExperimentSpec) -> None:
    paths = build_coco_paths()
    required = [
        paths["train_all_txt"],
        paths["labeled_5pct_txt"],
        paths["labeled_5pct_train_txt"],
        paths["labeled_5pct_val_txt"],
        paths["val_sample_20pct_txt"],
        paths["weak_95pct_txt"],
        paths["val_prompts_json"],
    ]
    if spec.requires_p0:
        for p in required:
            if not p.exists():
                raise FileNotFoundError(f"Missing P0 artifact: {p}. Run: python -m modules.wssis.prep.run_p0")
    if spec.use_gnn and not gnn_checkpoint(spec.gnn_checkpoint).exists():
        raise FileNotFoundError(
            f"Missing GNN checkpoint {spec.gnn_checkpoint}. Run P0.4 or P0.4b for 2C."
        )


def _split_file_for_spec(spec: ExperimentSpec) -> Path:
    return resolve_experiment_train_image_txt(spec.labeled_split, spec.weak_split)


def _write_experiment_config(spec: ExperimentSpec, out_dir: Path, ctx: Optional[RunContext] = None) -> Path:
    cfg = {
        "experiment_id": spec.id,
        "name": spec.name,
        "student": spec.student,
        "labeled_split": spec.labeled_split,
        "weak_split": spec.weak_split,
        "use_gnn": spec.use_gnn,
        "use_raw_sam_only": spec.use_raw_sam_only,
        "use_distillation": spec.use_distillation,
        "use_symmetric_loss": spec.use_symmetric_loss,
        "weak_signal": spec.weak_signal,
        "gnn_checkpoint": spec.gnn_checkpoint,
        "stage2_epochs": spec.stage2_epochs,
        "image_list": str(_split_file_for_spec(spec)),
        "notes": spec.notes,
        "teacher_pipeline": "SAM(cache) → decoder → GNN → pseudo-GT (see PLAN §3)",
        "gpu_policy": "GPU0: SAM teacher dataloader; other GPUs: student (RANDOM_NOTE.md)",
    }
    path = out_dir / "experiment_config.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if ctx is not None:
        ctx.save_config(cfg)
        ctx.log_metrics({"event": "experiment_config_written", "experiment_id": spec.id})
    return path


def _check_mask2former_ops() -> None:
    from modules.wssis.mask2former_ops import verify_msda_import

    try:
        verify_msda_import()
    except ImportError as e:
        ops = repo_root() / "modules" / "mask2former" / "mask2former" / "modeling" / "pixel_decoder" / "ops"
        raise RuntimeError(
            "MultiScaleDeformableAttention is not compiled (required for Mask2Former).\n"
            "Run: bash scripts/setup/03_compile_mask2former_ops.sh\n"
            f"  (or: cd {ops} && bash make.sh)"
        ) from e


def _mask2former_base_config(m2f_root: Path) -> Path:
    preferred = (
        m2f_root
        / "configs"
        / "coco"
        / "instance-segmentation"
        / "swin"
        / "maskformer2_swin_tiny_bs16_50ep.yaml"
    )
    if preferred.exists():
        return preferred
    config_dir = m2f_root / "configs" / "coco" / "instance-segmentation"
    candidates = sorted(config_dir.rglob("maskformer2_swin_tiny*.yaml"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "Mask2Former Swin-T COCO instance config not found under "
        f"{config_dir}. Expected maskformer2_swin_tiny_bs16_50ep.yaml"
    )


def _check_swin_weights() -> Path:
    ckpt = swin_tiny_checkpoint()
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Missing Swin-T backbone weights: {ckpt}\n"
            "Run: bash scripts/setup/04_download_swin_weights.sh"
        )
    return ckpt


def _mask2former_train(spec: ExperimentSpec, out_dir: Path, dry_run: bool = False) -> None:
    m2f_root = repo_root() / "modules" / "mask2former"
    train_net = m2f_root / "train_net.py"
    if not train_net.exists():
        raise FileNotFoundError(f"Mask2Former not found at {train_net}")

    base_yaml = _mask2former_base_config(m2f_root)
    swin_weights = swin_tiny_checkpoint()
    max_iter = spec.stage2_epochs * STAGE2_ITERS_PER_EPOCH
    eval_period = STAGE2_ITERS_PER_EPOCH
    lr_steps = (int(max_iter * 0.7), int(max_iter * 0.9))

    generated = out_dir / "mask2former_override.yaml"
    split_txt = _split_file_for_spec(spec)
    train_ds, val_ds = (
        f"wssis_train_{spec.id}",
        f"wssis_val_{spec.id}",
    )
    generated.write_text(
        f"""# Auto-generated for experiment {spec.id}
_BASE_: "{base_yaml.as_posix()}"
WSSIS:
  EXPERIMENT_ID: "{spec.id}"
  IMAGE_LIST: "{split_txt.as_posix()}"
  LABELED_SPLIT: "{spec.labeled_split}"
  WEAK_SPLIT: "{spec.weak_split}"
  USE_GNN: {str(spec.use_gnn).lower()}
  USE_DISTILL: {str(spec.use_distillation).lower()}
  WEAK_SIGNAL: "{spec.weak_signal}"
  USE_FULL_VAL_FINAL: true
  ITERS_PER_EPOCH: {STAGE2_ITERS_PER_EPOCH}
DATASETS:
  TRAIN: ("{train_ds}",)
  TEST: ("{val_ds}",)
MODEL:
  WEIGHTS: "{swin_weights.as_posix()}"
OUTPUT_DIR: "{(out_dir / 'mask2former').as_posix()}"
TEST:
  EVAL_PERIOD: {eval_period}
SOLVER:
  MAX_ITER: {max_iter}
  STEPS: ({lr_steps[0]}, {lr_steps[1]})
  IMS_PER_BATCH: {STAGE2_IMS_PER_BATCH}
  BASE_LR: {STAGE2_BASE_LR}
""",
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(train_net),
        "--num-gpus",
        os.environ.get("WSSIS_NUM_GPUS", "1"),
        "--dist-url",
        "auto",
        "--config-file",
        str(generated),
        "OUTPUT_DIR",
        str(out_dir / "mask2former"),
        "WSSIS.EXPERIMENT_ID",
        spec.id,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root()}{os.pathsep}{m2f_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["WSSIS_REPO_ROOT"] = str(repo_root())
    env["WSSIS_IMAGE_LIST"] = str(split_txt)
    env["WSSIS_LABELED_SPLIT"] = spec.labeled_split
    env["WSSIS_WEAK_SPLIT"] = spec.weak_split

    print("[stage2] Mask2Former command:")
    print(" ".join(cmd))
    if dry_run:
        return
    _check_swin_weights()
    _check_mask2former_ops()
    result = run_subprocess(cmd, cwd=str(m2f_root), env=env)
    if result != 0:
        raise RuntimeError(
            f"Mask2Former training failed for experiment {spec.id} (exit {result}). "
            "See log above; if import error, run: bash scripts/setup/03_compile_mask2former_ops.sh\n"
            "If GPUs still show memory in use after interrupt: "
            "python -m modules.wssis.proc_utils"
        )


def _yolo_train(spec: ExperimentSpec, out_dir: Path, dry_run: bool = False) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("Install ultralytics for Exp 4A: pip install ultralytics") from e

    data_yaml = out_dir / "yolo_data.yaml"
    paths = build_coco_paths()
    data_yaml.write_text(
        f"""# YOLO dataset stub for {spec.id} — export COCO→YOLO labels before training
path: {paths['coco_root'].as_posix()}
train: {paths['labeled_5pct_txt'].as_posix()}
val: {paths['val_all_txt'].as_posix()}
# Weak 95% pseudo-label export required for semi-supervised YOLO run
weak: {paths['weak_95pct_txt'].as_posix()}
""",
        encoding="utf-8",
    )
    print(f"[stage2] YOLO data config written: {data_yaml}")
    if dry_run:
        return
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data=str(data_yaml),
        epochs=spec.stage2_epochs,
        batch=32,
        project=str(out_dir),
        name="yolov8_seg",
        exist_ok=True,
    )


def train_experiment(
    spec: ExperimentSpec,
    dry_run: bool = False,
    skip_p0_check: bool = False,
    run_ctx: Optional[RunContext] = None,
) -> Path:
    if not skip_p0_check:
        _check_p0_artifacts(spec)

    ctx = run_ctx or RunContext(task=f"exp_{spec.id}", experiment_id=spec.id)
    out_dir = ctx.exp_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_experiment_config(spec, out_dir, ctx=ctx)

    step_key = f"exp_{spec.id}"
    if ctx.is_step_done(step_key) and not dry_run:
        ctx.log("Experiment %s already marked done in progress.json (skip or delete to rerun)", spec.id)
        return out_dir

    ctx.update_step(step_key, {"status": "running", "name": spec.name})
    ctx.log("Starting experiment %s: %s", spec.id, spec.name)
    if spec.student == "mask2former":
        _mask2former_train(spec, out_dir, dry_run=dry_run)
    elif spec.student == "yolov8":
        _yolo_train(spec, out_dir, dry_run=dry_run)
    else:
        raise ValueError(f"Unknown student: {spec.student}")

    if not dry_run:
        ctx.update_step(step_key, {"status": "done", "output_dir": str(out_dir)})
        ctx.finalize_report_bundle(
            extra_files={"experiment_config.json": out_dir / "experiment_config.json"}
        )
    return out_dir


def evaluate_experiment(
    spec: ExperimentSpec,
    dry_run: bool = False,
    run_ctx: Optional[RunContext] = None,
    *,
    full_val: bool = False,
    with_teacher_eval: bool = False,
) -> None:
    out_dir = experiment_output_dir(spec.id)
    print(f"[eval] Experiment {spec.id} — outputs at {out_dir}")

    if dry_run:
        msg = "[eval] Would run: student Mask2Former COCO AP"
        if with_teacher_eval:
            msg += " + teacher val report (use scripts/eval/run_teacher_eval.sh instead)"
        print(msg)
        return

    ctx = run_ctx or RunContext(task=f"eval_{spec.id}", experiment_id=spec.id)

    if with_teacher_eval:
        from modules.wssis.training.evaluate_teacher import evaluate_teacher_on_val

        gnn_ckpt = gnn_checkpoint(spec.gnn_checkpoint) if spec.use_gnn else None
        modes = ("raw_sam", "gnn_refined") if spec.use_gnn else ("raw_sam",)
        scope = "full val_all" if full_val else "val_sample_20pct (fast)"
        print(f"[eval] Teacher baseline on {scope} (--with-teacher-eval)...")
        evaluate_teacher_on_val(
            gnn_ckpt=gnn_ckpt,
            run_ctx=ctx,
            modes=modes,
            full_val=full_val,
        )

    print(
        "[eval] Student Mask2Former COCO AP — run train_net.py --eval-only on "
        f"{out_dir / 'mask2former'} when Stage-2 checkpoints exist."
    )
