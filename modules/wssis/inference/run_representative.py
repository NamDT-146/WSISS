"""
Run qualitative inference on the fixed representative val set (report figure grid).

Five settings (report matrix):
  1. Raw SAM (teacher, best of 3 decoder heads)
  2. SAM + Refiner refined (teacher pseudo)
  3. Exp 1C student (true semi-weak SWSIS)
  4. Exp 1D student (100% GT upper bound)
  5. Exp 4A student (YOLOv8-seg semi-weak)

Also saves GT overlay column.

Usage:
  python -m modules.wssis.inference.run_representative --build-list
  python -m modules.wssis.inference.run_representative --run-id wssis_main
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from modules.vig_refinenet.coco_sam_stage1_dataset import ann_to_mask
from modules.vig_refinenet.sam_stage1_common import (
    _best_mask_by_score,
    build_weak_signal_tensor,
    decode_sam_masks_3_batch,
    get_sam_pixel_stats,
    load_sam_vit_b,
)
from modules.vig_refinenet.sam_stage1_refiner import build_sam_stage1_refiner
from modules.wssis.inference.representative_val import (
    DEFAULT_LIST_PATH,
    RepresentativeSample,
    curate_representative_val_list,
    load_val_annotations_for_samples,
    parse_representative_list,
    write_representative_list,
)
from modules.wssis.experiments.registry import get_experiment
from modules.wssis.paths import (
    coco_root,
    gnn_checkpoint,
    repo_root,
    resolve_coco_image_dir,
    sam_vit_b_checkpoint,
)
from modules.wssis.run_context import RunContext
from modules.wssis.sam_cache import fetch_sam_embeddings_batch
from modules.wssis.stage2_constants import STAGE2_STUDENT_IMAGE_SIZE
from modules.wssis.training.visualize import _overlay_mask, save_refinement_grid
from modules.wssis.weak_prompts import build_instance_prompts, sam_prompt_for_signal

SETTING_LABELS: Tuple[str, ...] = (
    "Raw SAM",
    "SAM + Refiner",
    "1C (SWSIS)",
    "1D (upper)",
    "4A (YOLO semi-weak)",
)


def _ensure_repo_path() -> None:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _resolve_m2f_checkpoint(exp_dir: Path) -> Optional[Path]:
    m2f = exp_dir / "mask2former"
    for name in ("model_best.pth", "model_final.pth"):
        p = m2f / name
        if p.is_file():
            return p
    periodic = sorted(m2f.glob("model_*.pth"))
    return periodic[-1] if periodic else None


def _resolve_m2f_config(exp_dir: Path) -> Optional[Path]:
    p = exp_dir / "mask2former_override.yaml"
    return p if p.is_file() else None


def _combined_gt_overlay(
    image_rgb: np.ndarray,
    anns: Sequence[dict],
    height: int,
    width: int,
) -> np.ndarray:
    union = np.zeros((height, width), dtype=bool)
    for ann in anns:
        union |= ann_to_mask(ann, height, width).astype(bool)
    return _overlay_mask(image_rgb, union, color=(0.2, 0.9, 0.2))


def _teacher_overlays_for_image(
    image_rgb: np.ndarray,
    anns: Sequence[dict],
    *,
    sam_model: torch.nn.Module,
    refiner: Optional[torch.nn.Module],
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    image_id: int,
    device: torch.device,
    mask_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-instance oracle prompts → union raw SAM and GNN refined overlays."""
    h, w = image_rgb.shape[:2]
    raw_union = np.zeros((h, w), dtype=bool)
    gnn_union = np.zeros((h, w), dtype=bool)

    img1024 = cv2.resize(image_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    image_t = (
        torch.from_numpy(img1024)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )

    for ann in anns:
        mask_np = ann_to_mask(ann, h, w).astype(np.uint8)
        if mask_np.sum() == 0:
            continue
        mask1024 = cv2.resize(mask_np, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        prompts = build_instance_prompts(mask1024 > 0, policy="val_fixed", signal_type="mixed")
        meta = {"image_id": image_id, "ann_id": int(ann["id"]), "split": "val"}

        with torch.no_grad():
            embed, _ = fetch_sam_embeddings_batch(
                [meta],
                sam_model,
                image_t,
                pixel_mean,
                pixel_std,
                use_cache=True,
            )
            sam_prompt = sam_prompt_for_signal(prompts, "points_only")
            sam_masks_3, scores = decode_sam_masks_3_batch(
                sam_model,
                image_t,
                [sam_prompt],
                mask_size=mask_size,
                prompt_space=mask_size,
                image_embeddings=embed,
            )
            raw_best = _best_mask_by_score(sam_masks_3, scores)
            raw_np = (raw_best[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
            raw_full = cv2.resize(raw_np, (w, h), interpolation=cv2.INTER_NEAREST) > 0
            raw_union |= raw_full

            if refiner is not None:
                weak_signal = build_weak_signal_tensor(
                    [prompts],
                    spatial_size=mask_size,
                    device=device,
                    mask_np_list=[mask1024],
                    active_signal=None,
                    policy="val_fixed",
                )
                logits = refiner(embed, image_t, sam_masks_3, weak_signal)
                gnn_np = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
                gnn_full = cv2.resize(gnn_np, (w, h), interpolation=cv2.INTER_NEAREST) > 0
                gnn_union |= gnn_full
            else:
                gnn_union |= raw_full

    raw_vis = _overlay_mask(image_rgb, raw_union, color=(1.0, 0.35, 0.35))
    gnn_vis = _overlay_mask(image_rgb, gnn_union, color=(0.25, 0.5, 1.0))
    return raw_vis, gnn_vis


def _student_overlay(
    predictor,
    image_rgb: np.ndarray,
) -> np.ndarray:
    """Mask2Former DefaultPredictor → union of predicted instance masks."""
    h, w = image_rgb.shape[:2]
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    with torch.no_grad():
        outputs = predictor(bgr)
    instances = outputs["instances"].to("cpu")
    union = np.zeros((h, w), dtype=bool)
    if instances.has("pred_masks") and len(instances) > 0:
        masks = instances.pred_masks.numpy()
        for m in masks:
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            union |= m.astype(bool)
    return _overlay_mask(image_rgb, union, color=(0.95, 0.55, 0.1))


def _resolve_yolov8_checkpoint(exp_dir: Path) -> Optional[Path]:
    """
    Resolve YOLOv8-seg weights for Exp 4A.

    Stage-2 uses: model.train(project=<exp_dir>, name="yolov8_seg", exist_ok=True)
    so we expect: <exp_dir>/yolov8_seg/weights/{best,last}.pt
    """
    candidates = (
        exp_dir / "yolov8_seg" / "weights" / "best.pt",
        exp_dir / "yolov8_seg" / "weights" / "last.pt",
    )
    for p in candidates:
        if p.is_file():
            return p

    # Fallback: pick the newest .pt under weights/.
    weights = sorted(exp_dir.glob("**/weights/*.pt"))
    return weights[-1] if weights else None


def _build_yolov8_model(weights: Path):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics is required for Exp 4A (YOLOv8) inference. "
            "Install it in the current environment: pip install ultralytics"
        ) from e
    return YOLO(str(weights))


def _yolov8_student_overlay(
    model,
    image_rgb: np.ndarray,
    *,
    device: torch.device,
    imgsz: int = STAGE2_STUDENT_IMAGE_SIZE,
) -> np.ndarray:
    """Ultralytics YOLOv8-seg → union of predicted instance masks."""
    h, w = image_rgb.shape[:2]
    with torch.no_grad():
        results = model.predict(
            source=image_rgb,
            imgsz=imgsz,
            device=str(device),
            verbose=False,
        )

    union = np.zeros((h, w), dtype=bool)
    if not results:
        return _overlay_mask(image_rgb, union, color=(0.95, 0.55, 0.1))

    res0 = results[0]
    if getattr(res0, "masks", None) is None or getattr(res0.masks, "data", None) is None:
        return _overlay_mask(image_rgb, union, color=(0.95, 0.55, 0.1))

    masks = res0.masks.data
    for m in masks:
        m_np = m.detach().cpu().numpy()
        if m_np.shape != (h, w):
            m_np = cv2.resize(m_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        union |= m_np > 0.5

    return _overlay_mask(image_rgb, union, color=(0.95, 0.55, 0.1))


def _build_m2f_predictor(config_yaml: Path, weights: Path, device: str = "cuda"):
    # Ensure local Mask2Former repo is on sys.path *before* importing it.
    m2f_root = repo_root() / "modules" / "mask2former"
    if str(m2f_root) not in sys.path:
        sys.path.insert(0, str(m2f_root))

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    from modules.wssis.mask2former_config import add_wssis_config
    from modules.wssis.mask2former_datasets import ensure_wssis_datasets_in_cfg

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_wssis_config(cfg)
    cfg.merge_from_file(str(config_yaml))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.DEVICE = device
    ensure_wssis_datasets_in_cfg(cfg)
    cfg.freeze()
    return DefaultPredictor(cfg)


def _load_teacher_stack(device: torch.device, gnn_ckpt: Path):
    sam = load_sam_vit_b(str(sam_vit_b_checkpoint()), device)
    pixel_mean, pixel_std = get_sam_pixel_stats(device)
    refiner = None
    if gnn_ckpt.is_file():
        payload = torch.load(gnn_ckpt, map_location=device, weights_only=False)
        refiner = build_sam_stage1_refiner(payload.get("config", {})).to(device)
        refiner.load_state_dict(payload["state_dict"], strict=False)
        refiner.eval()
    return sam, refiner, pixel_mean, pixel_std


def build_list(out_path: Path) -> List[RepresentativeSample]:
    samples = curate_representative_val_list()
    write_representative_list(samples, out_path)
    print(f"[representative] Wrote {len(samples)} images -> {out_path}")
    for cat in ("easy_person", "multi_separated", "overlapping", "minor_class"):
        n = sum(1 for s in samples if s.category == cat)
        print(f"  {cat}: {n}")
    return samples


def run_inference(
    *,
    list_path: Path,
    run_id: str,
    run_dir: Optional[Path],
    device: str,
    max_images: Optional[int],
    skip_students: bool,
    gnn_ckpt: Path,
) -> Path:
    samples = parse_representative_list(list_path)
    if max_images is not None:
        samples = samples[:max_images]

    ctx = RunContext(run_id=run_id, run_dir=run_dir, task="representative_inference")
    out_dir = ctx.viz_dir / "representative_inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_map = load_val_annotations_for_samples(samples)
    val_image_root = resolve_coco_image_dir(coco_root(), "val")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    print("[representative] Loading teacher (SAM + Refiner)...")
    sam, refiner, pixel_mean, pixel_std = _load_teacher_stack(dev, gnn_ckpt)

    students: Dict[str, Tuple[str, object]] = {}
    if not skip_students:
        for exp_id in ("1C", "1D", "4A"):
            exp_dir = ctx.root / "experiments" / exp_id
            spec = get_experiment(exp_id)
            if spec.student == "mask2former":
                cfg_path = _resolve_m2f_config(exp_dir)
                ckpt = _resolve_m2f_checkpoint(exp_dir)
                if cfg_path is None or ckpt is None:
                    print(f"[representative] Skip student {exp_id}: missing Mask2Former config or checkpoint under {exp_dir}")
                    continue
                print(f"[representative] Loading {exp_id} (mask2former) from {ckpt.name}")
                model = _build_m2f_predictor(cfg_path, ckpt, device=str(dev))
                students[exp_id] = (spec.student, model)
            elif spec.student == "yolov8":
                ckpt = _resolve_yolov8_checkpoint(exp_dir)
                if ckpt is None:
                    print(f"[representative] Skip student {exp_id}: missing YOLOv8 weights under {exp_dir}")
                    continue
                print(f"[representative] Loading {exp_id} (yolov8) from {ckpt}")
                try:
                    model = _build_yolov8_model(ckpt)
                    students[exp_id] = (spec.student, model)
                except Exception as exc:
                    print(f"[representative] Skip student {exp_id}: {exc}")
            else:
                print(f"[representative] Skip student {exp_id}: unknown student type {spec.student}")

    manifest = []
    for idx, sample in enumerate(samples):
        info, anns = ann_map[sample.image_id]
        img_path = val_image_root / info["file_name"]
        if not img_path.is_file():
            img_path = val_image_root / Path(info["file_name"]).name
        if not img_path.is_file():
            print(f"[representative] Missing image {img_path}; skip")
            continue

        image_rgb = np.array(__import__("PIL").Image.open(img_path).convert("RGB"))
        h, w = image_rgb.shape[:2]

        gt_vis = _combined_gt_overlay(image_rgb, anns, info["height"], info["width"])
        raw_vis, gnn_vis = _teacher_overlays_for_image(
            image_rgb,
            anns,
            sam_model=sam,
            refiner=refiner,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            image_id=sample.image_id,
            device=dev,
        )

        panels: List[Tuple[str, np.ndarray]] = [
            ("Image", image_rgb),
            ("GT", gt_vis),
            ("Raw SAM", raw_vis),
            ("SAM + Refiner", gnn_vis),
        ]

        for exp_id, label in zip(("1C", "1D", "4A"), SETTING_LABELS[2:]):
            if exp_id in students:
                try:
                    student_type, model = students[exp_id]
                    if student_type == "mask2former":
                        panels.append((label, _student_overlay(model, image_rgb)))
                    elif student_type == "yolov8":
                        panels.append(
                            (
                                label,
                                _yolov8_student_overlay(
                                    model,
                                    image_rgb,
                                    device=dev,
                                    imgsz=STAGE2_STUDENT_IMAGE_SIZE,
                                ),
                            )
                        )
                    else:
                        panels.append((label, image_rgb.copy()))
                except Exception as exc:
                    print(f"[representative] {exp_id} failed on {sample.image_id}: {exc}")
                    panels.append((label, image_rgb.copy()))
            else:
                panels.append((label + " (n/a)", image_rgb.copy()))

        fname = out_dir / f"{sample.category}_{sample.image_id:012d}.png"
        save_refinement_grid(
            panels,
            fname,
            title=f"{sample.category} | img {sample.image_id} | {sample.n_objects} objs",
        )
        print(f"[representative] [{idx + 1}/{len(samples)}] {fname.name}")

        manifest.append(
            {
                "category": sample.category,
                "image_id": sample.image_id,
                "file_name": sample.file_name,
                "n_objects": sample.n_objects,
                "category_ids": list(sample.category_ids),
                "output_png": str(fname),
                "students_loaded": list(students.keys()),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ctx.log("Representative inference -> %s (%d images)", out_dir, len(manifest))
    print(f"[representative] Done. Outputs: {out_dir}")
    return out_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    _ensure_repo_path()

    parser = argparse.ArgumentParser(description="Representative val inference (5 settings + GT)")
    parser.add_argument(
        "--build-list",
        action="store_true",
        help="Curate and write fixed representative_val_inference.txt (then exit)",
    )
    parser.add_argument(
        "--list-path",
        type=Path,
        default=DEFAULT_LIST_PATH,
        help="Fixed image list (default: scripts/inference/representative_val_inference.txt)",
    )
    parser.add_argument("--run-id", default=None, help="Run bundle under outputs/runs/")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=None, help="Debug: cap number of images")
    parser.add_argument(
        "--skip-students",
        action="store_true",
        help="Teacher-only grids (Raw SAM + SAM+Refiner); skip 1C/1D/4A students",
    )
    parser.add_argument(
        "--gnn-checkpoint",
        type=Path,
        default=None,
        help="GNN ckpt (default: checkpoints/gnn_refiner_stage1.pt)",
    )
    args = parser.parse_args(argv)

    if args.build_list:
        build_list(args.list_path)
        return

    if not args.list_path.is_file():
        print(f"List missing: {args.list_path}. Run with --build-list first.")
        sys.exit(1)

    gnn_ckpt = args.gnn_checkpoint or gnn_checkpoint()
    run_inference(
        list_path=args.list_path,
        run_id=args.run_id or __import__("os").environ.get("WSSIS_RUN_ID", "wssis_main"),
        run_dir=Path(args.run_dir) if args.run_dir else None,
        device=args.device,
        max_images=args.max_images,
        skip_students=args.skip_students,
        gnn_ckpt=gnn_ckpt,
    )


if __name__ == "__main__":
    main()
