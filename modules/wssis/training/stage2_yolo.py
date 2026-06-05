"""
Stage-2 joint teacher-student training for YOLOv8-seg (Exp 4A).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from detectron2.data import detection_utils as utils
from detectron2.data.datasets.coco import load_coco_json

from modules.wssis.experiments.registry import ExperimentSpec
from modules.wssis.mask2former_datasets import coco_anns_to_masks_for_image
from modules.wssis.mask2former_teacher import WssisTeacherStack
from modules.wssis.paths import build_coco_paths, gnn_checkpoint, resolve_coco_image_dir
from modules.wssis.pseudo_label_confidence import build_threshold_policy, refined_probs_from_logits
from modules.wssis.run_context import EarlyStopping
from modules.wssis.stage2_constants import STAGE2_STUDENT_IMAGE_SIZE
from modules.wssis.training.stage2_augment import (
    GeomTransformParams,
    IMAGENET_MEAN,
    IMAGENET_STD,
    apply_geom_transform_to_mask,
    build_dual_views,
)
from modules.wssis.training.stage2_yolo_proto import yolo_mask_logits as _yolo_mask_logits
from modules.wssis.training.stage2_losses import (
    LossWeightSchedule,
    aggregate_refined_logits_per_image,
    aggregate_weak_signal_per_image,
    build_pce_valid_mask,
    partial_bce_loss,
    partial_dice_loss,
    student_feedback_loss,
    symmetric_sam_triplet_loss,
    voting_pseudo_mask,
)
from modules.vig_refinenet.sam_stage1_common import (
    compute_iou,
    forward_teacher_objects_impl,
    mask_ap_from_iou,
)

if TYPE_CHECKING:
    from modules.wssis.run_context import RunContext


class Stage2YoloDataset(Dataset):
    """Semi-weak 50/50 image records for YOLO joint training."""

    def __init__(self, records: List[dict], image_root: Path):
        self.records = records
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def _build_semi_weak_records(spec: ExperimentSpec, max_images: Optional[int] = None) -> tuple:
    from modules.wssis.mask2former_dataloader import WssisSemiWeakDataset
    from modules.wssis.mask2former_datasets import _ensure_filtered_coco_json
    from modules.wssis.paths import load_weak_95pct_signal_map

    paths = build_coco_paths()
    cache_dir = paths["coco_root"].parent / "cache" / "stage2_yolo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    labeled_json = _ensure_filtered_coco_json(
        paths["train_ann"],
        paths["labeled_5pct_txt"],
        cache_dir / f"labeled_{spec.id}.json",
    )
    weak_json = _ensure_filtered_coco_json(
        paths["train_ann"],
        paths["weak_95pct_txt"],
        cache_dir / f"weak_{spec.id}.json",
        strip_annotations=False,
    )
    image_root = resolve_coco_image_dir(paths["coco_root"], "train")
    labeled = load_coco_json(str(labeled_json), str(image_root), f"wssis_yolo_l_{spec.id}")
    weak = load_coco_json(str(weak_json), str(image_root), f"wssis_yolo_w_{spec.id}")
    signal_map = load_weak_95pct_signal_map()
    for rec in weak:
        iid = rec.get("image_id")
        rec["wssis_weak_signal_type"] = signal_map.get(iid, "points_only")
    if max_images:
        labeled = labeled[:max_images]
        weak = weak[: max_images]
    ds = WssisSemiWeakDataset(labeled, weak)
    records = [ds[i] for i in range(len(ds))]
    return records, image_root


def _union_mask(masks: List[np.ndarray], size: int) -> np.ndarray:
    out = np.zeros((size, size), dtype=np.float32)
    for m in masks:
        if m.shape != (size, size):
            m = apply_geom_transform_to_mask(m, _identity_geom(size, m.shape))
        out = np.maximum(out, (m > 0).astype(np.float32))
    return out


def _identity_geom(out_size: int, src_shape: tuple):
    h, w = src_shape[:2]
    return GeomTransformParams(h, w, 0, 0, h, w, False, out_size)


def _center_square_geom(h: int, w: int, out_size: int) -> GeomTransformParams:
    """Deterministic center square crop for validation (no random aug)."""
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return GeomTransformParams(h, w, y0, x0, side, side, False, out_size)


def _eval_image_tensor(image_rgb: np.ndarray, geom: GeomTransformParams) -> torch.Tensor:
    from modules.wssis.training.stage2_augment import _apply_geom_to_rgb

    arr = _apply_geom_to_rgb(image_rgb, geom)
    img_t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (img_t - mean) / std


def _build_val_records(
    spec: ExperimentSpec,
    *,
    val_image_txt,
    val_ann: Path,
    image_split: str,
    max_images: Optional[int] = None,
) -> List[dict]:
    from modules.wssis.mask2former_datasets import _ensure_filtered_coco_json

    paths = build_coco_paths()
    cache_dir = paths["coco_root"].parent / "cache" / "stage2_yolo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    scope = val_image_txt.stem
    val_json = _ensure_filtered_coco_json(
        val_ann,
        val_image_txt,
        cache_dir / f"val_{scope}_{spec.id}.json",
    )
    image_root = resolve_coco_image_dir(paths["coco_root"], image_split)
    records = load_coco_json(str(val_json), str(image_root), f"wssis_yolo_v_{spec.id}_{scope}")
    if max_images:
        records = records[:max_images]
    return records


def evaluate_stage2_yolo_val(
    student: torch.nn.Module,
    mask_proj: torch.nn.Module,
    val_records: List[dict],
    *,
    imgsz: int,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Fast subset validation for in-training monitoring (parity with Mask2Former EVAL_PERIOD).

    Uses union GT masks vs predicted union mask; reports COCO-style mask AP proxy per image.
    """
    if not val_records:
        return {"segm/AP": 0.0, "segm/AP50": 0.0, "val_mask_iou": 0.0, "n_val_images": 0}

    was_training = student.training
    student.eval()
    mask_proj.eval()

    loader = DataLoader(
        Stage2YoloDataset(val_records, Path(".")),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )

    iou_sum = 0.0
    ap_sum = 0.0
    ap50_sum = 0.0
    n_images = 0

    with torch.no_grad():
        for batch_recs in loader:
            images = []
            targets = []
            for rec in batch_recs:
                file_name = rec.get("file_name")
                if not file_name:
                    continue
                image_rgb = utils.read_image(file_name, format="RGB")
                h, w = image_rgb.shape[:2]
                geom = _center_square_geom(h, w, imgsz)
                anns = rec.get("annotations") or []
                mask_np_list, _, _ = coco_anns_to_masks_for_image(
                    anns, h, w, max_objects=32
                )
                if not mask_np_list:
                    continue
                warped = [apply_geom_transform_to_mask(m, geom) for m in mask_np_list]
                images.append(_eval_image_tensor(image_rgb, geom))
                targets.append(_union_mask(warped, imgsz))

            if not images:
                continue

            img_batch = torch.stack(images, dim=0).to(device)
            pred = _yolo_mask_logits(student, mask_proj, img_batch, imgsz)
            for i, tgt_np in enumerate(targets):
                tgt_t = torch.from_numpy(tgt_np).to(device).unsqueeze(0).unsqueeze(0)
                pm = pred[i : i + 1]
                if pm.shape[-2:] != tgt_t.shape[-2:]:
                    tgt_t = F.interpolate(tgt_t, size=pm.shape[-2:], mode="nearest")
                iou = compute_iou(pm, tgt_t)
                iou_sum += iou
                ap_sum += mask_ap_from_iou(iou)
                ap50_sum += 1.0 if iou >= 0.5 else 0.0
                n_images += 1

    if was_training:
        student.train()
        mask_proj.train()

    if n_images == 0:
        return {"segm/AP": 0.0, "segm/AP50": 0.0, "val_mask_iou": 0.0, "n_val_images": 0}

    return {
        "segm/AP": ap_sum / n_images,
        "segm/AP50": ap50_sum / n_images,
        "val_mask_iou": iou_sum / n_images,
        "n_val_images": float(n_images),
    }


def train_stage2_yolo(
    spec: ExperimentSpec,
    out_dir: Path,
    *,
    epochs: int = 50,
    batch_size: int = 8,
    imgsz: int = STAGE2_STUDENT_IMAGE_SIZE,
    max_images: Optional[int] = None,
    run_ctx: Optional["RunContext"] = None,
    early_stop_patience: int = 10,
    use_full_val_final: bool = True,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("Install ultralytics for Exp 4A: pip install ultralytics") from e

    from modules.wssis.eval_splits import resolve_eval_val_split
    from modules.wssis.smoke_profile import get_smoke_profile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoke = get_smoke_profile()
    val_limit = smoke.max_images if smoke else None
    paths = build_coco_paths()

    records, image_root = _build_semi_weak_records(spec, max_images=max_images)
    val_split = resolve_eval_val_split(full_val=False)
    val_records = _build_val_records(
        spec,
        val_image_txt=val_split["val_image_txt"],
        val_ann=val_split["val_ann"],
        image_split=str(val_split["image_split"]),
        max_images=val_limit,
    )
    full_val_records: Optional[List[dict]] = None
    if use_full_val_final:
        full_val_records = _build_val_records(
            spec,
            val_image_txt=paths["val_all_txt"],
            val_ann=paths["val_ann"],
            image_split="val",
            max_images=val_limit,
        )

    dataset = Stage2YoloDataset(records, image_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )

    pseudo_cfg = {
        "pseudo_label": {
            "threshold_mode": "fixed",
            "confidence_threshold": 0.9,
        }
    }
    teacher = WssisTeacherStack(
        device,
        gnn_ckpt_path=gnn_checkpoint(spec.gnn_checkpoint),
        use_gnn=spec.use_gnn,
        freeze_gnn=spec.freeze_gnn,
        threshold_policy=build_threshold_policy(pseudo_cfg),
    )
    teacher.train()
    teacher_opt = None
    if teacher.gnn is not None and not spec.freeze_gnn:
        teacher_opt = torch.optim.AdamW(teacher.gnn.parameters(), lr=1e-5)

    yolo = YOLO("yolov8n-seg.pt")
    student = yolo.model.to(device)
    student.train()
    proto_ch = 32
    mask_proj = torch.nn.Conv2d(proto_ch, 1, kernel_size=1, bias=True).to(device)
    student_opt = torch.optim.AdamW(
        list(student.parameters()) + list(mask_proj.parameters()),
        lr=1e-4,
    )

    schedule = LossWeightSchedule()
    total_steps = max(1, epochs * len(loader))
    step = 0
    ckpt_dir = out_dir / "yolov8_seg" / "weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    early_stop = EarlyStopping(
        patience=early_stop_patience,
        monitor="segm/AP",
        mode="max",
    )
    best_segm_ap = -1.0

    if run_ctx is not None:
        run_ctx.init_tensorboard()

    print(
        f"[stage2-yolo] start: epochs={epochs} batches/epoch={len(loader)} "
        f"total_steps={total_steps} batch_size={batch_size} imgsz={imgsz} "
        f"val_images={len(val_records)} early_stop_patience={early_stop_patience}"
    )
    train_t0 = time.perf_counter()

    epoch_pbar = tqdm(range(1, epochs + 1), desc="Stage-2 YOLO", unit="epoch")
    for epoch in epoch_pbar:
        epoch_loss_sum = 0.0
        n_train_steps = 0
        epoch_t0 = time.perf_counter()

        train_pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=False,
            unit="batch",
        )
        for batch_recs in train_pbar:
            step += 1
            weights = schedule.weights(step, total_steps)
            loss_total = torch.tensor(0.0, device=device)
            t_pce, t_sym, t_fb, s_semi = [], [], [], []

            images_s = []
            processed_recs = []
            for rec in batch_recs:
                file_name = rec.get("file_name")
                if not file_name:
                    continue
                image_rgb = utils.read_image(file_name, format="RGB")
                dual = build_dual_views(image_rgb, out_size=imgsz)
                is_labeled = rec.get("wssis_is_labeled", True)

                if not is_labeled:
                    anns = rec.get("wssis_teacher_anns") or rec.get("annotations") or []
                    mask_np_list, ann_ids, cats = coco_anns_to_masks_for_image(
                        anns, image_rgb.shape[0], image_rgb.shape[1], max_objects=8
                    )
                    if not mask_np_list:
                        continue
                    warped = [apply_geom_transform_to_mask(m, dual.geom) for m in mask_np_list]
                    masks_t = torch.stack(
                        [torch.from_numpy(m.astype(np.float32)) for m in warped],
                        dim=0,
                    ).to(device)
                    weak_sig = rec.get("wssis_weak_signal_type", "points_only")
                    meta = {
                        "image_id": rec.get("image_id", 0),
                        "ann_ids": ann_ids,
                        "category_ids": cats,
                    }
                    _, _, refined, sam3, weak_signal = forward_teacher_objects_impl(
                        teacher.sam,
                        teacher.gnn if teacher.use_gnn else None,
                        dual.image_weak.to(device),
                        masks_t,
                        teacher.pixel_mean,
                        teacher.pixel_std,
                        teacher.mask_size,
                        meta,
                        signal_type=weak_sig,
                        threshold_policy=teacher.threshold_policy,
                    )
                    valid_pce, target_pce = build_pce_valid_mask(weak_signal, weak_sig)
                    t_pce.append(partial_bce_loss(refined, target_pce, valid_pce))
                    t_sym.append(symmetric_sam_triplet_loss(sam3))
                    sam_probs = refined_probs_from_logits(sam3)
                    pseudo_vote, _ = voting_pseudo_mask(sam_probs, threshold=0.9, vote_min=2)
                    pseudo_np = [
                        (pseudo_vote[j, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                        for j in range(pseudo_vote.shape[0])
                    ]
                    rec["_target_mask"] = _union_mask(pseudo_np, imgsz)
                    rec["_weak_signal"] = weak_signal
                    rec["_weak_sig_type"] = weak_sig
                    rec["_refined"] = refined
                else:
                    anns = rec.get("annotations") or []
                    mask_np_list, _, _ = coco_anns_to_masks_for_image(
                        anns, image_rgb.shape[0], image_rgb.shape[1], max_objects=8
                    )
                    if not mask_np_list:
                        continue
                    warped = [apply_geom_transform_to_mask(m, dual.geom) for m in mask_np_list]
                    rec["_target_mask"] = _union_mask(warped, imgsz)

                images_s.append(dual.image_strong)
                processed_recs.append(rec)

            if not images_s:
                continue
            img_batch = torch.stack(images_s, dim=0).to(device)
            pred = _yolo_mask_logits(student, mask_proj, img_batch, imgsz)

            for i, rec in enumerate(processed_recs):
                tgt = torch.from_numpy(rec["_target_mask"]).to(device).unsqueeze(0).unsqueeze(0)
                if pred.shape[-2:] != tgt.shape[-2:]:
                    tgt = F.interpolate(tgt, size=pred.shape[-2:], mode="nearest")
                pm = pred[i : i + 1]
                loss_total = loss_total + partial_bce_loss(pm, tgt, torch.ones_like(tgt))
                loss_total = loss_total + partial_dice_loss(pm, tgt, torch.ones_like(tgt))

                if not rec.get("wssis_is_labeled", True) and "_weak_signal" in rec:
                    valid_pce, target_pce = aggregate_weak_signal_per_image(
                        rec["_weak_signal"], rec["_weak_sig_type"]
                    )
                    vm = valid_pce.to(pm.device)
                    if vm.shape[-2:] != pm.shape[-2:]:
                        vm = F.interpolate(vm, size=pm.shape[-2:], mode="nearest")
                        target_pce = F.interpolate(target_pce, size=pm.shape[-2:], mode="nearest")
                    s_semi.append(partial_bce_loss(pm, target_pce, vm) + partial_dice_loss(pm, target_pce, vm))
                    if weights["lambda_t_feedback"] > 0 and "_refined" in rec:
                        t_fb.append(
                            student_feedback_loss(
                                aggregate_refined_logits_per_image(rec["_refined"]),
                                pm.sigmoid(),
                                tau=0.95,
                            )
                        )

            if t_pce:
                loss_total = loss_total + weights["lambda_t_pce"] * torch.stack(t_pce).mean()
            if t_sym:
                loss_total = loss_total + weights["lambda_t_sym"] * torch.stack(t_sym).mean()
            if s_semi:
                loss_total = loss_total + weights["lambda_s_semi"] * torch.stack(s_semi).mean()
            if t_fb:
                loss_total = loss_total + weights["lambda_t_feedback"] * torch.stack(t_fb).mean()

            student_opt.zero_grad()
            if teacher_opt is not None:
                teacher_opt.zero_grad()
            loss_total.backward()
            student_opt.step()
            if teacher_opt is not None:
                teacher_opt.step()

            loss_val = float(loss_total.detach().item())
            epoch_loss_sum += loss_val
            n_train_steps += 1
            train_pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                step=f"{step}/{total_steps}",
            )

        epoch_elapsed = time.perf_counter() - epoch_t0
        mean_loss = epoch_loss_sum / max(n_train_steps, 1)
        elapsed_total = time.perf_counter() - train_t0
        eta_s = (elapsed_total / epoch) * (epochs - epoch) if epoch > 0 else 0.0

        val_metrics = evaluate_stage2_yolo_val(
            student,
            mask_proj,
            val_records,
            imgsz=imgsz,
            device=device,
            batch_size=batch_size,
        )
        segm_ap = val_metrics["segm/AP"]
        epoch_pbar.set_postfix(
            loss=f"{mean_loss:.4f}",
            segm_AP=f"{segm_ap:.4f}",
            eta=f"{eta_s / 60:.1f}m",
        )
        print(
            f"[stage2-yolo] epoch {epoch}/{epochs} "
            f"mean_loss={mean_loss:.4f} segm/AP={segm_ap:.4f} "
            f"segm/AP50={val_metrics['segm/AP50']:.4f} "
            f"val_mask_iou={val_metrics['val_mask_iou']:.4f} "
            f"train_steps={n_train_steps} elapsed={epoch_elapsed:.1f}s "
            f"eta={eta_s / 60:.1f}m"
        )

        payload = {
            "model": student.state_dict(),
            "mask_proj": mask_proj.state_dict(),
            "epoch": epoch,
            "segm/AP": segm_ap,
            "best_segm_ap": max(best_segm_ap, segm_ap),
        }
        torch.save(payload, ckpt_dir / "last.pt")

        is_best = segm_ap > best_segm_ap
        if is_best:
            best_segm_ap = segm_ap
            payload["best_segm_ap"] = best_segm_ap
            torch.save(payload, ckpt_dir / "best.pt")
            print(f"[stage2-yolo] new best segm/AP={best_segm_ap:.4f} -> {ckpt_dir / 'best.pt'}")

        metrics_row = {
            "event": "epoch",
            "epoch": epoch,
            "train_loss": mean_loss,
            "train_steps": n_train_steps,
            "epoch_time_s": epoch_elapsed,
            **val_metrics,
            "val_scope": val_split["scope"],
            "best_segm_ap": best_segm_ap,
        }
        if run_ctx is not None:
            run_ctx.log_metrics(metrics_row, step=epoch)
            run_ctx.update_step(
                f"exp_{spec.id}",
                {
                    "status": "running",
                    "epoch": epoch,
                    "max_epochs": epochs,
                    "segm/AP": segm_ap,
                },
            )

        if early_stop.patience > 0 and early_stop.step(metrics_row):
            print(
                f"[stage2-yolo] EarlyStopping: new best {early_stop.monitor}={early_stop.best:.4f}"
            )
        if early_stop.should_stop:
            print(
                f"[stage2-yolo] Early stopping at epoch {epoch} "
                f"(patience={early_stop.patience}, best {early_stop.monitor}={early_stop.best:.4f})"
            )
            break

    if use_full_val_final and full_val_records:
        final_metrics = evaluate_stage2_yolo_val(
            student,
            mask_proj,
            full_val_records,
            imgsz=imgsz,
            device=device,
            batch_size=batch_size,
        )
        print(
            f"[stage2-yolo] final full-val eval: segm/AP={final_metrics['segm/AP']:.4f} "
            f"segm/AP50={final_metrics['segm/AP50']:.4f} "
            f"n_images={int(final_metrics['n_val_images'])}"
        )
        if run_ctx is not None:
            run_ctx.log_metrics(
                {
                    "event": "final_eval",
                    "val_scope": "full_val",
                    **final_metrics,
                },
                step=epoch,
            )

    meta_path = ckpt_dir / "stage2_yolo_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "epochs": epochs,
                "imgsz": imgsz,
                "spec": spec.id,
                "best_segm_ap": best_segm_ap,
                "early_stop_patience": early_stop_patience,
            }
        ),
        encoding="utf-8",
    )
    best_path = ckpt_dir / "best.pt"
    return best_path if best_path.exists() else ckpt_dir / "last.pt"
