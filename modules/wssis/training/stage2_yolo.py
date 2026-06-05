"""
Stage-2 joint teacher-student training for YOLOv8-seg (Exp 4A).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

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
from modules.wssis.stage2_constants import STAGE2_STUDENT_IMAGE_SIZE
from modules.wssis.training.stage2_augment import apply_geom_transform_to_mask, build_dual_views
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
from modules.vig_refinenet.sam_stage1_common import forward_teacher_objects_impl


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
    from modules.wssis.training.stage2_augment import GeomTransformParams

    h, w = src_shape[:2]
    return GeomTransformParams(h, w, 0, 0, h, w, False, out_size)


def train_stage2_yolo(
    spec: ExperimentSpec,
    out_dir: Path,
    *,
    epochs: int = 50,
    batch_size: int = 8,
    imgsz: int = STAGE2_STUDENT_IMAGE_SIZE,
    max_images: Optional[int] = None,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("Install ultralytics for Exp 4A: pip install ultralytics") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, image_root = _build_semi_weak_records(spec, max_images=max_images)
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

    for epoch in range(1, epochs + 1):
        for batch_recs in loader:
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

        torch.save(
            {
                "model": student.state_dict(),
                "mask_proj": mask_proj.state_dict(),
                "epoch": epoch,
            },
            ckpt_dir / "last.pt",
        )

    meta_path = ckpt_dir / "stage2_yolo_meta.json"
    meta_path.write_text(json.dumps({"epochs": epochs, "imgsz": imgsz, "spec": spec.id}), encoding="utf-8")
    return ckpt_dir / "last.pt"
