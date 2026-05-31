"""Export labeled + pseudo labels for YOLO semi-weak training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from modules.wssis.experiments.registry import ExperimentSpec
from modules.wssis.mask2former_datasets import (
    _coco_thing_classes,
    coco_anns_to_masks_for_image,
)
from modules.wssis.paths import build_coco_paths, resolve_coco_image_dir
from modules.wssis.smoke_profile import get_smoke_profile
from modules.wssis.stage2_constants import STAGE2_STUDENT_IMAGE_SIZE
from modules.wssis.teacher_pseudo import map_teacher_pseudo_to_size, prepare_sam_teacher_inputs
from modules.vig_refinenet.coco_sam_stage1_dataset import ann_to_mask, filter_coco_json, load_image_ids_from_txt


def _coco_category_id_to_yolo_index() -> dict[int, int]:
    """Map COCO category_id (non-contiguous) to YOLO class index 0..79."""
    return {
        int(c["id"]): i
        for i, c in enumerate(COCO_CATEGORIES)
        if int(c.get("isthing", 1)) == 1
    }


def _mask_to_yolo_seg_line(cls_idx: int, mask: np.ndarray) -> Optional[str]:
    """Binary mask -> YOLOv8-seg label row: ``cls x1 y1 x2 y2 ...`` (normalized)."""
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return None
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1.0:
        return None
    eps = 0.002 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) < 3:
        return None
    pts = approx.reshape(-1, 2).astype(np.float64)
    xs = np.clip(pts[:, 0] / w, 0.0, 1.0)
    ys = np.clip(pts[:, 1] / h, 0.0, 1.0)
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
    return f"{cls_idx} {coords}"


def _resize_mask_to_student(mask: np.ndarray, student_size: int) -> np.ndarray:
    if mask.shape[0] == student_size and mask.shape[1] == student_size:
        return (mask > 0).astype(np.uint8)
    return cv2.resize(
        (mask > 0).astype(np.uint8),
        (student_size, student_size),
        interpolation=cv2.INTER_NEAREST,
    )


def _coco_ann_to_yolo_seg_line(
    ann: dict,
    img_h: int,
    img_w: int,
    student_size: int,
    cat_to_yolo: dict[int, int],
) -> Optional[str]:
    cls_idx = cat_to_yolo.get(int(ann["category_id"]))
    if cls_idx is None or ann.get("iscrowd"):
        return None
    mask = ann_to_mask(ann, img_h, img_w)
    mask_s = _resize_mask_to_student(mask, student_size)
    return _mask_to_yolo_seg_line(cls_idx, mask_s)


def prepare_yolo_semi_weak_dataset(
    export_dir: Path,
    spec: ExperimentSpec,
    max_images: int | None = None,
) -> Path:
    """
    Build YOLO directory with labeled GT + weak pseudo masks.

    Smoke mode exports only ``max_images`` per split.
    """
    paths = build_coco_paths()
    export_dir.mkdir(parents=True, exist_ok=True)
    images_train = export_dir / "images" / "train"
    labels_train = export_dir / "labels" / "train"
    images_val = export_dir / "images" / "val"
    labels_val = export_dir / "labels" / "val"
    for d in (images_train, labels_train, images_val, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    smoke = get_smoke_profile()
    if max_images is None and smoke:
        max_images = smoke.max_images

    with open(paths["train_ann"], encoding="utf-8") as f:
        coco = json.load(f)

    cat_to_yolo = _coco_category_id_to_yolo_index()

    def export_split(
        txt_path: Path,
        *,
        use_pseudo: bool,
        coco_dict: dict,
        img_root: Path,
        images_out: Path,
        labels_out: Path,
        split_tag: str,
    ) -> int:
        ids = sorted(load_image_ids_from_txt(txt_path))
        if max_images:
            ids = ids[:max_images]
        data = filter_coco_json(coco_dict, set(ids))
        images = {img["id"]: img for img in data["images"]}
        anns_by_img: dict = {}
        for ann in data["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        teacher = None
        device = None
        if use_pseudo and spec.use_gnn:
            import torch
            from modules.wssis.mask2former_teacher import WssisTeacherStack

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(
                "[yolo_export] Loading teacher (SAM + GNN) once for %s (%d images)...",
                txt_path.name,
                len(ids),
            )
            teacher = WssisTeacherStack(
                device,
                use_gnn=spec.use_gnn,
                freeze_gnn=True,
            )

        count = 0
        total = len(ids)
        for idx, img_id in enumerate(ids):
            info = images.get(img_id)
            if not info:
                continue
            src = img_root / info["file_name"]
            if not src.exists():
                src = img_root / Path(info["file_name"]).name
            stem = f"{img_id:012d}"
            dst_img = images_out / f"{stem}.jpg"
            pil_img = Image.open(src).convert("RGB")
            student_size = STAGE2_STUDENT_IMAGE_SIZE
            pil_student = pil_img.resize((student_size, student_size), Image.BILINEAR)
            pil_student.save(dst_img)

            label_path = labels_out / f"{stem}.txt"
            lines = []
            anns = anns_by_img.get(img_id, [])
            h, w = info["height"], info["width"]
            pseudo_mode = use_pseudo
            if pseudo_mode and teacher is not None and device is not None:
                try:
                    img_np = np.array(pil_img)
                    mask_np_list, _, _ = coco_anns_to_masks_for_image(anns, h, w, max_objects=3)
                    img_t, masks_sam, native_hw = prepare_sam_teacher_inputs(img_np, mask_np_list)
                    img_t = img_t.to(device)
                    meta = {
                        "image_id": img_id,
                        "ann_ids": [a["id"] for a in anns[: len(mask_np_list)]],
                        "split": split_tag,
                    }
                    pseudo, cats = teacher.generate_pseudo_for_image(
                        img_t, masks_sam, meta, prompt_policy="train_online"
                    )
                    pseudo = map_teacher_pseudo_to_size(
                        pseudo,
                        native_hw=native_hw,
                        target_size=student_size,
                    )
                    for pm, cat in zip(pseudo, cats):
                        line = _mask_to_yolo_seg_line(
                            cat_to_yolo.get(int(cat), 0),
                            pm,
                        )
                        if line is not None:
                            lines.append(line)
                except Exception as exc:
                    logger.warning(
                        "[yolo_export] pseudo failed image_id=%s: %s",
                        img_id,
                        exc,
                    )
                    pseudo_mode = False

            if not use_pseudo:
                for ann in anns:
                    line = _coco_ann_to_yolo_seg_line(
                        ann, h, w, student_size, cat_to_yolo
                    )
                    if line is not None:
                        lines.append(line)

            label_path.write_text("\n".join(lines), encoding="utf-8")
            count += 1
            if idx == 0 or (idx + 1) % 100 == 0 or idx + 1 == total:
                logger.info(
                    "[yolo_export] %s: %d/%d images exported",
                    txt_path.stem,
                    idx + 1,
                    total,
                )

        if teacher is not None:
            del teacher
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return count

    train_root = resolve_coco_image_dir(paths["coco_root"], "train")
    val_root = resolve_coco_image_dir(paths["coco_root"], "val")

    logger.info("[yolo_export] Exporting labeled split (GT segment polygons)...")
    n_l = export_split(
        paths["labeled_5pct_txt"],
        use_pseudo=False,
        coco_dict=coco,
        img_root=train_root,
        images_out=images_train,
        labels_out=labels_train,
        split_tag="train",
    )
    logger.info("[yolo_export] Exporting weak split (teacher pseudo-labels)...")
    n_w = export_split(
        paths["weak_95pct_txt"],
        use_pseudo=True,
        coco_dict=coco,
        img_root=train_root,
        images_out=images_train,
        labels_out=labels_train,
        split_tag="train",
    )

    with open(paths["val_ann"], encoding="utf-8") as f:
        val_coco = json.load(f)
    logger.info("[yolo_export] Exporting val split (GT segment polygons)...")
    n_v = export_split(
        paths["val_sample_20pct_txt"],
        use_pseudo=False,
        coco_dict=val_coco,
        img_root=val_root,
        images_out=images_val,
        labels_out=labels_val,
        split_tag="val",
    )

    for labels_dir in (labels_train, labels_val):
        for cache in labels_dir.glob("*.cache"):
            cache.unlink(missing_ok=True)

    class_names = _coco_thing_classes()
    if len(class_names) != 80:
        raise ValueError(f"Expected 80 COCO thing classes, got {len(class_names)}")
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))

    data_yaml = export_dir / "data.yaml"
    data_yaml.write_text(
        f"""path: {export_dir.as_posix()}
train: images/train
val: images/val
nc: 80
names:
{names_block}
# exported labeled={n_l} weak={n_w} val={n_v}
""",
        encoding="utf-8",
    )
    return data_yaml
