"""Export labeled + pseudo labels for YOLO semi-weak training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from modules.wssis.experiments.registry import ExperimentSpec
from modules.wssis.mask2former_datasets import (
    _ensure_filtered_coco_json,
    coco_anns_to_masks_for_image,
)
from modules.wssis.paths import build_coco_paths, resolve_coco_image_dir
from modules.wssis.smoke_profile import get_smoke_profile
from modules.vig_refinenet.coco_sam_stage1_dataset import filter_coco_json, load_image_ids_from_txt


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
    images_out = export_dir / "images" / "train"
    labels_out = export_dir / "labels" / "train"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    smoke = get_smoke_profile()
    if max_images is None and smoke:
        max_images = smoke.max_images

    with open(paths["train_ann"], encoding="utf-8") as f:
        coco = json.load(f)

    def export_split(txt_path: Path, use_pseudo: bool) -> int:
        ids = sorted(load_image_ids_from_txt(txt_path))
        if max_images:
            ids = ids[:max_images]
        data = filter_coco_json(coco, set(ids))
        images = {img["id"]: img for img in data["images"]}
        anns_by_img: dict = {}
        for ann in data["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        img_root = resolve_coco_image_dir(paths["coco_root"], "train")
        count = 0
        for img_id in ids:
            info = images.get(img_id)
            if not info:
                continue
            src = img_root / info["file_name"]
            if not src.exists():
                src = img_root / Path(info["file_name"]).name
            stem = f"{img_id:012d}"
            dst_img = images_out / f"{stem}.jpg"
            if not dst_img.exists():
                Image.open(src).convert("RGB").save(dst_img)

            label_path = labels_out / f"{stem}.txt"
            lines = []
            anns = anns_by_img.get(img_id, [])
            h, w = info["height"], info["width"]
            pseudo_mode = use_pseudo
            if pseudo_mode and spec.use_gnn:
                try:
                    import torch
                    from modules.wssis.mask2former_teacher import WssisTeacherStack

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    teacher = WssisTeacherStack(
                        device,
                        use_gnn=spec.use_gnn,
                        freeze_gnn=True,
                    )
                    img = Image.open(dst_img).convert("RGB").resize((1024, 1024))
                    img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    mask_np_list, _, _ = coco_anns_to_masks_for_image(anns, h, w, max_objects=3)
                    meta = {
                        "image_id": img_id,
                        "ann_ids": [a["id"] for a in anns[: len(mask_np_list)]],
                        "split": "train",
                    }
                    pseudo, cats = teacher.generate_pseudo_for_image(
                        img_t, mask_np_list, meta, prompt_policy="train_online"
                    )
                    for pm, cat in zip(pseudo, cats):
                        ys, xs = np.where(pm > 0)
                        if len(xs) == 0:
                            continue
                        cx = ((xs.min() + xs.max()) / 2) / w
                        cy = ((ys.min() + ys.max()) / 2) / h
                        bw = (xs.max() - xs.min() + 1) / w
                        bh = (ys.max() - ys.min() + 1) / h
                        lines.append(f"{max(0, cat)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                except Exception:
                    pseudo_mode = False

            if not pseudo_mode or not lines:
                for ann in anns:
                    if ann.get("iscrowd"):
                        continue
                    x, y, bw, bh = ann["bbox"]
                    cx = (x + bw / 2) / w
                    cy = (y + bh / 2) / h
                    lines.append(
                        f"{ann['category_id']} {cx:.6f} {cy:.6f} {bw/w:.6f} {bh/h:.6f}"
                    )

            label_path.write_text("\n".join(lines), encoding="utf-8")
            count += 1
        return count

    n_l = export_split(paths["labeled_5pct_txt"], use_pseudo=False)
    n_w = export_split(paths["weak_95pct_txt"], use_pseudo=True)

    data_yaml = export_dir / "data.yaml"
    data_yaml.write_text(
        f"""path: {export_dir.as_posix()}
train: images/train
val: {paths['val_all_txt'].as_posix()}
nc: 80
names: []  # COCO 80 classes
# exported labeled={n_l} weak={n_w}
""",
        encoding="utf-8",
    )
    return data_yaml
