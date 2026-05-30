"""
COCO 2017 minitrain instance dataset for Stage-1 SAM+GNN training.

Paths match data/EDA.ipynb and Kaggle dataset layout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


def load_image_ids_from_txt(txt_path: Path) -> set:
    ids = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.search(r"(\d{12})", line)
            if match:
                ids.add(int(match.group(1)))
    return ids


def filter_coco_json(data: dict, image_ids: set) -> dict:
    images = [img for img in data["images"] if img["id"] in image_ids]
    valid_ids = {img["id"] for img in images}
    annotations = [ann for ann in data["annotations"] if ann["image_id"] in valid_ids]
    return {
        "images": images,
        "annotations": annotations,
        "categories": data.get("categories", []),
    }


def ann_to_mask(ann: dict, height: int, width: int) -> np.ndarray:
    """Decode COCO annotation to binary HxW uint8 mask."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        if mask_utils is not None:
            rle = mask_utils.frPyObjects(seg, height, width)
            m = mask_utils.decode(mask_utils.merge(rle))
            if m.ndim == 3:
                m = m.max(axis=2)
            return m.astype(np.uint8)
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "pycocotools or opencv-python required for polygon masks"
            ) from e
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask
    if isinstance(seg, dict) and mask_utils:
        m = mask_utils.decode(seg)
        if m.ndim == 3:
            m = m.max(axis=2)
        return m.astype(np.uint8)
    raise ValueError("Unsupported segmentation format")


class CocoSamStage1Dataset(Dataset):
    """
    One sample = one COCO instance (image crop region implied by full image + mask).

    Returns:
        image: float tensor [3, img_size, img_size] in [0, 1]
        mask: float tensor [1, mask_size, mask_size] in {0, 1}
        meta: dict with image_id, ann_id (for debugging)
    """

    def __init__(
        self,
        coco_root: Path,
        ann_json: Path,
        image_id_txt: Path,
        split: str = "train",
        img_size: int = 1024,
        mask_size: int = 256,
        max_instances: Optional[int] = None,
    ):
        self.coco_root = Path(coco_root)
        self.img_size = img_size
        self.mask_size = mask_size
        self.split = split

        image_dir = self.coco_root / f"{split}2017"
        if not image_dir.exists():
            image_dir = self.coco_root / "images" / f"{split}2017"

        self.image_dir = image_dir

        with open(ann_json, "r", encoding="utf-8") as f:
            full_data = json.load(f)

        subset_ids = load_image_ids_from_txt(image_id_txt)
        data = filter_coco_json(full_data, subset_ids)

        self.images = {img["id"]: img for img in data["images"]}
        self.samples: List[Tuple[int, dict]] = []
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            img_id = ann["image_id"]
            if img_id not in self.images:
                continue
            self.samples.append((img_id, ann))

        if max_instances is not None and max_instances < len(self.samples):
            rng = np.random.RandomState(42)
            idx = rng.choice(len(self.samples), size=max_instances, replace=False)
            self.samples = [self.samples[i] for i in idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_id, ann = self.samples[idx]
        info = self.images[img_id]
        file_name = info["file_name"]
        path = self.image_dir / file_name
        if not path.exists():
            path = self.image_dir / Path(file_name).name

        image = Image.open(path).convert("RGB")
        w, h = image.size

        mask_np = ann_to_mask(ann, info["height"], info["width"])
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask_img = mask_img.resize((self.mask_size, self.mask_size), Image.NEAREST)

        image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(np.array(mask_img)).float().unsqueeze(0) / 255.0
        mask_t = (mask_t > 0.5).float()

        meta = {"image_id": img_id, "ann_id": ann["id"], "split": self.split}
        return image_t, mask_t, meta

    def get_raw_pair(self, idx: int):
        """
        Original-resolution RGB image and binary mask for SAM predictor / visualization.

        Returns:
            image_rgb: uint8 ndarray [H, W, 3]
            mask_np: uint8 ndarray [H, W] in {0, 1}
            meta: dict with image_id, ann_id, orig_size (h, w)
        """
        img_id, ann = self.samples[idx]
        info = self.images[img_id]
        file_name = info["file_name"]
        path = self.image_dir / file_name
        if not path.exists():
            path = self.image_dir / Path(file_name).name

        image = Image.open(path).convert("RGB")
        w, h = image.size
        mask_np = ann_to_mask(ann, info["height"], info["width"])
        image_rgb = np.array(image, dtype=np.uint8)
        meta = {
            "image_id": img_id,
            "ann_id": ann["id"],
            "split": self.split,
            "orig_size": (h, w),
        }
        return image_rgb, mask_np.astype(np.uint8), meta


def collate_stage1(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    masks = torch.stack([b[1] for b in batch], dim=0)
    meta = [b[2] for b in batch]
    return images, masks, meta


def build_coco_paths(
    kaggle: bool = True,
    coco_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """Default paths for Kaggle vs local."""
    if kaggle:
        coco_root = Path(
            "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017"
        )
        ann_dir = coco_root / "annotations"
        minitrain_root = Path(
            "/kaggle/input/datasets/banuprasadb/coco-minitrain-10k/coco_minitrain_10k"
        )
        return {
            "coco_root": coco_root,
            "train_ann": ann_dir / "instances_train2017.json",
            "val_ann": ann_dir / "instances_val2017.json",
            "train_txt": minitrain_root / "train2017.txt",
            "val_txt": minitrain_root / "val2017.txt",
        }

    root = coco_root or Path("./data/coco2017")
    mini = Path("./data/coco_minitrain_10k")
    return {
        "coco_root": root,
        "train_ann": root / "annotations" / "instances_train2017.json",
        "val_ann": root / "annotations" / "instances_val2017.json",
        "train_txt": mini / "train2017.txt",
        "val_txt": mini / "val2017.txt",
    }


def get_stage1_dataloaders(
    config: dict,
    kaggle: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    data_cfg = config.get("data", {})
    paths = build_coco_paths(kaggle=kaggle, coco_root=data_cfg.get("coco_root"))

    common = dict(
        coco_root=paths["coco_root"],
        img_size=data_cfg.get("img_size", 1024),
        mask_size=data_cfg.get("mask_size", 256),
        max_instances=data_cfg.get("max_instances"),
    )

    train_ds = CocoSamStage1Dataset(
        ann_json=paths["train_ann"],
        image_id_txt=paths["train_txt"],
        split="train",
        **common,
    )
    val_ds = CocoSamStage1Dataset(
        ann_json=paths["val_ann"],
        image_id_txt=paths["val_txt"],
        split="val",
        **common,
    )

    training_cfg = config.get("training", {})
    batch_size = training_cfg.get("batch_size", 2)
    num_workers = data_cfg.get("num_workers", 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_stage1,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_stage1,
    )
    return train_loader, val_loader
