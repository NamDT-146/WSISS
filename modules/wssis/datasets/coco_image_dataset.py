"""
Image-level COCO dataset: one sample = one image with N instances.

Training collate can flatten to per-object rows for SAM/GNN forward.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from modules.vig_refinenet.coco_sam_stage1_dataset import (
    ann_to_mask,
    filter_coco_json,
    load_image_ids_from_txt,
)
from modules.wssis.weak_prompts import WEAK_SIGNAL_TYPES


class CocoImageDataset(Dataset):
    """
    One sample = one image with all non-crowd COCO instances.

    Returns:
        image: [3, img_size, img_size]
        masks: [N, mask_size, mask_size]
        meta: {image_id, ann_ids, category_ids, split, n_objects}
    """

    def __init__(
        self,
        coco_root: Path,
        ann_json: Path,
        image_id_txt: Path,
        split: str = "train",
        img_size: int = 1024,
        mask_size: int = 256,
        max_images: Optional[int] = None,
        max_objects_per_image: Optional[int] = None,
    ):
        from modules.wssis.paths import resolve_coco_image_dir

        self.coco_root = Path(coco_root)
        self.img_size = img_size
        self.mask_size = mask_size
        self.split = split
        self.max_objects_per_image = max_objects_per_image
        self.image_dir = resolve_coco_image_dir(self.coco_root, split)

        with open(ann_json, "r", encoding="utf-8") as f:
            full_data = json.load(f)

        subset_ids = load_image_ids_from_txt(image_id_txt)
        data = filter_coco_json(full_data, subset_ids)

        self.images = {img["id"]: img for img in data["images"]}
        anns_by_image: Dict[int, list] = defaultdict(list)
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            img_id = ann["image_id"]
            if img_id in self.images:
                anns_by_image[img_id].append(ann)

        self.samples: List[Tuple[int, list]] = []
        for img_id in sorted(self.images.keys()):
            anns = anns_by_image.get(img_id, [])
            if not anns:
                continue
            self.samples.append((img_id, anns))

        if max_images is not None and max_images < len(self.samples):
            rng = np.random.RandomState(42)
            idx = rng.choice(len(self.samples), size=max_images, replace=False)
            self.samples = [self.samples[i] for i in sorted(idx)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_id, anns = self.samples[idx]
        if self.max_objects_per_image and len(anns) > self.max_objects_per_image:
            anns = anns[: self.max_objects_per_image]

        info = self.images[img_id]
        file_name = info["file_name"]
        path = self.image_dir / file_name
        if not path.exists():
            path = self.image_dir / Path(file_name).name

        image = Image.open(path).convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        mask_tensors = []
        ann_ids = []
        category_ids = []
        for ann in anns:
            mask_np = ann_to_mask(ann, info["height"], info["width"])
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_img = mask_img.resize((self.mask_size, self.mask_size), Image.NEAREST)
            m = torch.from_numpy(np.array(mask_img)).float().unsqueeze(0) / 255.0
            mask_tensors.append((m > 0.5).float())
            ann_ids.append(int(ann["id"]))
            category_ids.append(int(ann.get("category_id", 0)))

        if not mask_tensors:
            mask_tensors = [torch.zeros(1, self.mask_size, self.mask_size)]
            ann_ids = [0]
            category_ids = [0]

        masks = torch.cat(mask_tensors, dim=0)

        meta = {
            "image_id": img_id,
            "ann_ids": ann_ids,
            "category_ids": category_ids,
            "split": self.split,
            "n_objects": len(ann_ids),
        }
        return image_t, masks, meta

    def get_raw_image(self, idx: int):
        """Full-resolution RGB + list of instance masks for visualization."""
        img_id, anns = self.samples[idx]
        if self.max_objects_per_image and len(anns) > self.max_objects_per_image:
            anns = anns[: self.max_objects_per_image]
        info = self.images[img_id]
        path = self.image_dir / info["file_name"]
        if not path.exists():
            path = self.image_dir / Path(info["file_name"]).name
        image = Image.open(path).convert("RGB")
        w, h = image.size
        masks = [
            ann_to_mask(ann, info["height"], info["width"]).astype(np.uint8)
            for ann in anns
        ]
        meta = {
            "image_id": img_id,
            "ann_ids": [int(a["id"]) for a in anns],
            "split": self.split,
            "orig_size": (h, w),
        }
        return np.array(image, dtype=np.uint8), masks, meta


def collate_image_stage1(batch):
    """Batch of images; masks variable N per image (list)."""
    images = torch.stack([b[0] for b in batch], dim=0)
    masks = [b[1] for b in batch]
    meta = [b[2] for b in batch]
    return images, masks, meta


def collate_image_to_instances(batch):
    """
    Flatten image batch to per-object rows (reuses instance-level forward).

    Returns:
        images [B*O, 3, H, W], masks [B*O, 1, M, M], meta list
    """
    flat_images, flat_masks, flat_meta = [], [], []
    for img_t, mask_stack, meta in batch:
        n = mask_stack.shape[0]
        for j in range(n):
            flat_images.append(img_t)
            m = mask_stack[j]
            if m.dim() == 2:
                m = m.unsqueeze(0)
            flat_masks.append(m)
            flat_meta.append(
                {
                    "image_id": meta["image_id"],
                    "ann_id": meta["ann_ids"][j],
                    "category_id": meta["category_ids"][j],
                    "split": meta["split"],
                    "obj_idx": j,
                    "n_objects": meta["n_objects"],
                }
            )
    if not flat_images:
        raise ValueError("Empty batch in collate_image_to_instances")
    images = torch.stack(flat_images, dim=0)
    masks = torch.stack(flat_masks, dim=0)
    return images, masks, flat_meta


def collate_instance_triplets(batch):
    """
    Flatten to per-object rows, then expand each instance to 3 weak-signal types.

    Returns same tensor shapes as collate_image_to_instances but B' = 3 * num_instances.
    """
    images, masks, flat_meta = collate_image_to_instances(batch)
    tri_images, tri_masks, tri_meta = [], [], []
    for i in range(images.shape[0]):
        for sig in WEAK_SIGNAL_TYPES:
            tri_images.append(images[i])
            tri_masks.append(masks[i])
            m = dict(flat_meta[i])
            m["weak_signal_type"] = sig
            tri_meta.append(m)
    return (
        torch.stack(tri_images, dim=0),
        torch.stack(tri_masks, dim=0),
        tri_meta,
    )
