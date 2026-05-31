"""Detectron2 mapper: inject teacher pseudo-labels for weak semi-supervised images."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from mask2former import COCOInstanceNewBaselineDatasetMapper

from modules.wssis.mask2former_datasets import coco_anns_to_masks_for_image
from modules.wssis.teacher_pseudo import map_teacher_pseudo_to_size, prepare_sam_teacher_inputs


class WssisSemiWeakMapper(COCOInstanceNewBaselineDatasetMapper):
    """For weak images, replace empty annotations with GNN pseudo instances."""

    def __init__(self, cfg, is_train: bool = True, teacher=None):
        super().__init__(cfg, is_train)
        self._teacher = teacher
        self._use_semi = getattr(cfg.WSSIS, "USE_SEMI_WEAK", False)
        self._weak_signal = getattr(cfg.WSSIS, "WEAK_SIGNAL", "mixed")

    def __call__(self, dataset_dict):
        d = dict(dataset_dict)
        is_labeled = d.get("wssis_is_labeled", True)
        image_id = d.get("image_id", 0)
        split = "train"
        if (
            self._use_semi
            and self._teacher is not None
            and not d.get("wssis_is_labeled", True)
        ):
            d["annotations"] = self._pseudo_annotations(d)
        d.pop("wssis_teacher_anns", None)
        d.pop("wssis_is_labeled", None)
        out = super().__call__(d)
        if out is not None and self._use_semi:
            out["wssis_is_labeled"] = is_labeled
            out["image_id"] = image_id
            out["split"] = split
        return out

    def _pseudo_annotations(self, dataset_dict: dict) -> list:
        anns = dataset_dict.get("wssis_teacher_anns") or []
        if not anns:
            return []

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        h, w = image.shape[:2]
        mask_np_list, ann_ids, cats = coco_anns_to_masks_for_image(
            anns, h, w, max_objects=8
        )
        if not mask_np_list:
            return []

        img_t, masks_sam, native_hw = prepare_sam_teacher_inputs(image, mask_np_list)
        if self._teacher.sam is not None:
            img_t = img_t.to(next(self._teacher.sam.parameters()).device)

        meta = {
            "image_id": dataset_dict.get("image_id", 0),
            "ann_ids": ann_ids,
            "category_ids": cats,
            "split": "train",
        }
        pseudo_masks, cat_ids = self._teacher.generate_pseudo_for_image(
            img_t,
            masks_sam,
            meta,
            prompt_policy="train_online",
            signal_type=self._weak_signal if self._weak_signal != "none" else "mixed",
        )
        pseudo_masks = map_teacher_pseudo_to_size(
            pseudo_masks,
            native_hw=native_hw,
            target_size=None,
        )

        out = []
        for pm, cat, ann_id in zip(pseudo_masks, cat_ids, ann_ids):
            ys, xs = np.where(pm > 0)
            if len(xs) == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            out.append(
                {
                    "id": ann_id,
                    "image_id": dataset_dict.get("image_id", 0),
                    "category_id": int(cat) if cat else 1,
                    "bbox": [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)],
                    "area": float(pm.sum()),
                    "iscrowd": 0,
                    "segmentation": self._mask_to_seg(pm),
                }
            )
        return out

    @staticmethod
    def _mask_to_seg(pm: np.ndarray):
        try:
            from pycocotools import mask as mask_util

            rle = mask_util.encode(np.asfortranarray(pm.astype(np.uint8)))
            if isinstance(rle, list):
                return rle
            rle = dict(rle)
            rle["counts"] = rle["counts"].decode("ascii")
            return rle
        except Exception:
            return pm.astype(np.uint8).tolist()
