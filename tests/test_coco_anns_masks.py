"""Tests for COCO annotation mask extraction."""

import numpy as np

from modules.wssis.mask2former_datasets import coco_anns_to_masks_for_image


def test_coco_anns_without_id_field():
    """Detectron2-style anns omit COCO id; helper must not KeyError."""
    h, w = 32, 32
    ann = {
        "bbox": [2.0, 2.0, 8.0, 8.0],
        "category_id": 1,
        "iscrowd": 0,
        "segmentation": [[2, 2, 10, 2, 10, 10, 2, 10]],
    }
    masks, ann_ids, cats = coco_anns_to_masks_for_image([ann], h, w)
    assert len(masks) == 1
    assert ann_ids == [0]
    assert cats == [1]
    assert masks[0].shape == (h, w)
    assert masks[0].sum() > 0
