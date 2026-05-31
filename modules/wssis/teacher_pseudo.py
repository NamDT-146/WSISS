"""Resize helpers for teacher pseudo-labels vs student training resolution."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch

from modules.wssis.stage2_constants import SAM_TEACHER_IMAGE_SIZE


def resize_binary_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize for uint8/float binary masks."""
    mh, mw = mask.shape[:2]
    if mh == height and mw == width:
        return (mask > 0).astype(np.uint8)
    resized = cv2.resize(
        (mask > 0).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.uint8)


def resize_binary_masks(
    masks: Sequence[np.ndarray],
    height: int,
    width: int,
) -> List[np.ndarray]:
    return [resize_binary_mask(m, height, width) for m in masks]


def prepare_sam_teacher_inputs(
    image_rgb: np.ndarray,
    mask_np_list: Sequence[np.ndarray],
    *,
    sam_size: int = SAM_TEACHER_IMAGE_SIZE,
) -> Tuple[torch.Tensor, List[np.ndarray], Tuple[int, int]]:
    """
    Resize image + oracle masks to SAM input size for cache-aligned teacher forward.

    Returns:
        img_t: [3, sam_size, sam_size] float in [0, 1]
        masks_sam: masks at sam_size (for prompt generation)
        native_hw: original (H, W) for mapping pseudo labels back to COCO coords
    """
    native_h, native_w = image_rgb.shape[:2]
    if native_h == sam_size and native_w == sam_size:
        img_sam = image_rgb
        masks_sam = list(mask_np_list)
    else:
        img_sam = cv2.resize(image_rgb, (sam_size, sam_size), interpolation=cv2.INTER_LINEAR)
        masks_sam = resize_binary_masks(mask_np_list, sam_size, sam_size)

    img_t = torch.from_numpy(np.ascontiguousarray(img_sam)).permute(2, 0, 1).float() / 255.0
    return img_t, masks_sam, (native_h, native_w)


def map_teacher_pseudo_to_size(
    pseudo_masks: Sequence[np.ndarray],
    *,
    native_hw: Tuple[int, int],
    target_size: int | None = None,
) -> List[np.ndarray]:
    """
    Map teacher pseudo masks (typically 256×256) to annotation space.

    When ``target_size`` is set (student 512), resize native-space masks to a square
    student canvas; otherwise keep full native (H, W) for Detectron2 mappers.
    """
    native_h, native_w = native_hw
    in_native = resize_binary_masks(pseudo_masks, native_h, native_w)
    if target_size is None:
        return in_native
    return resize_binary_masks(in_native, target_size, target_size)
