"""
Stage-2 dual-view augmentation: weak (geom) for teacher, strong (+ RandAugment no-color) for student.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

import torch
import torch.nn.functional as F

from modules.wssis.stage2_constants import STAGE2_STUDENT_IMAGE_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class GeomTransformParams:
    """Record flip + crop for mask warping."""

    src_h: int
    src_w: int
    crop_y0: int
    crop_x0: int
    crop_h: int
    crop_w: int
    hflip: bool
    out_size: int


@dataclass
class DualViewResult:
    image_weak: torch.Tensor
    image_strong: torch.Tensor
    geom: GeomTransformParams
    student_mean: Tuple[float, ...] = IMAGENET_MEAN
    student_std: Tuple[float, ...] = IMAGENET_STD


def _rand_crop_box(h: int, w: int, out_size: int, scale: Tuple[float, float] = (0.5, 1.0)) -> Tuple[int, int, int, int]:
    """Return y0, x0, ch, cw."""
    target = out_size
    min_side = min(h, w)
    crop_side = int(round(min_side * random.uniform(scale[0], scale[1])))
    crop_side = max(target, min(crop_side, min_side))
    y0 = random.randint(0, max(0, h - crop_side))
    x0 = random.randint(0, max(0, w - crop_side))
    return y0, x0, crop_side, crop_side


def apply_geom_transform_to_mask(
    mask: np.ndarray,
    geom: GeomTransformParams,
) -> np.ndarray:
    """Warp binary/float mask with same geom as image."""
    m = mask.astype(np.float32)
    y0, x0, ch, cw = geom.crop_y0, geom.crop_x0, geom.crop_h, geom.crop_w
    m = m[y0 : y0 + ch, x0 : x0 + cw]
    if geom.hflip:
        m = np.flip(m, axis=1).copy()
    if cv2 is not None:
        m = cv2.resize(m, (geom.out_size, geom.out_size), interpolation=cv2.INTER_NEAREST)
    else:
        t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(geom.out_size, geom.out_size), mode="nearest")
        m = t[0, 0].numpy()
    return m


def _apply_geom_to_rgb(image: np.ndarray, geom: GeomTransformParams) -> np.ndarray:
    img = image.astype(np.float32)
    y0, x0, ch, cw = geom.crop_y0, geom.crop_x0, geom.crop_h, geom.crop_w
    img = img[y0 : y0 + ch, x0 : x0 + cw]
    if geom.hflip:
        img = np.flip(img, axis=1).copy()
    if cv2 is not None:
        img = cv2.resize(img, (geom.out_size, geom.out_size), interpolation=cv2.INTER_LINEAR)
    else:
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(geom.out_size, geom.out_size), mode="bilinear", align_corners=False)
        img = t[0].permute(1, 2, 0).numpy()
    return img


def _randaugment_no_color(img: np.ndarray, n_ops: int = 3, magnitude: int = 7) -> np.ndarray:
    """Geom-only RandAugment-style ops (no color jitter)."""
    if cv2 is None:
        return img
    out = img.copy()
    ops = ["rotate", "shear_x", "shear_y", "translate_x", "translate_y"]
    mag = magnitude / 10.0
    for _ in range(n_ops):
        op = random.choice(ops)
        if op == "rotate":
            angle = random.uniform(-15 * mag, 15 * mag)
            h, w = out.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        elif op == "shear_x":
            sh = random.uniform(-0.1 * mag, 0.1 * mag)
            h, w = out.shape[:2]
            m = np.array([[1, sh, 0], [0, 1, 0]], dtype=np.float32)
            out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        elif op == "shear_y":
            sh = random.uniform(-0.1 * mag, 0.1 * mag)
            h, w = out.shape[:2]
            m = np.array([[1, 0, 0], [sh, 1, 0]], dtype=np.float32)
            out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        elif op == "translate_x":
            tx = int(random.uniform(-0.1 * mag, 0.1 * mag) * out.shape[1])
            h, w = out.shape[:2]
            m = np.array([[1, 0, tx], [0, 1, 0]], dtype=np.float32)
            out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        elif op == "translate_y":
            ty = int(random.uniform(-0.1 * mag, 0.1 * mag) * out.shape[0])
            h, w = out.shape[:2]
            m = np.array([[1, 0, 0], [0, 1, ty]], dtype=np.float32)
            out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return out


def build_weak_geom(image_rgb: np.ndarray, out_size: int = STAGE2_STUDENT_IMAGE_SIZE) -> Tuple[np.ndarray, GeomTransformParams]:
    h, w = image_rgb.shape[:2]
    y0, x0, ch, cw = _rand_crop_box(h, w, out_size)
    hflip = random.random() < 0.5
    geom = GeomTransformParams(
        src_h=h,
        src_w=w,
        crop_y0=y0,
        crop_x0=x0,
        crop_h=ch,
        crop_w=cw,
        hflip=hflip,
        out_size=out_size,
    )
    weak = _apply_geom_to_rgb(image_rgb, geom)
    return weak, geom


def build_dual_views(
    image_rgb: np.ndarray,
    out_size: int = STAGE2_STUDENT_IMAGE_SIZE,
    strong_aug: bool = True,
) -> DualViewResult:
    """Weak geom view for teacher; strong = weak + RandAugment (no color) + normalize."""
    weak_np, geom = build_weak_geom(image_rgb, out_size)
    strong_np = _randaugment_no_color(weak_np, n_ops=3, magnitude=7) if strong_aug else weak_np.copy()

    weak_t = torch.from_numpy(np.ascontiguousarray(weak_np)).permute(2, 0, 1).float() / 255.0
    strong_t = torch.from_numpy(np.ascontiguousarray(strong_np)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    strong_norm = (strong_t - mean) / std

    return DualViewResult(
        image_weak=weak_t,
        image_strong=strong_norm,
        geom=geom,
    )


def warp_mask_list(masks: List[np.ndarray], geom: GeomTransformParams) -> List[np.ndarray]:
    return [apply_geom_transform_to_mask(m, geom) for m in masks]
