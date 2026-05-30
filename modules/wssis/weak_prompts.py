"""
Fixed and online weak prompt generation (report/PREPARATION.md).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def mask_centroid_point(mask: np.ndarray) -> Tuple[float, float]:
    """Geometric centroid of binary mask; falls back to center of mass."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def mask_interior_point(mask: np.ndarray) -> Tuple[float, float]:
    """Deepest interior point via distance transform (validation policy)."""
    if cv2 is None:
        return mask_centroid_point(mask)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(dist.argmax(), dist.shape)
    return float(x), float(y)


def mask_to_bbox_xywh(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return [0.0, 0.0, float(w), float(h)]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def build_instance_prompts(
    mask: np.ndarray,
    policy: str = "val_fixed",
    signal_type: str = "mixed",
    rng: Optional[np.random.RandomState] = None,
) -> Dict:
    """
    Build prompt dict for one instance.

    policy: 'val_fixed' | 'train_online'
    signal_type: 'mixed' | 'boxes_only' | 'points_only'
    """
    rng = rng or np.random.RandomState()
    h, w = mask.shape

    if policy == "val_fixed":
        px, py = mask_interior_point(mask)
        point = [px, py]
        bbox = mask_to_bbox_xywh(mask)
        return {
            "point": point,
            "bbox": bbox,
            "signal_type": signal_type if signal_type != "mixed" else "point",
        }

    # train_online — mixed / boxes / points
    bbox = mask_to_bbox_xywh(mask)
    if signal_type == "boxes_only":
        jitter = rng.randint(-5, 6, size=4)
        b = [bbox[0] + jitter[0], bbox[1] + jitter[1], bbox[2], bbox[3]]
        return {"bbox": b, "signal_type": "box"}

    if signal_type == "points_only":
        ys, xs = np.where(mask > 0)
        idx = rng.randint(0, len(xs))
        return {"point": [float(xs[idx]), float(ys[idx])], "signal_type": "point"}

    # mixed: random point with jitter
    px, py = mask_centroid_point(mask)
    px += rng.uniform(-5, 5)
    py += rng.uniform(-5, 5)
    px = float(np.clip(px, 0, w - 1))
    py = float(np.clip(py, 0, h - 1))
    return {
        "point": [px, py],
        "bbox": bbox,
        "signal_type": "mixed",
    }
