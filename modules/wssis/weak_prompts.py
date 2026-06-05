"""
Fixed and online weak prompt generation (report/PREPARATION.md).

Weak signals rasterize to 1×H×W maps at image resolution:
  - point: Gaussian blob centered on click
  - box: uniform fill (unit weight) inside bbox
  - scribble: polyline inside mask, widened with Gaussian filter
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

SignalType = Literal["mixed", "boxes_only", "points_only", "scribbles_only"]
WEAK_SIGNAL_TYPES: Tuple[str, ...] = ("boxes_only", "points_only", "scribbles_only")

# Default Gaussian sigma (pixels) for point / scribble widening
DEFAULT_GAUSSIAN_SIGMA = 4.0
DEFAULT_POINT_JITTER_PX = 5
DEFAULT_BOX_EXPAND_RATIO = 0.05
DEFAULT_SCRIBBLE_TRIM_RATIO = 0.15


def jitter_box_expand(
    bbox_xywh: List[float],
    expand_ratio: float = DEFAULT_BOX_EXPAND_RATIO,
    img_hw: Optional[Tuple[int, int]] = None,
) -> List[float]:
    """Expand box only (never shrink); outside remains strict background."""
    x, y, bw, bh = bbox_xywh
    dx = bw * expand_ratio
    dy = bh * expand_ratio
    x0 = x - dx
    y0 = y - dy
    bw2 = bw + 2 * dx
    bh2 = bh + 2 * dy
    if img_hw is not None:
        h, w = img_hw
        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        bw2 = min(bw2, w - x0)
        bh2 = min(bh2, h - y0)
    return [float(x0), float(y0), float(bw2), float(bh2)]


def jitter_point_manhattan(
    point: Tuple[float, float],
    radius: int = DEFAULT_POINT_JITTER_PX,
    rng: Optional[np.random.RandomState] = None,
    img_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    rng = rng or np.random.RandomState()
    px, py = point
    px += float(rng.randint(-radius, radius + 1))
    py += float(rng.randint(-radius, radius + 1))
    if img_hw is not None:
        h, w = img_hw
        px = float(np.clip(px, 0, w - 1))
        py = float(np.clip(py, 0, h - 1))
    return px, py


def jitter_scribble_trim(
    scribble_mask: np.ndarray,
    trim_ratio: float = DEFAULT_SCRIBBLE_TRIM_RATIO,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Trim head or tail segment (10–20% default) from scribble polyline mask."""
    rng = rng or np.random.RandomState()
    ys, xs = np.where(scribble_mask > 0)
    if len(xs) < 2:
        return scribble_mask
    coords = np.stack([ys, xs], axis=1)
    # order along principal axis
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()
    proj = (coords[:, 1] - cx) * 1.0 + (coords[:, 0] - cy) * 0.0
    order = np.argsort(proj)
    n = len(order)
    cut = max(1, int(round(n * trim_ratio)))
    if rng.rand() < 0.5:
        keep = order[cut:]
    else:
        keep = order[:-cut]
    out = np.zeros_like(scribble_mask)
    for idx in keep:
        out[coords[idx, 0], coords[idx, 1]] = 1
    return out


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


def generate_scribble_mask(
    mask: np.ndarray,
    policy: str = "val_fixed",
    rng: Optional[np.random.RandomState] = None,
    length_ratio: float = 0.7,
    ann_id: Optional[int] = None,
) -> np.ndarray:
    """
    Synthetic scribble polyline inside ``mask`` along the principal axis.

    val_fixed: ``length_ratio`` of longest bbox side (default 70%).
    train_online: random ratio in [0.3, 0.8].
    """
    rng = rng or np.random.RandomState()
    h, w = mask.shape
    out = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return out

    if policy == "val_fixed" and ann_id is not None:
        rng = np.random.RandomState(int(ann_id) % (2**31 - 1))

    cx, cy = xs.mean(), ys.mean()
    coords = np.stack([xs.astype(np.float64) - cx, ys.astype(np.float64) - cy])
    if coords.shape[1] < 2:
        out[ys[0], xs[0]] = 1
        return out

    cov = np.cov(coords)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / norm

    if policy == "val_fixed":
        ratio = length_ratio
    else:
        ratio = float(rng.uniform(0.3, 0.8))

    half_len = ratio * max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) / 2.0
    n_steps = max(2, int(2 * half_len) + 1)
    for t in np.linspace(-half_len, half_len, n_steps):
        px = int(round(cx + direction[0] * t))
        py = int(round(cy + direction[1] * t))
        if 0 <= px < w and 0 <= py < h and mask[py, px]:
            out[py, px] = 1
            if cv2 is not None:
                cv2.circle(out, (px, py), 1, 1, -1)
    return out


def _gaussian_blur_map(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Widen a sparse map with a Gaussian filter."""
    if cv2 is None:
        yy, xx = np.ogrid[: arr.shape[0], : arr.shape[1]]
        # fallback: find peak and apply analytic Gaussian
        peak = arr.argmax()
        py, px = np.unravel_index(peak, arr.shape) if arr.max() > 0 else (0, 0)
        dist2 = (xx - px) ** 2 + (yy - py) ** 2
        return np.exp(-dist2 / (2 * sigma**2)).astype(np.float32) * float(arr.max() > 0)
    k = int(max(3, round(sigma * 6))) | 1
    blurred = cv2.GaussianBlur(arr.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma)
    return blurred.astype(np.float32)


def rasterize_point_map(
    height: int,
    width: int,
    point: Tuple[float, float],
    sigma: float = DEFAULT_GAUSSIAN_SIGMA,
) -> np.ndarray:
    """1×H×W Gaussian click map."""
    px, py = float(point[0]), float(point[1])
    yy, xx = np.ogrid[:height, :width]
    dist2 = (xx - px) ** 2 + (yy - py) ** 2
    return np.exp(-dist2 / (2 * sigma**2)).astype(np.float32)


def rasterize_box_map(
    height: int,
    width: int,
    bbox_xywh: List[float],
) -> np.ndarray:
    """1×H×W uniform box map (unit weight everywhere inside bbox)."""
    out = np.zeros((height, width), dtype=np.float32)
    x, y, bw, bh = bbox_xywh
    x0 = int(np.clip(np.floor(x), 0, width - 1))
    y0 = int(np.clip(np.floor(y), 0, height - 1))
    x1 = int(np.clip(np.ceil(x + bw), 0, width))
    y1 = int(np.clip(np.ceil(y + bh), 0, height))
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = 1.0
    return out


def rasterize_scribble_map(
    mask: np.ndarray,
    policy: str = "val_fixed",
    rng: Optional[np.random.RandomState] = None,
    sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    length_ratio: float = 0.7,
    ann_id: Optional[int] = None,
) -> np.ndarray:
    """1×H×W scribble map with Gaussian widening."""
    line = generate_scribble_mask(
        mask, policy=policy, rng=rng, length_ratio=length_ratio, ann_id=ann_id
    ).astype(np.float32)
    if line.max() == 0:
        return line
    return _gaussian_blur_map(line, sigma)


def rasterize_weak_signal_maps(
    mask: np.ndarray,
    prompt: dict,
    spatial_size: int,
    active_signal: Optional[str] = None,
    policy: str = "val_fixed",
    rng: Optional[np.random.RandomState] = None,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
) -> np.ndarray:
    """
    Build stacked weak-signal maps [3, H, W]:
      ch0 point (Gaussian), ch1 box (uniform fill), ch2 scribble (Gaussian-widened).

    If ``active_signal`` is set (eval mode), only that channel is non-zero.
    If ``active_signal`` is None (unified training), all three channels are populated.
    """
    h, w = mask.shape
    scale = spatial_size / float(max(h, w))
    sh = sw = spatial_size
    mask_s = mask
    if scale != 1.0 and cv2 is not None:
        mask_s = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)

    ann_id = prompt.get("ann_id")
    point_map = np.zeros((sh, sw), dtype=np.float32)
    box_map = np.zeros((sh, sw), dtype=np.float32)
    scribble_map = np.zeros((sh, sw), dtype=np.float32)

    if "point" in prompt:
        pt = prompt["point"]
        pt_s = [pt[0] * scale, pt[1] * scale]
        point_map = rasterize_point_map(sh, sw, pt_s, sigma=gaussian_sigma)

    if "bbox" in prompt:
        b = prompt["bbox"]
        bbox_s = [b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale]
        box_map = rasterize_box_map(sh, sw, bbox_s)

    scribble_map = rasterize_scribble_map(
        mask_s,
        policy=policy,
        rng=rng,
        sigma=gaussian_sigma,
        ann_id=ann_id,
    )

    stack = np.stack([point_map, box_map, scribble_map], axis=0)

    if active_signal is not None:
        channel_idx = {
            "points_only": 0,
            "boxes_only": 1,
            "scribbles_only": 2,
            "point": 0,
            "box": 1,
            "scribble": 2,
        }.get(active_signal, 0)
        masked = np.zeros_like(stack)
        masked[channel_idx] = stack[channel_idx]
        return masked

    return stack


def build_instance_prompts(
    mask: np.ndarray,
    policy: str = "val_fixed",
    signal_type: str = "mixed",
    rng: Optional[np.random.RandomState] = None,
    ann_id: Optional[int] = None,
) -> Dict:
    """
    Build prompt dict for one instance (SAM decoder + weak-signal rasterization).

    policy: 'val_fixed' | 'train_online'
    signal_type: 'mixed' | 'boxes_only' | 'points_only' | 'scribbles_only'
    """
    rng = rng or np.random.RandomState()
    h, w = mask.shape
    bbox = mask_to_bbox_xywh(mask)

    if policy == "val_fixed":
        px, py = mask_interior_point(mask)
        point = [px, py]
        base = {
            "point": point,
            "bbox": bbox,
            "ann_id": ann_id,
            "signal_type": signal_type if signal_type != "mixed" else "point",
        }
        if signal_type == "boxes_only":
            return {"bbox": bbox, "ann_id": ann_id, "signal_type": "box"}
        if signal_type == "points_only":
            return {"point": point, "ann_id": ann_id, "signal_type": "point"}
        if signal_type == "scribbles_only":
            scrib = generate_scribble_mask(mask, policy=policy, ann_id=ann_id)
            sp = mask_centroid_point(scrib) if scrib.max() else (px, py)
            return {
                "point": [sp[0], sp[1]],
                "bbox": bbox,
                "ann_id": ann_id,
                "signal_type": "scribble",
            }
        return base

    # train_online (Stage-2 jitter policy)
    if signal_type == "boxes_only":
        b = jitter_box_expand(bbox, img_hw=(h, w))
        return {"bbox": b, "ann_id": ann_id, "signal_type": "box"}

    if signal_type == "points_only":
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            px, py = mask_interior_point(mask)
        else:
            idx = rng.randint(0, len(xs))
            px, py = float(xs[idx]), float(ys[idx])
        px, py = jitter_point_manhattan((px, py), img_hw=(h, w), rng=rng)
        return {
            "point": [px, py],
            "ann_id": ann_id,
            "signal_type": "point",
        }

    if signal_type == "scribbles_only":
        scrib = generate_scribble_mask(mask, policy=policy, rng=rng, ann_id=ann_id)
        trim_r = float(rng.uniform(0.10, 0.20))
        scrib = jitter_scribble_trim(scrib, trim_ratio=trim_r, rng=rng)
        sp = mask_centroid_point(scrib) if scrib.max() else mask_interior_point(mask)
        return {
            "point": [sp[0], sp[1]],
            "bbox": bbox,
            "ann_id": ann_id,
            "signal_type": "scribble",
            "scribble_mask": scrib,
        }

    # mixed: random point with jitter (+ all channels populated at rasterize)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        px, py = mask_interior_point(mask)
    else:
        idx = rng.randint(0, len(xs))
        px, py = float(xs[idx]), float(ys[idx])
    px, py = jitter_point_manhattan((px, py), img_hw=(h, w), rng=rng)
    return {
        "point": [px, py],
        "bbox": bbox,
        "ann_id": ann_id,
        "signal_type": "mixed",
    }


def sam_prompt_for_signal(prompt: dict, signal_type: str) -> dict:
    """Subset prompt dict for SAM decoder under a specific weak-signal eval type."""
    out = {"ann_id": prompt.get("ann_id")}
    if signal_type == "boxes_only" and "bbox" in prompt:
        out["bbox"] = prompt["bbox"]
    elif signal_type == "points_only" and "point" in prompt:
        out["point"] = prompt["point"]
    elif signal_type == "scribbles_only":
        if "point" in prompt:
            out["point"] = prompt["point"]
        if "bbox" in prompt:
            out["bbox"] = prompt["bbox"]
    else:
        out.update({k: v for k, v in prompt.items() if k in ("point", "bbox", "ann_id")})
    return out


def build_image_prompts(
    masks: List[np.ndarray],
    policy: str = "train_online",
    signal_type: str = "mixed",
    rng: Optional[np.random.RandomState] = None,
    ann_ids: Optional[List[int]] = None,
) -> List[Dict]:
    """Build one prompt dict per instance mask in an image."""
    prompts = []
    for i, mask in enumerate(masks):
        ann_id = ann_ids[i] if ann_ids and i < len(ann_ids) else None
        prompts.append(
            build_instance_prompts(
                mask,
                policy=policy,
                signal_type=signal_type,
                rng=rng,
                ann_id=ann_id,
            )
        )
    return prompts


def build_batch_prompts_from_image_masks(
    masks_list: List[List[np.ndarray]],
    policy: str = "train_online",
    signal_type: str = "mixed",
    metas: Optional[list] = None,
) -> list:
    """Build prompts for a batch of images (each with N instance masks)."""
    all_prompts = []
    for i, masks in enumerate(masks_list):
        ann_ids = None
        if metas and i < len(metas):
            ann_ids = metas[i].get("ann_ids")
        all_prompts.extend(
            build_image_prompts(
                masks,
                policy=policy,
                signal_type=signal_type,
                ann_ids=ann_ids,
            )
        )
    return all_prompts
