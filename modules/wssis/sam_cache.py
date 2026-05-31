"""
Load P0.2 SAM image embeddings and batch helpers for Stage-1 / teacher eval.

Avoids re-running the ViT-B encoder on every instance; deduplicates by image_id within a batch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from modules.wssis.paths import sam_embeddings_dir


def sam_embedding_cache_path(image_id: int, split: str) -> Path:
    key = f"{image_id:012d}"
    return sam_embeddings_dir() / split / f"{key}.fp16.npy"


def load_sam_embedding_cache(
    image_id: int,
    split: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Return [256, 64, 64] embedding or None if P0.2 cache file is missing."""
    path = sam_embedding_cache_path(image_id, split)
    if not path.exists():
        return None
    arr = np.load(path)
    t = torch.as_tensor(arr, device=device, dtype=dtype)
    del arr
    if t.dim() == 4:
        t = t.squeeze(0)
    return t.contiguous()


def fetch_sam_embeddings_batch(
    metas: List[dict],
    sam_model: torch.nn.Module,
    images: torch.Tensor,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    *,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Build [B, 256, 64, 64] SAM embeddings for a dataloader batch.

    Uses P0.2 npy cache when available; deduplicates encoder runs by (image_id, split).
    Falls back to live encoder for cache misses.
    """
    from modules.vig_refinenet.sam_stage1_common import encode_sam_embeddings

    B = len(metas)
    device = images.device
    dtype = images.dtype
    out: List[Optional[torch.Tensor]] = [None] * B
    unique: Dict[Tuple[int, str], torch.Tensor] = {}
    stats = {"cache_hits": 0, "cache_misses": 0, "unique_images": 0}

    miss_indices: List[int] = []
    miss_images: List[torch.Tensor] = []

    for i, meta in enumerate(metas):
        image_id = int(meta["image_id"])
        split = meta.get("split", "train")
        key = (image_id, split)

        if key in unique:
            out[i] = unique[key]
            stats["cache_hits"] += 1
            continue

        stats["unique_images"] += 1
        emb = None
        if use_cache:
            emb = load_sam_embedding_cache(image_id, split, device, dtype=dtype)

        if emb is not None:
            unique[key] = emb
            out[i] = emb
            stats["cache_hits"] += 1
        else:
            stats["cache_misses"] += 1
            miss_indices.append(i)
            miss_images.append(images[i])

    if miss_indices:
        stacked = torch.stack(miss_images, dim=0)
        encoded = encode_sam_embeddings(sam_model, stacked, pixel_mean, pixel_std)
        for j, idx in enumerate(miss_indices):
            meta = metas[idx]
            key = (int(meta["image_id"]), meta.get("split", "train"))
            unique[key] = encoded[j]
            out[idx] = encoded[j]

    if any(t is None for t in out):
        raise RuntimeError("Failed to resolve SAM embeddings for batch")

    return torch.stack(out, dim=0), stats
