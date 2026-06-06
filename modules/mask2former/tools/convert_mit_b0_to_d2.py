#!/usr/bin/env python3
"""Convert SegFormer MiT-B0 ImageNet weights to Detectron2 checkpoint format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _remap_state_dict(raw: dict) -> dict:
    """Map SegFormer / timm keys to ``backbone.model.*`` for D2MixVisionTransformer."""
    out: dict = {}
    for key, value in raw.items():
        if key.startswith("head"):
            continue
        if key.startswith("backbone.model."):
            out[key] = value
        elif key.startswith("model."):
            out[f"backbone.{key}"] = value
        else:
            out[f"backbone.model.{key}"] = value
    return out


def convert(src: Path, dst: Path) -> None:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raw = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        raw = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        raw = checkpoint
    else:
        raise TypeError(f"Unexpected checkpoint type from {src}")

    remapped = _remap_state_dict(raw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": remapped}, dst)
    print(f"Wrote {len(remapped)} tensors -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Source .pth (SegFormer mit_b0 release)")
    parser.add_argument("dst", type=Path, help="Output .pkl for Detectron2 MODEL.WEIGHTS")
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
