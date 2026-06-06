#!/usr/bin/env python3
"""Download / convert SegFormer MiT-B0 ImageNet weights to Detectron2 format."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable

import torch

# OpenMMLab mirror is usually more reliable than GitHub releases on cluster nodes.
MIT_B0_URLS: tuple[str, ...] = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220615-849e2549.pth",
    "https://github.com/NVlabs/SegFormer/releases/download/v1.0/mit_b0.pth",
)

# SegFormer MiT-B0 backbone-only checkpoint is a few MB; reject HTML/error stubs.
MIN_SOURCE_BYTES = 1_000_000


def _remap_state_dict(raw: dict) -> dict:
    """Map SegFormer / OpenMMLab keys to ``backbone.model.*`` for D2MixVisionTransformer."""
    out: dict = {}
    for key, value in raw.items():
        if key.startswith("head"):
            continue
        if key.startswith("backbone.model."):
            out[key] = value
        elif key.startswith("backbone."):
            out[f"backbone.model.{key[len('backbone.'):]}"] = value
        elif key.startswith("model."):
            out[f"backbone.{key}"] = value
        else:
            out[f"backbone.model.{key}"] = value
    return out


def _extract_raw_state(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raw = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        raw = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        raw = checkpoint
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)!r}")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Checkpoint contains no weight tensors")
    return raw


def _validate_source_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source weights: {path}")
    size = path.stat().st_size
    if size < MIN_SOURCE_BYTES:
        raise ValueError(
            f"{path} is too small ({size} bytes). "
            "Download likely failed or produced an HTML/error stub."
        )
    try:
        _extract_raw_state(torch.load(path, map_location="cpu", weights_only=False))
    except EOFError as exc:
        raise ValueError(f"{path} is truncated or not a valid PyTorch checkpoint") from exc


def _download_one(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "wssis-mit-b0-downloader/1.0"},
    )
    print(f"[mit_b0] downloading {url}")
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)
    _validate_source_file(dest)
    print(f"[mit_b0] saved {dest} ({dest.stat().st_size} bytes)")


def fetch_source_weights(dest: Path, urls: Iterable[str] = MIT_B0_URLS) -> Path:
    errors: list[str] = []
    for url in urls:
        try:
            if dest.exists():
                dest.unlink()
            _download_one(url, dest)
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
            if dest.exists():
                dest.unlink()
    raise RuntimeError(
        "Failed to download MiT-B0 weights from all sources:\n  "
        + "\n  ".join(errors)
    )


def convert(src: Path, dst: Path) -> None:
    _validate_source_file(src)
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    raw = _extract_raw_state(checkpoint)
    remapped = _remap_state_dict(raw)
    if not remapped:
        raise ValueError(f"No backbone tensors found in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": remapped}, dst)
    print(f"[mit_b0] wrote {len(remapped)} tensors -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "src",
        nargs="?",
        type=Path,
        help="Source .pth (SegFormer / OpenMMLab mit_b0 release)",
    )
    parser.add_argument(
        "dst",
        nargs="?",
        type=Path,
        help="Output .pkl for Detectron2 MODEL.WEIGHTS",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download source .pth when missing or invalid",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    src = args.src or (repo_root / "checkpoints" / "mit_b0.pth")
    dst = args.dst or (repo_root / "checkpoints" / "mit_b0_pretrained.pkl")

    need_fetch = args.fetch
    if src.exists():
        try:
            _validate_source_file(src)
        except ValueError as exc:
            print(f"[mit_b0] invalid source {src}: {exc}", file=sys.stderr)
            need_fetch = True
    else:
        need_fetch = True

    if need_fetch:
        fetch_source_weights(src)

    convert(src, dst)


if __name__ == "__main__":
    main()
