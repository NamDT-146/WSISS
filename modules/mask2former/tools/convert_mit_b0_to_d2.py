#!/usr/bin/env python3
"""Download / convert SegFormer MiT-B0 ImageNet weights to Detectron2 format."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

import torch

# GitHub first (often reachable on MSRA); OpenMMLab last (can hang on connect).
MIT_B0_HTTP_URLS: tuple[str, ...] = (
    "https://github.com/NVlabs/SegFormer/releases/download/v1.0/mit_b0.pth",
    "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220615-849e2549.pth",
)

HF_HUB_SOURCES: tuple[tuple[str, str], ...] = (
    ("openmmlab/mmsegmentation", "mit_b0_20220615-849e2549.pth"),
    ("openmmlab/mmsegmentation", "pretrain/segformer/mit_b0_20220615-849e2549.pth"),
)

TIMM_CANDIDATES: tuple[str, ...] = ("mit_b0", "mit_b0.in1k", "mit_b0.in1k224")

CONNECT_TIMEOUT_S = 20
READ_TIMEOUT_S = 300
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


def _looks_like_segformer_mit(raw: dict) -> bool:
    return any("patch_embed1" in k for k in raw)


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
        raw = _extract_raw_state(torch.load(path, map_location="cpu", weights_only=False))
    except EOFError as exc:
        raise ValueError(f"{path} is truncated or not a valid PyTorch checkpoint") from exc
    if not _looks_like_segformer_mit(raw):
        raise ValueError(f"{path} does not look like SegFormer MiT-B0 (missing patch_embed1 keys)")


def _download_via_cli(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    if shutil.which("wget"):
        cmd = [
            "wget",
            "-c",
            "--timeout=30",
            "--tries=2",
            "-O",
            str(tmp),
            url,
        ]
    elif shutil.which("curl"):
        cmd = [
            "curl",
            "-fL",
            "--connect-timeout",
            str(CONNECT_TIMEOUT_S),
            "--max-time",
            str(READ_TIMEOUT_S),
            "-o",
            str(tmp),
            url,
        ]
    else:
        raise OSError("Neither wget nor curl found on PATH")

    print(f"[mit_b0] downloading via {cmd[0]}: {url}")
    subprocess.run(cmd, check=True)
    tmp.replace(dest)
    _validate_source_file(dest)
    print(f"[mit_b0] saved {dest} ({dest.stat().st_size} bytes)")


def _download_via_urllib(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "wssis-mit-b0-downloader/1.0"},
    )
    print(f"[mit_b0] downloading via urllib: {url}")
    with urllib.request.urlopen(req, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)
    _validate_source_file(dest)
    print(f"[mit_b0] saved {dest} ({dest.stat().st_size} bytes)")


def _fetch_via_timm(dest: Path) -> None:
    import timm

    names: list[str] = []
    try:
        names = list(timm.list_models("*mit*b0*", pretrained=True))
    except Exception:
        pass
    candidates = list(dict.fromkeys([*names, *TIMM_CANDIDATES]))
    for name in candidates:
        try:
            print(f"[mit_b0] trying timm pretrained: {name}")
            model = timm.create_model(name, pretrained=True)
            raw = model.state_dict()
            if not _looks_like_segformer_mit(raw):
                raise ValueError(f"{name} layout incompatible with SegFormer MiT")
            torch.save(raw, dest)
            _validate_source_file(dest)
            print(f"[mit_b0] timm saved {dest} ({dest.stat().st_size} bytes)")
            return
        except Exception as exc:
            print(f"[mit_b0] timm {name} failed: {exc}")
    raise RuntimeError("timm could not fetch compatible MiT-B0 weights")


def _fetch_via_hf_hub(dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    last_err: Exception | None = None
    for repo_id, filename in HF_HUB_SOURCES:
        try:
            print(f"[mit_b0] hf_hub_download {repo_id}/{filename}")
            cached = Path(hf_hub_download(repo_id=repo_id, filename=filename))
            shutil.copy2(cached, dest)
            _validate_source_file(dest)
            print(f"[mit_b0] huggingface saved {dest} ({dest.stat().st_size} bytes)")
            return
        except Exception as exc:
            last_err = exc
            print(f"[mit_b0] hf_hub {repo_id}/{filename} failed: {exc}")
            if dest.exists():
                dest.unlink()
    raise RuntimeError(f"huggingface_hub fetch failed: {last_err}")


def _fetch_via_http(dest: Path, urls: Iterable[str] = MIT_B0_HTTP_URLS) -> None:
    errors: list[str] = []
    downloader: Callable[[str, Path], None]
    if shutil.which("wget") or shutil.which("curl"):
        downloader = _download_via_cli
    else:
        downloader = _download_via_urllib

    for url in urls:
        try:
            if dest.exists():
                dest.unlink()
            downloader(url, dest)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, subprocess.CalledProcessError) as exc:
            errors.append(f"{url}: {exc}")
            if dest.exists():
                dest.unlink()
    raise RuntimeError("HTTP fetch failed:\n  " + "\n  ".join(errors))


def fetch_source_weights(dest: Path) -> Path:
    manual = os.environ.get("WSSIS_MIT_B0_PTH", "").strip()
    if manual:
        src = Path(manual).expanduser().resolve()
        print(f"[mit_b0] using WSSIS_MIT_B0_PTH={src}")
        _validate_source_file(src)
        shutil.copy2(src, dest)
        return dest

    strategies: tuple[tuple[str, Callable[[Path], None]], ...] = (
        ("timm", _fetch_via_timm),
        ("huggingface_hub", _fetch_via_hf_hub),
        ("http", _fetch_via_http),
    )
    errors: list[str] = []
    for name, fn in strategies:
        try:
            if dest.exists():
                dest.unlink()
            fn(dest)
            return dest
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            if dest.exists():
                dest.unlink()

    raise RuntimeError(
        "Failed to fetch MiT-B0 weights.\n"
        "  " + "\n  ".join(errors)
        + "\n\nManual fallback:\n"
        "  1) Download mit_b0.pth on a machine with network access\n"
        "  2) scp to checkpoints/mit_b0.pth on this node\n"
        "  3) export WSSIS_MIT_B0_PTH=/path/to/mit_b0.pth\n"
        "  4) bash scripts/setup/05_download_mit_b0_weights.sh"
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
