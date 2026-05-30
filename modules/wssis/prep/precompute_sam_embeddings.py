"""
P0.2 — Precompute frozen SAM ViT-B image embeddings for train + val image lists.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from modules.wssis.paths import (
    build_coco_paths,
    coco_root,
    ensure_dirs,
    sam_embeddings_dir,
    sam_vit_b_checkpoint,
)
from modules.wssis.run_context import RunContext


def _image_ids_from_txt(txt: Path) -> list[int]:
    ids = []
    with open(txt, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"(\d{12})", line)
            if m:
                ids.append(int(m.group(1)))
    return sorted(set(ids))


def _image_path(image_id: int, split: str) -> Path:
    root = coco_root()
    for sub in (f"{split}2017", f"images/{split}2017"):
        p = root / sub / f"{image_id:012d}.jpg"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image {image_id:012d} not found under {root}")


def run(
    device: str = "cuda",
    batch_size: int = 1,
    limit: int | None = None,
    force: bool = False,
    run_ctx: RunContext | None = None,
) -> None:
    ensure_dirs()
    paths = build_coco_paths()

    for name in ("train_all_txt", "val_all_txt"):
        if not paths[name].exists():
            raise FileNotFoundError(f"Missing {paths[name]}. Run P0.1 first.")

    ckpt = sam_vit_b_checkpoint()
    if not ckpt.exists():
        raise FileNotFoundError(
            f"SAM checkpoint missing: {ckpt}\n"
            "Run scripts/setup/02_download_sam_weights.sh"
        )

    from modules.wssis.paths import repo_root
    import sys

    if str(repo_root()) not in sys.path:
        sys.path.insert(0, str(repo_root()))

    from modules.vig_refinenet.sam_stage1_common import (
        encode_sam_embeddings,
        get_sam_pixel_stats,
        load_sam_vit_b,
        resolve_device,
    )

    dev = resolve_device(prefer_cuda=device.startswith("cuda"))
    sam = load_sam_vit_b(str(ckpt), dev)
    pixel_mean, pixel_std = get_sam_pixel_stats(dev)

    manifest: dict = {}
    manifest_path = sam_embeddings_dir() / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    jobs: list[tuple[str, int]] = []
    train_ids = _image_ids_from_txt(paths["train_all_txt"])
    val_ids = _image_ids_from_txt(paths["val_all_txt"])
    for iid in train_ids:
        jobs.append(("train", iid))
    for iid in val_ids:
        jobs.append(("val", iid))
    if limit:
        jobs = jobs[:limit]

    total = len(jobs)

    for idx, (split, image_id) in enumerate(tqdm(jobs, desc="SAM embeddings"), start=1):
        key = f"{image_id:012d}"
        out_dir = sam_embeddings_dir() / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{key}.fp16.npy"
        if out_path.exists() and not force and key in manifest:
            continue

        img_path = _image_path(image_id, split if split == "val" else "train")
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        image = image.resize((1024, 1024), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(dev)

        with torch.no_grad():
            emb = encode_sam_embeddings(sam, tensor, pixel_mean, pixel_std)

        np.save(out_path, emb.squeeze(0).cpu().half().numpy())
        manifest[key] = {
            "path": str(out_path.relative_to(sam_embeddings_dir())).replace("\\", "/"),
            "split": split,
            "image_id": image_id,
            "orig_size": [h, w],
            "sam_input_size": 1024,
        }

        if run_ctx is not None and idx % 50 == 0:
            run_ctx.update_step(
                "p0_embeddings",
                {"status": "running", "done": idx, "total": total},
            )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if run_ctx is not None:
        run_ctx.update_step("p0_embeddings", {"status": "done", "done": total, "total": total})
        run_ctx.log("P0.2 embeddings: %d entries", len(manifest))
    print(f"[P0.2] Wrote {len(manifest)} entries to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="P0.2 precompute SAM embeddings")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap number of images")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(device=args.device, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
