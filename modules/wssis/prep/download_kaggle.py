"""
Download COCO 2017 and coco-minitrain-10k from Kaggle into data/.

Expects data/kaggle.json (copy from ~/.kaggle/kaggle.json).
Sets KAGGLE_CONFIG_DIR=data/ during download.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from modules.wssis.paths import coco_root, data_dir, kaggle_config_dir, minitrain_root, repo_root


COCO_DATASET = "awsaf49/coco-2017-dataset"
MINITRAIN_DATASET = "banuprasadb/coco-minitrain-10k"


def _run_kaggle(args: list[str], download_dir: Path) -> None:
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir())
    download_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", *args, "-p", str(download_dir), "--unzip"]
    print("[download]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(repo_root()))


def _find_subdir(root: Path, name: str) -> Path | None:
    for p in root.rglob(name):
        if p.is_dir():
            return p
    return None


def _organize_coco(extract_root: Path) -> None:
    """Move extracted COCO tree to data/coco2017/."""
    dest = coco_root()
    if (dest / "annotations" / "instances_train2017.json").exists():
        print(f"[download] COCO already at {dest}, skipping organize.")
        return

    candidates = [
        extract_root / "coco2017",
        extract_root / "coco-2017-dataset" / "coco2017",
        _find_subdir(extract_root, "coco2017"),
    ]
    src = next((c for c in candidates if c and c.exists()), None)
    if src is None:
        # Maybe annotations at top level
        if (extract_root / "annotations").exists():
            src = extract_root
        else:
            raise FileNotFoundError(
                f"Could not find coco2017 folder under {extract_root}. "
                "Check Kaggle zip layout manually."
            )

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    print(f"[download] COCO organized at {dest}")


def _organize_minitrain(extract_root: Path) -> None:
    dest = minitrain_root()
    if (dest / "train2017.txt").exists():
        print(f"[download] Minitrain already at {dest}, skipping organize.")
        return

    candidates = [
        extract_root / "coco_minitrain_10k",
        _find_subdir(extract_root, "coco_minitrain_10k"),
    ]
    src = next((c for c in candidates if c and c.exists()), None)
    if src is None:
        # Flat txt at root
        if (extract_root / "train2017.txt").exists():
            src = extract_root
        else:
            raise FileNotFoundError(
                f"Could not find coco_minitrain_10k under {extract_root}."
            )

    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        t = dest / item.name
        if item.is_dir():
            if t.exists():
                shutil.rmtree(t)
            shutil.copytree(item, t)
        else:
            shutil.copy2(item, t)
    print(f"[download] Minitrain organized at {dest}")


def _unzip_zips(folder: Path) -> None:
    for z in folder.glob("*.zip"):
        print(f"[download] Unzipping {z}")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(folder)


def run(skip_coco: bool = False, skip_minitrain: bool = False) -> None:
    data_dir().mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_config_dir() / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            f"Place your Kaggle API credentials at {kaggle_json}\n"
            "Copy from ~/.kaggle/kaggle.json"
        )

    dl_root = data_dir() / "downloads"
    dl_root.mkdir(parents=True, exist_ok=True)

    if not skip_coco:
        coco_dl = dl_root / "coco-2017"
        coco_dl.mkdir(parents=True, exist_ok=True)
        _run_kaggle([COCO_DATASET], coco_dl)
        _unzip_zips(coco_dl)
        _organize_coco(coco_dl)

    if not skip_minitrain:
        mini_dl = dl_root / "coco-minitrain-10k"
        mini_dl.mkdir(parents=True, exist_ok=True)
        _run_kaggle([MINITRAIN_DATASET], mini_dl)
        _unzip_zips(mini_dl)
        _organize_minitrain(mini_dl)

    print("[download] Done. Next: python -m modules.wssis.prep.generate_splits")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCO datasets via Kaggle API")
    parser.add_argument("--skip-coco", action="store_true")
    parser.add_argument("--skip-minitrain", action="store_true")
    args = parser.parse_args()
    run(skip_coco=args.skip_coco, skip_minitrain=args.skip_minitrain)


if __name__ == "__main__":
    main()
