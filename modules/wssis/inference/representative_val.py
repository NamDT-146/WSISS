"""
Curate and load a fixed representative COCO val2017 image list for qualitative inference.

Four categories × 5 images (20 total), drawn from full val — not limited to minitrain-10k.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from modules.vig_refinenet.coco_sam_stage1_dataset import ann_to_mask
from modules.wssis.paths import coco_root, repo_root, splits_dir

# COCO thing category ids (standard 80-class instance set).
PERSON_CAT_ID = 1

# Frequent / visually clear classes for multi-object and overlap demos.
MAJOR_CAT_IDS: Set[int] = {
    1,  # person
    3,  # car
    4,  # motorcycle
    6,  # bus
    8,  # truck
    16,  # bird
    17,  # cat
    18,  # dog
    62,  # chair
    63,  # couch
    64,  # potted plant
    44,  # bottle
    47,  # cup
    56,  # broccoli
    59,  # pizza
    73,  # laptop
}

# Rare thing classes — good for the minor-class bucket.
MINOR_CAT_IDS: Set[int] = {
    11,  # fire hydrant
    14,  # parking meter
    78,  # microwave
    79,  # oven
    80,  # toaster
    87,  # scissors
    89,  # hair drier
    90,  # toothbrush
}

CATEGORY_NAMES: Tuple[str, ...] = (
    "easy_person",
    "multi_separated",
    "overlapping",
    "minor_class",
)

DEFAULT_LIST_PATH = repo_root() / "scripts" / "inference" / "representative_val_inference.txt"
IMAGES_PER_CATEGORY = 5
CURATOR_SEED = 42


@dataclass(frozen=True)
class RepresentativeSample:
    category: str
    image_id: int
    file_name: str
    n_objects: int
    category_ids: Tuple[int, ...]
    score: float
    note: str = ""


def default_val_ann_path() -> Path:
    return coco_root() / "annotations" / "instances_val2017.json"


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter / max(union, 1))


def _image_masks(anns: Sequence[dict], height: int, width: int) -> List[np.ndarray]:
    return [ann_to_mask(ann, height, width).astype(np.uint8) for ann in anns]


def _max_pairwise_iou(masks: Sequence[np.ndarray]) -> float:
    best = 0.0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            best = max(best, _mask_iou(masks[i], masks[j]))
    return best


def _score_easy_person(anns: Sequence[dict], masks: Sequence[np.ndarray]) -> Optional[float]:
    if len(anns) != 1:
        return None
    if int(anns[0].get("category_id", 0)) != PERSON_CAT_ID:
        return None
    area = float(masks[0].sum())
    if area < 500:
        return None
    return area


def _score_multi_separated(anns: Sequence[dict], masks: Sequence[np.ndarray]) -> Optional[float]:
    if len(anns) < 3:
        return None
    cats = {int(a["category_id"]) for a in anns}
    if len(cats) < 3:
        return None
    max_iou = _max_pairwise_iou(masks)
    if max_iou > 0.05:
        return None
    major = len(cats & MAJOR_CAT_IDS)
    if major < 2:
        return None
    return major * 1000.0 + len(anns) * 10.0 + len(cats) - max_iou * 100.0


def _score_overlapping(anns: Sequence[dict], masks: Sequence[np.ndarray]) -> Optional[float]:
    if len(anns) < 2:
        return None
    max_iou = _max_pairwise_iou(masks)
    if max_iou < 0.12:
        return None
    cats = {int(a["category_id"]) for a in anns}
    major = len(cats & MAJOR_CAT_IDS)
    if major < 1:
        return None
    return max_iou * 1000.0 + major * 50.0 + len(anns)


def _score_minor_class(anns: Sequence[dict], masks: Sequence[np.ndarray]) -> Optional[float]:
    minor_areas = []
    for ann, mask in zip(anns, masks):
        cid = int(ann.get("category_id", 0))
        if cid in MINOR_CAT_IDS:
            minor_areas.append(float(mask.sum()))
    if not minor_areas:
        return None
    best_area = max(minor_areas)
    if best_area < 200:
        return None
    n_minor = sum(1 for a in anns if int(a.get("category_id", 0)) in MINOR_CAT_IDS)
    return best_area + n_minor * 100.0


def _rank_candidates(
    coco: dict,
    scorer,
    *,
    exclude: Set[int],
    top_k: int,
) -> List[Tuple[float, int, list, dict]]:
    images = {img["id"]: img for img in coco["images"]}
    anns_by_image: Dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[ann["image_id"]].append(ann)

    ranked: List[Tuple[float, int, list, dict]] = []
    for img_id, anns in anns_by_image.items():
        if img_id in exclude or img_id not in images:
            continue
        info = images[img_id]
        h, w = info["height"], info["width"]
        masks = _image_masks(anns, h, w)
        if not masks:
            continue
        score = scorer(anns, masks)
        if score is None:
            continue
        ranked.append((score, img_id, anns, info))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    return ranked[:top_k]


def curate_representative_val_list(
    ann_json: Optional[Path] = None,
    *,
    images_per_category: int = IMAGES_PER_CATEGORY,
    seed: int = CURATOR_SEED,
) -> List[RepresentativeSample]:
    """Select fixed representative val images (deterministic given COCO json)."""
    ann_json = Path(ann_json or default_val_ann_path())
    if not ann_json.is_file():
        raise FileNotFoundError(
            f"COCO val annotations not found: {ann_json}\n"
            "Run: bash scripts/setup/01_download_data.sh"
        )

    with open(ann_json, encoding="utf-8") as f:
        coco = json.load(f)

    scorers = {
        "easy_person": _score_easy_person,
        "multi_separated": _score_multi_separated,
        "overlapping": _score_overlapping,
        "minor_class": _score_minor_class,
    }

    selected: List[RepresentativeSample] = []
    used: Set[int] = set()
    rng = np.random.RandomState(seed)

    for category in CATEGORY_NAMES:
        ranked = _rank_candidates(
            coco,
            scorers[category],
            exclude=used,
            top_k=images_per_category * 4,
        )
        if len(ranked) < images_per_category:
            raise RuntimeError(
                f"Could only find {len(ranked)} candidates for {category!r}; "
                f"need {images_per_category}."
            )
        # Stable shuffle among top pool so ties break consistently but not only by image_id.
        pool = ranked[: images_per_category * 2]
        order = np.arange(len(pool))
        rng.shuffle(order)
        picks = [pool[i] for i in sorted(order[:images_per_category], key=lambda i: -pool[i][0])]

        for score, img_id, anns, info in picks:
            used.add(img_id)
            cats = tuple(int(a["category_id"]) for a in anns)
            selected.append(
                RepresentativeSample(
                    category=category,
                    image_id=img_id,
                    file_name=info["file_name"],
                    n_objects=len(anns),
                    category_ids=cats,
                    score=score,
                )
            )

    return selected


def write_representative_list(
    samples: Sequence[RepresentativeSample],
    out_path: Path,
    *,
    ann_json: Optional[Path] = None,
) -> Path:
    """Write human-readable fixed list (comments + TSV rows)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ann = ann_json or default_val_ann_path()

    lines = [
        "# WSSIS representative COCO val2017 inference set",
        "# Four categories × 5 images = 20 total (full val, not minitrain-10k).",
        f"# Source annotations: {ann.as_posix()}",
        "# Columns: category  image_id  file_name  n_objects  category_ids  score",
        "# Regenerate: python -m modules.wssis.inference.run_representative --build-list",
        "",
    ]
    current_cat = None
    for s in samples:
        if s.category != current_cat:
            current_cat = s.category
            lines.append(f"# --- {current_cat} ---")
        cat_str = ",".join(str(c) for c in s.category_ids)
        lines.append(
            f"{s.category}\t{s.image_id:012d}\t{s.file_name}\t{s.n_objects}\t{cat_str}\t{s.score:.2f}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def parse_representative_list(list_path: Path) -> List[RepresentativeSample]:
    """Load fixed list written by :func:`write_representative_list`."""
    list_path = Path(list_path)
    if not list_path.is_file():
        raise FileNotFoundError(f"Representative list not found: {list_path}")

    samples: List[RepresentativeSample] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        category, image_id_s, file_name, n_obj_s, cat_ids_s, score_s = parts[:6]
        cat_ids = tuple(int(x) for x in cat_ids_s.split(",") if x)
        samples.append(
            RepresentativeSample(
                category=category,
                image_id=int(image_id_s),
                file_name=file_name,
                n_objects=int(n_obj_s),
                category_ids=cat_ids,
                score=float(score_s),
            )
        )
    return samples


def load_val_annotations_for_samples(
    samples: Sequence[RepresentativeSample],
    ann_json: Optional[Path] = None,
) -> Dict[int, Tuple[dict, List[dict]]]:
    """Return image_id -> (image_info, anns) for listed samples."""
    ann_json = Path(ann_json or default_val_ann_path())
    with open(ann_json, encoding="utf-8") as f:
        coco = json.load(f)

    want = {s.image_id for s in samples}
    images = {img["id"]: img for img in coco["images"] if img["id"] in want}
    anns_by_image: Dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann["image_id"] in want:
            anns_by_image[ann["image_id"]].append(ann)

    out: Dict[int, Tuple[dict, List[dict]]] = {}
    for img_id in want:
        if img_id not in images:
            raise KeyError(f"image_id {img_id} missing from {ann_json}")
        out[img_id] = (images[img_id], anns_by_image.get(img_id, []))
    return out
