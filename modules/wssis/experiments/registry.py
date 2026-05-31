"""
Experiment registry aligned with report/PLAN.md (5-item report matrix).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

StudentType = Literal["mask2former", "yolov8"]
SignalType = Literal["none", "mixed", "boxes_only", "points_only"]

# Archived IDs (ablations / old baselines) — kept for backward-compatible lookups
ARCHIVED_EXPERIMENT_IDS = frozenset({"1B", "2A", "2B", "2C", "3A", "3B", "3C"})


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    name: str
    phase: str
    student: StudentType
    labeled_split: str  # none | labeled_5pct | train_all
    weak_split: str  # none | weak_95pct
    use_gnn: bool = False
    use_raw_sam_only: bool = False
    use_distillation: bool = False
    use_symmetric_loss: bool = True
    weak_signal: SignalType = "mixed"
    gnn_checkpoint: str = "gnn_refiner_stage1.pt"
    requires_p0: bool = True
    stage2_epochs: int = 50
    use_semi_weak: bool = False
    freeze_gnn: bool = False
    notes: str = ""


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "1A": ExperimentSpec(
        id="1A",
        name="Report 1 — 5% fully supervised Mask2Former",
        phase="R1",
        student="mask2former",
        labeled_split="labeled_5pct",
        weak_split="none",
        use_gnn=False,
        use_distillation=False,
        weak_signal="none",
        use_semi_weak=False,
        notes="Lower bound: GT only on labeled_5pct.",
    ),
    "1C": ExperimentSpec(
        id="1C",
        name="Report 3 — True semi-weak SWSIS (main)",
        phase="R3",
        student="mask2former",
        labeled_split="labeled_5pct",
        weak_split="weak_95pct",
        use_gnn=True,
        use_distillation=True,
        use_symmetric_loss=True,
        weak_signal="mixed",
        use_semi_weak=True,
        freeze_gnn=False,
        notes="50/50 labeled+weak; SAM cache + GNN pseudo + distill; GNN trainable.",
    ),
    "1D": ExperimentSpec(
        id="1D",
        name="Report 5 — 100% fully supervised upper bound",
        phase="R5",
        student="mask2former",
        labeled_split="train_all",
        weak_split="none",
        use_gnn=False,
        use_distillation=False,
        weak_signal="none",
        use_semi_weak=False,
        notes="Upper bound. Existing mislabeled 1C run can stand in for 1D.",
    ),
    "4A": ExperimentSpec(
        id="4A",
        name="Report 4 — YOLOv8-seg true semi-weak",
        phase="R4",
        student="yolov8",
        labeled_split="labeled_5pct",
        weak_split="weak_95pct",
        use_gnn=True,
        use_distillation=True,
        weak_signal="mixed",
        use_semi_weak=True,
        freeze_gnn=False,
        notes="Same teacher pipeline as 1C with YOLO student.",
    ),
}

# Backward compatibility: resolve archived IDs if referenced
_ARCHIVED_SPECS: Dict[str, ExperimentSpec] = {
    "1B": ExperimentSpec(
        id="1B",
        name="[archived] Raw SAM weak baseline",
        phase="archived",
        student="mask2former",
        labeled_split="none",
        weak_split="weak_95pct",
        use_raw_sam_only=True,
    ),
}

DEFAULT_RUN_ORDER: List[str] = ["1A", "1C", "4A"]


def get_experiment(exp_id: str) -> ExperimentSpec:
    key = exp_id.upper().replace("EXP", "").strip()
    if key in EXPERIMENTS:
        return EXPERIMENTS[key]
    if key in _ARCHIVED_SPECS:
        return _ARCHIVED_SPECS[key]
    raise KeyError(
        f"Unknown experiment '{exp_id}'. Active: {list(EXPERIMENTS)} "
        f"(archived: {sorted(ARCHIVED_EXPERIMENT_IDS)})"
    )
