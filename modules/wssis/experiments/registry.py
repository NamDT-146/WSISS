"""
Experiment registry aligned with report/PLAN.md and report/EXPERIMENT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

StudentType = Literal["mask2former", "yolov8"]
SignalType = Literal["none", "mixed", "boxes_only", "points_only"]


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
    notes: str = ""


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "1A": ExperimentSpec(
        id="1A",
        name="Lower bound — 5% fully supervised",
        phase="P2",
        student="mask2former",
        labeled_split="labeled_5pct",
        weak_split="none",
        use_gnn=False,
        use_distillation=False,
        weak_signal="none",
    ),
    "1B": ExperimentSpec(
        id="1B",
        name="Weak baseline — raw SAM pseudo-labels",
        phase="P2",
        student="mask2former",
        labeled_split="none",
        weak_split="weak_95pct",
        use_gnn=False,
        use_raw_sam_only=True,
        use_distillation=False,
    ),
    "1C": ExperimentSpec(
        id="1C",
        name="Full SWSIS (main result)",
        phase="P1",
        student="mask2former",
        labeled_split="labeled_5pct",
        weak_split="weak_95pct",
        use_gnn=True,
        use_distillation=True,
        use_symmetric_loss=True,
        weak_signal="mixed",
        notes="Primary experiment for report.",
    ),
    "1D": ExperimentSpec(
        id="1D",
        name="Upper bound — 100% fully supervised",
        phase="P2",
        student="mask2former",
        labeled_split="train_all",
        weak_split="none",
        use_gnn=False,
        use_distillation=False,
        weak_signal="none",
    ),
    "2A": ExperimentSpec(
        id="2A",
        name="Ablation — no GNN refiner",
        phase="P3",
        student="mask2former",
        labeled_split="labeled_5pct",
        weak_split="weak_95pct",
        use_gnn=False,
        use_distillation=True,
    ),
    "2B": ExperimentSpec(
        id="2B",
        name="Ablation — no feature distillation",
        phase="P3",
        student="mask2former",
        labeled_split="labeled_5pct",
        weak_split="weak_95pct",
        use_gnn=True,
        use_distillation=False,
    ),
    "2C": ExperimentSpec(
        id="2C",
        name="Ablation — no symmetric loss (GNN ckpt without sym)",
        phase="P3",
        student="mask2former",
        labeled_split="labeled_5pct",
        weak_split="weak_95pct",
        use_gnn=True,
        use_distillation=True,
        use_symmetric_loss=False,
        gnn_checkpoint="gnn_refiner_no_sym.pt",
        notes="Train Stage-1 without sym loss first (P0.4b).",
    ),
    "3A": ExperimentSpec(
        id="3A",
        name="Signal sensitivity — boxes only",
        phase="P4",
        student="mask2former",
        labeled_split="labeled_5pct",
        weak_split="weak_95pct",
        use_gnn=True,
        use_distillation=True,
        weak_signal="boxes_only",
    ),
    "3B": ExperimentSpec(
        id="3B",
        name="Signal sensitivity — points only",
        phase="P4",
        student="mask2former",
        labeled_split="labeled_5pct",
        weak_split="weak_95pct",
        use_gnn=True,
        use_distillation=True,
        weak_signal="points_only",
    ),
    "3C": ExperimentSpec(
        id="3C",
        name="Signal sensitivity — mixed (default)",
        phase="P4",
        student="mask2former",
        labeled_split="labeled_5pct",
        weak_split="weak_95pct",
        use_gnn=True,
        use_distillation=True,
        weak_signal="mixed",
    ),
    "4A": ExperimentSpec(
        id="4A",
        name="Cross-architecture — YOLOv8-seg",
        phase="P5",
        student="yolov8",
        labeled_split="labeled_5pct",
        weak_split="weak_95pct",
        use_gnn=True,
        use_distillation=True,
        weak_signal="mixed",
    ),
}

# Default execution order (report/PLAN.md §0.4)
DEFAULT_RUN_ORDER: List[str] = [
    "1C",
    "1A",
    "1B",
    "1D",
    "2A",
    "2B",
    "2C",
    "3A",
    "3B",
    "3C",
    "4A",
]


def get_experiment(exp_id: str) -> ExperimentSpec:
    key = exp_id.upper().replace("EXP", "").strip()
    if key not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment '{exp_id}'. Choose from: {list(EXPERIMENTS)}")
    return EXPERIMENTS[key]
