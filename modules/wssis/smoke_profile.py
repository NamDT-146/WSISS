"""Centralized smoke-test limits (target <10 min on 1 GPU)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SmokeProfile:
    max_images: int = 2
    max_objects_per_image: int = 3
    batch_size: int = 2
    stage1_epochs: int = 1
    stage1_max_steps: int = 2
    m2f_max_iter: int = 10
    m2f_eval_period: int = 10
    m2f_use_full_val_final: bool = False
    m2f_image_size: int = 512
    yolo_epochs: int = 1
    yolo_max_batches: int = 3
    viz_samples: int = 1
    teacher_max_images: int = 2
    num_gpus: int = 1


def is_smoke_mode() -> bool:
    return os.environ.get("WSSIS_SMOKE", "").strip() in ("1", "true", "yes")


def smoke_run_id(default: Optional[str] = None) -> str:
    if default:
        return default
    return os.environ.get("WSSIS_RUN_ID", "smoke_quick")


def get_smoke_profile() -> Optional[SmokeProfile]:
    if not is_smoke_mode():
        return None
    return SmokeProfile()


def apply_smoke_env() -> None:
    """Set GPU count for smoke runs."""
    if is_smoke_mode():
        os.environ.setdefault("WSSIS_NUM_GPUS", "1")
