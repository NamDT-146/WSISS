"""Detectron2 / Mask2Former config hooks for WSSIS experiments."""

from __future__ import annotations

import os

from detectron2.config import CfgNode as CN

from modules.wssis.stage2_constants import STAGE2_STUDENT_IMAGE_SIZE


def resolve_wssis_num_gpus(explicit: int | None = None) -> int:
    """GPU count from launch arg or WSSIS_NUM_GPUS (fallback: visible CUDA devices)."""
    if explicit is not None and explicit > 0:
        return int(explicit)
    env = os.environ.get("WSSIS_NUM_GPUS", "").strip()
    if env.isdigit():
        return max(1, int(env))
    try:
        import torch

        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
    except ImportError:
        pass
    return 1


def align_ims_per_batch(total_batch: int, num_gpus: int) -> int:
    """Detectron2 requires SOLVER.IMS_PER_BATCH divisible by world_size."""
    num_gpus = max(1, int(num_gpus))
    total_batch = max(num_gpus, int(total_batch))
    remainder = total_batch % num_gpus
    if remainder == 0:
        return total_batch
    return total_batch - remainder


def add_wssis_config(cfg: CN) -> None:
    """Register WSSIS keys so override YAML and CLI opts can merge safely."""
    cfg.WSSIS = CN()
    cfg.WSSIS.EXPERIMENT_ID = ""
    cfg.WSSIS.IMAGE_LIST = ""
    cfg.WSSIS.LABELED_SPLIT = ""
    cfg.WSSIS.WEAK_SPLIT = ""
    cfg.WSSIS.USE_GNN = False
    cfg.WSSIS.USE_DISTILL = False
    cfg.WSSIS.USE_SEMI_WEAK = False
    cfg.WSSIS.USE_RAW_SAM_ONLY = False
    cfg.WSSIS.WEAK_SIGNAL = "none"
    cfg.WSSIS.USE_FULL_VAL_FINAL = True
    cfg.WSSIS.ITERS_PER_EPOCH = 1000
    cfg.WSSIS.EARLY_STOPPING_PATIENCE = 5
    cfg.WSSIS.EARLY_STOPPING_MONITOR = "segm/AP"
    cfg.WSSIS.GNN_CHECKPOINT = ""
    cfg.WSSIS.FREEZE_GNN = False
    cfg.WSSIS.GNN_LR = 1e-5
    cfg.WSSIS.DISTILL_WEIGHT = 1.0
    cfg.WSSIS.LABELED_BATCH_RATIO = 0.5
    cfg.WSSIS.SMOKE = False
    cfg.WSSIS.DISTILL_BACKBONE_FEAT = "res4"
    cfg.WSSIS.DISTILL_FEAT_DIM = 0  # 0 = auto from backbone channels
    cfg.WSSIS.STUDENT_IMAGE_SIZE = STAGE2_STUDENT_IMAGE_SIZE


def apply_smoke_to_cfg(cfg: CN) -> None:
    from modules.wssis.smoke_profile import get_smoke_profile

    smoke = get_smoke_profile()
    if smoke is None:
        return
    cfg.WSSIS.SMOKE = True
    cfg.SOLVER.MAX_ITER = smoke.m2f_max_iter
    cfg.TEST.EVAL_PERIOD = smoke.m2f_eval_period
    cfg.WSSIS.USE_FULL_VAL_FINAL = smoke.m2f_use_full_val_final
    cfg.WSSIS.EARLY_STOPPING_PATIENCE = 0
    cfg.SOLVER.IMS_PER_BATCH = align_ims_per_batch(
        max(2, smoke.batch_size),
        resolve_wssis_num_gpus(),
    )
    if hasattr(cfg.INPUT, "IMAGE_SIZE"):
        cfg.INPUT.IMAGE_SIZE = smoke.m2f_image_size


def apply_gpu_batch_alignment(cfg: CN, num_gpus: int | None = None) -> None:
    """Ensure IMS_PER_BATCH is divisible by the distributed world size."""
    if not hasattr(cfg, "SOLVER") or not hasattr(cfg.SOLVER, "IMS_PER_BATCH"):
        return
    ng = resolve_wssis_num_gpus(num_gpus)
    current = int(cfg.SOLVER.IMS_PER_BATCH)
    aligned = align_ims_per_batch(current, ng)
    if aligned != current:
        import logging

        logging.getLogger("mask2former").info(
            "Adjusted SOLVER.IMS_PER_BATCH %d -> %d for %d GPU(s)",
            current,
            aligned,
            ng,
        )
        cfg.SOLVER.IMS_PER_BATCH = aligned
