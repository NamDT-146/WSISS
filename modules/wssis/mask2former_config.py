"""Detectron2 / Mask2Former config hooks for WSSIS experiments."""

from __future__ import annotations

from detectron2.config import CfgNode as CN


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
    cfg.WSSIS.EARLY_STOPPING_PATIENCE = 10
    cfg.WSSIS.EARLY_STOPPING_MONITOR = "segm/AP"
    cfg.WSSIS.GNN_CHECKPOINT = ""
    cfg.WSSIS.FREEZE_GNN = False
    cfg.WSSIS.GNN_LR = 1e-5
    cfg.WSSIS.DISTILL_WEIGHT = 1.0
    cfg.WSSIS.LABELED_BATCH_RATIO = 0.5
    cfg.WSSIS.SMOKE = False
    cfg.WSSIS.DISTILL_BACKBONE_FEAT = "res4"
    cfg.WSSIS.DISTILL_FEAT_DIM = 0  # 0 = auto from backbone channels


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
    cfg.SOLVER.IMS_PER_BATCH = max(2, smoke.batch_size)
    if hasattr(cfg.INPUT, "IMAGE_SIZE"):
        cfg.INPUT.IMAGE_SIZE = smoke.m2f_image_size
