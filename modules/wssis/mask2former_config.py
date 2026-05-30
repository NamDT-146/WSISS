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
    cfg.WSSIS.WEAK_SIGNAL = "none"
