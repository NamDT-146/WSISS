"""GNN checkpoint shape inference tests."""

import torch

from modules.vig_refinenet.sam_stage1_refiner import (
    infer_num_output_masks_from_state_dict,
    resolve_gnn_build_cfg_from_checkpoint,
)


def test_infer_num_output_masks_three_heads():
    sd = {"mask_head.2.weight": torch.zeros(3, 32, 1, 1)}
    assert infer_num_output_masks_from_state_dict(sd) == 3


def test_infer_num_output_masks_ddp_prefix():
    sd = {"module.mask_head.2.weight": torch.zeros(1, 32, 1, 1)}
    assert infer_num_output_masks_from_state_dict(sd) == 1


def test_resolve_cfg_prefers_state_dict_over_config():
    state = {
        "wssis_ckpt_version": 3,
        "config": {"model": {"num_output_masks": 1}},
        "state_dict": {"mask_head.2.weight": torch.zeros(3, 32, 1, 1)},
    }
    cfg = resolve_gnn_build_cfg_from_checkpoint(state, mask_size=256)
    assert cfg["model"]["num_output_masks"] == 3
