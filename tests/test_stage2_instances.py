"""Stage-2 Instances helpers for Mask2Former joint training."""

import numpy as np
import torch

from modules.wssis.training.stage2_trainer import _empty_instances, _masks_to_instances


def test_empty_instances_has_gt_masks():
    inst = _empty_instances(64, 64, torch.device("cpu"))
    assert inst.has("gt_masks")
    assert inst.has("gt_classes")
    assert inst.gt_masks.shape == (0, 64, 64)


def test_masks_to_instances_all_filtered():
    inst = _masks_to_instances([np.zeros((8, 8), dtype=np.uint8)], [1], (8, 8), torch.device("cpu"))
    assert inst.has("gt_masks")
    assert inst.gt_masks.shape[0] == 0
