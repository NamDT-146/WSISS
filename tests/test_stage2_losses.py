"""Unit tests for Stage-2 loss helpers."""

import torch

from modules.wssis.training.stage2_losses import (
    LossWeightSchedule,
    aggregate_weak_signal_per_image,
    build_pce_valid_mask,
    partial_bce_loss,
    symmetric_sam_triplet_loss,
    voting_pseudo_mask,
)


def test_pce_box_outside_background():
    weak = torch.zeros(1, 3, 8, 8)
    weak[:, 1, 2:6, 2:6] = 1.0  # box channel
    valid, target = build_pce_valid_mask(weak, "boxes_only")
    assert valid[:, :, 0, 0].item() == 1.0
    assert valid[:, :, 3, 3].item() == 0.0
    assert target[:, :, 0, 0].item() == 0.0


def test_voting_two_of_three():
    probs = torch.zeros(1, 3, 4, 4)
    probs[:, 0] = 0.95
    probs[:, 1] = 0.95
    probs[:, 2] = 0.1
    pseudo, valid = voting_pseudo_mask(probs, threshold=0.9, vote_min=2)
    assert pseudo.sum() == 16.0
    assert valid.sum() == 16.0


def test_symmetric_sam_triplet_zero_for_identical():
    m = torch.sigmoid(torch.randn(1, 3, 16, 16))
    m[:, 1] = m[:, 0]
    m[:, 2] = m[:, 0]
    loss = symmetric_sam_triplet_loss(m)
    assert loss.item() < 0.01


def test_loss_schedule_warmup():
    sched = LossWeightSchedule(warmup_frac=0.2)
    w0 = sched.weights(1, 100)
    assert w0["lambda_t_feedback"] == 0.0
    assert w0["lambda_s_unsup"] == 0.0
    w_end = sched.weights(100, 100)
    assert w_end["lambda_s_unsup"] > 0.0


def test_aggregate_weak_signal_multi_instance():
    """Per-instance [N,1,H,W] -> per-image [1,1,H,W] for YOLO semi-loss."""
    weak = torch.zeros(2, 1, 8, 8)
    weak[0, 0, 2, 2] = 1.0
    weak[1, 0, 5, 5] = 1.0
    valid, target = aggregate_weak_signal_per_image(weak, "points_only")
    assert valid.shape == (1, 1, 8, 8)
    assert valid[0, 0, 2, 2] == 1.0
    assert valid[0, 0, 5, 5] == 1.0
    bce = partial_bce_loss(torch.zeros(1, 1, 8, 8), target, valid)
    assert bce.ndim == 0


def test_partial_bce_three_channel_logits():
    """Legacy 3-head GNN logits are collapsed to 1 ch for PCE."""
    logits = torch.zeros(2, 3, 4, 4)
    target = torch.ones(2, 1, 4, 4)
    valid = torch.ones(2, 1, 4, 4)
    loss = partial_bce_loss(logits, target, valid)
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_partial_bce_masked():
    logits = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    valid = torch.zeros(1, 1, 4, 4)
    valid[:, :, 0, 0] = 1.0
    loss = partial_bce_loss(logits, target, valid)
    assert loss.ndim == 0
    assert loss.item() > 0.0
