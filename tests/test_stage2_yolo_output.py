"""Tests for YOLOv8-seg prototype extraction helpers."""

import torch

from modules.wssis.training.stage2_yolo_proto import (
    denorm_yolo_images,
    find_proto_tensor,
    yolo_mask_logits,
)


def test_find_proto_from_training_tuple():
    proto = torch.randn(2, 32, 40, 40)
    det = torch.randn(2, 116, 8400)
    mc = torch.randn(2, 32, 8400)
    found = find_proto_tensor((det, mc, proto))
    assert found is not None
    assert found.shape == proto.shape


def test_find_proto_from_nested_tuple():
    proto = torch.randn(1, 32, 64, 64)
    out = (torch.randn(1, 100, 10), proto)
    assert find_proto_tensor(out).shape == proto.shape


def test_denorm_clamps_to_unit_interval():
    x = torch.zeros(1, 3, 8, 8) - 5.0
    y = denorm_yolo_images(x)
    assert y.min() >= 0.0
    assert y.max() <= 1.0


def test_yolo_mask_logits_with_mock_student():
    proto = torch.randn(2, 32, 16, 16, requires_grad=True)

    class FakeStudent(torch.nn.Module):
        def _predict_once(self, x):
            return (torch.randn(2, 116, 100), torch.randn(2, 32, 100), proto)

    proj = torch.nn.Conv2d(32, 1, 1)
    imgs = torch.randn(2, 3, 32, 32)
    logits = yolo_mask_logits(FakeStudent(), proj, imgs, out_size=32)
    assert logits.shape == (2, 1, 32, 32)
    loss = logits.mean()
    loss.backward()
    assert proto.grad is not None
