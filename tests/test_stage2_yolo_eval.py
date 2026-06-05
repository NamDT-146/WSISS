"""Tests for Stage-2 YOLO in-training validation."""

import torch

from modules.wssis.training.stage2_yolo import evaluate_stage2_yolo_val
from modules.wssis.training.stage2_yolo_proto import yolo_mask_logits


def test_evaluate_stage2_yolo_val_with_mock_student():
    proto = torch.randn(1, 32, 16, 16)

    class FakeStudent(torch.nn.Module):
        def _predict_once(self, x):
            return (torch.randn(x.shape[0], 116, 100), torch.randn(x.shape[0], 32, 100), proto)

    student = FakeStudent()
    mask_proj = torch.nn.Conv2d(32, 1, 1)
    device = torch.device("cpu")

    val_records = [
        {
            "file_name": __file__,
            "annotations": [],
        }
    ]
    metrics = evaluate_stage2_yolo_val(
        student,
        mask_proj,
        val_records,
        imgsz=32,
        device=device,
        batch_size=1,
    )
    assert metrics["n_val_images"] == 0.0
    assert metrics["segm/AP"] == 0.0


def test_yolo_mask_logits_shape_for_eval():
    proto = torch.randn(2, 32, 16, 16)

    class FakeStudent(torch.nn.Module):
        def _predict_once(self, x):
            return (torch.randn(x.shape[0], 116, 100), torch.randn(x.shape[0], 32, 100), proto)

    proj = torch.nn.Conv2d(32, 1, 1)
    imgs = torch.randn(2, 3, 32, 32)
    logits = yolo_mask_logits(FakeStudent(), proj, imgs, out_size=32)
    assert logits.shape == (2, 1, 32, 32)
