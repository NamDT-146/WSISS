"""Unit tests for weak-signal jitter helpers."""

import numpy as np

from modules.wssis.weak_prompts import (
    jitter_box_expand,
    jitter_point_manhattan,
    jitter_scribble_trim,
    mask_to_bbox_xywh,
)


def test_box_expand_only():
    bbox = [10.0, 10.0, 100.0, 50.0]
    out = jitter_box_expand(bbox, expand_ratio=0.05)
    assert out[0] < bbox[0]
    assert out[1] < bbox[1]
    assert out[2] > bbox[2]
    assert out[3] > bbox[3]


def test_point_jitter_within_radius():
    rng = np.random.RandomState(0)
    px, py = jitter_point_manhattan((50.0, 50.0), radius=5, rng=rng, img_hw=(100, 100))
    assert abs(px - 50.0) <= 5
    assert abs(py - 50.0) <= 5


def test_scribble_trim_reduces_pixels():
    scrib = np.zeros((32, 32), dtype=np.uint8)
    scrib[16, 8:24] = 1
    rng = np.random.RandomState(42)
    trimmed = jitter_scribble_trim(scrib, trim_ratio=0.2, rng=rng)
    assert trimmed.sum() < scrib.sum()
    assert trimmed.sum() > 0
