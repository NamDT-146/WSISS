"""Shared Stage-2 student / teacher resolution constants."""

from __future__ import annotations

# Student training resolution (Mask2Former LSJ crop + YOLO imgsz).
STAGE2_STUDENT_IMAGE_SIZE = 512

# SAM ViT-B teacher + P0.2 embedding cache (fixed; do not change without re-running P0.2).
SAM_TEACHER_IMAGE_SIZE = 1024

# SAM image-encoder output spatial size (stride 16 on 1024 input).
SAM_EMBED_SPATIAL = 64
