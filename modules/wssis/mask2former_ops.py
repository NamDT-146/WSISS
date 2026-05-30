"""Mask2Former MultiScaleDeformableAttention CUDA extension helpers."""


def verify_msda_import() -> None:
    """
    Verify MSDeformAttn ops are importable.

    torch must be imported first — the .so links against libtorch and fails
    with a misleading ImportError if torch is not loaded yet.
    """
    import torch  # noqa: F401

    import MultiScaleDeformableAttention  # noqa: F401
