"""
GNN Refinement Network for Weakly Supervised Segmentation.

This package provides:
- Multi-stage GNN refinement model
- Weak signal generators (good/poor mask, box, point)
- DenseCRF baseline
- Training and evaluation utilities
"""

from .model import GNNRefineNet, build_model
from .backbone import ResNetBackbone, build_backbone
from .gnn_stages import GNNStage, FeatureInheritance
from .dataset import (
    OxfordPetBinary,
    WeakSupervisionDataset,
    get_dataloaders,
    get_test_dataloader
)
from .weak_signals import (
    GoodMaskGenerator,
    PoorMaskGenerator,
    BoxGenerator,
    PointGenerator,
    get_weak_signal_generator
)
from .losses import DiceLoss, CombinedLoss, build_loss
from .metrics import compute_iou, compute_dice, compute_metrics, MetricTracker
from .dense_crf import DenseCRF, build_dense_crf, PYDENSECRF_AVAILABLE

__version__ = '0.1.0'

__all__ = [
    # Model
    'GNNRefineNet',
    'build_model',
    'ResNetBackbone',
    'build_backbone',
    'GNNStage',
    'FeatureInheritance',
    
    # Dataset
    'OxfordPetBinary',
    'WeakSupervisionDataset',
    'get_dataloaders',
    'get_test_dataloader',
    
    # Weak signals
    'GoodMaskGenerator',
    'PoorMaskGenerator',
    'BoxGenerator',
    'PointGenerator',
    'get_weak_signal_generator',
    
    # Losses
    'DiceLoss',
    'CombinedLoss',
    'build_loss',
    
    # Metrics
    'compute_iou',
    'compute_dice',
    'compute_metrics',
    'MetricTracker',
    
    # CRF
    'DenseCRF',
    'build_dense_crf',
    'PYDENSECRF_AVAILABLE',
]