"""
Evaluation metrics for segmentation.

IoU (Intersection over Union) and Dice coefficient.
"""

import torch
import numpy as np
from typing import Dict, Tuple


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute IoU (Intersection over Union).
    
    Args:
        pred: Predictions [B, 1, H, W] or [1, H, W] (probabilities or logits)
        target: Ground truth [B, 1, H, W] or [1, H, W]
        threshold: Threshold for binarization
        
    Returns:
        IoU score
    """
    # Apply sigmoid if logits
    if pred.min() < 0:
        pred = torch.sigmoid(pred)
    
    # Binarize
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    intersection = (pred_flat * target_flat).sum().item()
    union = pred_flat.sum().item() + target_flat.sum().item() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Dice coefficient.
    
    Args:
        pred: Predictions [B, 1, H, W] or [1, H, W]
        target: Ground truth [B, 1, H, W] or [1, H, W]
        threshold: Threshold for binarization
        
    Returns:
        Dice score
    """
    # Apply sigmoid if logits
    if pred.min() < 0:
        pred = torch.sigmoid(pred)
    
    # Binarize
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    intersection = (pred_flat * target_flat).sum().item()
    total = pred_flat.sum().item() + target_flat.sum().item()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / total


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute all metrics.
    
    Args:
        pred: Predictions
        target: Ground truth
        
    Returns:
        Dictionary with 'iou' and 'dice' scores
    """
    return {
        'iou': compute_iou(pred, target),
        'dice': compute_dice(pred, target)
    }


class MetricTracker:
    """Track metrics over batches."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.iou_sum = 0.0
        self.dice_sum = 0.0
        self.count = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with a batch."""
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            metrics = compute_metrics(pred[i:i+1], target[i:i+1])
            self.iou_sum += metrics['iou']
            self.dice_sum += metrics['dice']
            self.count += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics."""
        if self.count == 0:
            return {'iou': 0.0, 'dice': 0.0}
        
        return {
            'iou': self.iou_sum / self.count,
            'dice': self.dice_sum / self.count
        }