"""
Loss functions for mask refinement.

Combined BCE + Dice loss for binary segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predictions (logits) [B, 1, H, W]
            target: Ground truth [B, 1, H, W]
            
        Returns:
            Dice loss (scalar)
        """
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss.
    
    L = bce_weight * BCE + dice_weight * Dice
    """
    
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predictions (logits) [B, 1, H, W]
            target: Ground truth [B, 1, H, W]
            
        Returns:
            Combined loss (scalar)
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def build_loss(config: dict) -> CombinedLoss:
    """Build loss function from config."""
    training_cfg = config.get('training', {})
    return CombinedLoss(
        bce_weight=training_cfg.get('bce_weight', 1.0),
        dice_weight=training_cfg.get('dice_weight', 1.0)
    )