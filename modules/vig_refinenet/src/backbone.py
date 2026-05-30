"""
ResNet-50 backbone with intermediate feature extraction.

Returns multi-scale features for GNN stages:
- feat4: 7x7, 2048-dim (Stage 1)
- feat2: 28x28, 512-dim (Stage 2)
- feat0: 112x112, 64-dim (Stage 3)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict


class ResNetBackbone(nn.Module):
    """
    ResNet-50 backbone with multi-scale feature extraction.
    
    For input 224x224:
        feat0: [B, 64, 112, 112]   -> Stage 3
        feat1: [B, 256, 56, 56]
        feat2: [B, 512, 28, 28]    -> Stage 2
        feat3: [B, 1024, 14, 14]
        feat4: [B, 2048, 7, 7]     -> Stage 1
    """
    
    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        
        # Load pretrained ResNet-50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.resnet = resnet50(weights=weights)
        else:
            self.resnet = resnet50(weights=None)
        
        # Remove FC layer
        del self.resnet.fc
        
        # Optionally freeze backbone
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input image [B, 3, 224, 224]
            
        Returns:
            Dictionary of feature maps
        """
        features = {}
        
        # Initial convolution
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        features['feat0'] = x  # [B, 64, 112, 112]
        
        x = self.resnet.maxpool(x)
        
        # Layer 1
        x = self.resnet.layer1(x)
        features['feat1'] = x  # [B, 256, 56, 56]
        
        # Layer 2
        x = self.resnet.layer2(x)
        features['feat2'] = x  # [B, 512, 28, 28]
        
        # Layer 3
        x = self.resnet.layer3(x)
        features['feat3'] = x  # [B, 1024, 14, 14]
        
        # Layer 4
        x = self.resnet.layer4(x)
        features['feat4'] = x  # [B, 2048, 7, 7]
        
        return features


def build_backbone(config: dict) -> ResNetBackbone:
    """Build backbone from config."""
    model_cfg = config.get('model', {})
    return ResNetBackbone(
        pretrained=model_cfg.get('pretrained', True),
        freeze=model_cfg.get('freeze_backbone', False)
    )