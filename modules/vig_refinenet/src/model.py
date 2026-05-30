"""
Full GNN Refinement Model.

Architecture:
    Backbone -> Stage1 (7x7) -> Inherit -> Stage2 (28x28) -> Inherit -> Stage3 (112x112) -> Mask Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .backbone import ResNetBackbone
from .gnn_stages import GNNStage, FeatureInheritance


class GNNRefineNet(nn.Module):
    """
    Multi-stage GNN refinement network for mask refinement.
    
    Takes an image and optional weak mask, outputs refined segmentation.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        model_cfg = config.get('model', {})
        stage1_cfg = model_cfg.get('stage1', {})
        stage2_cfg = model_cfg.get('stage2', {})
        stage3_cfg = model_cfg.get('stage3', {})
        
        self.img_size = model_cfg.get('img_size', 224)
        
        # Backbone
        self.backbone = ResNetBackbone(
            pretrained=model_cfg.get('pretrained', True),
            freeze=model_cfg.get('freeze_backbone', False)
        )
        
        # Weak signal encoder (optional)
        self.weak_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion for incorporating weak signal
        self.feat0_fusion = nn.Conv2d(64 + 64, 64, 1)
        self.feat2_fusion = nn.Conv2d(512 + 64, 512, 1)
        self.feat4_fusion = nn.Conv2d(2048 + 64, 2048, 1)
        
        # Stage 1: Global reasoning (7x7)
        self.stage1 = GNNStage(
            in_dim=stage1_cfg.get('in_dim', 2048),
            hidden_dim=stage1_cfg.get('hidden_dim', 512),
            out_dim=stage1_cfg.get('out_dim', 512),
            connectivity=stage1_cfg.get('connectivity', 'full'),
            k_neighbors=8
        )
        
        # Inheritance 1->2
        self.inherit1 = FeatureInheritance(
            child_dim=512,  # feat2 dim
            parent_dim=512,  # stage1 output
            out_dim=512
        )
        
        # Stage 2: Mid-level local reasoning (28x28)
        self.stage2 = GNNStage(
            in_dim=stage2_cfg.get('in_dim', 512),
            hidden_dim=stage2_cfg.get('hidden_dim', 256),
            out_dim=stage2_cfg.get('out_dim', 256),
            connectivity=stage2_cfg.get('connectivity', 'local'),
            k_neighbors=stage2_cfg.get('k_neighbors', 8)
        )
        
        # Inheritance 2->3
        self.inherit2 = FeatureInheritance(
            child_dim=64,   # feat0 dim
            parent_dim=256,  # stage2 output
            out_dim=64
        )
        
        # Stage 3: Fine alignment (112x112)
        self.stage3 = GNNStage(
            in_dim=stage3_cfg.get('in_dim', 64),
            hidden_dim=stage3_cfg.get('hidden_dim', 64),
            out_dim=stage3_cfg.get('out_dim', 64),
            connectivity=stage3_cfg.get('connectivity', 'grid'),
            k_neighbors=8
        )
        
        # Mask prediction head
        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
    
    def _encode_weak_signal(self, weak_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode weak signal at multiple scales."""
        # weak_mask: [B, 1, H, W]
        weak_feat = self.weak_encoder(weak_mask)  # [B, 64, H, W]
        
        return {
            'scale_112': F.interpolate(weak_feat, size=(112, 112), mode='bilinear', align_corners=False),
            'scale_28': F.interpolate(weak_feat, size=(28, 28), mode='bilinear', align_corners=False),
            'scale_7': F.interpolate(weak_feat, size=(7, 7), mode='bilinear', align_corners=False),
        }
    
    def forward(self, x: torch.Tensor, weak_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, 3, 224, 224]
            weak_mask: Optional weak supervision mask [B, 1, 224, 224]
            
        Returns:
            Refined mask [B, 1, 224, 224]
        """
        B = x.shape[0]
        
        # Extract backbone features
        feats = self.backbone(x)
        feat0 = feats['feat0']  # [B, 64, 112, 112]
        feat2 = feats['feat2']  # [B, 512, 28, 28]
        feat4 = feats['feat4']  # [B, 2048, 7, 7]
        
        # Fuse weak signal if provided
        if weak_mask is not None:
            weak_feats = self._encode_weak_signal(weak_mask)
            feat0 = self.feat0_fusion(torch.cat([feat0, weak_feats['scale_112']], dim=1))
            feat2 = self.feat2_fusion(torch.cat([feat2, weak_feats['scale_28']], dim=1))
            feat4 = self.feat4_fusion(torch.cat([feat4, weak_feats['scale_7']], dim=1))
        
        # Stage 1: Global GNN (7x7)
        f1 = feat4.flatten(2).transpose(1, 2)  # [B, 49, 2048]
        f1 = self.stage1(f1, h=7, w=7)  # [B, 49, 512]
        f1_spatial = f1.transpose(1, 2).reshape(B, 512, 7, 7)  # [B, 512, 7, 7]
        
        # Inheritance 1 -> 2
        f2_input = self.inherit1(feat2, f1_spatial)  # [B, 512, 28, 28]
        
        # Stage 2: Local GNN (28x28)
        f2 = f2_input.flatten(2).transpose(1, 2)  # [B, 784, 512]
        f2 = self.stage2(f2, h=28, w=28)  # [B, 784, 256]
        f2_spatial = f2.transpose(1, 2).reshape(B, 256, 28, 28)  # [B, 256, 28, 28]
        
        # Inheritance 2 -> 3
        f3_input = self.inherit2(feat0, f2_spatial)  # [B, 64, 112, 112]
        
        # Stage 3: Fine GNN (112x112)
        f3 = f3_input.flatten(2).transpose(1, 2)  # [B, 12544, 64]
        f3 = self.stage3(f3, h=112, w=112)  # [B, 12544, 64]
        f3_spatial = f3.transpose(1, 2).reshape(B, 64, 112, 112)  # [B, 64, 112, 112]
        
        # Mask prediction
        mask_logits = self.mask_head(f3_spatial)  # [B, 1, 112, 112]
        
        # Upsample to original size
        mask_logits = F.interpolate(mask_logits, size=(self.img_size, self.img_size), 
                                    mode='bilinear', align_corners=False)
        
        return mask_logits


def build_model(config: dict) -> GNNRefineNet:
    """Build model from config."""
    return GNNRefineNet(config)