"""
Unified weak signal generator for training and type-specific generators for evaluation.

Training: Uses UnifiedWeakSignalGenerator that samples across all corruption levels
Evaluation: Uses type-specific generators (GoodMask, PoorMask, Box, Point) for separate reporting
"""

import torch
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional
import cv2


class WeakSignalGenerator:
    """Base class for weak signal generation."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, gt_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UnifiedWeakSignalGenerator(WeakSignalGenerator):
    """
    Unified generator that samples across all corruption levels.
    
    Randomly selects from:
    - Good masks (mild corruption, IoU ~80-90%)
    - Poor masks (moderate corruption, IoU ~60-80%)
    - Very poor masks (heavy corruption, IoU ~40-60%)
    - Bounding boxes
    - Point supervision
    
    This ensures the model learns to handle the full spectrum of weak supervision.
    """
    
    def __init__(
        self,
        good_mask_prob: float = 0.25,
        poor_mask_prob: float = 0.25,
        very_poor_mask_prob: float = 0.25,
        box_prob: float = 0.15,
        point_prob: float = 0.10,
        seed: Optional[int] = None
    ):
        super().__init__(seed)
        
        # Normalize probabilities
        total = good_mask_prob + poor_mask_prob + very_poor_mask_prob + box_prob + point_prob
        self.good_mask_prob = good_mask_prob / total
        self.poor_mask_prob = poor_mask_prob / total
        self.very_poor_mask_prob = very_poor_mask_prob / total
        self.box_prob = box_prob / total
        self.point_prob = point_prob / total
        
        # Initialize sub-generators
        self.good_mask_gen = GoodMaskGenerator(seed=seed)
        self.poor_mask_gen = PoorMaskGenerator(
            erosion_range=(2, 5),
            dilation_range=(2, 5),
            dropout_prob=0.2,
            noise_prob=0.2,
            seed=seed
        )
        self.very_poor_mask_gen = PoorMaskGenerator(
            erosion_range=(5, 10),
            dilation_range=(5, 10),
            dropout_prob=0.4,
            noise_prob=0.4,
            seed=seed
        )
        self.box_gen = BoxGenerator(seed=seed)
        self.point_gen = PointGenerator(seed=seed)
    
    def __call__(self, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate weak signal by randomly sampling corruption type.
        
        Args:
            gt_mask: Ground truth mask [1, H, W] or [H, W]
            
        Returns:
            Weak supervision mask [1, H, W]
        """
        # Sample corruption type
        rand_val = self.rng.random()
        
        if rand_val < self.good_mask_prob:
            return self.good_mask_gen(gt_mask)
        elif rand_val < self.good_mask_prob + self.poor_mask_prob:
            return self.poor_mask_gen(gt_mask)
        elif rand_val < self.good_mask_prob + self.poor_mask_prob + self.very_poor_mask_prob:
            return self.very_poor_mask_gen(gt_mask)
        elif rand_val < self.good_mask_prob + self.poor_mask_prob + self.very_poor_mask_prob + self.box_prob:
            return self.box_gen(gt_mask)
        else:
            return self.point_gen(gt_mask)


class GoodMaskGenerator(WeakSignalGenerator):
    """
    Generate mildly corrupted mask (target IoU: 80-90%).
    
    Operations:
    - Mild erosion/dilation
    - Boundary noise
    """
    
    def __init__(
        self,
        erosion_range: Tuple[int, int] = (1, 3),
        dilation_range: Tuple[int, int] = (1, 3),
        noise_prob: float = 0.1,
        seed: Optional[int] = None
    ):
        super().__init__(seed)
        self.erosion_range = erosion_range
        self.dilation_range = dilation_range
        self.noise_prob = noise_prob
    
    def __call__(self, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate good (mildly corrupted) mask.
        
        Args:
            gt_mask: Ground truth mask [1, H, W] or [H, W]
            
        Returns:
            Corrupted mask [1, H, W]
        """
        if gt_mask.dim() == 3:
            mask = gt_mask[0].numpy()
        else:
            mask = gt_mask.numpy()
        
        mask = mask.astype(np.float32)
        
        # Random erosion or dilation
        if self.rng.random() > 0.5:
            kernel_size = self.rng.randint(*self.erosion_range)
            if kernel_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size*2+1, kernel_size*2+1))
                mask = cv2.erode(mask, kernel)
        else:
            kernel_size = self.rng.randint(*self.dilation_range)
            if kernel_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size*2+1, kernel_size*2+1))
                mask = cv2.dilate(mask, kernel)
        
        # Add boundary noise
        if self.rng.random() < self.noise_prob:
            boundary = cv2.Laplacian(mask, cv2.CV_32F)
            boundary = np.abs(boundary) > 0
            
            noise = self.rng.random(mask.shape) < 0.3
            flip_mask = boundary & noise
            mask = np.where(flip_mask, 1 - mask, mask)
        
        result = torch.from_numpy(mask).float().unsqueeze(0)
        return result


class PoorMaskGenerator(WeakSignalGenerator):
    """
    Generate heavily corrupted mask (target IoU: 40-60%).
    
    Operations:
    - Aggressive erosion/dilation
    - Random region dropout
    - Heavy boundary noise
    """
    
    def __init__(
        self,
        erosion_range: Tuple[int, int] = (3, 7),
        dilation_range: Tuple[int, int] = (3, 7),
        dropout_prob: float = 0.3,
        noise_prob: float = 0.3,
        seed: Optional[int] = None
    ):
        super().__init__(seed)
        self.erosion_range = erosion_range
        self.dilation_range = dilation_range
        self.dropout_prob = dropout_prob
        self.noise_prob = noise_prob
    
    def __call__(self, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate poor (heavily corrupted) mask.
        
        Args:
            gt_mask: Ground truth mask [1, H, W] or [H, W]
            
        Returns:
            Corrupted mask [1, H, W]
        """
        if gt_mask.dim() == 3:
            mask = gt_mask[0].numpy()
        else:
            mask = gt_mask.numpy()
        
        mask = mask.astype(np.float32)
        
        # Aggressive morphological operations
        if self.rng.random() > 0.5:
            kernel_size = self.rng.randint(*self.erosion_range)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size*2+1, kernel_size*2+1))
            mask = cv2.erode(mask, kernel)
        else:
            kernel_size = self.rng.randint(*self.dilation_range)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size*2+1, kernel_size*2+1))
            mask = cv2.dilate(mask, kernel)
        
        # Random region dropout
        if self.rng.random() < self.dropout_prob:
            H, W = mask.shape
            num_drops = self.rng.randint(1, 4)
            for _ in range(num_drops):
                cx, cy = self.rng.randint(0, W), self.rng.randint(0, H)
                radius = self.rng.randint(10, 30)
                y, x = np.ogrid[:H, :W]
                drop_mask = (x - cx)**2 + (y - cy)**2 <= radius**2
                if self.rng.random() > 0.5:
                    mask[drop_mask] = 0
                else:
                    mask[drop_mask] = 1
        
        # Heavy boundary noise
        noise = self.rng.random(mask.shape) < self.noise_prob
        boundary = cv2.Laplacian(mask, cv2.CV_32F)
        boundary = np.abs(boundary) > 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel).astype(bool)
        
        flip_mask = boundary & noise
        mask = np.where(flip_mask, 1 - mask, mask)
        
        result = torch.from_numpy(mask).float().unsqueeze(0)
        return result


class BoxGenerator(WeakSignalGenerator):
    """
    Generate bounding box supervision.
    
    Creates a mask where:
    - Inside box: foreground (1)
    - Outside box: background (0)
    """
    
    def __init__(
        self,
        expansion_ratio: Tuple[float, float] = (0.0, 0.2),
        seed: Optional[int] = None
    ):
        super().__init__(seed)
        self.expansion_ratio = expansion_ratio
    
    def __call__(self, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate box supervision mask.
        
        Args:
            gt_mask: Ground truth mask [1, H, W] or [H, W]
            
        Returns:
            Box mask [1, H, W]
        """
        if gt_mask.dim() == 3:
            mask = gt_mask[0].numpy()
        else:
            mask = gt_mask.numpy()
        
        H, W = mask.shape
        
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return torch.zeros(1, H, W)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        expansion = self.rng.uniform(*self.expansion_ratio)
        box_h = y_max - y_min
        box_w = x_max - x_min
        
        y_min = max(0, int(y_min - box_h * expansion))
        y_max = min(H - 1, int(y_max + box_h * expansion))
        x_min = max(0, int(x_min - box_w * expansion))
        x_max = min(W - 1, int(x_max + box_w * expansion))
        
        box_mask = np.zeros((H, W), dtype=np.float32)
        box_mask[y_min:y_max+1, x_min:x_max+1] = 1.0
        
        result = torch.from_numpy(box_mask).float().unsqueeze(0)
        return result


class PointGenerator(WeakSignalGenerator):
    """
    Generate point supervision.
    
    Creates a mask with Gaussian blobs at foreground points.
    """
    
    def __init__(
        self,
        num_points: Tuple[int, int] = (1, 3),
        sigma: float = 5.0,
        seed: Optional[int] = None
    ):
        super().__init__(seed)
        self.num_points = num_points
        self.sigma = sigma
    
    def __call__(self, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate point supervision mask.
        
        Args:
            gt_mask: Ground truth mask [1, H, W] or [H, W]
            
        Returns:
            Point mask [1, H, W] with Gaussian blobs
        """
        if gt_mask.dim() == 3:
            mask = gt_mask[0].numpy()
        else:
            mask = gt_mask.numpy()
        
        H, W = mask.shape
        
        fg_coords = np.where(mask > 0.5)
        if len(fg_coords[0]) == 0:
            return torch.zeros(1, H, W)
        
        num_pts = self.rng.randint(*self.num_points)
        num_pts = min(num_pts, len(fg_coords[0]))
        
        indices = self.rng.choice(len(fg_coords[0]), size=num_pts, replace=False)
        points = [(fg_coords[0][i], fg_coords[1][i]) for i in indices]
        
        point_mask = np.zeros((H, W), dtype=np.float32)
        y, x = np.ogrid[:H, :W]
        
        for py, px in points:
            gaussian = np.exp(-((x - px)**2 + (y - py)**2) / (2 * self.sigma**2))
            point_mask = np.maximum(point_mask, gaussian)
        
        point_mask = point_mask / (point_mask.max() + 1e-8)
        
        result = torch.from_numpy(point_mask).float().unsqueeze(0)
        return result


def get_weak_signal_generator(signal_type: str, config: dict, seed: Optional[int] = None):
    """
    Factory function for weak signal generators.
    
    Args:
        signal_type: 'unified' for training, or 'good_mask', 'poor_mask', 'box', 'point' for evaluation
        config: Configuration dictionary
        seed: Random seed
        
    Returns:
        WeakSignalGenerator instance
    """
    weak_cfg = config.get('weak_signal', {})
    
    if signal_type == 'unified':
        cfg = weak_cfg.get('unified', {})
        return UnifiedWeakSignalGenerator(
            good_mask_prob=cfg.get('good_mask_prob', 0.25),
            poor_mask_prob=cfg.get('poor_mask_prob', 0.25),
            very_poor_mask_prob=cfg.get('very_poor_mask_prob', 0.25),
            box_prob=cfg.get('box_prob', 0.15),
            point_prob=cfg.get('point_prob', 0.10),
            seed=seed
        )
    elif signal_type == 'good_mask':
        cfg = weak_cfg.get('good_mask', {})
        return GoodMaskGenerator(
            erosion_range=tuple(cfg.get('erosion_range', [1, 3])),
            dilation_range=tuple(cfg.get('dilation_range', [1, 3])),
            noise_prob=cfg.get('noise_prob', 0.1),
            seed=seed
        )
    elif signal_type == 'poor_mask':
        cfg = weak_cfg.get('poor_mask', {})
        return PoorMaskGenerator(
            erosion_range=tuple(cfg.get('erosion_range', [3, 7])),
            dilation_range=tuple(cfg.get('dilation_range', [3, 7])),
            dropout_prob=cfg.get('dropout_prob', 0.3),
            noise_prob=cfg.get('noise_prob', 0.3),
            seed=seed
        )
    elif signal_type == 'box':
        cfg = weak_cfg.get('box', {})
        return BoxGenerator(
            expansion_ratio=tuple(cfg.get('expansion_ratio', [0.0, 0.2])),
            seed=seed
        )
    elif signal_type == 'point':
        cfg = weak_cfg.get('point', {})
        return PointGenerator(
            num_points=tuple(cfg.get('num_points', [1, 3])),
            sigma=cfg.get('sigma', 5.0),
            seed=seed
        )
    else:
        raise ValueError(f"Unknown weak signal type: {signal_type}")