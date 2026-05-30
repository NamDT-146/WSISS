"""
Dataset utilities for Oxford-IIIT Pet Dataset with weak signal generation.

Training: Uses unified weak signal generator that covers all corruption levels
Evaluation: Uses type-specific generators for separate reporting
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from typing import Optional, Tuple, List
import os

from .weak_signals import get_weak_signal_generator


class OxfordPetBinary(datasets.OxfordIIITPet):
    """
    Oxford-IIIT Pet Dataset with binary segmentation masks.
    
    The original masks have values:
        1: Pet (Foreground)
        2: Background
        3: Border/Trimap
    
    We convert to binary: 1 = Pet, 0 = Background (including border).
    """
    
    def __init__(
        self,
        root: str = './data',
        split: str = 'trainval',
        img_size: int = 224,
        augment: bool = False,
        download: bool = False
    ):
        super().__init__(
            root=root,
            split=split,
            target_types='segmentation',
            download=download
        )
        self.img_size = img_size
        self.augment = augment
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample."""
        img, mask = super().__getitem__(idx)
        
        # Apply augmentation if enabled
        if self.augment:
            img, mask = self._augment(img, mask)
        
        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # Convert to tensor
        img = transforms.ToTensor()(img)
        mask = torch.from_numpy(np.array(mask)).long()
        
        # Binary mask: Pet(1) -> 1, else -> 0
        mask_binary = torch.where(mask == 1, 1.0, 0.0).float()
        mask_binary = mask_binary.unsqueeze(0)  # [1, H, W]
        
        # Normalize image
        img = self.normalize(img)
        
        return img, mask_binary
    
    def _augment(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Apply data augmentation."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
        
        return img, mask


class WeakSupervisionDataset(Dataset):
    """
    Wrapper dataset that adds weak supervision signals.
    
    Returns: (image, gt_mask, weak_mask)
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        weak_signal_type: str,
        config: dict,
        seed: Optional[int] = None
    ):
        self.base_dataset = base_dataset
        self.weak_signal_type = weak_signal_type
        self.generator = get_weak_signal_generator(weak_signal_type, config, seed)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get sample with weak supervision.
        
        Returns:
            img: Image tensor [3, H, W]
            gt_mask: Ground truth mask [1, H, W]
            weak_mask: Weak supervision mask [1, H, W]
        """
        img, gt_mask = self.base_dataset[idx]
        weak_mask = self.generator(gt_mask)
        return img, gt_mask, weak_mask


def create_train_val_split(
    dataset: Dataset,
    train_ratio: float = 0.75,
    seed: int = 42
) -> Tuple[Subset, Subset]:
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset: Full dataset
        train_ratio: Ratio for training set
        seed: Random seed for reproducibility
        
    Returns:
        train_subset, val_subset
    """
    n = len(dataset)
    indices = list(range(n))
    
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    
    split_idx = int(n * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def get_dataloaders(
    config: dict,
    weak_signal_type: str = 'unified',
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders with weak supervision.
    
    Args:
        config: Configuration dictionary
        weak_signal_type: 'unified' for training (covers all corruption levels),
                         or specific type for evaluation
        seed: Random seed
        
    Returns:
        train_loader, val_loader
    """
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    
    img_size = model_cfg.get('img_size', 224)
    batch_size = training_cfg.get('batch_size', 8)
    num_workers = data_cfg.get('num_workers', 4)
    data_root = data_cfg.get('root', './data')
    
    # Create base dataset (trainval split from torchvision)
    base_dataset = OxfordPetBinary(
        root=data_root,
        split='trainval',
        img_size=img_size,
        augment=False,
        download=False
    )
    
    # Split into train/val (75/25)
    train_base, val_base = create_train_val_split(base_dataset, train_ratio=0.75, seed=seed)
    
    # Wrap with weak supervision
    train_dataset = WeakSupervisionDataset(
        train_base,
        weak_signal_type=weak_signal_type,
        config=config,
        seed=seed
    )
    
    val_dataset = WeakSupervisionDataset(
        val_base,
        weak_signal_type=weak_signal_type,
        config=config,
        seed=seed + 1000  # Different seed for validation
    )
    
    # Create augmented wrapper for training
    class AugmentedDataset(Dataset):
        def __init__(self, dataset, augment=True):
            self.dataset = dataset
            self.augment = augment
            
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, gt_mask, weak_mask = self.dataset[idx]
            
            if self.augment and np.random.random() > 0.5:
                # Horizontal flip
                img = torch.flip(img, dims=[-1])
                gt_mask = torch.flip(gt_mask, dims=[-1])
                weak_mask = torch.flip(weak_mask, dims=[-1])
            
            return img, gt_mask, weak_mask
    
    train_dataset = AugmentedDataset(train_dataset, augment=True)
    val_dataset = AugmentedDataset(val_dataset, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def get_test_dataloader(
    config: dict,
    weak_signal_type: str = 'good_mask',
    seed: int = 42
) -> DataLoader:
    """
    Create test DataLoader (uses the held-out 25% from trainval).
    
    Note: This uses the same split logic as training to ensure
    the test set is the same 25% held out during training.
    
    Args:
        weak_signal_type: Specific type for evaluation ('good_mask', 'poor_mask', 'box', 'point')
    """
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    
    img_size = model_cfg.get('img_size', 224)
    batch_size = training_cfg.get('batch_size', 8)
    num_workers = data_cfg.get('num_workers', 4)
    data_root = data_cfg.get('root', './data')
    
    # Create base dataset
    base_dataset = OxfordPetBinary(
        root=data_root,
        split='trainval',
        img_size=img_size,
        augment=False,
        download=False
    )
    
    # Get the same split
    _, test_base = create_train_val_split(base_dataset, train_ratio=0.75, seed=seed)
    
    # Wrap with weak supervision
    test_dataset = WeakSupervisionDataset(
        test_base,
        weak_signal_type=weak_signal_type,
        config=config,
        seed=seed + 2000
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return test_loader