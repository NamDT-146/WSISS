"""
Training script for GNN Refinement Network with unified weak signal generation.

Usage:
    # Train with unified weak signal generator (covers all corruption levels)
    uv run python vig_refinenet/scripts/train.py --config vig_refinenet/configs/default.yaml
    
    # Train with specific weak signal type (for ablation studies)
    uv run python vig_refinenet/scripts/train.py --config vig_refinenet/configs/default.yaml --weak_signal good_mask
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_model
from src.dataset import get_dataloaders
from src.losses import build_loss
from src.metrics import MetricTracker, compute_metrics


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def freeze_backbone(model: nn.Module, freeze_bn: bool = True) -> int:
    """
    Freeze ResNet backbone parameters.
    
    Args:
        model: PyTorch model
        freeze_bn: Whether to freeze batch normalization layers
        
    Returns:
        Number of frozen parameters
    """
    frozen_count = 0
    
    # Freeze backbone (usually named 'backbone' or 'encoder')
    backbone_names = ['backbone', 'encoder', 'resnet', 'base']
    
    for name, module in model.named_modules():
        # Check if this is a backbone module
        is_backbone = any(backbone_name in name.lower() for backbone_name in backbone_names)
        
        if is_backbone:
            # Freeze weights and biases
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += param.numel()
            
            # Optionally freeze batch norm
            if freeze_bn and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
    
    return frozen_count


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_iou: float,
    checkpoint_dir: str,
    filename: str
):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_iou': best_iou
    }
    
    path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    checkpoint_path: str,
    device: torch.device
):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_iou']


def get_layer_info(model: nn.Module) -> Tuple[str, int, int]:
    """
    Get detailed information about each layer.
    
    Returns:
        Formatted string with layer information and parameter counts
    """
    total_params = 0
    trainable_params = 0
    
    lines = []
    lines.append("\n" + "="*120)
    lines.append(f"{'Layer Name':<60} {'Parameters':<25} {'Trainable':<20} {'Device':<15}")
    lines.append("-"*120)
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
            trainable_str = "Yes"
        else:
            trainable_str = "No (Frozen)"
        
        device_str = str(param.device)
        param_str = f"{num_params:,}"
        
        lines.append(f"{name:<60} {param_str:>24} {trainable_str:<20} {device_str:<15}")
    
    lines.append("-"*120)
    total_str = f"{total_params:,}"
    trainable_str = f"{trainable_params:,}"
    frozen_str = f"{(total_params - trainable_params):,}"
    trainable_pct = f"{(trainable_params/total_params*100):.2f}%"
    
    lines.append(f"{'TOTAL':<60} {total_str:>24}")
    lines.append(f"{'Trainable':<60} {trainable_str:>24}")
    lines.append(f"{'Frozen':<60} {frozen_str:>24}")
    lines.append(f"{'Trainable Percentage':<60} {trainable_pct:>24}")
    lines.append("="*120 + "\n")
    
    return "\n".join(lines), total_params, trainable_params


def get_module_summary(model: nn.Module, logger: logging.Logger):
    """Print detailed module-level parameter summary."""
    module_params = {}
    module_trainable = {}
    
    for name, module in model.named_modules():
        if name == '':
            continue
        
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        if params > 0:
            module_params[name] = params
            module_trainable[name] = trainable
    
    # Sort by parameter count
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("\n" + "="*120)
    logger.info("MODULE-LEVEL PARAMETER SUMMARY (Top 20)")
    logger.info("="*120)
    logger.info(f"{'Module Name':<70} {'Parameters':<25} {'Trainable':<25}")
    logger.info("-"*120)
    
    for idx, (module_name, param_count) in enumerate(sorted_modules[:20]):
        trainable_count = module_trainable[module_name]
        params_str = f"{param_count:,}"
        trainable_str = f"{trainable_count:,}"
        logger.info(f"{module_name:<70} {params_str:>24} {trainable_str:>24}")
    
    logger.info("="*120 + "\n")


def get_architecture_summary(model: nn.Module, logger: logging.Logger):
    """Print model architecture with layer details."""
    logger.info("\n" + "="*120)
    logger.info("MODEL ARCHITECTURE")
    logger.info("="*120)
    logger.info(str(model))
    logger.info("="*120 + "\n")


def log_parameter_report(model: nn.Module, logger: logging.Logger):
    """Log comprehensive parameter report."""
    # Get detailed layer information
    layer_info, total_params, trainable_params = get_layer_info(model)
    logger.info(layer_info)
    
    # Log parameter statistics
    logger.info("\n" + "="*120)
    logger.info("PARAMETER STATISTICS")
    logger.info("="*120)
    logger.info(f"Total Parameters:          {total_params:>20,}")
    logger.info(f"Trainable Parameters:      {trainable_params:>20,}")
    logger.info(f"Frozen Parameters:         {(total_params - trainable_params):>20,}")
    logger.info(f"Trainable Percentage:      {(trainable_params/total_params*100):>19.2f}%")
    
    # Calculate parameter breakdown by layer type
    param_by_type = {}
    for name, param in model.named_parameters():
        layer_type = name.split('.')[0] if '.' in name else 'other'
        if layer_type not in param_by_type:
            param_by_type[layer_type] = {'total': 0, 'trainable': 0}
        
        param_by_type[layer_type]['total'] += param.numel()
        if param.requires_grad:
            param_by_type[layer_type]['trainable'] += param.numel()
    
    logger.info("\n" + "-"*120)
    logger.info("PARAMETER BREAKDOWN BY COMPONENT")
    logger.info("-"*120)
    logger.info(f"{'Component':<50} {'Total':<25} {'Trainable':<25} {'Percentage':<20}")
    logger.info("-"*120)
    
    for comp_name in sorted(param_by_type.keys()):
        total = param_by_type[comp_name]['total']
        trainable = param_by_type[comp_name]['trainable']
        percentage = (total / total_params * 100) if total_params > 0 else 0
        
        total_str = f"{total:,}"
        trainable_str = f"{trainable:,}"
        
        logger.info(
            f"{comp_name:<50} {total_str:>24} {trainable_str:>24} {percentage:>18.2f}%"
        )
    
    logger.info("="*120 + "\n")


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger
) -> dict:
    """Train for one epoch."""
    model.train()
    
    loss_sum = 0.0
    metric_tracker = MetricTracker()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, gt_masks, weak_masks) in enumerate(pbar):
        images = images.to(device)
        gt_masks = gt_masks.to(device)
        weak_masks = weak_masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_logits = model(images, weak_masks)
        
        # Compute loss (supervise on GT, not weak mask)
        loss = criterion(pred_logits, gt_masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        loss_sum += loss.item()
        with torch.no_grad():
            pred_probs = torch.sigmoid(pred_logits)
            metric_tracker.update(pred_probs, gt_masks)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{loss_sum / (batch_idx + 1):.4f}"
        })
    
    metrics = metric_tracker.compute()
    metrics['loss'] = loss_sum / len(train_loader)
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device: torch.device,
    epoch: int,
    logger: logging.Logger
) -> dict:
    """Validate the model."""
    model.eval()
    
    loss_sum = 0.0
    metric_tracker = MetricTracker()
    
    # Also track metrics for weak signal baseline
    weak_metric_tracker = MetricTracker()
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    for images, gt_masks, weak_masks in pbar:
        images = images.to(device)
        gt_masks = gt_masks.to(device)
        weak_masks = weak_masks.to(device)
        
        # Forward pass
        pred_logits = model(images, weak_masks)
        
        # Compute loss
        loss = criterion(pred_logits, gt_masks)
        loss_sum += loss.item()
        
        # Track metrics for model prediction
        pred_probs = torch.sigmoid(pred_logits)
        metric_tracker.update(pred_probs, gt_masks)
        
        # Track metrics for weak signal baseline
        weak_metric_tracker.update(weak_masks, gt_masks)
    
    metrics = metric_tracker.compute()
    metrics['loss'] = loss_sum / len(val_loader)
    
    weak_metrics = weak_metric_tracker.compute()
    metrics['weak_iou'] = weak_metrics['iou']
    metrics['weak_dice'] = weak_metrics['dice']
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train GNN Refinement Network')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--weak_signal', type=str, default='unified',
                        choices=['unified', 'good_mask', 'poor_mask', 'box', 'point'],
                        help='Type of weak supervision signal (default: unified for all types)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_freeze_backbone', action='store_true',
                        help='Do not freeze backbone parameters')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup experiment name and logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"gnn_refine_{args.weak_signal}_{timestamp}"
    
    checkpoint_dir = config['training'].get('checkpoint_dir', './checkpoints')
    checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    
    log_dir = os.path.join(checkpoint_dir, 'logs')
    logger = setup_logging(log_dir, experiment_name)
    
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Weak signal type: {args.weak_signal}")
    if args.weak_signal == 'unified':
        logger.info("  -> Training with UNIFIED weak signal generator (covers all corruption levels)")
    logger.info(f"Config: {config}")
    
    # Save config
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        config,
        weak_signal_type=args.weak_signal,
        seed=args.seed
    )
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = build_model(config)
    model = model.to(device)
    
    # Freeze backbone if not disabled
    if not args.no_freeze_backbone:
        logger.info("Freezing ResNet backbone...")
        frozen_params = freeze_backbone(model, freeze_bn=True)
        logger.info(f"Frozen {frozen_params:,} backbone parameters")
    else:
        logger.info("Training with unfrozen backbone")
    
    # Log model architecture
    get_architecture_summary(model, logger)
    
    # Log detailed parameter report
    log_parameter_report(model, logger)
    
    # Log module summary
    get_module_summary(model, logger)
    
    # Print to console as well
    layer_info, total_params, trainable_params = get_layer_info(model)
    print(layer_info)
    
    # Create loss function
    criterion = build_loss(config)
    
    # Create optimizer
    training_cfg = config.get('training', {})
    
    lr = training_cfg.get('lr', 1e-4)
    if isinstance(lr, str):
        lr = float(lr)
    
    weight_decay = training_cfg.get('weight_decay', 1e-4)
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Create scheduler
    epochs = training_cfg.get('epochs', 100)
    if isinstance(epochs, str):
        epochs = int(epochs)
    
    scheduler = None
    scheduler_type = training_cfg.get('scheduler', 'cosine')
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Resume from checkpoint
    start_epoch = 1
    best_iou = 0.0
    
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        start_epoch, best_iou = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
        start_epoch += 1
        logger.info(f"Resumed from epoch {start_epoch - 1}, best IoU: {best_iou:.4f}")
    
    # Training loop
    logger.info("Starting training...")
    save_freq = training_cfg.get('save_freq', 10)
    if isinstance(save_freq, str):
        save_freq = int(save_freq)
    
    for epoch in range(start_epoch, epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, logger
        )
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Log metrics
        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train IoU: {train_metrics['iou']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Weak IoU: {val_metrics['weak_iou']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_iou,
                checkpoint_dir, 'best_model.pth'
            )
            logger.info(f"New best model saved! IoU: {best_iou:.4f}")
        
        # Save periodic checkpoint
        if epoch % save_freq == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_iou,
                checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, epochs, best_iou,
        checkpoint_dir, 'final_model.pth'
    )
    
    logger.info(f"Training complete! Best IoU: {best_iou:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()