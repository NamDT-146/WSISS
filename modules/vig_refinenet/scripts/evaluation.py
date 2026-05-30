"""
Evaluation script for comparing refinement methods.

Compares:
1. Original weak signal (baseline)
2. DenseCRF refinement
3. Proposed GNN refinement

Usage:
    python scripts/evaluate.py --checkpoint path/to/model.pth --weak_signal good_mask
    python scripts/evaluate.py --checkpoint path/to/model.pth --weak_signal all
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_model
from src.dataset import get_test_dataloader
from src.metrics import compute_metrics, MetricTracker
from src.dense_crf import DenseCRF, PYDENSECRF_AVAILABLE


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained model from checkpoint."""
    model = build_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_method(
    method_name: str,
    images: torch.Tensor,
    gt_masks: torch.Tensor,
    weak_masks: torch.Tensor,
    model=None,
    crf=None,
    device=None
) -> dict:
    """
    Evaluate a single method.
    
    Args:
        method_name: 'original', 'densecrf', or 'ours'
        images: Input images [B, 3, H, W]
        gt_masks: Ground truth masks [B, 1, H, W]
        weak_masks: Weak supervision masks [B, 1, H, W]
        model: GNN model (for 'ours')
        crf: DenseCRF instance (for 'densecrf')
        device: torch device
        
    Returns:
        Dictionary with IoU and Dice scores per sample
    """
    B = images.shape[0]
    results = {'iou': [], 'dice': []}
    
    for i in range(B):
        if method_name == 'original':
            pred = weak_masks[i:i+1]
        
        elif method_name == 'densecrf':
            if crf is None:
                raise ValueError("CRF instance required for densecrf method")
            pred = crf(images[i], weak_masks[i]).unsqueeze(0)
        
        elif method_name == 'ours':
            if model is None:
                raise ValueError("Model required for ours method")
            pred_logits = model(images[i:i+1].to(device), weak_masks[i:i+1].to(device))
            pred = torch.sigmoid(pred_logits).cpu()
        
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        metrics = compute_metrics(pred, gt_masks[i:i+1])
        results['iou'].append(metrics['iou'])
        results['dice'].append(metrics['dice'])
    
    return results


def run_evaluation(
    model,
    test_loader,
    device: torch.device,
    use_crf: bool = True
) -> dict:
    """
    Run full evaluation on test set.
    
    Returns:
        Dictionary with results for each method
    """
    # Initialize trackers
    methods = ['original', 'ours']
    if use_crf and PYDENSECRF_AVAILABLE:
        methods.insert(1, 'densecrf')
        crf = DenseCRF()
    else:
        crf = None
    
    results = {m: {'iou': [], 'dice': []} for m in methods}
    
    pbar = tqdm(test_loader, desc="Evaluating")
    
    for images, gt_masks, weak_masks in pbar:
        for method in methods:
            method_results = evaluate_method(
                method,
                images, gt_masks, weak_masks,
                model=model, crf=crf, device=device
            )
            results[method]['iou'].extend(method_results['iou'])
            results[method]['dice'].extend(method_results['dice'])
    
    # Compute mean metrics
    summary = {}
    for method in methods:
        summary[method] = {
            'iou_mean': np.mean(results[method]['iou']),
            'iou_std': np.std(results[method]['iou']),
            'dice_mean': np.mean(results[method]['dice']),
            'dice_std': np.std(results[method]['dice']),
            'num_samples': len(results[method]['iou'])
        }
    
    return summary


def print_results_table(results: dict, weak_signal_type: str):
    """Print results as a formatted table."""
    headers = ['Method', 'IoU (mean±std)', 'Dice (mean±std)']
    rows = []
    
    for method, metrics in results.items():
        iou_str = f"{metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}"
        dice_str = f"{metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}"
        rows.append([method.capitalize(), iou_str, dice_str])
    
    print(f"\n{'='*60}")
    print(f"Results for weak signal type: {weak_signal_type}")
    print(f"{'='*60}")
    print(tabulate(rows, headers=headers, tablefmt='grid'))
    print()


def save_results(all_results: dict, output_dir: str):
    """Save results to JSON and generate summary table."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    results_path = os.path.join(output_dir, 'results.json')
    
    # Convert numpy types to Python types for JSON serialization
    json_results = {}
    for signal_type, methods in all_results.items():
        json_results[signal_type] = {}
        for method, metrics in methods.items():
            json_results[signal_type][method] = {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                for k, v in metrics.items()
            }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Generate summary markdown table
    summary_path = os.path.join(output_dir, 'results_summary.md')
    
    with open(summary_path, 'w') as f:
        f.write("# Evaluation Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for signal_type, methods in all_results.items():
            f.write(f"## {signal_type.replace('_', ' ').title()}\n\n")
            f.write("| Method | IoU | Dice |\n")
            f.write("|--------|-----|------|\n")
            
            for method, metrics in methods.items():
                f.write(f"| {method.capitalize()} | "
                       f"{metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f} | "
                       f"{metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f} |\n")
            
            f.write("\n")
    
    print(f"Summary saved to: {summary_path}")


def visualize_samples(
    model,
    test_loader,
    device: torch.device,
    output_dir: str,
    num_samples: int = 5
):
    """Generate visualization of sample predictions."""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    crf = DenseCRF() if PYDENSECRF_AVAILABLE else None
    
    # Get some samples
    images_list, gt_list, weak_list = [], [], []
    for images, gt_masks, weak_masks in test_loader:
        images_list.append(images)
        gt_list.append(gt_masks)
        weak_list.append(weak_masks)
        if len(images_list) * images.shape[0] >= num_samples:
            break
    
    images = torch.cat(images_list, dim=0)[:num_samples]
    gt_masks = torch.cat(gt_list, dim=0)[:num_samples]
    weak_masks = torch.cat(weak_list, dim=0)[:num_samples]
    
    # Generate predictions
    with torch.no_grad():
        pred_logits = model(images.to(device), weak_masks.to(device))
        pred_masks = torch.sigmoid(pred_logits).cpu()
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images_vis = images * std + mean
    images_vis = torch.clamp(images_vis, 0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))
    
    titles = ['Image', 'GT Mask', 'Weak Signal', 'DenseCRF', 'Ours']
    
    for i in range(num_samples):
        # Image
        axes[i, 0].imshow(images_vis[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title(titles[0] if i == 0 else '')
        axes[i, 0].axis('off')
        
        # GT Mask
        axes[i, 1].imshow(gt_masks[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(titles[1] if i == 0 else '')
        axes[i, 1].axis('off')
        
        # Weak Signal
        axes[i, 2].imshow(weak_masks[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(titles[2] if i == 0 else '')
        axes[i, 2].axis('off')
        
        # DenseCRF
        if crf:
            crf_mask = crf(images[i], weak_masks[i])
            axes[i, 3].imshow(crf_mask[0].numpy(), cmap='gray', vmin=0, vmax=1)
        else:
            axes[i, 3].text(0.5, 0.5, 'N/A', ha='center', va='center')
        axes[i, 3].set_title(titles[3] if i == 0 else '')
        axes[i, 3].axis('off')
        
        # Ours
        axes[i, 4].imshow(pred_masks[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 4].set_title(titles[4] if i == 0 else '')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    
    vis_path = os.path.join(output_dir, 'visualization.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {vis_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GNN Refinement Network')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (auto-detected from checkpoint dir if not provided)')
    parser.add_argument('--weak_signal', type=str, default='all',
                        choices=['good_mask', 'poor_mask', 'box', 'point', 'all'],
                        help='Type of weak supervision signal to evaluate')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Find config
    if args.config is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(checkpoint_dir, 'config.yaml')
        if os.path.exists(config_path):
            args.config = config_path
        else:
            args.config = 'configs/default.yaml'
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)
    
    # Determine which weak signals to evaluate
    if args.weak_signal == 'all':
        weak_signals = ['good_mask', 'poor_mask', 'box', 'point']
    else:
        weak_signals = [args.weak_signal]
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'eval_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation for each weak signal type
    all_results = {}
    
    for signal_type in weak_signals:
        print(f"\n{'='*60}")
        print(f"Evaluating weak signal type: {signal_type}")
        print(f"{'='*60}")
        
        # Create test dataloader
        test_loader = get_test_dataloader(
            config,
            weak_signal_type=signal_type,
            seed=args.seed
        )
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Run evaluation
        results = run_evaluation(model, test_loader, device)
        all_results[signal_type] = results
        
        # Print results
        print_results_table(results, signal_type)
        
        # Generate visualization for this signal type
        if args.visualize:
            vis_dir = os.path.join(output_dir, f'vis_{signal_type}')
            visualize_samples(model, test_loader, device, vis_dir)
    
    # Save all results
    save_results(all_results, output_dir)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    # Create comparison table
    headers = ['Signal Type', 'Original IoU', 'CRF IoU', 'Ours IoU', 'Improvement']
    rows = []
    
    for signal_type, methods in all_results.items():
        original_iou = methods['original']['iou_mean']
        ours_iou = methods['ours']['iou_mean']
        crf_iou = methods.get('densecrf', {}).get('iou_mean', 'N/A')
        
        if isinstance(crf_iou, float):
            crf_str = f"{crf_iou:.4f}"
        else:
            crf_str = crf_iou
        
        improvement = ours_iou - original_iou
        rows.append([
            signal_type,
            f"{original_iou:.4f}",
            crf_str,
            f"{ours_iou:.4f}",
            f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt='grid'))
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()