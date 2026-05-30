"""
DenseCRF baseline for mask refinement.

Pure PyTorch implementation - no external CRF dependencies required.
Implements bilateral and spatial Gaussian potentials for iterative refinement.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


class DenseCRF:
    """
    Dense CRF for mask refinement using PyTorch.
    
    Implements:
    - Unary potentials from input probabilities
    - Bilateral potential (appearance + spatial)
    - Spatial potential (position only)
    - Mean-field inference
    """
    
    def __init__(
        self,
        num_iterations: int = 5,
        bilateral_weight: float = 10.0,
        bilateral_xy_std: float = 80.0,
        bilateral_rgb_std: float = 13.0,
        spatial_weight: float = 3.0,
        spatial_std: float = 3.0
    ):
        self.num_iterations = num_iterations
        self.bilateral_weight = bilateral_weight
        self.bilateral_xy_std = bilateral_xy_std
        self.bilateral_rgb_std = bilateral_rgb_std
        self.spatial_weight = spatial_weight
        self.spatial_std = spatial_std
    
    def _compute_spatial_kernel(
        self,
        H: int,
        W: int,
        std: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute spatial Gaussian kernel.
        
        K(i,j) = exp(-||p_i - p_j||^2 / (2 * std^2))
        
        Args:
            H, W: Image dimensions
            std: Standard deviation for spatial distance
            device: torch device
            
        Returns:
            Kernel weights [H*W, H*W] (sparse representation via convolution)
        """
        # Create coordinate grid
        y = torch.arange(H, dtype=torch.float32, device=device)
        x = torch.arange(W, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Flatten coordinates [H*W, 2]
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
        
        return coords, std
    
    def _compute_bilateral_kernel(
        self,
        image: torch.Tensor,
        xy_std: float,
        rgb_std: float
    ) -> tuple:
        """
        Compute bilateral kernel features (position + appearance).
        
        Args:
            image: RGB image [3, H, W]
            xy_std: Spatial standard deviation
            rgb_std: Color standard deviation
            
        Returns:
            Features for bilateral filtering
        """
        device = image.device
        C, H, W = image.shape
        
        # Spatial coordinates
        y = torch.arange(H, dtype=torch.float32, device=device)
        x = torch.arange(W, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Normalize spatial coords by std
        yy = yy / xy_std
        xx = xx / xy_std
        
        # Normalize color by std
        color = image / rgb_std  # [3, H, W]
        
        # Concatenate features [5, H, W] = [x, y, r, g, b]
        features = torch.cat([
            xx.unsqueeze(0),
            yy.unsqueeze(0),
            color
        ], dim=0)
        
        return features
    
    def _message_passing(
        self,
        Q: torch.Tensor,
        features: torch.Tensor,
        compatibility: float
    ) -> torch.Tensor:
        """
        Perform message passing using permutohedral lattice approximation.
        
        For efficiency, we use a Gaussian blur approximation.
        
        Args:
            Q: Current beliefs [2, H, W]
            features: Feature map [C, H, W]
            compatibility: Compatibility weight
            
        Returns:
            Messages [2, H, W]
        """
        # Simple Gaussian blur approximation
        # In full CRF, this would be permutohedral lattice filtering
        
        # Create Gaussian kernel based on feature similarity
        kernel_size = 5
        sigma = 1.0
        
        # Apply separable Gaussian blur to Q
        messages = Q.unsqueeze(0)  # [1, 2, H, W]
        
        # Approximate bilateral filtering with guided filtering
        # For spatial: simple Gaussian blur
        padding = kernel_size // 2
        messages = F.avg_pool2d(
            messages,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        messages = messages.squeeze(0)  # [2, H, W]
        messages = messages * compatibility
        
        return messages
    
    def _spatial_message_passing(
        self,
        Q: torch.Tensor,
        kernel_size: int = 5
    ) -> torch.Tensor:
        """
        Spatial message passing (position-only Gaussian).
        
        Args:
            Q: Current beliefs [2, H, W]
            kernel_size: Size of Gaussian kernel
            
        Returns:
            Messages [2, H, W]
        """
        # Apply Gaussian blur
        padding = kernel_size // 2
        Q_expanded = Q.unsqueeze(0)  # [1, 2, H, W]
        
        # Create Gaussian kernel
        sigma = self.spatial_std / 10.0  # Scale for kernel
        device = Q.device
        
        # 1D Gaussian kernel
        ax = torch.arange(-padding, padding + 1, dtype=torch.float32, device=device)
        gauss = torch.exp(-ax**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        
        # Separable convolution
        kernel_h = gauss.view(1, 1, -1, 1).repeat(2, 1, 1, 1)
        kernel_w = gauss.view(1, 1, 1, -1).repeat(2, 1, 1, 1)
        
        # Apply horizontal then vertical
        Q_filtered = F.conv2d(Q_expanded, kernel_h, padding=(padding, 0), groups=2)
        Q_filtered = F.conv2d(Q_filtered, kernel_w, padding=(0, padding), groups=2)
        
        messages = Q_filtered.squeeze(0)  # [2, H, W]
        
        return messages
    
    def _bilateral_message_passing(
        self,
        Q: torch.Tensor,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Bilateral message passing (appearance + spatial).
        
        Uses joint bilateral filtering approximation.
        
        Args:
            Q: Current beliefs [2, H, W]
            image: RGB image [3, H, W]
            
        Returns:
            Messages [2, H, W]
        """
        device = Q.device
        
        # Simple bilateral filter approximation
        # In practice, this is a color-guided spatial filter
        
        kernel_size = 5
        padding = kernel_size // 2
        
        # Expand dimensions
        Q_expanded = Q.unsqueeze(0)  # [1, 2, H, W]
        img_expanded = image.unsqueeze(0)  # [1, 3, H, W]
        
        # Extract patches
        patches_Q = F.unfold(Q_expanded, kernel_size, padding=padding)  # [1, 2*K*K, H*W]
        patches_I = F.unfold(img_expanded, kernel_size, padding=padding)  # [1, 3*K*K, H*W]
        
        B, _, N = patches_Q.shape
        K2 = kernel_size * kernel_size
        
        # Reshape patches
        patches_Q = patches_Q.view(1, 2, K2, N).permute(0, 3, 1, 2)  # [1, H*W, 2, K*K]
        patches_I = patches_I.view(1, 3, K2, N).permute(0, 3, 1, 2)  # [1, H*W, 3, K*K]
        
        # Center pixel color
        center_I = image.permute(1, 2, 0).reshape(1, -1, 3, 1)  # [1, H*W, 3, 1]
        
        # Compute color difference weights
        color_diff = patches_I - center_I  # [1, H*W, 3, K*K]
        color_dist = (color_diff ** 2).sum(dim=2)  # [1, H*W, K*K]
        
        # Spatial weights (Gaussian)
        sigma_xy = self.bilateral_xy_std / 50.0
        cy, cx = kernel_size // 2, kernel_size // 2
        y_grid = torch.arange(kernel_size, device=device).view(-1, 1).repeat(1, kernel_size)
        x_grid = torch.arange(kernel_size, device=device).view(1, -1).repeat(kernel_size, 1)
        spatial_dist = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2).view(1, 1, K2)
        
        # Color weights (Gaussian)
        sigma_rgb = self.bilateral_rgb_std / 50.0
        color_weights = torch.exp(-color_dist / (2 * sigma_rgb ** 2))
        spatial_weights = torch.exp(-spatial_dist / (2 * sigma_xy ** 2))
        
        # Combined weights
        weights = color_weights * spatial_weights  # [1, H*W, K*K]
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
        
        # Apply weights to Q patches
        weighted_Q = patches_Q * weights.unsqueeze(2)  # [1, H*W, 2, K*K]
        filtered_Q = weighted_Q.sum(dim=3)  # [1, H*W, 2]
        
        # Reshape back
        H, W = Q.shape[1], Q.shape[2]
        messages = filtered_Q.view(1, H, W, 2).permute(0, 3, 1, 2).squeeze(0)  # [2, H, W]
        
        return messages
    
    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Refine mask using DenseCRF.
        
        Args:
            image: RGB image [3, H, W] (ImageNet normalized)
            mask: Initial mask [1, H, W] (probabilities in [0, 1])
            
        Returns:
            Refined mask [1, H, W]
        """
        device = image.device
        
        # Denormalize image for CRF processing
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        img = image * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Initialize Q (beliefs) from input mask
        prob = mask[0]  # [H, W]
        prob = torch.clamp(prob, 1e-6, 1 - 1e-6)
        
        # Q has shape [2, H, W] where Q[0] = bg, Q[1] = fg
        Q = torch.stack([1 - prob, prob], dim=0)  # [2, H, W]
        
        # Unary energy (negative log likelihood)
        unary = -torch.log(Q + 1e-8)
        
        # Mean-field inference
        for iteration in range(self.num_iterations):
            # Store previous Q
            Q_prev = Q.clone()
            
            # Compute messages from pairwise potentials
            
            # 1. Spatial message passing
            spatial_msg = self._spatial_message_passing(Q)
            spatial_msg = spatial_msg * self.spatial_weight
            
            # 2. Bilateral message passing
            bilateral_msg = self._bilateral_message_passing(Q, img)
            bilateral_msg = bilateral_msg * self.bilateral_weight
            
            # 3. Compatibility transform (simple subtraction for Potts model)
            # For binary segmentation with Potts model:
            # mu(xi, xj) = w if xi != xj, else 0
            total_msg = spatial_msg + bilateral_msg
            
            # Apply compatibility: -1 for same label, +1 for different label
            compat_msg = torch.stack([
                total_msg[1],  # Message to class 0 from class 1
                total_msg[0]   # Message to class 1 from class 0
            ], dim=0)
            
            # Update Q using mean-field approximation
            # Q_new = exp(-E_unary - E_pairwise)
            energy = unary + compat_msg
            Q = F.softmax(-energy, dim=0)
            
            # Check convergence (optional)
            diff = (Q - Q_prev).abs().mean()
            if diff < 1e-5:
                break
        
        # Extract foreground probability
        refined_mask = Q[1]  # [H, W]
        
        return refined_mask.unsqueeze(0)  # [1, H, W]
    
    def refine_batch(
        self,
        images: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Refine a batch of masks.
        
        Args:
            images: RGB images [B, 3, H, W]
            masks: Initial masks [B, 1, H, W]
            
        Returns:
            Refined masks [B, 1, H, W]
        """
        B = images.shape[0]
        refined = []
        
        for i in range(B):
            refined_mask = self(images[i], masks[i])
            refined.append(refined_mask)
        
        return torch.stack(refined, dim=0)


class FastDenseCRF:
    """
    Faster approximate version using simple guided filtering.
    Suitable for real-time applications.
    """
    
    def __init__(
        self,
        num_iterations: int = 3,
        radius: int = 5,
        eps: float = 0.1
    ):
        self.num_iterations = num_iterations
        self.radius = radius
        self.eps = eps
    
    def _guided_filter(
        self,
        guide: torch.Tensor,
        src: torch.Tensor,
        radius: int,
        eps: float
    ) -> torch.Tensor:
        """
        Fast guided filter for edge-aware smoothing.
        
        Args:
            guide: Guidance image [3, H, W]
            src: Source signal [C, H, W]
            radius: Filter radius
            eps: Regularization
            
        Returns:
            Filtered signal [C, H, W]
        """
        # Box filter
        def box_filter(x, r):
            return F.avg_pool2d(
                x.unsqueeze(0),
                kernel_size=2*r+1,
                stride=1,
                padding=r
            ).squeeze(0)
        
        # Mean of guide
        mean_I = box_filter(guide, radius)
        
        # Mean of source
        mean_p = box_filter(src, radius)
        
        # Correlation
        corr_I = box_filter(guide * guide.unsqueeze(1), radius)
        corr_Ip = box_filter(guide.unsqueeze(0) * src.unsqueeze(1), radius)
        
        # Variance and covariance
        var_I = corr_I - mean_I * mean_I.unsqueeze(1)
        cov_Ip = corr_Ip - mean_I.unsqueeze(0) * mean_p.unsqueeze(1)
        
        # Coefficients a and b
        a = cov_Ip / (var_I + eps)
        b = mean_p - (a * mean_I.unsqueeze(0)).sum(dim=1)
        
        # Mean of a and b
        mean_a = box_filter(a, radius)
        mean_b = box_filter(b, radius)
        
        # Output
        output = (mean_a * guide.unsqueeze(0)).sum(dim=1) + mean_b
        
        return output
    
    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Fast refinement using guided filtering.
        
        Args:
            image: RGB image [3, H, W]
            mask: Initial mask [1, H, W]
            
        Returns:
            Refined mask [1, H, W]
        """
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        img = (image * std + mean).clamp(0, 1)
        
        refined = mask
        
        # Iterative refinement
        for _ in range(self.num_iterations):
            refined = self._guided_filter(img, refined, self.radius, self.eps)
            refined = torch.clamp(refined, 0, 1)
        
        return refined
    
    def refine_batch(
        self,
        images: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """Refine batch."""
        B = images.shape[0]
        refined = []
        
        for i in range(B):
            refined_mask = self(images[i], masks[i])
            refined.append(refined_mask)
        
        return torch.stack(refined, dim=0)


def build_dense_crf(config: Optional[dict] = None, fast: bool = False):
    """
    Build DenseCRF from config.
    
    Args:
        config: Optional configuration dictionary
        fast: If True, use FastDenseCRF (guided filter approximation)
        
    Returns:
        DenseCRF or FastDenseCRF instance
    """
    if fast:
        if config is None:
            return FastDenseCRF()
        
        crf_cfg = config.get('dense_crf', {})
        return FastDenseCRF(
            num_iterations=crf_cfg.get('num_iterations', 3),
            radius=crf_cfg.get('radius', 5),
            eps=crf_cfg.get('eps', 0.1)
        )
    
    else:
        if config is None:
            return DenseCRF()
        
        crf_cfg = config.get('dense_crf', {})
        return DenseCRF(
            num_iterations=crf_cfg.get('num_iterations', 5),
            bilateral_weight=crf_cfg.get('bilateral_weight', 10.0),
            bilateral_xy_std=crf_cfg.get('bilateral_xy_std', 80.0),
            bilateral_rgb_std=crf_cfg.get('bilateral_rgb_std', 13.0),
            spatial_weight=crf_cfg.get('spatial_weight', 3.0),
            spatial_std=crf_cfg.get('spatial_std', 3.0)
        )


# For backward compatibility
PYDENSECRF_AVAILABLE = True  # Our pure PyTorch version is always available