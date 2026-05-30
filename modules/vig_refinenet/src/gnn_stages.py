"""
GNN Stage modules for multi-scale mask refinement.

Each stage follows the same template:
1. Build edges (connectivity depends on stage)
2. Edge MLP -> edge weights
3. Message aggregation
4. Node MLP -> updated features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class EdgeMLP(nn.Module):
    """MLP for computing edge weights from node pair features."""
    
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Input: [f_i, f_j, delta_x, delta_y] -> 2*in_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, fi: torch.Tensor, fj: torch.Tensor, 
                delta_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute edge weight.
        
        Args:
            fi: Source node features [E, D]
            fj: Target node features [E, D]
            delta_pos: Position difference [E, 2]
            
        Returns:
            Edge weights [E, 1]
        """
        edge_input = torch.cat([fi, fj, delta_pos], dim=-1)
        return self.mlp(edge_input)


class NodeMLP(nn.Module):
    """MLP for updating node features after aggregation."""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GNNStage(nn.Module):
    """
    Base GNN stage module.
    
    Performs one round of message passing:
    1. Compute edge weights via EdgeMLP
    2. Aggregate messages (attention-weighted sum)
    3. Update node features via NodeMLP (with residual)
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        connectivity: str = "full",
        k_neighbors: int = 8
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.connectivity = connectivity
        self.k_neighbors = k_neighbors
        
        # Edge MLP
        self.edge_mlp = EdgeMLP(in_dim, hidden_dim)
        
        # Node MLP
        self.node_mlp = NodeMLP(in_dim, hidden_dim, out_dim)
        
        # Projection for residual if dimensions differ
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            self.proj = nn.Identity()
    
    def _get_positions(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Generate normalized grid positions."""
        y = torch.linspace(0, 1, h, device=device)
        x = torch.linspace(0, 1, w, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        pos = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [N, 2]
        return pos
    
    def _build_edges_full(self, num_nodes: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build fully connected edges."""
        # All pairs (i, j) where i != j
        idx = torch.arange(num_nodes, device=device)
        src = idx.repeat_interleave(num_nodes)
        dst = idx.repeat(num_nodes)
        # Remove self-loops
        mask = src != dst
        return src[mask], dst[mask]
    
    def _build_edges_local(self, h: int, w: int, k: int, 
                           device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build k-NN edges based on spatial proximity."""
        num_nodes = h * w
        pos = self._get_positions(h, w, device)  # [N, 2]
        
        # Compute pairwise distances
        dist = torch.cdist(pos, pos)  # [N, N]
        
        # Get k nearest neighbors (excluding self)
        dist.fill_diagonal_(float('inf'))
        _, indices = dist.topk(k, dim=1, largest=False)  # [N, k]
        
        src = torch.arange(num_nodes, device=device).repeat_interleave(k)
        dst = indices.flatten()
        
        return src, dst
    
    def _build_edges_grid(self, h: int, w: int, 
                          device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build 8-connected grid edges."""
        num_nodes = h * w
        src_list, dst_list = [], []
        
        for i in range(h):
            for j in range(w):
                node_idx = i * w + j
                # 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            neighbor_idx = ni * w + nj
                            src_list.append(node_idx)
                            dst_list.append(neighbor_idx)
        
        src = torch.tensor(src_list, device=device)
        dst = torch.tensor(dst_list, device=device)
        return src, dst
    
    def forward(self, features: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Node features [B, N, D] where N = h*w
            h, w: Spatial dimensions
            
        Returns:
            Updated features [B, N, out_dim]
        """
        B, N, D = features.shape
        device = features.device
        
        # Get node positions
        pos = self._get_positions(h, w, device)  # [N, 2]
        
        # Build edges based on connectivity type
        if self.connectivity == "full":
            src, dst = self._build_edges_full(N, device)
        elif self.connectivity == "local":
            src, dst = self._build_edges_local(h, w, self.k_neighbors, device)
        else:  # grid
            src, dst = self._build_edges_grid(h, w, device)
        
        num_edges = src.shape[0]
        
        # Compute position differences
        delta_pos = pos[dst] - pos[src]  # [E, 2]
        
        # Process each batch item
        out_features = []
        for b in range(B):
            f = features[b]  # [N, D]
            
            # Get node features for edges
            fi = f[src]  # [E, D]
            fj = f[dst]  # [E, D]
            
            # Compute edge weights
            edge_weights = self.edge_mlp(fi, fj, delta_pos)  # [E, 1]
            
            # Softmax over incoming edges for each node
            edge_weights_exp = torch.exp(edge_weights.squeeze(-1))  # [E]
            
            # Aggregate messages
            messages = torch.zeros(N, D, device=device)
            weights_sum = torch.zeros(N, device=device)
            
            # Weighted sum of neighbor features
            weighted_fj = edge_weights_exp.unsqueeze(-1) * fj  # [E, D]
            messages.index_add_(0, src, weighted_fj)
            weights_sum.index_add_(0, src, edge_weights_exp)
            
            # Normalize
            messages = messages / (weights_sum.unsqueeze(-1) + 1e-8)
            
            # Update nodes with residual
            updated = self.node_mlp(f + messages)  # [N, out_dim]
            updated = updated + self.proj(f)  # Residual
            
            out_features.append(updated)
        
        return torch.stack(out_features, dim=0)  # [B, N, out_dim]


class FeatureInheritance(nn.Module):
    """
    Feature inheritance from parent stage to child stage.
    
    f_child = Conv1x1(f_child_raw) + Upsample(f_parent)
    """
    
    def __init__(self, child_dim: int, parent_dim: int, out_dim: int):
        super().__init__()
        self.child_proj = nn.Conv2d(child_dim, out_dim, 1)
        self.parent_proj = nn.Conv2d(parent_dim, out_dim, 1)
    
    def forward(self, child_feat: torch.Tensor, parent_feat: torch.Tensor) -> torch.Tensor:
        """
        Inherit features from parent to child.
        
        Args:
            child_feat: Child features [B, C_child, H_child, W_child]
            parent_feat: Parent features [B, C_parent, H_parent, W_parent]
            
        Returns:
            Combined features [B, out_dim, H_child, W_child]
        """
        H, W = child_feat.shape[2:]
        
        # Project child features
        child_proj = self.child_proj(child_feat)
        
        # Upsample and project parent features
        parent_up = F.interpolate(parent_feat, size=(H, W), mode='bilinear', align_corners=False)
        parent_proj = self.parent_proj(parent_up)
        
        # Combine
        return child_proj + parent_proj