"""
Multi-Head Latent Attention (MHLA) Implementation

This module implements the Multi-Head Latent Attention mechanism, which is designed
to reduce computational complexity in transformer models by using window-based
local attention. MHLA can be used as a drop-in replacement for standard Multi-Head
Attention in Vision Transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention mechanism that uses window-based local attention
    to reduce computational complexity.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        window_size (int): Size of local attention window
        dropout (float): Dropout probability
    """
    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 7, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for query, key, and value
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Latent projection for reducing KV cache size
        self.latent_proj = nn.Linear(self.head_dim, self.head_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def _get_window_indices(self, seq_len: int) -> torch.Tensor:
        """
        Get indices for window-based attention.
        
        Args:
            seq_len (int): Length of the sequence
            
        Returns:
            torch.Tensor: Window indices of shape (seq_len, window_size)
        """
        # Create indices for each position
        indices = torch.arange(seq_len)
        
        # Create window indices for each position
        window_indices = []
        half_window = self.window_size // 2
        
        for i in range(seq_len):
            # Calculate window start and end
            window_start = max(0, i - half_window)
            window_end = min(seq_len, i + half_window + 1)
            
            # Create window for this position
            window = indices[window_start:window_end]
            
            # Pad if necessary
            if len(window) < self.window_size:
                padding = self.window_size - len(window)
                if window_start == 0:
                    # Pad at the end
                    window = torch.cat([window, torch.ones(padding, dtype=torch.long) * (seq_len - 1)])
                else:
                    # Pad at the beginning
                    window = torch.cat([torch.zeros(padding, dtype=torch.long), window])
            
            window_indices.append(window)
        
        return torch.stack(window_indices)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for multi-head latent attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Linear projections and reshape for multi-head attention
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply latent projection to keys and values
        k_latent = self.latent_proj(k)
        v_latent = self.latent_proj(v)
        
        # Get window indices for local attention
        window_indices = self._get_window_indices(seq_len).to(device)  # (seq_len, window_size)
        
        # Gather keys and values based on window indices
        # Reshape for batch and head dimensions
        window_indices = window_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        
        # Gather keys and values for each window
        # Shape: (batch_size, num_heads, seq_len, window_size, head_dim)
        k_windows = torch.gather(
            k_latent.unsqueeze(3).expand(-1, -1, -1, self.window_size, -1),
            dim=2,
            index=window_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        )
        v_windows = torch.gather(
            v_latent.unsqueeze(3).expand(-1, -1, -1, self.window_size, -1),
            dim=2,
            index=window_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        )
        
        # Compute attention scores within each window
        # Shape: (batch_size, num_heads, seq_len, window_size)
        attn = torch.matmul(
            q.unsqueeze(3),  # (batch_size, num_heads, seq_len, 1, head_dim)
            k_windows.transpose(-2, -1)  # (batch_size, num_heads, seq_len, head_dim, window_size)
        ).squeeze(3) / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Create window-based attention mask
            window_mask = torch.gather(
                attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1),
                dim=3,
                index=window_indices
            )
            attn = attn.masked_fill(window_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        out = torch.matmul(
            attn.unsqueeze(3),  # (batch_size, num_heads, seq_len, 1, window_size)
            v_windows  # (batch_size, num_heads, seq_len, window_size, head_dim)
        ).squeeze(3)
        
        # Reshape and apply output projection
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out


class MHLATransformerBlock(nn.Module):
    """
    Transformer block with Multi-Head Latent Attention.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        window_size (int): Size of local attention window
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        dropout (float): Dropout probability
        attn_dropout (float): Attention dropout probability
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        window_size: int = 7,
        mlp_ratio: float = 4.0, 
        dropout: float = 0.0, 
        attn_dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadLatentAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=attn_dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP block
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for transformer block with MHLA.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Attention block with residual connection
        x = x + self.attn(self.norm1(x), attention_mask)
        
        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x