"""
Superpixel-Based Patch Pooling (SPPP) Vision Transformer with Multi-Head Latent Attention (MHLA)

This module implements a modified SPPP Vision Transformer that can use either
standard Multi-Head Attention or Multi-Head Latent Attention (MHLA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange, repeat
from typing import Dict, List, Tuple, Optional, Union

from models.sppp import SuperpixelSegmentation, PatchToSuperpixelMapper, SuperpixelPooling, DynamicPositionalEncoding
from models.vit import PatchEmbedding, MLP
from models.mhla import MultiHeadLatentAttention


class TransformerBlock(nn.Module):
    """
    Transformer encoder block that can use either standard Multi-Head Attention
    or Multi-Head Latent Attention.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        dropout (float): Dropout probability
        attn_dropout (float): Attention dropout probability
        window_size (int): Size of local attention window for MHLA
        use_mhla (bool): Whether to use Multi-Head Latent Attention
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.0, 
        attn_dropout: float = 0.0,
        window_size: int = 7,
        use_mhla: bool = False
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Choose between standard Multi-Head Attention and Multi-Head Latent Attention
        if use_mhla:
            self.attn = MultiHeadLatentAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                dropout=attn_dropout
            )
        else:
            # Standard Multi-Head Attention
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True
            )
            
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP block
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            out_features=embed_dim,
            dropout=dropout
        )
        
        self.use_mhla = use_mhla
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask (Optional[torch.Tensor]): Attention mask
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        x_norm = self.norm1(x)
        
        # Apply attention with the appropriate interface
        if self.use_mhla:
            # MHLA interface
            attn_output = self.attn(x_norm, attention_mask)
        else:
            # Standard MultiheadAttention interface
            attn_output, _ = self.attn(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                key_padding_mask=None if attention_mask is None else ~attention_mask
            )
        
        # Attention block with residual connection
        x = x + attn_output
        
        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class SPPPViTMHLA(nn.Module):
    """
    SPPP Vision Transformer with optional Multi-Head Latent Attention.
    
    Args:
        img_size (int): Input image size (assumed to be square)
        patch_size (int): Patch size (assumed to be square)
        in_channels (int): Number of input channels
        num_classes (int): Number of classes for classification
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        dropout (float): Dropout probability
        attn_dropout (float): Attention dropout probability
        embed_dropout (float): Embedding dropout probability
        num_superpixels (int): Number of superpixels
        compactness (float): Compactness parameter for SLIC
        pooling_type (str): Type of pooling ('mean', 'max', or 'attention')
        window_size (int): Size of local attention window for MHLA
        use_mhla (bool): Whether to use Multi-Head Latent Attention
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        embed_dropout: float = 0.0,
        num_superpixels: int = 16,
        compactness: float = 0.1,
        pooling_type: str = 'mean',
        window_size: int = 7,
        use_mhla: bool = False
    ):
        super().__init__()
        
        # Store configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_superpixels = num_superpixels
        self.use_mhla = use_mhla
        
        # SPPP components
        self.segmentation = SuperpixelSegmentation(
            num_segments=num_superpixels,
            compactness=compactness
        )
        self.patch_mapper = PatchToSuperpixelMapper(patch_size=patch_size)
        self.pooling = SuperpixelPooling(pooling_type=pooling_type)
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = DynamicPositionalEncoding(embed_dim, embed_dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                window_size=window_size,
                use_mhla=use_mhla
            )
            for _ in range(depth)
        ])
        
        # Final normalization and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for the model."""
        # Initialize patch embedding, class token, and position embedding
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize transformer blocks
        self.apply(self._init_weights_recursive)
        
    def _init_weights_recursive(self, m):
        """Initialize weights recursively."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def _calculate_superpixel_centroids(self, segmentation_maps):
        """
        Calculate centroids of superpixels.
        
        Args:
            segmentation_maps (torch.Tensor): Segmentation maps of shape [B, H, W]
            
        Returns:
            torch.Tensor: Centroids of shape [B, R, 2], where R is the number of superpixels
        """
        batch_size = segmentation_maps.shape[0]
        device = segmentation_maps.device
        centroids = torch.zeros(batch_size, self.num_superpixels, 2, device=device)
        
        for b in range(batch_size):
            segmap = segmentation_maps[b]  # [H, W]
            h, w = segmap.shape
            
            # Create coordinate grid
            y_coords = torch.arange(h, device=device).float() / h
            x_coords = torch.arange(w, device=device).float() / w
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Calculate centroids for each superpixel
            for s in range(self.num_superpixels):
                mask = (segmap == s).float()
                if mask.sum() > 0:
                    y_centroid = (y_grid * mask).sum() / mask.sum()
                    x_centroid = (x_grid * mask).sum() / mask.sum()
                    centroids[b, s, 0] = x_centroid
                    centroids[b, s, 1] = y_centroid
                else:
                    # Default to center if superpixel is empty
                    centroids[b, s, 0] = 0.5
                    centroids[b, s, 1] = 0.5
        
        return centroids
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SPPP Vision Transformer with optional MHLA.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Step 1: Segment images into superpixels
        segmentation_maps = self.segmentation.segment(x)  # [B, H, W]
        
        # Step 2: Create patch embeddings
        patch_embeddings = self.patch_embed(x)  # [B, N, D]
        
        # Step 3: Map patches to superpixels and pool
        pooled_embeddings_list = []
        
        for b in range(batch_size):
            # Map patches to superpixels for this image
            superpixel_to_patches = self.patch_mapper.map_patches(
                segmentation_maps[b], self.img_size
            )
            
            # Pool patch embeddings based on superpixel regions
            pooled_embeddings = self.pooling.pool(
                patch_embeddings[b], superpixel_to_patches
            )  # [R, D]
            
            pooled_embeddings_list.append(pooled_embeddings)
        
        # Stack pooled embeddings
        pooled_embeddings = torch.stack(pooled_embeddings_list)  # [B, R, D]
        
        # Step 4: Add class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, pooled_embeddings), dim=1)  # [B, R+1, D]
        
        # Step 5: Calculate superpixel centroids for positional encoding
        superpixel_centroids = self._calculate_superpixel_centroids(segmentation_maps)
        
        # Step 6: Add positional encoding
        x = self.pos_embed(x, superpixel_centroids)
        
        # Step 7: Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Step 8: Apply final normalization
        x = self.norm(x)
        
        # Step 9: Extract class token as feature representation
        x = x[:, 0]
        
        # Step 10: Classify
        x = self.head(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """
        Calculate the number of parameters in the model.
        
        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)