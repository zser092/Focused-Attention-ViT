"""
Vision Transformer Models with Multi-Head Local Attention (MHLA)

This module implements Vision Transformer models that incorporate MHLA,
including both standard ViT and SPPP ViT variants with pretrained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, Dict, List, Union

# Import from original models
from models.vit import PatchEmbedding
from models.sppp import SuperpixelSegmentation, PatchToSuperpixelMapper, SuperpixelPooling, DynamicPositionalEncoding

# Import MHLA components
from models.mhla import MHLATransformerBlock


class PretrainedViTWithMHLA(nn.Module):
    """
    Pretrained Vision Transformer with Multi-Head Local Attention.
    
    Args:
        img_size (int): Input image size (assumed to be square)
        patch_size (int): Patch size (assumed to be square)
        in_channels (int): Number of input channels
        num_classes (int): Number of classes for classification
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        window_size (int): Size of local attention window
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        dropout (float): Dropout probability
        attn_dropout (float): Attention dropout probability
        embed_dropout (float): Embedding dropout probability
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
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        embed_dropout: float = 0.0
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
        self.window_size = window_size
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(embed_dropout)
        
        # Transformer blocks with MHLA
        self.blocks = nn.ModuleList([
            MHLATransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
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
        nn.init.normal_(self.pos_embed, std=0.02)
        
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
            
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Features tensor
        """
        batch_size = x.shape[0]
        
        # Create patch embeddings
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks with MHLA
        for block in self.blocks:
            x = block(x)
            
        # Apply final normalization
        x = self.norm(x)
        
        # Extract class token as feature representation
        return x[:, 0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Vision Transformer with MHLA.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Extract features and classify
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    def get_num_parameters(self) -> int:
        """
        Calculate the number of parameters in the model.
        
        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PretrainedSPPPViTWithMHLA(nn.Module):
    """
    Pretrained SPPP Vision Transformer with Multi-Head Local Attention.
    
    Args:
        img_size (int): Input image size (assumed to be square)
        patch_size (int): Patch size (assumed to be square)
        in_channels (int): Number of input channels
        num_classes (int): Number of classes for classification
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        window_size (int): Size of local attention window
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        dropout (float): Dropout probability
        attn_dropout (float): Attention dropout probability
        embed_dropout (float): Embedding dropout probability
        num_superpixels (int): Number of superpixels
        compactness (float): Compactness parameter for SLIC
        pooling_type (str): Type of pooling ('mean', 'max', or 'attention')
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
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        embed_dropout: float = 0.0,
        num_superpixels: int = 16,
        compactness: float = 0.1,
        pooling_type: str = 'mean'
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
        self.window_size = window_size
        self.num_superpixels = num_superpixels
        
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
        
        # Transformer blocks with MHLA
        self.blocks = nn.ModuleList([
            MHLATransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
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
        Forward pass for the SPPP Vision Transformer with MHLA.
        
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
        
        # Step 7: Apply transformer blocks with MHLA
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