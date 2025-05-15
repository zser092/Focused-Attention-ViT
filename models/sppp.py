"""
Superpixel-Based Patch Pooling (SPPP) Implementation

This module implements the SPPP strategy for Vision Transformers as described in the
SPPP algorithm documentation. The strategy involves:
1. Segmentation: Using superpixels to segment the image
2. Patching: Creating patches from the image
3. Pooling: Pooling patches based on superpixel regions

This approach reduces the number of tokens in a Vision Transformer, making it more efficient
while maintaining most of the accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from skimage.segmentation import slic
from einops import rearrange, repeat
from typing import Dict, List, Tuple, Optional, Union

from models.vit import VisionTransformer, PatchEmbedding


class SuperpixelSegmentation:
    """
    Implements superpixel segmentation using the SLIC algorithm.
    """
    
    def __init__(self, num_segments: int = 16, compactness: float = 0.1, sigma: float = 1.0):
        """
        Initialize the superpixel segmentation.
        
        Args:
            num_segments (int): Number of superpixels to generate
            compactness (float): Compactness parameter for SLIC
            sigma (float): Width of Gaussian smoothing kernel for pre-processing
        """
        self.num_segments = num_segments
        self.compactness = compactness
        self.sigma = sigma
    
    def segment(self, image: torch.Tensor) -> torch.Tensor:
        """
        Segment an image into superpixels.
        
        Args:
            image (torch.Tensor): Image tensor of shape [B, C, H, W] or [C, H, W]
            
        Returns:
            torch.Tensor: Segmentation map of shape [B, H, W] or [H, W]
        """
        # Handle batch dimension
        batch_mode = len(image.shape) == 4
        
        if batch_mode:
            batch_size = image.shape[0]
            segmentation_maps = []
            
            for i in range(batch_size):
                # Process each image in the batch
                img = image[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                segments = slic(img, n_segments=self.num_segments, compactness=self.compactness, 
                               sigma=self.sigma, start_label=0)
                segmentation_maps.append(torch.from_numpy(segments).to(image.device))
            
            return torch.stack(segmentation_maps)  # [B, H, W]
        else:
            # Process single image
            img = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            segments = slic(img, n_segments=self.num_segments, compactness=self.compactness, 
                           sigma=self.sigma, start_label=0)
            return torch.from_numpy(segments).to(image.device)  # [H, W]


class PatchToSuperpixelMapper:
    """
    Maps patches to superpixels based on overlap.
    """
    
    def __init__(self, patch_size: int):
        """
        Initialize the patch to superpixel mapper.
        
        Args:
            patch_size (int): Size of each patch
        """
        self.patch_size = patch_size
    
    def map_patches(self, segmentation_map: torch.Tensor, img_size: int) -> Dict[int, List[int]]:
        """
        Map patches to superpixels.
        
        Args:
            segmentation_map (torch.Tensor): Segmentation map of shape [H, W]
            img_size (int): Size of the image
            
        Returns:
            Dict[int, List[int]]: Dictionary mapping superpixel indices to lists of patch indices
        """
        num_patches = img_size // self.patch_size
        superpixel_to_patches = {}
        
        # Iterate through patches
        for i in range(num_patches):
            for j in range(num_patches):
                # Get patch coordinates
                patch_top = i * self.patch_size
                patch_left = j * self.patch_size
                
                # Extract the segmentation map for this patch
                patch_segmap = segmentation_map[patch_top:patch_top+self.patch_size, 
                                               patch_left:patch_left+self.patch_size]
                
                # Count occurrences of each superpixel in this patch
                unique_segments, counts = torch.unique(patch_segmap, return_counts=True)
                
                # Assign patch to the superpixel with the most overlap
                dominant_segment = unique_segments[counts.argmax()].item()
                
                # Add patch index to the corresponding superpixel
                patch_idx = i * num_patches + j
                if dominant_segment not in superpixel_to_patches:
                    superpixel_to_patches[dominant_segment] = []
                superpixel_to_patches[dominant_segment].append(patch_idx)
        
        return superpixel_to_patches


class SuperpixelPooling:
    """
    Pools patch embeddings based on superpixel regions.
    """
    
    def __init__(self, pooling_type: str = 'mean'):
        """
        Initialize the superpixel pooling.
        
        Args:
            pooling_type (str): Type of pooling ('mean', 'max', or 'attention')
        """
        self.pooling_type = pooling_type
    
    def pool(self, patch_embeddings: torch.Tensor, superpixel_to_patches: Dict[int, List[int]]) -> torch.Tensor:
        """
        Pool patch embeddings based on superpixel regions.
        
        Args:
            patch_embeddings (torch.Tensor): Patch embeddings of shape [B, N, D] or [N, D]
            superpixel_to_patches (Dict[int, List[int]]): Dictionary mapping superpixel indices to lists of patch indices
            
        Returns:
            torch.Tensor: Pooled embeddings of shape [B, R, D] or [R, D], where R is the number of superpixels
        """
        # Handle batch dimension
        batch_mode = len(patch_embeddings.shape) == 3
        
        if batch_mode:
            batch_size, num_patches, embed_dim = patch_embeddings.shape
            device = patch_embeddings.device
            
            # Initialize pooled embeddings
            num_superpixels = len(superpixel_to_patches)
            pooled_embeddings = torch.zeros(batch_size, num_superpixels, embed_dim, device=device)
            
            # Pool embeddings for each superpixel
            for i, (superpixel_idx, patch_indices) in enumerate(superpixel_to_patches.items()):
                if not patch_indices:  # Skip empty superpixels
                    continue
                    
                # Extract embeddings for patches in this superpixel
                superpixel_embeddings = patch_embeddings[:, patch_indices, :]
                
                # Apply pooling
                if self.pooling_type == 'mean':
                    pooled = torch.mean(superpixel_embeddings, dim=1)  # [B, D]
                elif self.pooling_type == 'max':
                    pooled = torch.max(superpixel_embeddings, dim=1)[0]  # [B, D]
                elif self.pooling_type == 'attention':
                    # Simple attention pooling
                    attn_weights = F.softmax(torch.sum(superpixel_embeddings, dim=-1), dim=-1)  # [B, P]
                    attn_weights = attn_weights.unsqueeze(-1)  # [B, P, 1]
                    pooled = torch.sum(superpixel_embeddings * attn_weights, dim=1)  # [B, D]
                else:
                    raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
                
                # Store pooled embedding
                pooled_embeddings[:, i, :] = pooled
            
            return pooled_embeddings  # [B, R, D]
        else:
            num_patches, embed_dim = patch_embeddings.shape
            device = patch_embeddings.device
            
            # Initialize pooled embeddings
            num_superpixels = len(superpixel_to_patches)
            pooled_embeddings = torch.zeros(num_superpixels, embed_dim, device=device)
            
            # Pool embeddings for each superpixel
            for i, (superpixel_idx, patch_indices) in enumerate(superpixel_to_patches.items()):
                if not patch_indices:  # Skip empty superpixels
                    continue
                    
                # Extract embeddings for patches in this superpixel
                superpixel_embeddings = patch_embeddings[patch_indices, :]
                
                # Apply pooling
                if self.pooling_type == 'mean':
                    pooled = torch.mean(superpixel_embeddings, dim=0)  # [D]
                elif self.pooling_type == 'max':
                    pooled = torch.max(superpixel_embeddings, dim=0)[0]  # [D]
                elif self.pooling_type == 'attention':
                    # Simple attention pooling
                    attn_weights = F.softmax(torch.sum(superpixel_embeddings, dim=-1), dim=-1)  # [P]
                    pooled = torch.sum(superpixel_embeddings * attn_weights.unsqueeze(-1), dim=0)  # [D]
                else:
                    raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
                
                # Store pooled embedding
                pooled_embeddings[i, :] = pooled
            
            return pooled_embeddings  # [R, D]


class DynamicPositionalEncoding(nn.Module):
    """
    Dynamic positional encoding for variable number of tokens.
    """
    
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        """
        Initialize the dynamic positional encoding.
        
        Args:
            embed_dim (int): Embedding dimension
            dropout (float): Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, superpixel_centroids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for dynamic positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, D]
            superpixel_centroids (Optional[torch.Tensor]): Centroids of superpixels of shape [B, N, 2]
            
        Returns:
            torch.Tensor: Output tensor with positional encoding added
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if superpixel_centroids is None:
            # Use standard sinusoidal encoding if centroids are not provided
            position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float, device=device) * 
                                (-math.log(10000.0) / self.embed_dim))
            
            pe = torch.zeros(seq_len, self.embed_dim, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # Use centroid-based encoding
            # Normalize centroids to [0, 1]
            centroids_norm = superpixel_centroids.clone()
            if centroids_norm.shape[1] < seq_len:
                # Add a dummy centroid for the class token at (0.5, 0.5)
                cls_centroid = torch.ones(batch_size, 1, 2, device=device) * 0.5
                centroids_norm = torch.cat([cls_centroid, centroids_norm], dim=1)
            
            # Create positional encoding based on centroids
            pe = torch.zeros(batch_size, seq_len, self.embed_dim, device=device)
            
            # Use different frequencies for different dimensions
            freq_x = torch.exp(torch.arange(0, self.embed_dim // 2, dtype=torch.float, device=device) * 
                              (-math.log(10000.0) / (self.embed_dim // 2)))
            freq_y = torch.exp(torch.arange(0, self.embed_dim // 2, dtype=torch.float, device=device) * 
                              (-math.log(10000.0) / (self.embed_dim // 2)))
            
            # Apply sinusoidal encoding
            x_pos = centroids_norm[:, :, 0].unsqueeze(-1)  # [B, N, 1]
            y_pos = centroids_norm[:, :, 1].unsqueeze(-1)  # [B, N, 1]
            
            pe_x = torch.zeros(batch_size, seq_len, self.embed_dim // 2, device=device)
            pe_y = torch.zeros(batch_size, seq_len, self.embed_dim // 2, device=device)
            
            pe_x = torch.sin(x_pos * freq_x)
            pe_y = torch.cos(y_pos * freq_y)
            
            # Interleave x and y encodings
            pe = torch.cat([pe_x, pe_y], dim=-1)
        
        # Add positional encoding to input
        x = x + pe
        return self.dropout(x)


class SPPPViT(nn.Module):
    """
    Vision Transformer with Superpixel-Based Patch Pooling (SPPP).
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
        pooling_type: str = 'mean'
    ):
        """
        Initialize the SPPP Vision Transformer.
        
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
        """
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
            VisionTransformer.TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
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
            
    def forward(self, x):
        """
        Forward pass for the SPPP Vision Transformer.
        
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
    
    def get_num_parameters(self):
        """
        Calculate the number of parameters in the model.
        
        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)