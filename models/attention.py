"""
Cross-Attention Mechanisms for Vision Transformers

This module implements various cross-attention mechanisms for Vision Transformers,
including basic cross-attention and multi-head cross-attention. These mechanisms
allow one set of tokens to attend to another set, enabling more flexible attention
patterns than standard self-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple


class CrossAttention(nn.Module):
    """
    Basic cross-attention mechanism that allows one set of tokens to attend to another set.
    
    Args:
        embed_dim (int): Embedding dimension
        dropout (float): Dropout probability
    """
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key_value: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, query_len, embed_dim)
            key_value (torch.Tensor): Key-value tensor of shape (batch_size, kv_len, embed_dim)
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, query_len, kv_len)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, query_len, embed_dim)
        """
        batch_size, query_len, _ = query.shape
        _, kv_len, _ = key_value.shape
        
        # Linear projections
        q = self.q_proj(query)  # (batch_size, query_len, embed_dim)
        k = self.k_proj(key_value)  # (batch_size, kv_len, embed_dim)
        v = self.v_proj(key_value)  # (batch_size, kv_len, embed_dim)
        
        # Compute attention scores
        attn = torch.bmm(q, k.transpose(1, 2))  # (batch_size, query_len, kv_len)
        attn = attn / (self.embed_dim ** 0.5)  # Scale dot-product
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.bmm(attn, v)  # (batch_size, query_len, embed_dim)
        out = self.out_proj(out)
        
        return out


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism that extends basic cross-attention with multiple attention heads.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key_value: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multi-head cross-attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, query_len, embed_dim)
            key_value (torch.Tensor): Key-value tensor of shape (batch_size, kv_len, embed_dim)
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, query_len, kv_len)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, query_len, embed_dim)
        """
        batch_size, query_len, _ = query.shape
        _, kv_len, _ = key_value.shape
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query).reshape(batch_size, query_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(batch_size, kv_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(batch_size, kv_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, query_len, kv_len)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multi-head attention
            expanded_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn.masked_fill(expanded_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch_size, num_heads, query_len, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, query_len, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class CrossAttentionTransformerBlock(nn.Module):
    """
    Transformer block with cross-attention mechanism.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        dropout (float): Dropout probability
        attn_dropout (float): Attention dropout probability
        use_multi_head (bool): Whether to use multi-head cross-attention
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.0, 
        attn_dropout: float = 0.0,
        use_multi_head: bool = False
    ):
        super().__init__()
        self.norm1_query = nn.LayerNorm(embed_dim)
        self.norm1_kv = nn.LayerNorm(embed_dim)
        
        # Choose between basic cross-attention and multi-head cross-attention
        if use_multi_head:
            self.attn = MultiHeadCrossAttention(embed_dim, num_heads, attn_dropout)
        else:
            self.attn = CrossAttention(embed_dim, attn_dropout)
            
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
        
    def forward(
        self, 
        query: torch.Tensor, 
        key_value: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention transformer block.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, query_len, embed_dim)
            key_value (torch.Tensor): Key-value tensor of shape (batch_size, kv_len, embed_dim)
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, query_len, kv_len)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, query_len, embed_dim)
        """
        # Cross-attention block with residual connection
        query_norm = self.norm1_query(query)
        kv_norm = self.norm1_kv(key_value)
        query = query + self.attn(query_norm, kv_norm, attention_mask)
        
        # MLP block with residual connection
        query = query + self.mlp(self.norm2(query))
        
        return query


class CrossAttentionViT(nn.Module):
    """
    Vision Transformer with cross-attention mechanism.
    
    This model extends the standard Vision Transformer by replacing self-attention
    with cross-attention, allowing different sets of tokens to interact.
    
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
        use_multi_head (bool): Whether to use multi-head cross-attention
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
        use_multi_head: bool = False
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
        self.use_multi_head = use_multi_head
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            # Rearrange image into patches and flatten
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            nn.Transpose(1, 2)
        )
        num_patches = (img_size // patch_size) ** 2
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(embed_dropout)
        
        # Cross-attention transformer blocks
        self.blocks = nn.ModuleList([
            CrossAttentionTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_multi_head=use_multi_head
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
        
        # Apply cross-attention transformer blocks
        # In this implementation, we use the same tensor for query and key-value
        # This effectively makes it behave like self-attention
        for block in self.blocks:
            x = block(x, x)
            
        # Apply final normalization
        x = self.norm(x)
        
        # Extract class token as feature representation
        return x[:, 0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Vision Transformer with cross-attention.
        
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


class CrossAttentionSPPPViT(nn.Module):
    """
    SPPP Vision Transformer with cross-attention mechanism.
    
    This model extends the SPPP Vision Transformer by replacing self-attention
    with cross-attention, allowing different sets of tokens to interact.
    
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
        use_multi_head (bool): Whether to use multi-head cross-attention
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
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
        use_multi_head: bool = False
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
        self.use_multi_head = use_multi_head
        
        # Import SPPP components
        from models.sppp import SuperpixelSegmentation, PatchToSuperpixelMapper, SuperpixelPooling, DynamicPositionalEncoding
        
        # SPPP components
        self.segmentation = SuperpixelSegmentation(
            num_segments=num_superpixels,
            compactness=compactness
        )
        self.patch_mapper = PatchToSuperpixelMapper(patch_size=patch_size)
        self.pooling = SuperpixelPooling(pooling_type=pooling_type)
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            # Rearrange image into patches and flatten
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            nn.Transpose(1, 2)
        )
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = DynamicPositionalEncoding(embed_dim, embed_dropout)
        
        # Cross-attention transformer blocks
        self.blocks = nn.ModuleList([
            CrossAttentionTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_multi_head=use_multi_head
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
        Forward pass for the SPPP Vision Transformer with cross-attention.
        
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
        
        # Step 7: Apply cross-attention transformer blocks
        # In this implementation, we use the same tensor for query and key-value
        # This effectively makes it behave like self-attention
        for block in self.blocks:
            x = block(x, x)
        
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