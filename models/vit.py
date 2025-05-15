"""
Vision Transformer (ViT) Implementation

This module implements the Vision Transformer architecture as described in the paper
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
(Dosovitskiy et al., 2020).

The implementation is based on PyTorch and follows the original architecture
with configurable parameters for flexibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """
    Converts images into patches and linearly embeds them.
    
    Args:
        img_size (int): Size of the input image (assumed to be square)
        patch_size (int): Size of each patch (assumed to be square)
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.projection = nn.Sequential(
            # Rearrange image into patches and flatten
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
    
    def forward(self, x):
        """
        Forward pass for patch embedding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Embedded patches of shape (batch_size, num_patches, embed_dim)
        """
        return self.projection(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projection for query, key, value
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # Scale dot-product
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out


class MLP(nn.Module):
    """
    Multilayer Perceptron for Transformer blocks.
    
    Args:
        in_features (int): Input feature dimension
        hidden_features (int): Hidden feature dimension
        out_features (int): Output feature dimension
        dropout (float): Dropout probability
    """
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for MLP.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        dropout (float): Dropout probability
        attn_dropout (float): Attention dropout probability
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout=dropout
        )
        
    def forward(self, x):
        """
        Forward pass for transformer block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Attention block with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model.
    
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
    """
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
        embed_dropout=0.0
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
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
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
            
    def forward_features(self, x):
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
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Apply final normalization
        x = self.norm(x)
        
        # Extract class token as feature representation
        return x[:, 0]
    
    def forward(self, x):
        """
        Forward pass for the Vision Transformer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Extract features and classify
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    def get_num_parameters(self):
        """
        Calculate the number of parameters in the model.
        
        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)