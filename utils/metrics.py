"""
Metrics Utility Functions

This module provides utility functions for calculating theoretical space and time complexity,
as well as measuring actual training/inference time and memory usage for deep learning models.
"""

import time
import numpy as np
import torch
import psutil
import os
from typing import Dict, Any, Union, List, Tuple, Optional


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate the model size in different units.
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        Dict[str, float]: Dictionary containing model size in different units
    """
    num_params = count_parameters(model)
    
    # Calculate size in different units
    size_bytes = num_params * 4  # Assuming float32 (4 bytes per parameter)
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    
    return {
        "parameters": num_params,
        "size_bytes": size_bytes,
        "size_kb": size_kb,
        "size_mb": size_mb
    }


def calculate_vit_complexity(
    img_size: int,
    patch_size: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    in_channels: int = 3
) -> Dict[str, Any]:
    """
    Calculate the theoretical complexity of a Vision Transformer model.
    
    Args:
        img_size (int): Size of the input image (assumed to be square)
        patch_size (int): Size of each patch (assumed to be square)
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        in_channels (int): Number of input channels
        
    Returns:
        Dict[str, Any]: Dictionary containing complexity metrics
    """
    # Calculate number of patches
    num_patches = (img_size // patch_size) ** 2
    seq_len = num_patches + 1  # +1 for class token
    
    # Patch embedding complexity
    patch_embed_params = patch_size * patch_size * in_channels * embed_dim + embed_dim  # weights + bias
    patch_embed_flops = num_patches * patch_size * patch_size * in_channels * embed_dim
    
    # Position embedding complexity
    pos_embed_params = seq_len * embed_dim
    
    # Transformer blocks complexity
    block_params = 0
    block_flops = 0
    
    # For each transformer block
    for _ in range(depth):
        # Layer normalization parameters
        ln_params = 2 * embed_dim  # gamma and beta
        
        # Multi-head attention parameters
        mha_params = 3 * embed_dim * embed_dim + embed_dim * embed_dim + 2 * embed_dim  # QKV + projection + biases
        
        # Multi-head attention FLOPs
        # QKV projection: 3 * seq_len * embed_dim * embed_dim
        # Attention: 2 * num_heads * seq_len * seq_len * (embed_dim // num_heads)
        # Attention output: num_heads * seq_len * seq_len * (embed_dim // num_heads)
        # Output projection: seq_len * embed_dim * embed_dim
        mha_flops = (
            3 * seq_len * embed_dim * embed_dim +  # QKV projection
            2 * num_heads * seq_len * seq_len * (embed_dim // num_heads) +  # Q*K and softmax
            num_heads * seq_len * seq_len * (embed_dim // num_heads) +  # attention * V
            seq_len * embed_dim * embed_dim  # output projection
        )
        
        # MLP parameters
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        mlp_params = embed_dim * mlp_hidden_dim + mlp_hidden_dim + mlp_hidden_dim * embed_dim + embed_dim  # weights + biases
        
        # MLP FLOPs
        mlp_flops = seq_len * (embed_dim * mlp_hidden_dim + mlp_hidden_dim * embed_dim)
        
        # Total for one block
        block_params += ln_params * 2 + mha_params + mlp_params
        block_flops += mha_flops + mlp_flops
    
    # Final layer normalization and classification head
    final_ln_params = 2 * embed_dim
    head_params = embed_dim * 1000 + 1000  # Assuming 1000 classes for ImageNet
    
    # Total parameters and FLOPs
    total_params = patch_embed_params + pos_embed_params + block_params + final_ln_params + head_params
    total_flops = patch_embed_flops + block_flops
    
    # Space complexity (memory usage during inference)
    # Activation memory: store activations for each layer
    activation_memory = seq_len * embed_dim * 4 * (depth + 2)  # +2 for input and output
    
    # Total memory usage during inference (parameters + activations)
    inference_memory = total_params * 4 + activation_memory  # Assuming float32 (4 bytes)
    
    # Time complexity
    # Proportional to FLOPs
    time_complexity = total_flops
    
    return {
        "parameters": total_params,
        "flops": total_flops,
        "time_complexity": time_complexity,
        "space_complexity_bytes": inference_memory,
        "space_complexity_mb": inference_memory / (1024 * 1024)
    }


def measure_inference_time(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_iterations: int = 100,
    warm_up: int = 250
) -> Dict[str, float]:
    """
    Measure the inference time of a model.
    
    Args:
        model (torch.nn.Module): PyTorch model
        input_tensor (torch.Tensor): Input tensor
        num_iterations (int): Number of iterations to measure
        warm_up (int): Number of warm-up iterations
        
    Returns:
        Dict[str, float]: Dictionary containing inference time metrics
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(input_tensor)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = num_iterations / total_time
    
    return {
        "total_time": total_time,
        "avg_time": avg_time,
        "fps": fps
    }


def measure_training_time(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_iterations: int = 10
) -> Dict[str, float]:
    """
    Measure the training time of a model.
    
    Args:
        model (torch.nn.Module): PyTorch model
        input_tensor (torch.Tensor): Input tensor
        target (torch.Tensor): Target tensor
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        num_iterations (int): Number of iterations to measure
        
    Returns:
        Dict[str, float]: Dictionary containing training time metrics
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    target = target.to(device)
    
    # Measure training time
    start_time = time.time()
    for _ in range(num_iterations):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    
    return {
        "total_time": total_time,
        "avg_time": avg_time,
        "iterations_per_second": num_iterations / total_time
    }


def measure_memory_usage(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    backward: bool = False
) -> Dict[str, float]:
    """
    Measure the memory usage of a model during inference or training.
    
    Args:
        model (torch.nn.Module): PyTorch model
        input_tensor (torch.Tensor): Input tensor
        backward (bool): Whether to perform backward pass
        
    Returns:
        Dict[str, float]: Dictionary containing memory usage metrics
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Get initial memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # CPU memory before
    process = psutil.Process(os.getpid())
    cpu_mem_before = process.memory_info().rss
    
    # GPU memory before
    if torch.cuda.is_available():
        gpu_mem_before = torch.cuda.memory_allocated()
    else:
        gpu_mem_before = 0
    
    # Run model
    if backward:
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
    else:
        with torch.no_grad():
            _ = model(input_tensor)
    
    # CPU memory after
    cpu_mem_after = process.memory_info().rss
    
    # GPU memory after
    if torch.cuda.is_available():
        gpu_mem_after = torch.cuda.memory_allocated()
        gpu_mem_peak = torch.cuda.max_memory_allocated()
    else:
        gpu_mem_after = 0
        gpu_mem_peak = 0
    
    return {
        "cpu_memory_before_bytes": cpu_mem_before,
        "cpu_memory_after_bytes": cpu_mem_after,
        "cpu_memory_used_bytes": cpu_mem_after - cpu_mem_before,
        "cpu_memory_used_mb": (cpu_mem_after - cpu_mem_before) / (1024 * 1024),
        "gpu_memory_before_bytes": gpu_mem_before,
        "gpu_memory_after_bytes": gpu_mem_after,
        "gpu_memory_used_bytes": gpu_mem_after - gpu_mem_before,
        "gpu_memory_used_mb": (gpu_mem_after - gpu_mem_before) / (1024 * 1024),
        "gpu_memory_peak_bytes": gpu_mem_peak,
        "gpu_memory_peak_mb": gpu_mem_peak / (1024 * 1024)
    }


def benchmark_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    num_classes: int = 1000,
    batch_size: int = 1,
    num_inference_iterations: int = 100,
    num_training_iterations: int = 10
) -> Dict[str, Any]:
    """
    Comprehensive benchmark of a model including theoretical complexity and actual measurements.
    
    Args:
        model (torch.nn.Module): PyTorch model
        input_shape (Tuple[int, ...]): Input shape (C, H, W)
        num_classes (int): Number of classes
        batch_size (int): Batch size
        num_inference_iterations (int): Number of iterations for inference measurement
        num_training_iterations (int): Number of iterations for training measurement
        
    Returns:
        Dict[str, Any]: Dictionary containing benchmark results
    """
    device = next(model.parameters()).device
    
    # Create dummy input and target
    input_tensor = torch.randn(batch_size, *input_shape).to(device)
    target = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # Theoretical metrics
    model_size = calculate_model_size(model)
    
    # Actual measurements
    inference_time = measure_inference_time(
        model, input_tensor, num_iterations=num_inference_iterations
    )
    
    memory_usage_inference = measure_memory_usage(model, input_tensor, backward=False)
    
    # Training measurements
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    training_time = measure_training_time(
        model, input_tensor, target, criterion, optimizer, num_iterations=num_training_iterations
    )
    
    memory_usage_training = measure_memory_usage(model, input_tensor, backward=True)
    
    # Combine all metrics
    return {
        "theoretical": {
            "model_size": model_size,
        },
        "actual": {
            "inference_time": inference_time,
            "training_time": training_time,
            "memory_usage_inference": memory_usage_inference,
            "memory_usage_training": memory_usage_training
        }
    }