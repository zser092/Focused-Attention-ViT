"""
Data Utility Functions

This module provides utility functions for loading and preprocessing common image datasets,
creating data loaders, and visualizing sample images and patches.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, ImageFolder
from typing import Dict, Any, Union, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from PIL import Image


def get_transforms(dataset_name: str, img_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Get standard transforms for common datasets.
    
    Args:
        dataset_name (str): Name of the dataset ('cifar10', 'imagenet', etc.)
        img_size (int): Target image size
        
    Returns:
        Dict[str, transforms.Compose]: Dictionary containing train and test transforms
    """
    if dataset_name.lower() == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    
    elif dataset_name.lower() == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),  # 256 / 224 = 1.14
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    else:
        # Default transforms
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    return {
        'train': train_transform,
        'test': test_transform
    }


def load_cifar10(
    data_dir: str = './project/data',
    img_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    subset_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Load CIFAR-10 dataset.
    
    Args:
        data_dir (str): Directory to store the dataset
        img_size (int): Target image size
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loaders
        subset_size (Optional[int]): Size of subset to use (for quick testing)
        
    Returns:
        Dict[str, Any]: Dictionary containing dataset and data loaders
    """
    transforms_dict = get_transforms('cifar10', img_size)
    
    # Load datasets
    train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms_dict['train']
    )
    
    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms_dict['test']
    )
    
    # Create subset if specified
    if subset_size is not None:
        train_indices = torch.randperm(len(train_dataset))[:subset_size]
        test_indices = torch.randperm(len(test_dataset))[:subset_size // 5]
        
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'class_names': class_names,
        'num_classes': 10
    }


def load_imagenet_subset(
    data_dir: str = './project/data/imagenet',
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    subset_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Load ImageNet dataset or a subset.
    
    Args:
        data_dir (str): Directory containing ImageNet data
        img_size (int): Target image size
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loaders
        subset_size (Optional[int]): Size of subset to use (for quick testing)
        
    Returns:
        Dict[str, Any]: Dictionary containing dataset and data loaders
    """
    transforms_dict = get_transforms('imagenet', img_size)
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"ImageNet directory not found: {data_dir}")
    
    # Load datasets
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"ImageNet train or validation directory not found in {data_dir}")
    
    train_dataset = ImageFolder(
        root=train_dir,
        transform=transforms_dict['train']
    )
    
    val_dataset = ImageFolder(
        root=val_dir,
        transform=transforms_dict['test']
    )
    
    # Create subset if specified
    if subset_size is not None:
        train_indices = torch.randperm(len(train_dataset))[:subset_size]
        val_indices = torch.randperm(len(val_dataset))[:subset_size // 5]
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get class names and mapping
    try:
        idx_to_class = {i: cls for i, cls in enumerate(train_dataset.classes)}
    except AttributeError:
        # Handle Subset case
        if isinstance(train_dataset, Subset):
            idx_to_class = {i: cls for i, cls in enumerate(train_dataset.dataset.classes)}
        else:
            idx_to_class = {i: f"Class {i}" for i in range(1000)}
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'idx_to_class': idx_to_class,
        'num_classes': 1000
    }


def download_pretrained_vit_weights(
    model_variant: str = 'vit_b_16',
    source: str = 'torchvision',
    save_dir: str = './project/data/pretrained_weights',
    force_download: bool = False
) -> Dict[str, Any]:
    """
    Download pre-trained Vision Transformer weights.
    
    Args:
        model_variant (str): Model variant to download
            For torchvision: 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14'
            For huggingface: 'google/vit-base-patch16-224', 'google/vit-large-patch16-224', etc.
        source (str): Source of pre-trained weights ('torchvision' or 'huggingface')
        save_dir (str): Directory to save the weights
        force_download (bool): Whether to force download even if weights already exist
        
    Returns:
        Dict[str, Any]: Dictionary containing model information and weights path
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if source.lower() == 'torchvision':
        # Use PyTorch's TorchVision models
        try:
            import torchvision.models as models
            
            # Map model variant to function
            model_map = {
                'vit_b_16': models.vit_b_16,
                'vit_b_32': models.vit_b_32,
                'vit_l_16': models.vit_l_16,
                'vit_l_32': models.vit_l_32,
                'vit_h_14': models.vit_h_14
            }
            
            if model_variant not in model_map:
                raise ValueError(f"Unsupported model variant: {model_variant}. "
                                f"Supported variants: {list(model_map.keys())}")
            
            # Get model function
            model_fn = model_map[model_variant]
            
            # Create weights path
            weights_path = os.path.join(save_dir, f"{model_variant}_torchvision.pth")
            
            # Check if weights already exist
            if os.path.exists(weights_path) and not force_download:
                print(f"Pre-trained weights already exist at {weights_path}")
            else:
                print(f"Downloading pre-trained weights for {model_variant} from TorchVision...")
                # Load pre-trained model
                model = model_fn(pretrained=True)
                
                # Save weights
                torch.save(model.state_dict(), weights_path)
                print(f"Weights saved to {weights_path}")
            
            # Get model configuration
            temp_model = model_fn(pretrained=False)
            config = {
                'img_size': 224,  # Default for most ViT models
                'patch_size': int(model_variant.split('_')[-1]),
                'embed_dim': temp_model.hidden_dim,
                'depth': len(temp_model.encoder.layers),
                'num_heads': temp_model.encoder.layers[0].self_attention.num_heads,
                'num_classes': 1000  # ImageNet classes
            }
            
            return {
                'weights_path': weights_path,
                'config': config,
                'source': 'torchvision',
                'model_variant': model_variant
            }
            
        except ImportError:
            print("TorchVision not available. Please install it with: pip install torchvision")
            raise
    
    elif source.lower() == 'huggingface':
        # Use Hugging Face Transformers
        try:
            from transformers import ViTModel, ViTConfig
            
            # Create weights path
            model_name = model_variant.split('/')[-1] if '/' in model_variant else model_variant
            weights_path = os.path.join(save_dir, f"{model_name}_huggingface")
            
            # Check if weights already exist
            if os.path.exists(weights_path) and not force_download:
                print(f"Pre-trained weights already exist at {weights_path}")
                # Load config
                config = ViTConfig.from_pretrained(weights_path)
            else:
                print(f"Downloading pre-trained weights for {model_variant} from Hugging Face...")
                # Download pre-trained model
                model = ViTModel.from_pretrained(model_variant)
                
                # Save weights
                model.save_pretrained(weights_path)
                print(f"Weights saved to {weights_path}")
                
                # Get config
                config = model.config
            
            # Extract model configuration
            patch_size = 16  # Default
            if 'patch' in model_variant.lower():
                # Try to extract patch size from model name
                try:
                    patch_str = model_variant.lower().split('patch')[1].split('-')[0]
                    patch_size = int(patch_str)
                except:
                    pass
            
            model_config = {
                'img_size': config.image_size,
                'patch_size': patch_size,
                'embed_dim': config.hidden_size,
                'depth': config.num_hidden_layers,
                'num_heads': config.num_attention_heads,
                'num_classes': 1000  # ImageNet classes
            }
            
            return {
                'weights_path': weights_path,
                'config': model_config,
                'source': 'huggingface',
                'model_variant': model_variant
            }
            
        except ImportError:
            print("Hugging Face Transformers not available. Please install it with: pip install transformers")
            raise
    
    else:
        raise ValueError(f"Unsupported source: {source}. Supported sources: 'torchvision', 'huggingface'")


def load_pretrained_weights_to_model(
    model: torch.nn.Module,
    weights_info: Dict[str, Any],
    num_classes: int = None,
    freeze_layers: Union[bool, List[str]] = False
) -> torch.nn.Module:
    """
    Load pre-trained weights into a model.
    
    Args:
        model (torch.nn.Module): Model to load weights into
        weights_info (Dict[str, Any]): Dictionary containing weights information from download_pretrained_vit_weights
        num_classes (int): Number of classes for the model (if None, use the original number)
        freeze_layers (Union[bool, List[str]]): Whether to freeze layers or list of layer names to freeze
        
    Returns:
        torch.nn.Module: Model with loaded weights
    """
    weights_path = weights_info['weights_path']
    source = weights_info['source']
    
    if source == 'torchvision':
        # Load weights from TorchVision
        state_dict = torch.load(weights_path)
        
        # Handle different number of classes
        if num_classes is not None and num_classes != 1000:
            # Remove the classification head
            keys_to_remove = [k for k in state_dict.keys() if 'head' in k]
            for key in keys_to_remove:
                del state_dict[key]
            
            # Load partial state dict
            model.load_state_dict(state_dict, strict=False)
            
            # Reinitialize the classification head
            model.head = nn.Linear(model.embed_dim, num_classes)
        else:
            # Load full state dict
            model.load_state_dict(state_dict)
    
    elif source == 'huggingface':
        # Load weights from Hugging Face
        from transformers import ViTModel
        
        # Load pre-trained model
        pretrained_model = ViTModel.from_pretrained(weights_path)
        pretrained_state_dict = pretrained_model.state_dict()
        
        # Map Hugging Face state dict to our model
        # This requires careful mapping between different model structures
        # Here's a simplified example that assumes similar structure
        model_state_dict = model.state_dict()
        
        # Create a mapping between HF keys and our model keys
        key_mapping = {
            'embeddings.patch_embeddings.projection': 'patch_embed.projection',
            'embeddings.cls_token': 'cls_token',
            'embeddings.position_embeddings': 'pos_embed',
            'encoder.layer': 'blocks',
            'layernorm_before': 'norm1',
            'layernorm_after': 'norm2',
            'attention.attention.query': 'attn.qkv',  # Need special handling
            'attention.attention.key': 'attn.qkv',    # Need special handling
            'attention.attention.value': 'attn.qkv',  # Need special handling
            'attention.output.dense': 'attn.proj',
            'intermediate.dense': 'mlp.fc1',
            'output.dense': 'mlp.fc2'
        }
        
        # Create a new state dict with mapped keys
        new_state_dict = {}
        
        # Special handling for QKV
        qkv_q_dict = {}
        qkv_k_dict = {}
        qkv_v_dict = {}
        
        for key, value in pretrained_state_dict.items():
            # Skip certain keys
            if key.startswith('pooler') or key.startswith('classifier'):
                continue
                
            # Handle QKV weights specially
            if 'attention.attention.query' in key:
                layer_idx = key.split('.')[2]
                qkv_q_dict[layer_idx] = value
                continue
            elif 'attention.attention.key' in key:
                layer_idx = key.split('.')[2]
                qkv_k_dict[layer_idx] = value
                continue
            elif 'attention.attention.value' in key:
                layer_idx = key.split('.')[2]
                qkv_v_dict[layer_idx] = value
                continue
                
            # Map other keys
            mapped_key = key
            for hf_key, our_key in key_mapping.items():
                if hf_key in key:
                    mapped_key = key.replace(hf_key, our_key)
                    break
                    
            # Check if the mapped key exists in our model
            if mapped_key in model_state_dict:
                # Check if shapes match
                if value.shape == model_state_dict[mapped_key].shape:
                    new_state_dict[mapped_key] = value
                else:
                    print(f"Shape mismatch for {mapped_key}: {value.shape} vs {model_state_dict[mapped_key].shape}")
            else:
                print(f"Key {mapped_key} not found in model state dict")
                
        # Handle QKV weights
        for layer_idx in qkv_q_dict.keys():
            q = qkv_q_dict[layer_idx]
            k = qkv_k_dict[layer_idx]
            v = qkv_v_dict[layer_idx]
            
            # Concatenate Q, K, V weights
            qkv = torch.cat([q, k, v], dim=0)
            
            # Map to our model's key
            mapped_key = f'blocks.{layer_idx}.attn.qkv.weight'
            
            if mapped_key in model_state_dict:
                if qkv.shape == model_state_dict[mapped_key].shape:
                    new_state_dict[mapped_key] = qkv
                else:
                    print(f"Shape mismatch for {mapped_key}: {qkv.shape} vs {model_state_dict[mapped_key].shape}")
            else:
                print(f"Key {mapped_key} not found in model state dict")
                
        # Load partial state dict
        model.load_state_dict(new_state_dict, strict=False)
        
        # Handle classification head
        if num_classes is not None and hasattr(model, 'head'):
            model.head = nn.Linear(model.embed_dim, num_classes)
    
    # Freeze layers if specified
    if freeze_layers:
        if isinstance(freeze_layers, bool) and freeze_layers:
            # Freeze all layers except the head
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        elif isinstance(freeze_layers, list):
            # Freeze specific layers
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in freeze_layers):
                    param.requires_grad = False
    
    return model


def visualize_images(
    dataset: Dataset,
    num_images: int = 5,
    figsize: Tuple[int, int] = (15, 3),
    denormalize: bool = True
) -> None:
    """
    Visualize sample images from a dataset.
    
    Args:
        dataset (Dataset): PyTorch dataset
        num_images (int): Number of images to visualize
        figsize (Tuple[int, int]): Figure size
        denormalize (bool): Whether to denormalize images
    """
    # Get random indices
    indices = torch.randperm(len(dataset))[:num_images]
    
    # Create figure
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    # Function to denormalize image
    def denormalize_image(img):
        # Assuming img is a tensor with shape [3, H, W] and values in [0, 1]
        # This is a simple denormalization, might need adjustment for specific datasets
        return img * 0.5 + 0.5
    
    # Plot images
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        
        # Denormalize if needed
        if denormalize:
            img = denormalize_image(img)
        
        # Convert tensor to numpy and transpose
        img = img.numpy().transpose(1, 2, 0)
        
        # Clip values to [0, 1] for display
        img = np.clip(img, 0, 1)
        
        # Plot
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_patches(
    image: torch.Tensor,
    patch_size: int,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Visualize an image divided into patches.
    
    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W]
        patch_size (int): Size of each patch
        figsize (Tuple[int, int]): Figure size
    """
    # Convert tensor to numpy and transpose
    img = image.numpy().transpose(1, 2, 0)
    
    # Clip values to [0, 1] for display
    img = np.clip(img, 0, 1)
    
    # Get image dimensions
    _, h, w = image.shape
    
    # Calculate number of patches
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    # Create figure
    fig, axes = plt.subplots(num_patches_h, num_patches_w, figsize=figsize)
    
    # Plot original image
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()
    
    # Plot patches
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Extract patch
            patch = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
            
            # Plot
            if num_patches_h > 1 and num_patches_w > 1:
                axes[i, j].imshow(patch)
                axes[i, j].set_title(f"Patch ({i}, {j})")
                axes[i, j].axis('off')
            else:
                axes[i*num_patches_w + j].imshow(patch)
                axes[i*num_patches_w + j].set_title(f"Patch ({i}, {j})")
                axes[i*num_patches_w + j].axis('off')
    
    plt.tight_layout()
    plt.show()


def patchify_image(
    image: torch.Tensor,
    patch_size: int
) -> torch.Tensor:
    """
    Convert an image into patches.
    
    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W] or [B, C, H, W]
        patch_size (int): Size of each patch
        
    Returns:
        torch.Tensor: Patches tensor of shape [B, N, P*P*C] or [N, P*P*C]
    """
    if len(image.shape) == 3:
        # Single image
        c, h, w = image.shape
        
        # Ensure image dimensions are divisible by patch size
        assert h % patch_size == 0 and w % patch_size == 0, \
            f"Image dimensions ({h}, {w}) must be divisible by patch size {patch_size}"
        
        # Calculate number of patches
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Reshape image into patches
        patches = image.reshape(c, num_patches_h, patch_size, num_patches_w, patch_size)
        patches = patches.permute(1, 3, 2, 4, 0)  # [num_patches_h, num_patches_w, patch_size, patch_size, c]
        patches = patches.reshape(num_patches, patch_size * patch_size * c)
        
        return patches
    
    elif len(image.shape) == 4:
        # Batch of images
        b, c, h, w = image.shape
        
        # Ensure image dimensions are divisible by patch size
        assert h % patch_size == 0 and w % patch_size == 0, \
            f"Image dimensions ({h}, {w}) must be divisible by patch size {patch_size}"
        
        # Calculate number of patches
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Reshape image into patches
        patches = image.reshape(b, c, num_patches_h, patch_size, num_patches_w, patch_size)
        patches = patches.permute(0, 2, 4, 3, 5, 1)  # [b, num_patches_h, num_patches_w, patch_size, patch_size, c]
        patches = patches.reshape(b, num_patches, patch_size * patch_size * c)
        
        return patches
    
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def unpatchify_image(
    patches: torch.Tensor,
    patch_size: int,
    img_size: Tuple[int, int],
    channels: int = 3
) -> torch.Tensor:
    """
    Convert patches back to an image.
    
    Args:
        patches (torch.Tensor): Patches tensor of shape [B, N, P*P*C] or [N, P*P*C]
        patch_size (int): Size of each patch
        img_size (Tuple[int, int]): Original image size (H, W)
        channels (int): Number of channels
        
    Returns:
        torch.Tensor: Image tensor of shape [B, C, H, W] or [C, H, W]
    """
    h, w = img_size
    
    # Ensure image dimensions are divisible by patch size
    assert h % patch_size == 0 and w % patch_size == 0, \
        f"Image dimensions ({h}, {w}) must be divisible by patch size {patch_size}"
    
    # Calculate number of patches
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    num_patches = num_patches_h * num_patches_w
    
    if len(patches.shape) == 2:
        # Single image
        n, p = patches.shape
        assert n == num_patches, f"Number of patches {n} doesn't match expected {num_patches}"
        assert p == patch_size * patch_size * channels, f"Patch dimension {p} doesn't match expected {patch_size * patch_size * channels}"
        
        # Reshape patches into image
        patches = patches.reshape(num_patches_h, num_patches_w, patch_size, patch_size, channels)
        patches = patches.permute(4, 0, 2, 1, 3)  # [c, num_patches_h, patch_size, num_patches_w, patch_size]
        image = patches.reshape(channels, h, w)
        
        return image
    
    elif len(patches.shape) == 3:
        # Batch of images
        b, n, p = patches.shape
        assert n == num_patches, f"Number of patches {n} doesn't match expected {num_patches}"
        assert p == patch_size * patch_size * channels, f"Patch dimension {p} doesn't match expected {patch_size * patch_size * channels}"
        
        # Reshape patches into image
        patches = patches.reshape(b, num_patches_h, num_patches_w, patch_size, patch_size, channels)
        patches = patches.permute(0, 5, 1, 3, 2, 4)  # [b, c, num_patches_h, patch_size, num_patches_w, patch_size]
        image = patches.reshape(b, channels, h, w)
        
        return image
    
    else:
        raise ValueError(f"Unsupported patches shape: {patches.shape}")


def get_sample_batch(
    dataset_name: str = 'cifar10',
    batch_size: int = 4,
    img_size: int = 224
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a sample batch from a dataset.
    
    Args:
        dataset_name (str): Name of the dataset ('cifar10', 'imagenet', etc.)
        batch_size (int): Batch size
        img_size (int): Target image size
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batch of images and labels
    """
    if dataset_name.lower() == 'cifar10':
        data = load_cifar10(img_size=img_size, batch_size=batch_size, subset_size=batch_size)
        dataloader = data['train_loader']
    else:
        # Default to random tensors
        images = torch.randn(batch_size, 3, img_size, img_size)
        labels = torch.randint(0, 10, (batch_size,))
        return images, labels
    
    # Get first batch
    for images, labels in dataloader:
        return images, labels