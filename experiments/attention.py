"""
Cross-Attention Vision Transformer Experiments

This module implements experiments for Vision Transformers with cross-attention
and multi-head cross-attention mechanisms. It supports both traditional ViT and
SPPP approaches, with options for training from scratch or using pre-trained weights.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attention import CrossAttentionViT, CrossAttentionSPPPViT
from utils.data_utils import load_cifar10, download_pretrained_vit_weights, load_pretrained_weights_to_model
from utils.metrics import (
    calculate_vit_complexity,
    measure_memory_usage,
    count_parameters,
    calculate_model_size
)


class CrossAttentionExperiment:
    """
    Experiment class for Vision Transformer with cross-attention.
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0,
        embed_dropout=0.0,
        batch_size=128,
        learning_rate=1e-4,
        weight_decay=0.05,
        epochs=50,
        device=None,
        data_dir='./project/data',
        results_dir='./project/results',
        subset_size=None,
        use_sppp=False,
        num_superpixels=16,
        compactness=0.1,
        pooling_type='mean',
        use_pretrained=False,
        pretrained_model_variant='vit_b_16',
        pretrained_source='torchvision',
        freeze_layers=False,
        use_multi_head=False
    ):
        """
        Initialize the experiment.
        
        Args:
            img_size (int): Input image size
            patch_size (int): Patch size
            in_channels (int): Number of input channels
            num_classes (int): Number of classes
            embed_dim (int): Embedding dimension
            depth (int): Number of transformer blocks
            num_heads (int): Number of attention heads
            mlp_ratio (float): MLP ratio
            dropout (float): Dropout rate
            attn_dropout (float): Attention dropout rate
            embed_dropout (float): Embedding dropout rate
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay
            epochs (int): Number of epochs
            device (torch.device): Device to use
            data_dir (str): Data directory
            results_dir (str): Results directory
            subset_size (int): Size of subset to use
            use_sppp (bool): Whether to use SPPP approach
            num_superpixels (int): Number of superpixels (for SPPP)
            compactness (float): Compactness parameter for SLIC (for SPPP)
            pooling_type (str): Type of pooling (for SPPP)
            use_pretrained (bool): Whether to use pre-trained weights
            pretrained_model_variant (str): Pre-trained model variant
            pretrained_source (str): Source of pre-trained weights
            freeze_layers (bool): Whether to freeze pre-trained layers
            use_multi_head (bool): Whether to use multi-head cross-attention
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.embed_dropout = embed_dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.subset_size = subset_size
        self.use_sppp = use_sppp
        self.num_superpixels = num_superpixels
        self.compactness = compactness
        self.pooling_type = pooling_type
        self.use_pretrained = use_pretrained
        self.pretrained_model_variant = pretrained_model_variant
        self.pretrained_source = pretrained_source
        self.freeze_layers = freeze_layers
        self.use_multi_head = use_multi_head
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize model, data, and metrics
        self.model = None
        self.data = None
        self.metrics = {}
        self.weights_info = None
        
        # Determine experiment type for file naming
        self.experiment_type = "multihead_cross_attention" if self.use_multi_head else "cross_attention"
        self.model_type = "sppp" if self.use_sppp else "traditional"
        self.weight_type = "pretrained" if self.use_pretrained else ""
        
    def setup(self):
        """Set up the model and data."""
        # Load data
        self.data = load_cifar10(
            data_dir=self.data_dir,
            img_size=self.img_size,
            batch_size=self.batch_size,
            subset_size=self.subset_size
        )
        
        # Download pre-trained weights if needed
        if self.use_pretrained:
            print(f"Downloading pre-trained weights for {self.pretrained_model_variant} from {self.pretrained_source}...")
            self.weights_info = download_pretrained_vit_weights(
                model_variant=self.pretrained_model_variant,
                source=self.pretrained_source,
                save_dir=os.path.join(self.data_dir, 'pretrained_weights')
            )
        
        # Create model based on configuration
        if self.use_sppp:
            print("Creating Cross-Attention SPPP Vision Transformer model...")
            self.model = CrossAttentionSPPPViT(
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                embed_dropout=self.embed_dropout,
                num_superpixels=self.num_superpixels,
                compactness=self.compactness,
                pooling_type=self.pooling_type,
                use_multi_head=self.use_multi_head
            )
        else:
            print("Creating Cross-Attention Vision Transformer model...")
            self.model = CrossAttentionViT(
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                embed_dropout=self.embed_dropout,
                use_multi_head=self.use_multi_head
            )
        
        # Load pre-trained weights if specified
        if self.use_pretrained:
            print("Loading pre-trained weights...")
            # For cross-attention models, we need to handle weight mapping differently
            # First, load weights into a standard ViT model
            from models.vit import VisionTransformer
            temp_model = VisionTransformer(
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio
            )
            
            temp_model = load_pretrained_weights_to_model(
                model=temp_model,
                weights_info=self.weights_info,
                num_classes=self.num_classes,
                freeze_layers=False  # Don't freeze yet, we'll handle this after mapping
            )
            
            # Now map weights from temp_model to our cross-attention model
            # This is a simplified mapping that works for both traditional and SPPP models
            # Patch embedding weights
            if hasattr(self.model.patch_embed, '0'):  # Sequential with Conv2d
                self.model.patch_embed[0].weight.data.copy_(temp_model.patch_embed.projection.weight.data.view(
                    self.embed_dim, self.in_channels, self.patch_size, self.patch_size
                ))
                if hasattr(self.model.patch_embed[0], 'bias') and self.model.patch_embed[0].bias is not None:
                    self.model.patch_embed[0].bias.data.copy_(temp_model.patch_embed.projection.bias.data)
            
            # Class token
            self.model.cls_token.data.copy_(temp_model.cls_token.data)
            
            # Position embedding (for traditional model)
            if hasattr(self.model, 'pos_embed') and not self.use_sppp:
                self.model.pos_embed.data.copy_(temp_model.pos_embed.data)
            
            # Transformer blocks
            for i in range(self.depth):
                # For cross-attention blocks, we need to map the weights differently
                # Query projection
                self.model.blocks[i].attn.q_proj.weight.data.copy_(
                    temp_model.blocks[i].attn.qkv.weight.data[:self.embed_dim, :]
                )
                self.model.blocks[i].attn.q_proj.bias.data.copy_(
                    temp_model.blocks[i].attn.qkv.bias.data[:self.embed_dim]
                )
                
                # Key projection
                self.model.blocks[i].attn.k_proj.weight.data.copy_(
                    temp_model.blocks[i].attn.qkv.weight.data[self.embed_dim:2*self.embed_dim, :]
                )
                self.model.blocks[i].attn.k_proj.bias.data.copy_(
                    temp_model.blocks[i].attn.qkv.bias.data[self.embed_dim:2*self.embed_dim]
                )
                
                # Value projection
                self.model.blocks[i].attn.v_proj.weight.data.copy_(
                    temp_model.blocks[i].attn.qkv.weight.data[2*self.embed_dim:, :]
                )
                self.model.blocks[i].attn.v_proj.bias.data.copy_(
                    temp_model.blocks[i].attn.qkv.bias.data[2*self.embed_dim:]
                )
                
                # Output projection
                self.model.blocks[i].attn.out_proj.weight.data.copy_(temp_model.blocks[i].attn.proj.weight.data)
                self.model.blocks[i].attn.out_proj.bias.data.copy_(temp_model.blocks[i].attn.proj.bias.data)
                
                # MLP weights
                self.model.blocks[i].mlp[0].weight.data.copy_(temp_model.blocks[i].mlp.fc1.weight.data)
                self.model.blocks[i].mlp[0].bias.data.copy_(temp_model.blocks[i].mlp.fc1.bias.data)
                self.model.blocks[i].mlp[3].weight.data.copy_(temp_model.blocks[i].mlp.fc2.weight.data)
                self.model.blocks[i].mlp[3].bias.data.copy_(temp_model.blocks[i].mlp.fc2.bias.data)
                
                # LayerNorm weights
                self.model.blocks[i].norm1_query.weight.data.copy_(temp_model.blocks[i].norm1.weight.data)
                self.model.blocks[i].norm1_query.bias.data.copy_(temp_model.blocks[i].norm1.bias.data)
                self.model.blocks[i].norm1_kv.weight.data.copy_(temp_model.blocks[i].norm1.weight.data)
                self.model.blocks[i].norm1_kv.bias.data.copy_(temp_model.blocks[i].norm1.bias.data)
                self.model.blocks[i].norm2.weight.data.copy_(temp_model.blocks[i].norm2.weight.data)
                self.model.blocks[i].norm2.bias.data.copy_(temp_model.blocks[i].norm2.bias.data)
            
            # Final LayerNorm
            self.model.norm.weight.data.copy_(temp_model.norm.weight.data)
            self.model.norm.bias.data.copy_(temp_model.norm.bias.data)
            
            # Classification head
            if self.num_classes == 1000:  # If using ImageNet classes
                self.model.head.weight.data.copy_(temp_model.head.weight.data)
                self.model.head.bias.data.copy_(temp_model.head.bias.data)
            
            # Apply freezing if specified
            if self.freeze_layers:
                if isinstance(self.freeze_layers, bool) and self.freeze_layers:
                    # Freeze all layers except the head
                    for name, param in self.model.named_parameters():
                        if 'head' not in name:
                            param.requires_grad = False
                elif isinstance(self.freeze_layers, list):
                    # Freeze specific layers
                    for name, param in self.model.named_parameters():
                        if any(layer_name in name for layer_name in self.freeze_layers):
                            param.requires_grad = False
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Calculate theoretical complexity
        if self.use_sppp:
            # For SPPP, we modify the complexity calculation to account for reduced tokens
            num_patches = (self.img_size // self.patch_size) ** 2
            num_tokens_traditional = num_patches + 1  # +1 for class token
            num_tokens_sppp = self.num_superpixels + 1  # +1 for class token
            
            # Calculate traditional complexity for comparison
            traditional_complexity = calculate_vit_complexity(
                img_size=self.img_size,
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                in_channels=self.in_channels
            )
            
            # Adjust complexity for SPPP
            # The main difference is in the attention mechanism, which is O(N²)
            # So we scale the FLOPs and time complexity by (num_tokens_sppp / num_tokens_traditional)²
            token_ratio = num_tokens_sppp / num_tokens_traditional
            attention_scaling = token_ratio ** 2
            
            # For SPPP, we need to add the overhead of superpixel segmentation and pooling
            # This is a rough estimate based on the SLIC algorithm complexity
            superpixel_overhead_flops = self.img_size * self.img_size * 10  # Approximate SLIC complexity
            pooling_overhead_flops = num_patches * self.embed_dim  # Pooling complexity
            
            # Calculate SPPP complexity
            sppp_complexity = {
                'parameters': traditional_complexity['parameters'],  # Same number of parameters
                'flops': traditional_complexity['flops'] * attention_scaling + superpixel_overhead_flops + pooling_overhead_flops,
                'time_complexity': traditional_complexity['time_complexity'] * attention_scaling + superpixel_overhead_flops + pooling_overhead_flops,
                'space_complexity_bytes': traditional_complexity['space_complexity_bytes'] * token_ratio,  # Reduced activation memory
                'space_complexity_mb': traditional_complexity['space_complexity_bytes'] * token_ratio / (1024 * 1024)
            }
            
            self.metrics['theoretical'] = sppp_complexity
            self.metrics['traditional_complexity'] = traditional_complexity
            self.metrics['token_reduction'] = {
                'traditional_tokens': num_tokens_traditional,
                'sppp_tokens': num_tokens_sppp,
                'reduction_factor': num_tokens_traditional / num_tokens_sppp
            }
        else:
            # For traditional ViT with cross-attention, use standard complexity calculation
            self.metrics['theoretical'] = calculate_vit_complexity(
                img_size=self.img_size,
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                in_channels=self.in_channels
            )
        
        # Calculate model size
        self.metrics['model_size'] = calculate_model_size(self.model)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.metrics['trainable_params'] = trainable_params
        self.metrics['total_params'] = total_params
        self.metrics['frozen_params'] = total_params - trainable_params
        
        print(f"Model setup complete. Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        
    def train(self):
        """Train the model and record metrics."""
        # Set up optimizer with different learning rates for pre-trained layers and new layers
        if self.use_pretrained and self.freeze_layers:
            # If layers are frozen, only optimize the head
            optimizer = optim.AdamW(
                self.model.head.parameters(),
                lr=self.learning_rate * 10,  # Higher learning rate for head
                weight_decay=self.weight_decay
            )
        else:
            # Use standard optimization
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        epoch_times = []
        memory_usage = []
        
        # Get a sample batch for memory measurement
        sample_images, sample_labels = next(iter(self.data['train_loader']))
        sample_images = sample_images.to(self.device)
        sample_labels = sample_labels.to(self.device)
        
        # Measure initial memory usage
        initial_memory = measure_memory_usage(self.model, sample_images)
        memory_usage.append(initial_memory)
        
        # Training loop
        total_start_time = time.time()
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Training
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in self.data['train_loader']:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(self.data['train_loader'])
            train_acc = 100.0 * correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in self.data['test_loader']:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(self.data['test_loader'])
            val_acc = 100.0 * correct / total
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Measure memory usage during training
            if epoch == self.epochs // 2:  # Measure in the middle of training
                memory_usage.append(measure_memory_usage(self.model, sample_images, backward=True))
            
            # Record epoch time
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_time)
            
            # Print progress
            print(f'Epoch {epoch+1}/{self.epochs} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
                  f'Time: {epoch_time:.2f}s')
        
        # Measure final memory usage
        final_memory = measure_memory_usage(self.model, sample_images)
        memory_usage.append(final_memory)
        
        # Record total training time
        total_end_time = time.time()
        total_training_time = total_end_time - total_start_time
        
        # Record metrics
        self.metrics['training'] = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'epoch_times': epoch_times,
            'avg_epoch_time': np.mean(epoch_times),
            'total_training_time': total_training_time,
            'memory_usage': memory_usage,
            'final_val_acc': val_accs[-1],
            'final_val_loss': val_losses[-1]
        }
        
    def evaluate(self):
        """Evaluate the model on the test set."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        # Measure inference time
        inference_times = []
        
        with torch.no_grad():
            for images, labels in self.data['test_loader']:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = self.model(images)
                end_time = time.time()
                inference_times.append(end_time - start_time)
                
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_loss /= len(self.data['test_loader'])
        test_acc = 100.0 * correct / total
        avg_inference_time = np.mean(inference_times)
        avg_inference_time_per_image = avg_inference_time / self.batch_size
        
        # Record metrics
        self.metrics['evaluation'] = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'avg_inference_time': avg_inference_time,
            'avg_inference_time_per_image': avg_inference_time_per_image
        }
        
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | '
              f'Avg Inference Time per Batch: {avg_inference_time:.4f}s | '
              f'Avg Inference Time per Image: {avg_inference_time_per_image:.4f}s')
        
    def save_results(self):
        """Save the results to a CSV file."""
        # Determine file name based on experiment configuration
        if self.use_pretrained:
            file_name = f"exp{5 if self.use_multi_head else 4}_{self.experiment_type}_pretrained_{self.model_type}.csv"
        else:
            file_name = f"exp{5 if self.use_multi_head else 4}_{self.experiment_type}_{self.model_type}.csv"
        
        # Combine all metrics into a single dictionary
        results = {
            'model': f"{'MultiHead ' if self.use_multi_head else ''}CrossAttention {'SPPP ' if self.use_sppp else ''}ViT",
            'use_pretrained': self.use_pretrained,
            'pretrained_source': self.pretrained_source if self.use_pretrained else "None",
            'pretrained_model_variant': self.pretrained_model_variant if self.use_pretrained else "None",
            'freeze_layers': str(self.freeze_layers) if self.use_pretrained else "False",
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'use_multi_head': self.use_multi_head,
            'total_parameters': self.metrics['total_params'],
            'trainable_parameters': self.metrics['trainable_params'],
            'frozen_parameters': self.metrics['frozen_params'],
            'flops': self.metrics['theoretical']['flops'],
            'time_complexity': self.metrics['theoretical']['time_complexity'],
            'space_complexity_mb': self.metrics['theoretical']['space_complexity_mb'],
            'model_size_mb': self.metrics['model_size']['size_mb'],
            'avg_epoch_time': self.metrics['training']['avg_epoch_time'],
            'total_training_time': self.metrics['training']['total_training_time'],
            'final_val_acc': self.metrics['training']['final_val_acc'],
            'final_val_loss': self.metrics['training']['final_val_loss'],
            'test_acc': self.metrics['evaluation']['test_acc'],
            'test_loss': self.metrics['evaluation']['test_loss'],
            'avg_inference_time_per_image': self.metrics['evaluation']['avg_inference_time_per_image'],
            'peak_gpu_memory_mb': max([m['gpu_memory_peak_mb'] for m in self.metrics['training']['memory_usage'] if 'gpu_memory_peak_mb' in m])
        }
        
        # Add SPPP-specific metrics if applicable
        if self.use_sppp:
            results.update({
                'num_superpixels': self.num_superpixels,
                'traditional_tokens': self.metrics['token_reduction']['traditional_tokens'],
                'sppp_tokens': self.metrics['token_reduction']['sppp_tokens'],
                'token_reduction_factor': self.metrics['token_reduction']['reduction_factor'],
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame([results])
        
        # Save to CSV
        csv_path = os.path.join(self.results_dir, file_name)
        results_df.to_csv(csv_path, index=False)
        print(f'Results saved to {csv_path}')
        
    def run(self):
        """Run the experiment."""
        print("Setting up experiment...")
        self.setup()
        
        print("Starting training...")
        self.train()
        
        print("Evaluating model...")
        self.evaluate()
        
        print("Saving results...")
        self.save_results()
        
        print("Experiment completed!")


def run_cross_attention_experiments(args):
    """
    Run all cross-attention experiments.
    
    Args:
        args: Command line arguments
    """
    # Experiment 4: Cross-Attention
    # Traditional ViT with Cross-Attention
    print("\n" + "="*50)
    print("Running Experiment 4A: Traditional ViT with Cross-Attention")
    print("="*50)
    
    traditional_ca_experiment = CrossAttentionExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        subset_size=args.subset_size,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        use_sppp=False,
        use_pretrained=False,
        use_multi_head=False
    )
    traditional_ca_experiment.run()
    
    # SPPP ViT with Cross-Attention
    print("\n" + "="*50)
    print("Running Experiment 4B: SPPP ViT with Cross-Attention")
    print("="*50)
    
    sppp_ca_experiment = CrossAttentionExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        subset_size=args.subset_size,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        use_sppp=True,
        num_superpixels=args.num_superpixels,
        compactness=args.compactness,
        pooling_type=args.pooling_type,
        use_pretrained=False,
        use_multi_head=False
    )
    sppp_ca_experiment.run()
    
    # Traditional ViT with Cross-Attention and Pre-trained Weights
    print("\n" + "="*50)
    print("Running Experiment 4C: Traditional ViT with Cross-Attention and Pre-trained Weights")
    print("="*50)
    
    traditional_ca_pretrained_experiment = CrossAttentionExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        subset_size=args.subset_size,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        use_sppp=False,
        use_pretrained=True,
        pretrained_model_variant=args.pretrained_model_variant,
        pretrained_source=args.pretrained_source,
        freeze_layers=args.freeze_layers,
        use_multi_head=False
    )
    traditional_ca_pretrained_experiment.run()
    
    # SPPP ViT with Cross-Attention and Pre-trained Weights
    print("\n" + "="*50)
    print("Running Experiment 4D: SPPP ViT with Cross-Attention and Pre-trained Weights")
    print("="*50)
    
    sppp_ca_pretrained_experiment = CrossAttentionExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        subset_size=args.subset_size,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        use_sppp=True,
        num_superpixels=args.num_superpixels,
        compactness=args.compactness,
        pooling_type=args.pooling_type,
        use_pretrained=True,
        pretrained_model_variant=args.pretrained_model_variant,
        pretrained_source=args.pretrained_source,
        freeze_layers=args.freeze_layers,
        use_multi_head=False
    )
    sppp_ca_pretrained_experiment.run()


def run_multihead_cross_attention_experiments(args):
    """
    Run all multi-head cross-attention experiments.
    
    Args:
        args: Command line arguments
    """
    # Experiment 5: Multi-Head Cross-Attention
    # Traditional ViT with Multi-Head Cross-Attention
    print("\n" + "="*50)
    print("Running Experiment 5A: Traditional ViT with Multi-Head Cross-Attention")
    print("="*50)
    
    traditional_mhca_experiment = CrossAttentionExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        subset_size=args.subset_size,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        use_sppp=False,
        use_pretrained=False,
        use_multi_head=True
    )
    traditional_mhca_experiment.run()
    
    # SPPP ViT with Multi-Head Cross-Attention
    print("\n" + "="*50)
    print("Running Experiment 5B: SPPP ViT with Multi-Head Cross-Attention")
    print("="*50)
    
    sppp_mhca_experiment = CrossAttentionExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        subset_size=args.subset_size,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        use_sppp=True,
        num_superpixels=args.num_superpixels,
        compactness=args.compactness,
        pooling_type=args.pooling_type,
        use_pretrained=False,
        use_multi_head=True
    )
    sppp_mhca_experiment.run()
    
    # Traditional ViT with Multi-Head Cross-Attention and Pre-trained Weights
    print("\n" + "="*50)
    print("Running Experiment 5C: Traditional ViT with Multi-Head Cross-Attention and Pre-trained Weights")
    print("="*50)
    
    traditional_mhca_pretrained_experiment = CrossAttentionExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        subset_size=args.subset_size,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        use_sppp=False,
        use_pretrained=True,
        pretrained_model_variant=args.pretrained_model_variant,
        pretrained_source=args.pretrained_source,
        freeze_layers=args.freeze_layers,
        use_multi_head=True
    )
    traditional_mhca_pretrained_experiment.run()
    
    # SPPP ViT with Multi-Head Cross-Attention and Pre-trained Weights
    print("\n" + "="*50)
    print("Running Experiment 5D: SPPP ViT with Multi-Head Cross-Attention and Pre-trained Weights")
    print("="*50)
    
    sppp_mhca_pretrained_experiment = CrossAttentionExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        subset_size=args.subset_size,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        use_sppp=True,
        num_superpixels=args.num_superpixels,
        compactness=args.compactness,
        pooling_type=args.pooling_type,
        use_pretrained=True,
        pretrained_model_variant=args.pretrained_model_variant,
        pretrained_source=args.pretrained_source,
        freeze_layers=args.freeze_layers,
        use_multi_head=True
    )
    sppp_mhca_pretrained_experiment.run()


def main():
    """Main function to run the experiments."""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Cross-Attention Vision Transformer Experiments')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--num_superpixels', type=int, default=16, help='Number of superpixels')
    parser.add_argument('--compactness', type=float, default=0.1, help='Compactness parameter for SLIC')
    parser.add_argument('--pooling_type', type=str, default='mean', choices=['mean', 'max', 'attention'], help='Type of pooling')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--subset_size', type=int, default=None, help='Size of subset to use')
    parser.add_argument('--data_dir', type=str, default='./project/data', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='./project/results', help='Results directory')
    parser.add_argument('--pretrained_model_variant', type=str, default='vit_b_16', help='Pretrained model variant')
    parser.add_argument('--pretrained_source', type=str, default='torchvision', choices=['torchvision', 'huggingface'], help='Source of pretrained weights')
    parser.add_argument('--freeze_layers', action='store_true', help='Whether to freeze pretrained layers')
    parser.add_argument('--experiment', type=str, default='all', choices=['all', 'cross_attention', 'multihead_cross_attention'], help='Experiment to run')
    args = parser.parse_args()
    
    # Run experiments based on selection
    if args.experiment == 'all' or args.experiment == 'cross_attention':
        run_cross_attention_experiments(args)
    
    if args.experiment == 'all' or args.experiment == 'multihead_cross_attention':
        run_multihead_cross_attention_experiments(args)


if __name__ == '__main__':
    main()