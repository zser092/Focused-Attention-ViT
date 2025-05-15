"""
Traditional Vision Transformer Training Experiment

This module implements a standard training pipeline for the Vision Transformer model.
It calculates and records theoretical space/time complexity and actual resource usage.
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

from models.vit import VisionTransformer
from utils.data_utils import load_cifar10
from utils.metrics import (
    calculate_vit_complexity,
    measure_memory_usage,
    count_parameters,
    calculate_model_size
)


class TraditionalViTExperiment:
    """
    Experiment class for traditional Vision Transformer training.
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
        subset_size=None
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
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize model, data, and metrics
        self.model = None
        self.data = None
        self.metrics = {}
        
    def setup(self):
        """Set up the model and data."""
        # Load data
        self.data = load_cifar10(
            data_dir=self.data_dir,
            img_size=self.img_size,
            batch_size=self.batch_size,
            subset_size=self.subset_size
        )
        
        # Create model
        self.model = VisionTransformer(
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
            embed_dropout=self.embed_dropout
        ).to(self.device)
        
        # Calculate theoretical complexity
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
        
    def train(self):
        """Train the model and record metrics."""
        # Set up optimizer and loss function
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
        # Combine all metrics into a single dictionary
        results = {
            'model': 'Traditional ViT',
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'parameters': self.metrics['theoretical']['parameters'],
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
        
        # Convert to DataFrame
        results_df = pd.DataFrame([results])
        
        # Save to CSV
        csv_path = os.path.join(self.results_dir, 'exp1_traditional.csv')
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


def main():
    """Main function to run the experiment."""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Traditional ViT Experiment')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--subset_size', type=int, default=None, help='Size of subset to use')
    parser.add_argument('--data_dir', type=str, default='./project/data', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='./project/results', help='Results directory')
    args = parser.parse_args()
    
    # Run experiment
    experiment = TraditionalViTExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        subset_size=args.subset_size,
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    experiment.run()


if __name__ == '__main__':
    main()