"""
Main Script for Vision Transformer Experiments

This script serves as the entry point for running all experiments related to
Vision Transformers with different strategies.
"""

import os
import torch
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import sys
import time

# Import project modules
from models.vit import VisionTransformer
from models.sppp import SPPPViT
from models.attention import CrossAttentionViT, CrossAttentionSPPPViT
from models.vit_mhla import VisionTransformerMHLA
from models.sppp_mhla import SPPPViTMHLA
from models.mhla_models import PretrainedViTWithMHLA, PretrainedSPPPViTWithMHLA

from utils.metrics import (
    calculate_model_size,
    calculate_vit_complexity,
    benchmark_model
)
from utils.data_utils import (
    load_cifar10,
    get_sample_batch,
    visualize_images,
    visualize_patches,
    patchify_image
)

# Import experiment modules
from experiments.traditional import TraditionalViTExperiment
from experiments.traditional_pretrained import TraditionalPretrainedViTExperiment
from experiments.sppp import SPPPViTExperiment
from experiments.sppp_pretrained import SPPPPretrainedViTExperiment
from experiments.attention import (
    run_cross_attention_experiments,
    run_multihead_cross_attention_experiments
)
# Import new MHLA experiment modules
from experiments.mhla_pretrained import PretrainedMHLAViTExperiment
from experiments.sppp_mhla_pretrained import PretrainedSPPPMHLAExperiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vit_experiments.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Vision Transformer Experiments')
    
    # General settings
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['traditional', 'traditional_pretrained', 
                                 'sppp', 'sppp_pretrained', 
                                 'cross_attention', 'multihead_cross_attention',
                                 'mhla_pretrained', 'sppp_mhla_pretrained'],
                        help='Experiment to run')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to store datasets')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to store results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Dataset to use')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Size of subset to use (for debugging)')
    
    # Model settings
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=12,
                        help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help='MLP ratio')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--attn_dropout', type=float, default=0.0,
                        help='Attention dropout rate')
    parser.add_argument('--embed_dropout', type=float, default=0.0,
                        help='Embedding dropout rate')
    
    # SPPP settings
    parser.add_argument('--num_superpixels', type=int, default=16,
                        help='Number of superpixels')
    parser.add_argument('--compactness', type=float, default=0.1,
                        help='Compactness parameter for SLIC')
    parser.add_argument('--pooling_type', type=str, default='mean',
                        choices=['mean', 'max', 'attention'],
                        help='Type of pooling')
    
    # MHLA settings
    parser.add_argument('--window_size', type=int, default=7,
                        help='Size of local attention window for MHLA')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    
    # Pretrained settings
    parser.add_argument('--pretrained_model_variant', type=str, default='vit_b_16',
                        help='Pretrained model variant')
    parser.add_argument('--pretrained_source', type=str, default='torchvision',
                        choices=['torchvision', 'huggingface'],
                        help='Source of pretrained weights')
    parser.add_argument('--freeze_layers', action='store_true',
                        help='Whether to freeze pretrained layers')
    parser.add_argument('--head_learning_rate', type=float, default=1e-3,
                        help='Learning rate for classification head')
    
    # Visualization settings
    parser.add_argument('--visualize', action='store_true',
                        help='Whether to visualize results')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_traditional_experiment(args):
    """Run traditional ViT experiment."""
    logger.info("Running traditional ViT experiment...")
    
    experiment = TraditionalViTExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=10 if args.dataset == 'cifar10' else 100,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        embed_dropout=args.embed_dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        subset_size=args.subset_size
    )
    experiment.run()


def run_traditional_pretrained_experiment(args):
    """Run traditional ViT experiment with pretrained weights."""
    logger.info("Running traditional ViT experiment with pretrained weights...")
    
    experiment = TraditionalPretrainedViTExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=10 if args.dataset == 'cifar10' else 100,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        embed_dropout=args.embed_dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        subset_size=args.subset_size,
        pretrained_model_variant=args.pretrained_model_variant,
        pretrained_source=args.pretrained_source,
        freeze_layers=args.freeze_layers,
        head_learning_rate=args.head_learning_rate
    )
    experiment.run()


def run_sppp_experiment(args):
    """Run SPPP ViT experiment."""
    logger.info("Running SPPP ViT experiment...")
    
    experiment = SPPPViTExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=10 if args.dataset == 'cifar10' else 100,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        embed_dropout=args.embed_dropout,
        num_superpixels=args.num_superpixels,
        compactness=args.compactness,
        pooling_type=args.pooling_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        subset_size=args.subset_size
    )
    experiment.run()


def run_sppp_pretrained_experiment(args):
    """Run SPPP ViT experiment with pretrained weights."""
    logger.info("Running SPPP ViT experiment with pretrained weights...")
    
    experiment = SPPPPretrainedViTExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=10 if args.dataset == 'cifar10' else 100,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        embed_dropout=args.embed_dropout,
        num_superpixels=args.num_superpixels,
        compactness=args.compactness,
        pooling_type=args.pooling_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        subset_size=args.subset_size,
        pretrained_model_variant=args.pretrained_model_variant,
        pretrained_source=args.pretrained_source,
        freeze_layers=args.freeze_layers,
        head_learning_rate=args.head_learning_rate
    )
    experiment.run()


def run_mhla_pretrained_experiment(args):
    """Run ViT + MHLA experiment with pretrained weights."""
    logger.info("Running ViT + MHLA experiment with pretrained weights...")
    
    experiment = PretrainedMHLAViTExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=10 if args.dataset == 'cifar10' else 100,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        embed_dropout=args.embed_dropout,
        window_size=args.window_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        subset_size=args.subset_size,
        pretrained_model_variant=args.pretrained_model_variant,
        pretrained_source=args.pretrained_source,
        freeze_layers=args.freeze_layers,
        head_learning_rate=args.head_learning_rate
    )
    experiment.run()


def run_sppp_mhla_pretrained_experiment(args):
    """Run SPPP + MHLA experiment with pretrained weights."""
    logger.info("Running SPPP + MHLA experiment with pretrained weights...")
    
    experiment = PretrainedSPPPMHLAExperiment(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=10 if args.dataset == 'cifar10' else 100,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        embed_dropout=args.embed_dropout,
        num_superpixels=args.num_superpixels,
        compactness=args.compactness,
        pooling_type=args.pooling_type,
        window_size=args.window_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        subset_size=args.subset_size,
        pretrained_model_variant=args.pretrained_model_variant,
        pretrained_source=args.pretrained_source,
        freeze_layers=args.freeze_layers,
        head_learning_rate=args.head_learning_rate
    )
    experiment.run()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Log experiment settings
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    
    # Run experiment
    if args.experiment == 'traditional':
        run_traditional_experiment(args)
    elif args.experiment == 'traditional_pretrained':
        run_traditional_pretrained_experiment(args)
    elif args.experiment == 'sppp':
        run_sppp_experiment(args)
    elif args.experiment == 'sppp_pretrained':
        run_sppp_pretrained_experiment(args)
    elif args.experiment == 'cross_attention':
        run_cross_attention_experiments(args)
    elif args.experiment == 'multihead_cross_attention':
        run_multihead_cross_attention_experiments(args)
    elif args.experiment == 'mhla_pretrained':
        run_mhla_pretrained_experiment(args)
    elif args.experiment == 'sppp_mhla_pretrained':
        run_sppp_mhla_pretrained_experiment(args)
    else:
        logger.error(f"Unknown experiment: {args.experiment}")


if __name__ == '__main__':
    main()