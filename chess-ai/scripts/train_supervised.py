#!/usr/bin/env python3
"""
Main script for supervised learning training.
"""

import os
import sys
import argparse
import logging
import torch
from pathlib import Path

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, os.path.join(project_root, "chess-ai"))

from models.architecture import HybridChessNet
from training.supervised import SupervisedTrainer
from data.dataset import create_data_loaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train chess AI with supervised learning')
    parser.add_argument('--train-data', type=str, default=None,
                       help='Path to training dataset file/directory')
    parser.add_argument('--val-data', type=str, default=None,
                       help='Path to validation dataset file/directory')
    parser.add_argument('--train-dataset-name', type=str, default=None,
                       help='Name of training dataset in data/processed/')
    parser.add_argument('--val-dataset-name', type=str, default=None,
                       help='Name of validation dataset in data/processed/')
    parser.add_argument('--format', type=str, default='auto',
                       choices=['auto', 'hdf5', 'datasets'],
                       help='Dataset format')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size (default: 512, increase for better VRAM utilization)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to train on (cuda/cpu/auto). auto will use CUDA if available, otherwise CPU')
    parser.add_argument('--min-elo', type=int, default=2600,
                       help='Minimum ELO rating for filtering')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of worker processes for data loading (default: 8, use more for lazy loading)')
    parser.add_argument('--wandb-project', type=str, default='chess-ai',
                       help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='Wandb run name (default: auto-generated)')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Wandb entity/team name')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Get script path for project root detection
    script_path = os.path.abspath(__file__)
    
    # Create data loaders
    logger.info("Loading datasets...")
    train_loader, val_loader = create_data_loaders(
        train_path=args.train_data,
        val_path=args.val_data,
        train_dataset_name=args.train_dataset_name,
        val_dataset_name=args.val_dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=True,
        format=args.format,
        min_elo=args.min_elo,
        script_path=script_path
    )
    
    # Create model
    logger.info("Initializing model...")
    model = HybridChessNet()
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Training configuration
    config = {
        'device': args.device,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'epochs': args.epochs,
        'value_weight': 1.0,
        'use_mixed_precision': True,
        'checkpoint_dir': args.checkpoint_dir,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'use_wandb': not args.no_wandb,
        'wandb_project': args.wandb_project,
        'wandb_run_name': args.wandb_run_name,
        'wandb_entity': args.wandb_entity,
    }
    
    # Log effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    logger.info(f"Batch size: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {effective_batch_size}")
    
    # Create trainer
    trainer = SupervisedTrainer(model, train_loader, val_loader, config)
    
    # Train
    logger.info("Starting training...")
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
