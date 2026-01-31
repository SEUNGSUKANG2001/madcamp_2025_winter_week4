#!/usr/bin/env python3
"""
Training script with enhanced logging for quick testing.
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
from data.dataset import ChessDataset
from torch.utils.data import DataLoader, Subset

# Setup logging to both console and file
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
logger.info("Logging to both console and training.log file")


def main():
    parser = argparse.ArgumentParser(description='Train chess AI with enhanced logging')
    parser.add_argument('--train-dataset-name', type=str, default='lichess-2500-180_dataset',
                       help='Name of training dataset in data/processed/')
    parser.add_argument('--val-dataset-name', type=str, default=None,
                       help='Name of validation dataset (default: same as train)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to train on (cuda/cpu/auto)')
    parser.add_argument('--min-elo', type=int, default=2600,
                       help='Minimum ELO rating for filtering')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to use (for quick testing)')
    parser.add_argument('--max-games', type=int, default=None,
                       help='Maximum number of games to process (for quick testing)')
    parser.add_argument('--cache-in-memory', action='store_true',
                       help='Cache dataset in memory (faster but uses more RAM)')
    
    args = parser.parse_args()
    
    script_path = os.path.abspath(__file__)
    
    # Use same dataset for validation if not specified
    val_dataset_name = args.val_dataset_name or args.train_dataset_name
    
    logger.info("=" * 60)
    logger.info("Chess AI Training - Enhanced Logging")
    logger.info("=" * 60)
    logger.info(f"Train dataset: {args.train_dataset_name}")
    logger.info(f"Val dataset: {val_dataset_name}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max samples: {args.max_samples or 'All'}")
    logger.info(f"Max games: {args.max_games or 'All'}")
    logger.info(f"Cache in memory: {args.cache_in_memory}")
    logger.info("=" * 60)
    
    # Create datasets
    logger.info("\n[1/4] Loading training dataset...")
    train_dataset = ChessDataset(
        dataset_name=args.train_dataset_name,
        format='auto',
        augment=True,
        cache_in_memory=args.cache_in_memory,
        min_elo=args.min_elo,
        script_path=script_path,
        max_games=args.max_games
    )
    
    logger.info(f"Training dataset loaded: {len(train_dataset)} samples")
    
    logger.info("\n[2/4] Loading validation dataset...")
    val_dataset = ChessDataset(
        dataset_name=val_dataset_name,
        format='auto',
        augment=False,
        cache_in_memory=args.cache_in_memory,
        min_elo=args.min_elo,
        script_path=script_path,
        max_games=(args.max_games // 10) if args.max_games else None
    )
    logger.info(f"Validation dataset loaded: {len(val_dataset)} samples")
    
    # Limit samples if specified
    if args.max_samples:
        logger.info(f"\nLimiting to {args.max_samples} samples for quick testing...")
        train_indices = list(range(min(args.max_samples, len(train_dataset))))
        val_indices = list(range(min(args.max_samples // 10, len(val_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create data loaders
    logger.info("\n[3/4] Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if not args.cache_in_memory else 0,
        pin_memory=True,
        persistent_workers=(2 > 0) if not args.cache_in_memory else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1 if not args.cache_in_memory else 0,
        pin_memory=True,
        persistent_workers=(1 > 0) if not args.cache_in_memory else False
    )
    
    logger.info(f"Data loaders created: train batches={len(train_loader)}, val batches={len(val_loader)}")
    
    # Create model
    logger.info("\n[4/4] Initializing model...")
    model = HybridChessNet()
    param_count = model.count_parameters()
    logger.info(f"Model initialized: {param_count:,} parameters")
    
    # Training configuration
    config = {
        'device': args.device,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'epochs': args.epochs,
        'value_weight': 1.0,
        'use_mixed_precision': True,
        'checkpoint_dir': args.checkpoint_dir
    }
    
    # Create trainer
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60 + "\n")
    
    trainer = SupervisedTrainer(model, train_loader, val_loader, config)
    trainer.train(num_epochs=args.epochs, save_every=10)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
