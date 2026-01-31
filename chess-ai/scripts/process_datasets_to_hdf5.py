#!/usr/bin/env python3
"""
Convert datasets format to HDF5 for faster training.

This script processes games from HuggingFace datasets format
and converts them to HDF5 format for efficient training.
"""

import os
import argparse
import logging
from pathlib import Path

# Add parent directory to path
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, os.path.join(project_root, "chess-ai"))

from data.preprocessing import (
    get_project_root, get_dataset_path, load_dataset_from_datasets,
    process_game_from_datasets, save_dataset
)
from datasets import load_from_disk

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Convert datasets format to HDF5')
    parser.add_argument('--dataset-name', type=str, default='chess_dataset',
                       help='Name of dataset in data/processed/')
    parser.add_argument('--output-name', type=str, default='train.h5',
                       help='Output HDF5 filename')
    parser.add_argument('--min-elo', type=int, default=2600,
                       help='Minimum ELO rating')
    parser.add_argument('--max-games', type=int, default=None,
                       help='Maximum number of games to process (for testing)')
    
    args = parser.parse_args()
    
    # Get dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = get_dataset_path(args.dataset_name, script_path=script_dir)
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset_from_datasets(dataset_path)
    
    logger.info(f"Dataset info:")
    logger.info(f"  Total games: {len(dataset)}")
    logger.info(f"  Columns: {dataset.column_names}")
    
    # Process games
    all_positions = []
    processed_games = 0
    
    max_games = args.max_games if args.max_games else len(dataset)
    
    for idx, game_data in enumerate(dataset):
        if idx >= max_games:
            break
        
        # Filter by ELO
        white_elo = game_data.get('WhiteElo', '')
        black_elo = game_data.get('BlackElo', '')
        
        try:
            white_elo_int = int(white_elo) if white_elo and str(white_elo).isdigit() else 0
            black_elo_int = int(black_elo) if black_elo and str(black_elo).isdigit() else 0
        except (ValueError, TypeError):
            white_elo_int = 0
            black_elo_int = 0
        
        if white_elo_int < args.min_elo or black_elo_int < args.min_elo:
            continue
        
        # Process game
        try:
            positions = process_game_from_datasets(game_data)
            all_positions.extend(positions)
            processed_games += 1
            
            if (processed_games) % 100 == 0:
                logger.info(f"Processed {processed_games} games, "
                          f"{len(all_positions)} positions so far")
        except Exception as e:
            logger.warning(f"Error processing game {idx}: {e}")
            continue
    
    logger.info(f"\nProcessing complete:")
    logger.info(f"  Processed games: {processed_games}")
    logger.info(f"  Total positions: {len(all_positions)}")
    
    # Save to HDF5
    project_root = get_project_root(script_dir)
    output_path = os.path.join(project_root, "data", "processed", args.output_name)
    
    logger.info(f"Saving to: {output_path}")
    save_dataset(all_positions, output_path)
    
    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
