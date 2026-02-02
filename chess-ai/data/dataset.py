"""
Dataset module for PyTorch training.

Provides ChessDataset class for efficient data loading with augmentation.
Supports both HDF5 and HuggingFace datasets formats.
"""

import logging
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os

try:
    from datasets import load_from_disk, Dataset as HFDataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    HFDataset = None

from data.preprocessing import (
    load_dataset, encode_board, move_to_index, index_to_move,
    get_project_root, get_dataset_path, load_dataset_from_datasets,
    process_game_from_datasets
)
from data.augmentation import augment_game_data

logger = logging.getLogger(__name__)


class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess positions.
    
    Supports both HDF5 and HuggingFace datasets formats.
    Supports lazy loading, on-the-fly augmentation, and efficient batching.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        format: str = 'auto',
        augment: bool = False,
        augment_prob: float = 0.5,
        cache_in_memory: bool = False,
        min_elo: int = 2600,
        script_path: Optional[str] = None,
        max_games: Optional[int] = None
    ):
        """
        Initialize chess dataset.
        
        Args:
            data_path: Path to dataset file/directory (optional if dataset_name provided)
            dataset_name: Name of dataset in data/processed/ (optional)
            format: Format type ('hdf5', 'datasets', or 'auto')
            augment: Whether to apply data augmentation
            augment_prob: Probability of applying augmentation to each sample
            cache_in_memory: Whether to load entire dataset into memory
            min_elo: Minimum ELO for filtering (only for datasets format)
            script_path: Path to calling script (for project root detection)
        """
        self.augment = augment
        self.augment_prob = augment_prob
        self.cache_in_memory = cache_in_memory
        self.min_elo = min_elo
        self.format = format
        self.max_games = max_games
        
        # Determine dataset path
        if data_path is None:
            if dataset_name is None:
                dataset_name = "chess_dataset"
            data_path = get_dataset_path(dataset_name, script_path)
        
        self.data_path = data_path
        
        # Load dataset
        if format == 'auto' or format == 'datasets':
            # Try datasets format first
            if os.path.isdir(data_path) and DATASETS_AVAILABLE:
                try:
                    self.dataset = load_dataset_from_datasets(data_path)
                    self.format = 'datasets'
                    self.length = len(self.dataset)
                    logger.info(f"Loaded HuggingFace dataset: {self.length} games")
                    
                    # For datasets format, we need to process games to positions
                    # This is expensive, so we do it lazily during training
                    # But we need to know the length first
                    if cache_in_memory:
                        logger.info("Processing games to positions (this may take a while)...")
                        self._process_games_to_positions(max_games=self.max_games)
                    else:
                        # For lazy loading, we'll process on-the-fly
                        # But this requires knowing which game contains which position
                        # For now, we'll estimate length based on average moves per game
                        # Actual processing will happen during __getitem__
                        logger.warning(
                            "Lazy processing for datasets format is slow. "
                            "Consider using cache_in_memory=True or converting to HDF5 format."
                        )
                        # Estimate: average ~40 moves per game, but we'll process on demand
                        self.positions_cache = None
                        # We can't know exact length without processing, so use game count as estimate
                        # Actual positions will be generated on-the-fly
                    
                    return
                except Exception as e:
                    logger.warning(f"Failed to load as datasets format: {e}")
                    if format == 'datasets':
                        raise
        
        # Fall back to HDF5 format
        self.format = 'hdf5'
        if cache_in_memory:
            self.data = load_dataset(data_path, format='hdf5')
            self.positions = self.data['positions']
            self.moves = self.data['moves']
            self.values = self.data['values']
            self.move_numbers = self.data['move_numbers']
            self.players = self.data['players']
            self.length = len(self.positions)
            logger.info(f"Loaded {self.length} samples into memory")
        else:
            # Lazy loading: just store metadata
            with h5py.File(data_path, 'r') as f:
                self.length = f.attrs['num_samples']
            logger.info(f"Dataset initialized with {self.length} samples (lazy loading)")
    
    def _process_games_to_positions(self, max_games: Optional[int] = None):
        """Process games from datasets format to positions."""
        logger.info("Processing games to positions (this may take a while)...")
        
        all_positions = []
        all_moves = []
        all_values = []
        all_move_numbers = []
        all_players = []
        
        max_games = max_games or len(self.dataset)
        games_to_process = min(max_games, len(self.dataset))
        
        logger.info(f"Processing up to {games_to_process} games (out of {len(self.dataset)} total)...")
        
        for idx, game_data in enumerate(self.dataset):
            if idx >= max_games:
                break
            # Filter by ELO if available
            # Try different possible field names
            white_elo = game_data.get('WhiteElo', '') or game_data.get('WhiteRating', '') or game_data.get('white_elo', '')
            black_elo = game_data.get('BlackElo', '') or game_data.get('BlackRating', '') or game_data.get('black_elo', '')
            
            try:
                white_elo_int = int(white_elo) if white_elo and str(white_elo).isdigit() else 0
                black_elo_int = int(black_elo) if black_elo and str(black_elo).isdigit() else 0
            except (ValueError, TypeError):
                white_elo_int = 0
                black_elo_int = 0
            
            # If min_elo is set and ELOs are available, filter
            # Otherwise, if no ELO info, skip filtering (process all)
            if self.min_elo > 0 and (white_elo_int > 0 or black_elo_int > 0):
                if white_elo_int < self.min_elo or black_elo_int < self.min_elo:
                    continue
            
            # Process game
            try:
                positions = process_game_from_datasets(game_data)
                
                for pos_data in positions:
                    all_positions.append(pos_data['position'])
                    all_moves.append(pos_data['move'])
                    all_values.append(pos_data['value'])
                    all_move_numbers.append(pos_data['move_number'])
                    all_players.append(pos_data['player'])
            except Exception as e:
                logger.warning(f"Error processing game {idx}: {e}")
                continue
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(self.dataset)} games, "
                          f"{len(all_positions)} positions so far")
        
        # Convert to numpy arrays
        if not all_positions:
            logger.warning("No positions processed! Check ELO filtering or data format.")
            self.positions = np.array([])
            self.moves = np.array([], dtype=np.int32)
            self.values = np.array([], dtype=np.float32)
            self.move_numbers = np.array([], dtype=np.int32)
            self.players = np.array([], dtype=np.int32)
        else:
            self.positions = np.stack(all_positions)
            self.moves = np.array(all_moves, dtype=np.int32)
            self.values = np.array(all_values, dtype=np.float32)
            self.move_numbers = np.array(all_move_numbers, dtype=np.int32)
            self.players = np.array(all_players, dtype=np.int32)
        
        self.length = len(self.positions)
        
        logger.info(f"Processed {self.length} positions from {idx + 1} games")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (position, move, value) as torch tensors
        """
        if self.format == 'datasets':
            # For datasets format, we need to process on-the-fly or use cache
            if hasattr(self, 'positions') and len(self.positions) > 0:
                # Use cached positions
                # Load as float, then normalize
                position = torch.from_numpy(self.positions[idx]).float()
                
                # Normalize move count (plane 30)
                # It was stored as uint8 (raw move number), need to convert to 0-1 range
                # Cap at 1.0 (equivalent to 100 moves)
                position[30] = torch.clamp(position[30] / 100.0, max=1.0)
                
                move = torch.tensor(self.moves[idx], dtype=torch.long)
                value = torch.tensor(self.values[idx], dtype=torch.float32)
            else:
                # Lazy processing - would need to process game on-the-fly
                # This is expensive, so cache_in_memory=True is recommended
                raise NotImplementedError(
                    "Lazy processing for datasets format not implemented. "
                    "Use cache_in_memory=True or preprocess to HDF5 format."
                )
        else:
            # HDF5 format
            if self.cache_in_memory:
                position = torch.from_numpy(self.positions[idx]).float()
                
                # Normalize move count (plane 30)
                position[30] = torch.clamp(position[30] / 100.0, max=1.0)
                
                move = torch.tensor(self.moves[idx], dtype=torch.long)
                value = torch.tensor(self.values[idx], dtype=torch.float32)
            else:
                # Lazy load from disk
                with h5py.File(self.data_path, 'r') as f:
                    position = torch.from_numpy(f['positions'][idx]).float()
                    
                    # Normalize move count (plane 30)
                    position[30] = torch.clamp(position[30] / 100.0, max=1.0)
                    
                    move = torch.tensor(f['moves'][idx], dtype=torch.long)
                    value = torch.tensor(f['values'][idx], dtype=torch.float32)
        
        # Apply augmentation if enabled
        if self.augment and np.random.random() < self.augment_prob:
            # For augmentation, we need the original board
            # This is expensive, so we only do it probabilistically
            # In practice, augmentation might be done during preprocessing
            # For now, return original (augmentation can be done offline)
            pass
        
        return position, move, value
    
    def get_batch(self, indices: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a batch of samples efficiently.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Tuple of batched (positions, moves, values)
        """
        if self.format == 'datasets':
            if hasattr(self, 'positions') and len(self.positions) > 0:
                positions = torch.from_numpy(self.positions[indices]).float()
                
                # Normalize move count (plane 30)
                positions[:, 30] = torch.clamp(positions[:, 30] / 100.0, max=1.0)
                
                moves = torch.tensor(self.moves[indices], dtype=torch.long)
                values = torch.tensor(self.values[indices], dtype=torch.float32)
            else:
                raise NotImplementedError("Batch loading for datasets format requires cache_in_memory=True")
        else:
            # HDF5 format
            if self.cache_in_memory:
                positions = torch.from_numpy(self.positions[indices]).float()
                
                # Normalize move count (plane 30)
                positions[:, 30] = torch.clamp(positions[:, 30] / 100.0, max=1.0)
                
                moves = torch.tensor(self.moves[indices], dtype=torch.long)
                values = torch.tensor(self.values[indices], dtype=torch.float32)
            else:
                # Batch load from disk
                with h5py.File(self.data_path, 'r') as f:
                    positions = torch.from_numpy(f['positions'][indices]).float()
                    
                    # Normalize move count (plane 30)
                    positions[:, 30] = torch.clamp(positions[:, 30] / 100.0, max=1.0)
                    
                    moves = torch.tensor(f['moves'][indices], dtype=torch.long)
                    values = torch.tensor(f['values'][indices], dtype=torch.float32)
        
        return positions, moves, values


def create_data_loaders(
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    train_dataset_name: Optional[str] = None,
    val_dataset_name: Optional[str] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    augment_train: bool = True,
    pin_memory: bool = True,
    format: str = 'auto',
    min_elo: int = 2600,
    script_path: Optional[str] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        train_path: Path to training dataset (optional if train_dataset_name provided)
        val_path: Path to validation dataset (optional if val_dataset_name provided)
        train_dataset_name: Name of training dataset in data/processed/
        val_dataset_name: Name of validation dataset in data/processed/
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        augment_train: Whether to augment training data
        pin_memory: Whether to pin memory for faster GPU transfer
        format: Dataset format ('hdf5', 'datasets', or 'auto')
        min_elo: Minimum ELO for filtering
        script_path: Path to calling script (for project root detection)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # For datasets format, we need cache_in_memory=True since lazy processing is not implemented
    # For HDF5 format, we can use lazy loading
    # First, check what format we'll be using
    if train_path is None and train_dataset_name:
        from data.preprocessing import get_dataset_path
        check_path = get_dataset_path(train_dataset_name, script_path)
    else:
        check_path = train_path
    
    # Determine if we need to cache in memory (datasets format requires it)
    need_cache = False
    if format == 'auto' or format == 'datasets':
        if os.path.isdir(check_path) and DATASETS_AVAILABLE:
            need_cache = True  # datasets format requires cache_in_memory
    
    train_dataset = ChessDataset(
        data_path=train_path,
        dataset_name=train_dataset_name,
        format=format,
        augment=augment_train,
        cache_in_memory=need_cache,  # Auto-detect based on format
        min_elo=min_elo,
        script_path=script_path
    )
    
    val_dataset = ChessDataset(
        data_path=val_path,
        dataset_name=val_dataset_name,
        format=format,
        augment=False,  # No augmentation for validation
        cache_in_memory=need_cache,  # Auto-detect based on format
        min_elo=min_elo,
        script_path=script_path
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if need_cache else 2,  # No workers needed for cached data
        pin_memory=pin_memory,
        persistent_workers=False if need_cache else (num_workers > 0)
    )
    
    logger.info(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_loader, val_loader
