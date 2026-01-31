"""
Self-play game generation module.

Generates training data by playing games against itself using MCTS.
"""

import logging
from typing import List, Dict, Optional
import numpy as np
import chess
import torch
from multiprocessing import Pool, Manager
import time

from mcts.mcts import MCTS
from data.preprocessing import encode_board, move_to_index
from models.architecture import HybridChessNet

logger = logging.getLogger(__name__)


class SelfPlayWorker:
    """
    Worker for generating self-play games.
    """
    
    def __init__(
        self,
        model_path: str,
        num_simulations: int = 800,
        device: str = 'cuda'
    ):
        """
        Initialize self-play worker.
        
        Args:
            model_path: Path to model checkpoint
            num_simulations: Number of MCTS simulations per move
            device: Device to run model on
        """
        self.device = device
        self.num_simulations = num_simulations
        
        # Load model
        self.model = HybridChessNet()
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Create MCTS
        self.mcts = MCTS(
            model=self.model,
            num_simulations=num_simulations,
            device=device
        )
    
    def play_game(self, temperature_schedule: Optional[List[float]] = None) -> List[Dict]:
        """
        Play a single self-play game.
        
        Args:
            temperature_schedule: Temperature for each move (None = use default)
            
        Returns:
            List of training samples (position, policy, value)
        """
        board = chess.Board()
        game_data = []
        move_number = 0
        
        # Default temperature schedule: high for first 30 moves, then low
        if temperature_schedule is None:
            temperature_schedule = [1.0] * 30 + [0.1] * 200
        
        while not board.is_game_over():
            # Get temperature for this move
            temperature = temperature_schedule[min(move_number, len(temperature_schedule) - 1)]
            
            # MCTS search
            try:
                move, policy_dist = self.mcts.search(board, temperature=temperature)
            except Exception as e:
                logger.error(f"Error in MCTS search: {e}")
                # Fallback: random legal move
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                move = np.random.choice(legal_moves)
                policy_dist = np.ones(len(legal_moves)) / len(legal_moves)
            
            # Encode position
            position = encode_board(board)
            
            # Convert policy to move index format
            # For now, store the full policy distribution
            # In practice, we'd store visit counts or normalized distribution
            
            # Store training sample
            game_data.append({
                'position': position,
                'policy': policy_dist,  # Full distribution from MCTS
                'move_number': move_number,
                'player': board.turn
            })
            
            # Make move
            board.push(move)
            move_number += 1
            
            # Update MCTS root
            self.mcts.update_root(move)
            
            # Limit game length
            if move_number >= 200:
                break
        
        # Determine game outcome
        result = self._get_game_result(board)
        
        # Assign values to all positions
        for sample in game_data:
            if sample['player'] == chess.WHITE:
                sample['value'] = result
            else:
                sample['value'] = -result
        
        return game_data
    
    def _get_game_result(self, board: chess.Board) -> float:
        """
        Get game result from white's perspective.
        
        Args:
            board: Final board position
            
        Returns:
            +1.0 for white win, -1.0 for black win, 0.0 for draw
        """
        if board.is_checkmate():
            # Last player to move won (opponent is checkmated)
            return 1.0 if board.turn == chess.BLACK else -1.0
        else:
            # Draw
            return 0.0


def parallel_self_play(
    model_path: str,
    num_workers: int = 8,
    games_per_worker: int = 125,
    num_simulations: int = 800,
    device: str = 'cuda'
) -> List[Dict]:
    """
    Generate self-play games in parallel.
    
    Args:
        model_path: Path to model checkpoint
        num_workers: Number of parallel workers
        games_per_worker: Number of games per worker
        num_simulations: MCTS simulations per move
        device: Device to run models on
        
    Returns:
        List of all game data samples
    """
    logger.info(f"Starting parallel self-play: {num_workers} workers, "
               f"{games_per_worker} games each")
    
    # Create workers
    workers = [
        SelfPlayWorker(model_path, num_simulations, device)
        for _ in range(num_workers)
    ]
    
    all_game_data = []
    
    # Generate games
    for worker_idx, worker in enumerate(workers):
        logger.info(f"Worker {worker_idx+1}/{num_workers} generating games...")
        
        for game_idx in range(games_per_worker):
            try:
                game_data = worker.play_game()
                all_game_data.extend(game_data)
                
                if (game_idx + 1) % 10 == 0:
                    logger.info(f"Worker {worker_idx+1}: Generated {game_idx+1}/{games_per_worker} games")
            except Exception as e:
                logger.error(f"Error generating game {game_idx} on worker {worker_idx}: {e}")
                continue
    
    logger.info(f"Generated {len(all_game_data)} training samples from "
               f"{num_workers * games_per_worker} games")
    
    return all_game_data


def collect_and_save_data(
    game_data: List[Dict],
    output_path: str
):
    """
    Collect and save self-play data.
    
    Args:
        game_data: List of game data samples
        output_path: Path to save HDF5 file
    """
    from data.preprocessing import save_dataset
    
    # Convert to format expected by save_dataset
    # Note: policy is stored as full distribution, may need conversion
    processed_data = []
    
    for sample in game_data:
        # For now, use the first move from policy distribution
        # In practice, we'd store the full policy or convert to move index
        move_idx = 0  # Placeholder - would need proper conversion
        
        processed_data.append({
            'position': sample['position'],
            'move': move_idx,
            'value': sample['value'],
            'move_number': sample['move_number'],
            'player': sample['player']
        })
    
    save_dataset(processed_data, output_path)
    logger.info(f"Saved {len(processed_data)} samples to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate self-play games
    # game_data = parallel_self_play(
    #     model_path='checkpoints/best_model.pt',
    #     num_workers=8,
    #     games_per_worker=125
    # )
    # 
    # collect_and_save_data(game_data, 'data/processed/self_play_data.h5')
    pass
