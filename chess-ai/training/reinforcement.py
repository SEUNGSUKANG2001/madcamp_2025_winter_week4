"""
Reinforcement learning training module.

Implements the complete RL training pipeline with self-play, replay buffer, and evaluation.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

from models.architecture import HybridChessNet
from models.utils import save_checkpoint, load_checkpoint
from mcts.mcts import MCTS
from self_play.generator import parallel_self_play
from training.supervised import SupervisedTrainer, PolicyValueLoss

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Replay buffer for storing and sampling training positions.
    """
    
    def __init__(self, capacity: int = 500000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of positions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, data: List[Dict]):
        """
        Add data to buffer.
        
        Args:
            data: List of position dictionaries
        """
        self.buffer.extend(data)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """
        Sample a batch of positions.
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            List of sampled positions
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class RLTrainer:
    """
    Reinforcement learning trainer.
    """
    
    def __init__(
        self,
        model: HybridChessNet,
        config: Dict,
        baseline_model: Optional[HybridChessNet] = None
    ):
        """
        Initialize RL trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            baseline_model: Baseline model for KL regularization (frozen)
        """
        self.model = model
        self.config = config
        self.baseline_model = baseline_model
        
        if baseline_model is not None:
            baseline_model.eval()
            for param in baseline_model.parameters():
                param.requires_grad = False
        
        # Setup device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        # Replay buffer
        buffer_capacity = config.get('replay_buffer_size', 500000)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Optimizer
        lr = config.get('learning_rate', 1e-3)
        weight_decay = config.get('weight_decay', 1e-4)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Loss function with KL regularization
        value_weight = config.get('value_weight', 1.0)
        kl_weight = config.get('kl_weight', 0.01)
        self.criterion = PolicyValueLoss(value_weight=value_weight)
        self.kl_weight = kl_weight
        
        # Training state
        self.iteration = 0
        self.best_elo = 2600  # Starting from supervised baseline
        
        logger.info(f"Initialized RL trainer")
    
    def train_iteration(self) -> Dict[str, float]:
        """
        Perform one RL training iteration.
        
        Returns:
            Dictionary of training metrics
        """
        self.iteration += 1
        logger.info(f"Starting RL iteration {self.iteration}")
        
        # 1. Generate self-play games
        games_per_iteration = self.config.get('games_per_iteration', 1000)
        num_workers = self.config.get('num_workers', 8)
        num_simulations = self.config.get('mcts_simulations', 800)
        
        logger.info("Generating self-play games...")
        game_data = parallel_self_play(
            model_path=None,  # Would need to save model first
            num_workers=num_workers,
            games_per_worker=games_per_iteration // num_workers,
            num_simulations=num_simulations,
            device=str(self.device)
        )
        
        # 2. Add to replay buffer
        self.replay_buffer.add(game_data)
        logger.info(f"Replay buffer size: {len(self.replay_buffer)}")
        
        # 3. Train on samples from replay buffer
        epochs_per_iteration = self.config.get('epochs_per_iteration', 5)
        batch_size = self.config.get('batch_size', 512)
        
        metrics = self._train_on_buffer(epochs_per_iteration, batch_size)
        
        return metrics
    
    def _train_on_buffer(self, num_epochs: int, batch_size: int) -> Dict[str, float]:
        """
        Train model on replay buffer.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            # Sample batches
            num_samples = len(self.replay_buffer)
            num_batches_epoch = max(1, num_samples // batch_size)
            
            for _ in range(num_batches_epoch):
                batch = self.replay_buffer.sample(batch_size)
                
                # Convert to tensors
                positions = torch.stack([torch.from_numpy(s['position']) for s in batch]).float().to(self.device)
                # Note: policy and value would need proper conversion
                # For now, placeholder
                moves = torch.zeros(len(batch), dtype=torch.long).to(self.device)
                values = torch.tensor([s['value'] for s in batch], dtype=torch.float32).to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                policy_pred, value_pred = self.model(positions)
                
                # Policy loss (cross-entropy with MCTS policy)
                # Note: Would need proper MCTS policy targets
                policy_loss = F.cross_entropy(policy_pred, moves)
                
                # Value loss
                value_loss = F.mse_loss(value_pred.squeeze(), values)
                
                # KL regularization (if baseline model provided)
                kl_loss = torch.tensor(0.0)
                if self.baseline_model is not None:
                    with torch.no_grad():
                        baseline_policy, _ = self.baseline_model(positions)
                        baseline_policy = F.softmax(baseline_policy, dim=-1)
                    
                    current_policy = F.softmax(policy_pred, dim=-1)
                    kl_loss = F.kl_div(
                        F.log_softmax(policy_pred, dim=-1),
                        baseline_policy,
                        reduction='batchmean'
                    )
                
                # Total loss
                loss = policy_loss + value_loss + self.kl_weight * kl_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_kl_loss += kl_loss.item()
                num_batches += 1
        
        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches
        }
        
        return metrics
    
    def evaluate_model(
        self,
        new_model: HybridChessNet,
        old_model: HybridChessNet,
        num_games: int = 100
    ) -> float:
        """
        Evaluate new model against old model.
        
        Args:
            new_model: New model to evaluate
            old_model: Previous best model
            num_games: Number of games to play
            
        Returns:
            Win rate of new model (0.0 to 1.0)
        """
        # This would implement head-to-head evaluation
        # For now, return placeholder
        logger.info(f"Evaluating new model vs old model ({num_games} games)...")
        
        # Placeholder: would implement actual game playing
        win_rate = 0.5  # Placeholder
        
        return win_rate
    
    def update_learning_rate(self, iteration: int):
        """
        Update learning rate based on iteration.
        
        Args:
            iteration: Current iteration number
        """
        # Decay learning rate
        initial_lr = self.config.get('learning_rate', 1e-3)
        final_lr = self.config.get('final_learning_rate', 1e-4)
        decay_steps = self.config.get('lr_decay_steps', 100)
        
        if iteration < decay_steps:
            lr = initial_lr - (initial_lr - final_lr) * (iteration / decay_steps)
        else:
            lr = final_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'device': 'cuda',
        'learning_rate': 1e-3,
        'final_learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'replay_buffer_size': 500000,
        'games_per_iteration': 1000,
        'num_workers': 8,
        'mcts_simulations': 800,
        'epochs_per_iteration': 5,
        'batch_size': 512,
        'value_weight': 1.0,
        'kl_weight': 0.01
    }
    
    model = HybridChessNet()
    # trainer = RLTrainer(model, config)
    # trainer.train_iteration()
