"""
Supervised learning training module.

Trains the chess model on human game data with policy and value prediction.
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.architecture import HybridChessNet
from models.utils import save_checkpoint, load_checkpoint, clip_gradients
from data.dataset import ChessDataset, create_data_loaders

logger = logging.getLogger(__name__)


class PolicyValueLoss(nn.Module):
    """
    Combined policy and value loss.
    
    Loss = policy_loss + value_loss
    - policy_loss: Cross-entropy between predicted and target moves
    - value_loss: MSE between predicted and actual game outcomes
    """
    
    def __init__(self, value_weight: float = 1.0):
        """
        Initialize loss function.
        
        Args:
            value_weight: Weight for value loss relative to policy loss
        """
        super(PolicyValueLoss, self).__init__()
        self.value_weight = value_weight
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
    
    def forward(
        self,
        policy_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_pred: torch.Tensor,
        value_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            policy_pred: Predicted policy logits (B, 4096)
            policy_target: Target move indices (B,)
            value_pred: Predicted values (B, 1)
            value_target: Target values (B,)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Policy loss
        policy_loss = self.policy_loss_fn(policy_pred, policy_target)
        
        # Value loss
        value_loss = self.value_loss_fn(value_pred.squeeze(), value_target)
        
        # Combined loss
        total_loss = policy_loss + self.value_weight * value_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
        
        return total_loss, loss_dict


def compute_policy_accuracy(
    policy_pred: torch.Tensor,
    policy_target: torch.Tensor,
    k: list = [1, 3, 5]
) -> Dict[str, float]:
    """
    Compute top-k policy accuracy.
    
    Args:
        policy_pred: Predicted policy logits (B, 4096)
        policy_target: Target move indices (B,)
        k: List of k values for top-k accuracy
        
    Returns:
        Dictionary of accuracy metrics
    """
    with torch.no_grad():
        _, top_indices = torch.topk(policy_pred, max(k), dim=1)
        
        accuracies = {}
        for k_val in k:
            correct = (top_indices[:, :k_val] == policy_target.unsqueeze(1)).any(dim=1)
            accuracies[f'top_{k_val}_accuracy'] = correct.float().mean().item()
        
        return accuracies


class SupervisedTrainer:
    """
    Supervised learning trainer for chess model.
    """
    
    def __init__(
        self,
        model: HybridChessNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict
    ):
        """
        Initialize trainer.
        
        Args:
            model: Chess neural network
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        device_config = config.get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            # Check if requested device is available
            if device_config == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device(device_config)
        
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Setup optimizer
        lr = config.get('learning_rate', 1e-3)
        weight_decay = config.get('weight_decay', 1e-4)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        T_max = config.get('epochs', 50)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_max // 2,
            T_mult=2
        )
        
        # Setup loss
        value_weight = config.get('value_weight', 1.0)
        self.criterion = PolicyValueLoss(value_weight=value_weight)
        
        # Mixed precision
        self.use_amp = config.get('use_mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        
        logger.info(f"Initialized trainer on device: {self.device}")
        logger.info(f"Model parameters: {model.count_parameters():,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_samples = 0
        
        # Accumulate accuracies
        accuracies_sum = {f'top_{k}_accuracy': 0.0 for k in [1, 3, 5]}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (positions, moves, values) in enumerate(pbar):
            positions = positions.to(self.device)
            moves = moves.to(self.device)
            values = values.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    policy_pred, value_pred = self.model(positions)
                    loss, loss_dict = self.criterion(
                        policy_pred, moves, value_pred, values
                    )
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                clip_gradients(self.model, max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                policy_pred, value_pred = self.model(positions)
                loss, loss_dict = self.criterion(
                    policy_pred, moves, value_pred, values
                )
                
                loss.backward()
                clip_gradients(self.model, max_norm=1.0)
                self.optimizer.step()
            
            # Update metrics
            batch_size = positions.size(0)
            total_loss += loss_dict['total_loss'] * batch_size
            total_policy_loss += loss_dict['policy_loss'] * batch_size
            total_value_loss += loss_dict['value_loss'] * batch_size
            total_samples += batch_size
            
            # Compute accuracies
            accuracies = compute_policy_accuracy(policy_pred, moves)
            for key in accuracies_sum:
                accuracies_sum[key] += accuracies[key] * batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['total_loss'],
                'acc': accuracies['top_1_accuracy']
            })
        
        # Average metrics
        metrics = {
            'train_loss': total_loss / total_samples,
            'train_policy_loss': total_policy_loss / total_samples,
            'train_value_loss': total_value_loss / total_samples,
        }
        for key in accuracies_sum:
            metrics[key] = accuracies_sum[key] / total_samples
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_samples = 0
        
        accuracies_sum = {f'top_{k}_accuracy': 0.0 for k in [1, 3, 5]}
        
        with torch.no_grad():
            for positions, moves, values in tqdm(self.val_loader, desc="Validation"):
                positions = positions.to(self.device)
                moves = moves.to(self.device)
                values = values.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        policy_pred, value_pred = self.model(positions)
                        loss, loss_dict = self.criterion(
                            policy_pred, moves, value_pred, values
                        )
                else:
                    policy_pred, value_pred = self.model(positions)
                    loss, loss_dict = self.criterion(
                        policy_pred, moves, value_pred, values
                    )
                
                # Update metrics
                batch_size = positions.size(0)
                total_loss += loss_dict['total_loss'] * batch_size
                total_policy_loss += loss_dict['policy_loss'] * batch_size
                total_value_loss += loss_dict['value_loss'] * batch_size
                total_samples += batch_size
                
                # Compute accuracies
                accuracies = compute_policy_accuracy(policy_pred, moves)
                for key in accuracies_sum:
                    accuracies_sum[key] += accuracies[key] * batch_size
        
        # Average metrics
        metrics = {
            'val_loss': total_loss / total_samples,
            'val_policy_loss': total_policy_loss / total_samples,
            'val_value_loss': total_value_loss / total_samples,
        }
        for key in accuracies_sum:
            metrics[f'val_{key}'] = accuracies_sum[key] / total_samples
        
        return metrics
    
    def train(self, num_epochs: int, save_every: int = 10):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train - Loss: {train_metrics['train_loss']:.4f}, "
                       f"Top-1 Acc: {train_metrics['top_1_accuracy']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['val_loss']:.4f}, "
                       f"Top-1 Acc: {val_metrics['val_top_1_accuracy']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            if (epoch + 1) % save_every == 0 or is_best:
                checkpoint_path = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    checkpoint_path,
                    scheduler=self.scheduler,
                    metrics={**train_metrics, **val_metrics},
                    is_best=is_best
                )


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data loaders for testing
    # In practice, use create_data_loaders with real data paths
    
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 50,
        'value_weight': 1.0,
        'use_mixed_precision': True,
        'checkpoint_dir': 'checkpoints'
    }
    
    model = HybridChessNet()
    # trainer = SupervisedTrainer(model, train_loader, val_loader, config)
    # trainer.train(num_epochs=50)
