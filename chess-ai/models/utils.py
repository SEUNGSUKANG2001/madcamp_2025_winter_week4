"""
Model utilities: checkpointing, initialization, mixed precision, etc.
"""

import logging
import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def initialize_weights(model: nn.Module):
    """
    Initialize model weights using best practices.
    
    - Kaiming init for Conv layers (ReLU)
    - Xavier init for Linear layers
    - BatchNorm: weight=1, bias=0
    
    Args:
        model: PyTorch model
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    metrics: Optional[Dict] = None,
    is_best: bool = False,
    wandb_run_id: Optional[str] = None,
    global_step: int = 0
):
    """
    Save model checkpoint with optimizer state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        metrics: Optional dictionary of metrics to save
        is_best: Whether this is the best model so far
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'is_best': is_best
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if wandb_run_id is not None:
        checkpoint['wandb_run_id'] = wandb_run_id
    
    checkpoint['global_step'] = global_step
    
    torch.save(checkpoint, path)
    
    # Also save as best model if applicable
    if is_best:
        best_path = os.path.join(os.path.dirname(path), 'best_model.pt')
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")
    
    logger.info(f"Saved checkpoint to {path} (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cuda'
) -> Tuple[int, Optional[Dict]]:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint on
        
    Returns:
        Tuple of (epoch, metrics_dict)
        
    Raises:
        FileNotFoundError: if checkpoint file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model weights from {path}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Loaded optimizer state")
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Loaded scheduler state")
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', None)
    
    return epoch, metrics


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count number of parameters in model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        
    Returns:
        String summary of the model
    """
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    
    summary = f"""
Model Summary:
==============
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Input shape: {input_shape}

Architecture:
"""
    
    # Add layer-by-layer summary if needed
    # This is a simplified version; could use torchsummary or similar
    
    return summary


class MixedPrecisionWrapper:
    """
    Wrapper for mixed precision training.
    """
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """
        Initialize mixed precision wrapper.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
        """
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, loss_fn, *args, **kwargs):
        """
        Perform a training step with mixed precision.
        
        Args:
            loss_fn: Loss function that takes model output and targets
            *args, **kwargs: Arguments to pass to loss function
            
        Returns:
            Loss value
        """
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            loss = loss_fn(*args, **kwargs)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def update_scaler(self):
        """Update gradient scaler (call after each step)."""
        self.scaler.update()


def clip_gradients(model: nn.Module, max_norm: float = 1.0):
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


if __name__ == "__main__":
    # Test utilities
    from .architecture import HybridChessNet
    
    model = HybridChessNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print(f"Parameters: {count_parameters(model):,}")
    print(get_model_summary(model, (33, 8, 8)))
    
    # Test checkpointing
    test_path = "/tmp/test_checkpoint.pt"
    save_checkpoint(model, optimizer, epoch=10, path=test_path, 
                   metrics={'loss': 0.5, 'accuracy': 0.6})
    
    # Test loading
    new_model = HybridChessNet()
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    epoch, metrics = load_checkpoint(test_path, new_model, new_optimizer)
    
    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Metrics: {metrics}")
