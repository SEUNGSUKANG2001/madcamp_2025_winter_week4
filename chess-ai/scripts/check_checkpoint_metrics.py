#!/usr/bin/env python3
"""
Script to inspect checkpoint metrics and diagnose training issues.
"""

import os
import sys
import torch

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def check_checkpoint(checkpoint_path):
    """Load and display checkpoint metrics."""
    print(f"\n{'='*80}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"{'='*80}")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # Basic info
        print(f"\nEpoch: {ckpt.get('epoch', 'N/A')}")
        print(f"Is Best: {ckpt.get('is_best', False)}")
        
        # Metrics
        metrics = ckpt.get('metrics', {})
        if metrics:
            print("\nTraining Metrics:")
            print(f"  Train Loss: {metrics.get('train_loss', 'N/A'):.6f}")
            print(f"  Train Policy Loss: {metrics.get('train_policy_loss', 'N/A'):.6f}")
            print(f"  Train Value Loss: {metrics.get('train_value_loss', 'N/A'):.6f}")
            print(f"  Top-1 Accuracy: {metrics.get('top_1_accuracy', 'N/A'):.4f}")
            print(f"  Top-3 Accuracy: {metrics.get('top_3_accuracy', 'N/A'):.4f}")
            print(f"  Top-5 Accuracy: {metrics.get('top_5_accuracy', 'N/A'):.4f}")
            
            print("\nValidation Metrics:")
            print(f"  Val Loss: {metrics.get('val_loss', 'N/A'):.6f}")
            print(f"  Val Policy Loss: {metrics.get('val_policy_loss', 'N/A'):.6f}")
            print(f"  Val Value Loss: {metrics.get('val_value_loss', 'N/A'):.6f}")
            print(f"  Val Top-1 Accuracy: {metrics.get('val_top_1_accuracy', 'N/A'):.4f}")
            print(f"  Val Top-3 Accuracy: {metrics.get('val_top_3_accuracy', 'N/A'):.4f}")
            print(f"  Val Top-5 Accuracy: {metrics.get('val_top_5_accuracy', 'N/A'):.4f}")
        else:
            print("\nNo metrics found in checkpoint!")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    
    # Check if best model exists
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        check_checkpoint(best_model_path)
    
    # Check epoch 15 (currently used)
    epoch_15_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_15.pt')
    if os.path.exists(epoch_15_path):
        check_checkpoint(epoch_15_path)
    
    # Check latest checkpoint (epoch 34)
    epoch_34_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_34.pt')
    if os.path.exists(epoch_34_path):
        check_checkpoint(epoch_34_path)
    
    print(f"\n{'='*80}\n")
