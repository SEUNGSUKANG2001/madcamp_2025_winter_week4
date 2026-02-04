"""
Script to load a checkpoint file and verify model output.

This script:
1. Loads a .pt checkpoint file from the check_points folder
2. Creates a model instance with the same architecture
3. Loads the weights into the model
4. Performs a test forward pass to verify the model works correctly
5. Displays model information and output statistics
"""

import os
import sys
import torch
import torch.nn.functional as F

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.architecture import HybridChessNet


def load_checkpoint(checkpoint_path: str):
    """
    Load a checkpoint file and return the state dict.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        
    Returns:
        Loaded checkpoint (could be dict or state_dict)
    """
    print(f"\n{'='*60}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load checkpoint (CPU for compatibility)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"✓ Checkpoint loaded successfully")
    print(f"  File size: {os.path.getsize(checkpoint_path) / (1024**2):.2f} MB\n")
    
    return checkpoint


def extract_model_state(checkpoint):
    """
    Extract model state dict from checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint object
        
    Returns:
        Model state dictionary
    """
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            print("Checkpoint format: Dictionary with 'model_state_dict' key")
            state_dict = checkpoint['model_state_dict']
            
            # Print other keys if available
            print("\nCheckpoint contains:")
            for key, value in checkpoint.items():
                if key != 'model_state_dict':
                    if isinstance(value, (int, float, str)):
                        print(f"  - {key}: {value}")
                    else:
                        print(f"  - {key}: {type(value)}")
            print()
            
            return state_dict
        elif 'state_dict' in checkpoint:
            print("Checkpoint format: Dictionary with 'state_dict' key")
            return checkpoint['state_dict']
        else:
            # Assume the dict itself is the state dict
            print("Checkpoint format: Direct state dictionary")
            return checkpoint
    else:
        # Assume it's already a state dict
        print("Checkpoint format: Direct state dictionary")
        return checkpoint


def create_model():
    """
    Create a HybridChessNet model with default architecture.
    
    Returns:
        Initialized model
    """
    print(f"\n{'='*60}")
    print("Creating HybridChessNet model")
    print(f"{'='*60}\n")
    
    model = HybridChessNet(
        input_channels=33,
        cnn_channels=256,
        num_res_blocks=8,
        transformer_embed_dim=512,
        transformer_num_heads=8,
        transformer_num_blocks=4,
        num_moves=4096,
        dropout=0.1
    )
    
    num_params = model.count_parameters()
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Parameter memory: {num_params * 4 / (1024**2):.2f} MB (float32)\n")
    
    return model


def load_weights(model, state_dict):
    """
    Load weights into the model.
    
    Args:
        model: Model instance
        state_dict: State dictionary to load
        
    Returns:
        Model with loaded weights
    """
    print(f"\n{'='*60}")
    print("Loading weights into model")
    print(f"{'='*60}\n")
    
    try:
        # Try to load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠ Missing keys in checkpoint: {len(missing_keys)}")
            for key in missing_keys[:5]:  # Show first 5
                print(f"    - {key}")
            if len(missing_keys) > 5:
                print(f"    ... and {len(missing_keys) - 5} more")
            print()
        
        if unexpected_keys:
            print(f"⚠ Unexpected keys in checkpoint: {len(unexpected_keys)}")
            for key in unexpected_keys[:5]:  # Show first 5
                print(f"    - {key}")
            if len(unexpected_keys) > 5:
                print(f"    ... and {len(unexpected_keys) - 5} more")
            print()
        
        if not missing_keys and not unexpected_keys:
            print("✓ All weights loaded successfully (perfect match)")
        else:
            print("⚠ Weights loaded with some mismatches")
        
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        raise
    
    return model


def test_model(model):
    """
    Test the model with random input and display output statistics.
    
    Args:
        model: Model to test
    """
    print(f"\n{'='*60}")
    print("Testing model forward pass")
    print(f"{'='*60}\n")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create random input (batch_size=2, channels=33, height=8, width=8)
    batch_size = 2
    x = torch.randn(batch_size, 33, 8, 8)
    
    print(f"Input shape: {x.shape}")
    print(f"Input stats: mean={x.mean():.4f}, std={x.std():.4f}, "
          f"min={x.min():.4f}, max={x.max():.4f}\n")
    
    # Forward pass
    with torch.no_grad():
        policy_logits, value = model(x)
    
    # Display results
    print("Policy Output:")
    print(f"  Shape: {policy_logits.shape}")
    print(f"  Stats: mean={policy_logits.mean():.4f}, std={policy_logits.std():.4f}")
    print(f"         min={policy_logits.min():.4f}, max={policy_logits.max():.4f}")
    
    # Convert to probabilities
    policy_probs = F.softmax(policy_logits, dim=-1)
    top_k = 5
    top_probs, top_indices = torch.topk(policy_probs[0], k=top_k)
    print(f"\n  Top {top_k} move probabilities (batch 0):")
    for i in range(top_k):
        print(f"    Move {top_indices[i]:4d}: {top_probs[i]:.6f}")
    
    print("\nValue Output:")
    print(f"  Shape: {value.shape}")
    print(f"  Stats: mean={value.mean():.4f}, std={value.std():.4f}")
    print(f"         min={value.min():.4f}, max={value.max():.4f}")
    print(f"  Values (should be in [-1, 1]):")
    for i in range(batch_size):
        print(f"    Batch {i}: {value[i, 0]:.6f}")
    
    print()
    
    # Check if value is in valid range
    if value.min() >= -1 and value.max() <= 1:
        print("✓ Value output is in valid range [-1, 1]")
    else:
        print("⚠ WARNING: Value output is outside expected range [-1, 1]")
    
    # Check if policy probabilities sum to 1
    prob_sum = policy_probs.sum(dim=-1)
    if torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-6):
        print("✓ Policy probabilities sum to 1.0")
    else:
        print(f"⚠ WARNING: Policy probabilities sum to {prob_sum[0]:.6f} (expected 1.0)")


def main():
    """Main function to run checkpoint verification."""
    
    # Get checkpoint file path
    # Default to check_points/best_model.pt
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'checkpoints'
    )
    
    # Find all .pt files
    if os.path.exists(checkpoint_dir):
        pt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        
        if not pt_files:
            print(f"No .pt files found in {checkpoint_dir}")
            return
        
        print(f"\nFound {len(pt_files)} checkpoint file(s):")
        for i, f in enumerate(pt_files):
            file_path = os.path.join(checkpoint_dir, f)
            file_size = os.path.getsize(file_path) / (1024**2)
            print(f"  {i+1}. {f} ({file_size:.2f} MB)")
        
        # Use the first one (or you can add selection logic)
        checkpoint_path = os.path.join(checkpoint_dir, pt_files[0])
    else:
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    try:
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path)
        
        # Extract model state
        state_dict = extract_model_state(checkpoint)
        
        # Create model
        model = create_model()
        
        # Load weights
        model = load_weights(model, state_dict)
        
        # Test model
        test_model(model)
        
        print(f"\n{'='*60}")
        print("✓ Checkpoint verification completed successfully!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Error during checkpoint verification:")
        print(f"  {type(e).__name__}: {e}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
