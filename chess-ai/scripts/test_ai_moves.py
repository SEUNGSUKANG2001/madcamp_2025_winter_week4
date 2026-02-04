#!/usr/bin/env python3
"""
Test the AI's move selection in a simple position.
This will help identify if there's a bug in the inference logic.
"""

import os
import sys
import torch
import torch.nn.functional as F
import chess

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from models.architecture import HybridChessNet
from data.preprocessing import encode_board, move_to_index, index_to_move

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the trained model."""
    checkpoint_path = os.path.join(project_root, 'checkpoints', 'checkpoint_epoch_15.pt')
    
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model

def test_position(model, fen, description):
    """Test AI's move selection in a given position."""
    print("\n" + "=" * 80)
    print(f"Test: {description}")
    print("=" * 80)
    
    board = chess.Board(fen)
    print(f"\nPosition:\n{board}")
    print(f"\nFEN: {fen}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    
    # Encode board
    encoded_board = encode_board(board)
    x = torch.from_numpy(encoded_board).unsqueeze(0).float().to(device)
    
    # Get predictions
    with torch.no_grad():
        policy_logits, value = model(x)
    
    print(f"\nPosition value: {value.item():.4f}")
    print(f"(+1.0 = winning for current player, -1.0 = losing)")
    
    # Get policy probabilities
    policy_probs = F.softmax(policy_logits, dim=-1)[0]
    
    # Evaluate legal moves
    legal_moves = list(board.legal_moves)
    print(f"\nNumber of legal moves: {len(legal_moves)}")
    
    move_scores = []
    for move in legal_moves:
        try:
            move_idx = move_to_index(move, board)
            prob = policy_probs[move_idx].item()
            move_scores.append((move, prob, move_idx))
        except Exception as e:
            print(f"Error evaluating move {move}: {e}")
    
    # Sort by probability
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Show top 10 moves
    print(f"\nTop 10 moves by policy probability:")
    for i, (move, prob, idx) in enumerate(move_scores[:10]):
        san = board.san(move)
        print(f"  {i+1}. {san:10s} (UCI: {move.uci():6s}) - prob: {prob:.6f} - idx: {idx}")
    
    # Check if obvious moves have high probability
    return move_scores[0][0]  # Return best move

def main():
    print("Loading model...")
    model = load_model()
    print("Model loaded!")
    
    # Test 1: Starting position
    best_move = test_position(
        model,
        chess.STARTING_FEN,
        "Starting Position - Should prefer center pawns (e4, d4, Nf3, etc.)"
    )
    
    # Test 2: Queen can be captured
    # Position: White queen on e4, Black can capture with pawn f5
    test_position(
        model,
        "rnbqkbnr/pppp1ppp/8/4p3/3PQ3/8/PPP1PPPP/RNB1KBNR b KQkq - 0 1",
        "Black to move - Can capture White's queen with pawn! Should play exd4 (capturing queen)"
    )
    
    # Test 3: Simple checkmate in 1
    test_position(
        model,
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/ PPPP1PPP/RNB1K1NR w KQkq - 0 1",
        "White to move - Checkmate in 1 with Qxf7#"
    )
    
    # Test 4: Position where queen sacrifice would be bad
    test_position(
        model,
        "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPPQPPP/RNB1KBNR w KQkq - 0 1",
        "White to move - Queen on e2 is SAFE, should NOT sacrifice it"
    )

if __name__ == "__main__":
    main()
