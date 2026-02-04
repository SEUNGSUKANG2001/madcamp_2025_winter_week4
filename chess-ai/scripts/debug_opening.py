#!/usr/bin/env python3
"""
Debug Opening: 1. e4 response.
Check probabilities of common responses vs 'h6'.
"""
import os
import sys
import torch
import torch.nn.functional as F
import chess

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from models.architecture import HybridChessNet
from data.preprocessing import encode_board, move_to_index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def debug_opening(epoch):
    checkpoint_path = os.path.join(project_root, 'checkpoints', f'checkpoint_epoch_{epoch}.pt')
    if not os.path.exists(checkpoint_path):
        return

    print(f"\n{'='*60}")
    print(f"DEBUGGING EPOCH {epoch}")
    print(f"{'='*60}")
    
    model = HybridChessNet()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    # Setup: 1. e4
    board = chess.Board()
    board.push_san("e4")
    
    # Predict
    x = torch.from_numpy(encode_board(board)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        policy_logits, value = model(x)
        
    policy_probs = F.softmax(policy_logits, dim=-1)[0]
    
    # Check specific moves
    targets = ["e5", "c5", "e6", "c6", "d5", "Nf6", "h6", "a6"]
    print(f"\n{'Move':<10} {'Prob':<10} {'Rank':<10}")
    print("-" * 30)
    
    # Get all legal moves and rank them
    all_moves = []
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        prob = policy_probs[idx].item()
        all_moves.append((move, idx, prob))
        
    all_moves.sort(key=lambda x: x[2], reverse=True)
    
    # Print targets
    for san in targets:
        try:
            move = board.parse_san(san)
            idx = move_to_index(move, board)
            prob = policy_probs[idx].item()
            
            # Find rank
            rank = -1
            for i, (m, _, _) in enumerate(all_moves):
                if m == move:
                    rank = i + 1
                    break
            
            print(f"{san:<10} {prob:.6f}   #{rank}")
        except ValueError:
            print(f"{san:<10} INVALID")

    print("\nTop 5 Predictions:")
    for i, (move, idx, prob) in enumerate(all_moves[:5], 1):
        print(f"{i}. {board.san(move):<6} ({prob:.6f})")

if __name__ == "__main__":
    for epoch in [10, 15, 20, 25, 30, 34]:
        debug_opening(epoch)
