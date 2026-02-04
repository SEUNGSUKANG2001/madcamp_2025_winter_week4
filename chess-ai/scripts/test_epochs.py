#!/usr/bin/env python3
"""
Test with later checkpoints to see if more training helps.
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

def test_checkpoint(epoch):
    """Test a specific checkpoint."""
    checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch}.pt'
    
    print(f"\n{'='*80}")
    print(f"Testing Checkpoint: Epoch {epoch}")
    print(f"{'='*80}")
    
    model = HybridChessNet()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    # Test position: Queen on e4, black knight on f6 can capture
    fen = "rnbqkb1r/pppppppp/5n2/8/4Q3/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
    board = chess.Board(fen)
    
    # Encode and predict
    x = torch.from_numpy(encode_board(board)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        policy_logits, value = model(x)
    
    policy_probs = F.softmax(policy_logits, dim=-1)[0]
    
    # Get top moves
    all_moves = []
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        prob = policy_probs[idx].item()
        san = board.san(move)
        all_moves.append((san, move.uci(), prob, move.to_square == chess.E4))
    
    all_moves.sort(key=lambda x: x[2], reverse=True)
    
    # Find rank of queen capture
    queen_rank = None
    queen_prob = None
    for i, (san, uci, prob, is_capture) in enumerate(all_moves, 1):
        if is_capture:
            queen_rank = i
            queen_prob = prob
            break
    
    print(f"\n퀸 잡기 순위: {queen_rank}위 (확률: {queen_prob:.6f})")
    print(f"Value: {value.item():.4f}")
    
    print(f"\n상위 5개:")
    for i, (san, uci, prob, is_capture) in enumerate(all_moves[:5], 1):
        marker = "⭐ 퀸 잡기!" if is_capture else ""
        print(f"{i}. {san:10s} - {prob:.6f} {marker}")
    
    return queen_rank, queen_prob

# Test multiple checkpoints
print("퀸 잡기 순위 변화 추이:")
print("="*80)

results = []
for epoch in [10, 15, 20, 25, 30, 34]:
    if os.path.exists(f'checkpoints/checkpoint_epoch_{epoch}.pt'):
        rank, prob = test_checkpoint(epoch)
        results.append((epoch, rank, prob))

print(f"\n\n{'='*80}")
print("요약:")
print(f"{'='*80}")
print(f"{'Epoch':<10} {'퀸 잡기 순위':<15} {'확률':<10}")
print("-"*80)
for epoch, rank, prob in results:
    print(f"{epoch:<10} {rank:<15} {prob:.6f}")
