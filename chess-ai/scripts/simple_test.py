#!/usr/bin/env python3
"""
Test with a CORRECT position where queen can be captured.
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

# Load model
print("Loading model...")
model = HybridChessNet()
checkpoint = torch.load('checkpoints/checkpoint_epoch_15.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.to(device)
model.eval()
print("✓ Model loaded\n")

# Corrected test position: Queen on d4, black pawn on e5 can capture diagonally
# Or simpler: Queen on e4, black knight can capture it
fen = "rnbqkb1r/pppppppp/5n2/8/4Q3/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
board = chess.Board(fen)

print("Position:")
print(board)
print(f"\n흑 차례 - 나이트(f6)로 백의 퀸(e4)를 공짜로 잡을 수 있음!")
print("(퀸이 보호받지 않음)")

# Encode and predict
x = torch.from_numpy(encode_board(board)).unsqueeze(0).float().to(device)
with torch.no_grad():
    policy_logits, value = model(x)

policy_probs = F.softmax(policy_logits, dim=-1)[0]

# Find the queen capture move
print("\n\n퀸을 잡는 수:")
for move in board.legal_moves:
    if move.to_square == chess.E4:  # Captures queen on e4
        idx = move_to_index(move, board)
        prob = policy_probs[idx].item()
        san = board.san(move)
        print(f"  ⭐ {san:10s} ({move.uci()}) - index: {idx:4d} - prob: {prob:.6f}")

# Get top moves
print("\n상위 10개 예측 수:")
all_moves = []
for move in board.legal_moves:
    idx = move_to_index(move, board)
    prob = policy_probs[idx].item()
    san = board.san(move)
    all_moves.append((san, move.uci(), idx, prob, move.to_square == chess.E4))

all_moves.sort(key=lambda x: x[3], reverse=True)

for i, (san, uci, idx, prob, captures_queen) in enumerate(all_moves[:10], 1):
    marker = "⭐ 퀸 잡기!" if captures_queen else ""
    print(f"{i:2d}. {san:10s} ({uci}) - prob: {prob:.6f} {marker}")

print(f"\nValue: {value.item():.4f}")
