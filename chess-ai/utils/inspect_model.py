import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("Maxlegrec/ChessBot", trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Example usage
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Sample move from policy
move = model.get_move_from_fen_no_thinking(fen, T=0.1, device=device)
print(f"Policy-based move: {move}")
#e2e4

# Get the best move using value analysis
value_move = model.get_best_move_value(fen, T=0, device=device)
print(f"Value-based move: {value_move}")
#e2e4

# Get position evaluation
position_value = model.get_position_value(fen, device=device)
print(f"Position value [black_win, draw, white_win]: {position_value}")
#[0.2318, 0.4618, 0.3064]

# Get move probabilities
probs = model.get_move_from_fen_no_thinking(fen, T=1, device=device, return_probs=True)
top_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 moves:")
for move, prob in top_moves:
    print(f"  {move}: {prob:.4f}")
#Top 5 moves:
#  e2e4: 0.9285
#  d2d4: 0.0712
#  g1f3: 0.0001
#  e2e3: 0.0000
#  c2c3: 0.0000