import torch
from transformers import AutoModel
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from peft import LoraConfig, get_peft_model
import chess
import os

app = Flask(__name__)
CORS(app)

# --- LoRA 및 체크포인트 설정 ---
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["query_proj", "key_proj", "value_proj", "out_proj", "ff1", "ff2", "policy_head"]
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")

# Load model
print("Loading model and latest checkpoint...")
try:
    # 1. 베이스 모델 로드
    model = AutoModel.from_pretrained("Maxlegrec/ChessBot", trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. LoRA 설정 적용
    config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias="none", task_type=None
    )
    model = get_peft_model(model, config)
    
    # 3. 특정 체크포인트 로드 (ep2900)
    target_checkpoint = "chess_ai_rl_ep2900.pt"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, target_checkpoint)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {target_checkpoint}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Successfully loaded checkpoint: {target_checkpoint}")
    else:
        print(f"Warning: Checkpoint {target_checkpoint} not found at {checkpoint_path}")
        print("Using base model with LoRA layers initialized.")

    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Failed to load model/checkpoint: {e}")
    model = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/move", methods=["POST"])
def get_ai_move():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    fen = data.get("fen")
    
    if not fen:
        return jsonify({"error": "No FEN provided"}), 400
    
    try:
        # Get AI's move (Policy-based)
        # Using T=0.1 for more deterministic best move
        ai_move = model.get_move_from_fen_no_thinking(fen, T=0.1, device=device)

        # Get position evaluation [black_win, draw, white_win]
        position_value = model.get_position_value(fen, device=device)
        if torch.is_tensor(position_value):
            position_value = position_value.tolist()
        
        # Get top 5 moves probabilities
        # Using T=1 to see a wider distribution of move probabilities
        probs = model.get_move_from_fen_no_thinking(fen, T=1, device=device, return_probs=True)
        top_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        top_moves_list = [{"move": m, "prob": round(float(p) * 100, 2)} for m, p in top_moves]
        
        # Simplify position value for display: White win probability - Black win probability
        score = float(position_value[2]) - float(position_value[0])
        
        return jsonify({
            "move": ai_move,
            "position_value": {
                "raw": position_value,
                "score": round(float(score), 4)
            },
            "top_moves": top_moves_list
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "templates"), exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
