import torch
from transformers import AutoModel
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import chess
import os

app = Flask(__name__)
CORS(app)

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
    
    # 3. 최신 체크포인트 찾기 및 로드
    if os.path.exists(CHECKPOINT_DIR):
        cps = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
        if cps:
            latest_cp = max(cps, key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)))
            print(f"Applying checkpoint: {latest_cp}")
            checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, latest_cp), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
except Exception as e:
    print(f"Failed to load model/checkpoint: {e}")
    model = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/move", methods=["POST"])
def get_ai_move():
    data = request.json
    fen = data.get("fen")
    
    if not fen:
        return jsonify({"error": "No FEN provided"}), 400
    
    try:
        # Get AI move
        ai_move = model.get_move_from_fen_no_thinking(fen, T=0.1, device=device)
        
        # Get position evaluation
        position_value = model.get_position_value(fen, device=device)
        
        # Get top 5 moves
        probs = model.get_move_from_fen_no_thinking(fen, T=1, device=device, return_probs=True)
        top_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        top_moves_list = [{"move": m, "prob": float(p)} for m, p in top_moves]
        
        return jsonify({
            "move": ai_move,
            "position_value": position_value, # [black_win, draw, white_win]
            "top_moves": top_moves_list
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
