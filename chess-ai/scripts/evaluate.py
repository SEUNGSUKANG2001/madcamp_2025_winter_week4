import torch
import torch.nn as nn
import chess
import os
import random
from tqdm import tqdm
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import sys

# === 설정 (CONFIGURATION) ===
MODEL_NAME = "Maxlegrec/ChessBot"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GAMES_PER_MATCH = 20  # 각 체크포인트당 20판 대결

# 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "checkpoints")
EP0_PATH = os.path.join(CHECKPOINT_DIR, "chess_ai_rl_ep0.pt")

# LoRA 설정
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["query_proj", "key_proj", "value_proj", "out_proj", "ff1", "ff2", "policy_head"]

def load_chess_model(checkpoint_path=None):
    """모델 로드 및 가중치 적용"""
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
    config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias="none", task_type=None
    )
    model = get_peft_model(model, config)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        cp = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(cp['model_state_dict'], strict=False)
    model.eval()
    return model

def play_match(model_a, model_b, n_games=20):
    """두 모델 간의 대결 진행 (Model A vs Model B)"""
    wins, draws, losses = 0, 0, 0
    pbar = tqdm(range(n_games), desc="Match Progress", leave=False)
    for i in pbar:
        white_model = model_a if i % 2 == 0 else model_b
        black_model = model_b if i % 2 == 0 else model_a
        
        board = chess.Board()
        while not board.is_game_over():
            curr_model = white_model if board.turn == chess.WHITE else black_model
            try:
                res = curr_model.get_move_from_fen_no_thinking(board.fen(), T=0.1, device=DEVICE)
                if not res:
                    board.push(random.choice(list(board.legal_moves)))
                else:
                    board.push_uci(res)
            except Exception:
                if not board.is_game_over():
                    board.push(random.choice(list(board.legal_moves)))
        
        res_str = board.result()
        # model_a 기준 승무패 계산
        a_is_white = (i % 2 == 0)
        if (a_is_white and res_str == "1-0") or (not a_is_white and res_str == "0-1"):
            wins += 1
        elif res_str == "1/2-1/2":
            draws += 1
        else:
            losses += 1
        
        pbar.set_postfix({"W-D-L": f"{wins}-{draws}-{losses}"})
    return wins, draws, losses

def main():
    print(f"[{DEVICE}] RL 체크포인트 성능 평가 (Vs Pure HuggingFace Base Model)")
    print(f"매칭당 경기 수: {NUM_GAMES_PER_MATCH}판")
    
    # 1. Pure Base 모델 로드 (체크포인트 적용 안 함)
    print("Pure Base 모델 로드 중...")
    model_base = load_chess_model(checkpoint_path=None)
    
    # 2. 체크포인트 목록 수집 및 정렬
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"오류: 체크포인트 디레토리 {CHECKPOINT_DIR}가 없습니다.")
        return
        
    all_files = os.listdir(CHECKPOINT_DIR)
    # RL 체크포인트 필터링 (chess_ai_rl_epXXX.pt)
    cps = [f for f in all_files if f.startswith('chess_ai_rl_ep') and f.endswith('.pt')]
    
    def extract_ep(name):
        try: return int(name.split('ep')[-1].split('.pt')[0])
        except: return 0
    
    cps.sort(key=extract_ep)
    
    if not cps:
        print("평가할 체크포인트가 없습니다.")
        return

    print("-" * 80)
    print(f"{'Checkpoint Name':<35} | {'Wins':<5} | {'Draws':<5} | {'Losses':<6} | {'Win Rate'}")
    print("-" * 80)

    # 3. 평가용 모델 구조 미리 생성
    eval_model = load_chess_model()
    
    results = []
    for cp_name in cps:
        print(f"Evaluating {cp_name}...")
        cp_path = os.path.join(CHECKPOINT_DIR, cp_name)
        checkpoint = torch.load(cp_path, map_location=DEVICE)
        eval_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        eval_model.eval()
        
        w, d, l = play_match(eval_model, model_base, n_games=NUM_GAMES_PER_MATCH)
        win_rate = (w + 0.5 * d) / NUM_GAMES_PER_MATCH
        results.append((cp_name, w, d, l, win_rate))
        
        # 중간 결과 출력
        print(f"결과: {w}승 {d}무 {l}패 (승률: {win_rate:.1%})")
        print("-" * 80)

    print("\n" + "=" * 80)
    print(f"{'Final Summary: RL Progress (Vs ep0)':^80}")
    print("-" * 80)
    print(f"{'Checkpoint Name':<35} | {'Wins':<5} | {'Draws':<5} | {'Losses':<6} | {'Win Rate'}")
    print("-" * 80)
    for res in results:
        cp_name, w, d, l, wr = res
        print(f"{cp_name:<35} | {w:<5} | {d:<5} | {l:<6} | {wr:.1%}")
    print("-" * 80)

    print("-" * 75)
    print("RL 평가가 완료되었습니다.")

if __name__ == "__main__":
    main()
