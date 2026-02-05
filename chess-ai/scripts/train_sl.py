import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.engine
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
import os
import random
from tqdm import tqdm
import wandb
from collections import deque
import sys

# === 설정 (CONFIGURATION) ===
MODEL_NAME = "Maxlegrec/ChessBot"
STOCKFISH_PATH = "stockfish"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
TOTAL_STEPS = 50000 
BATCH_SIZE = 64
STOCKFISH_DEPTH = 10 
SAVE_INTERVAL = 1000
BUFFER_SIZE = 50000 
MIN_BUFFER_SIZE = 1000 # 512에서 5000으로 상향 (데이터 다양성 확보)
UPDATES_PER_GAME = 10 
CHECKPOINT_DIR = "checkpoints"
WANDB_PROJECT = "chess-ai-sl-kd"

# 지식 증류 설정
TEMPERATURE = 5.0  

# LoRA 설정
USE_LORA = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["query_proj", "key_proj", "value_proj", "out_proj", "ff1", "ff2", "policy_head"]

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === 모델 로드 ===
print(f"SL 학습용 모델 로드 중 ({DEVICE})...")
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)

# 모델 내부 유틸리티 접근
model_module = sys.modules[model.__class__.__module__]
fen_to_tensor = model_module.fen_to_tensor
policy_index = model_module.policy_index

if USE_LORA:
    config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias="none", task_type=None
    )
    model = get_peft_model(model, config)

model.train()
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# === WandB 초기화 ===
wandb.init(project=WANDB_PROJECT, config={
    "lr": LR, "steps": TOTAL_STEPS, "batch_size": BATCH_SIZE, "depth": STOCKFISH_DEPTH
})

def get_stockfish_distribution(engine, board, depth):
    """Stockfish로부터 모든 합법적 수의 평가치를 받아 확률 분포로 반환합니다."""
    legal_moves = list(board.legal_moves)
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=len(legal_moves))
    
    move_scores = {}
    for entry in info:
        move_uci = entry["pv"][0].uci()
        score = entry["score"].relative.score(mate_score=10000)
        move_scores[move_uci] = score
    
    target_dist = torch.zeros(1929)
    uci_list = []
    score_list = []
    for uci, s in move_scores.items():
        uci_list.append(uci)
        score_list.append(s / TEMPERATURE)
        
    scores_t = torch.tensor(score_list)
    probs_t = torch.softmax(scores_t, dim=0)
    
    for uci, p in zip(uci_list, probs_t):
        lookup = uci[:-1] if (uci.endswith('n') and len(uci) == 5) else uci
        try:
            idx = policy_index.index(lookup)
        except ValueError:
            try: idx = policy_index.index(uci)
            except ValueError: continue
        target_dist[idx] = p
        
    return target_dist

def train_sl():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    pbar = tqdm(range(1, TOTAL_STEPS + 1), desc="KD 학습")
    
    # 1. 리플레이 버퍼 및 손실 함수 초기화
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    # 체크포인트 로드
    start_step = 1
    cps = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
    if cps:
        sl_cps = [f for f in cps if 'kd' in f]
        target_cps = sl_cps if sl_cps else cps
        latest = max(target_cps, key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)))
        cp = torch.load(os.path.join(CHECKPOINT_DIR, latest), map_location=DEVICE)
        model.load_state_dict(cp['model_state_dict'], strict=False)
        start_step = cp.get('step', 0) + 1
        print(f"체크포인트 로드: {latest} (Step: {start_step})")

    board = chess.Board()
    game_count = 0
    global_step = start_step
    
    while global_step <= TOTAL_STEPS:
        # A. 데이터 생성 (Knowledge Distillation용 Soft Targets 수집)
        sf_side = chess.WHITE if game_count % 2 == 0 else chess.BLACK
        game_count += 1
        board.reset()
        
        while not board.is_game_over():
            try:
                if board.turn == sf_side:
                    # 모든 수의 분포를 계산하여 버퍼에 저장
                    target_dist = get_stockfish_distribution(engine, board, STOCKFISH_DEPTH)
                    replay_buffer.append((board.fen(), target_dist))
                    
                    # 다음 수를 두기 위해 Stockfish가 고른 가장 좋은 수 선택
                    best_move_idx = torch.argmax(target_dist).item()
                    move_uci = policy_index[best_move_idx]
                    
                    # 승진 보정
                    move_obj = chess.Move.from_uci(move_uci)
                    if board.piece_at(move_obj.from_square).piece_type == chess.PAWN:
                        if (chess.square_rank(move_obj.to_square) == 7 and sf_side == chess.WHITE) or \
                           (chess.square_rank(move_obj.to_square) == 0 and sf_side == chess.BLACK):
                            if not move_obj.promotion: move_uci += 'q'
                    board.push_uci(move_uci)
                else:
                    # 대국 진행을 위한 모델의 수
                    with torch.no_grad():
                        res = model.get_move_from_fen_no_thinking(board.fen(), T=1.0, device=DEVICE)
                    if not res:
                        board.push(random.choice(list(board.legal_moves)))
                    else:
                        board.push_uci(res)
            except Exception:
                break
        
        # B. 학습 단계
        if len(replay_buffer) >= MIN_BUFFER_SIZE:
            for _ in range(UPDATES_PER_GAME):
                if global_step > TOTAL_STEPS: break
                
                # 배치 샘플링
                samples = random.sample(list(replay_buffer), BATCH_SIZE)
                batch_fens, batch_targets = zip(*samples)
                
                # 텐서 변환
                input_tensors = []
                for f in batch_fens:
                    arr = fen_to_tensor(f)
                    input_tensors.append(torch.from_numpy(arr).to(DEVICE).to(torch.float32).view(1, 8, 8, 19))
                
                inputs = torch.stack(input_tensors)
                targets = torch.stack(batch_targets).to(DEVICE) # [B, 1929]
                
                optimizer.zero_grad()
                outputs = model(inputs)
                logits = (outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs).view(BATCH_SIZE, -1)
                
                # KLDivLoss 학습 (Log Softmax vs Softmax Distribution)
                log_probs = F.log_softmax(logits, dim=-1)
                loss = criterion(log_probs, targets)
                loss.backward()
                optimizer.step()
                
                # 정확도 (가장 높은 확률의 수가 Stockfish의 1등 수와 일치하는가)
                acc = (torch.argmax(logits, 1) == torch.argmax(targets, 1)).sum().item() / BATCH_SIZE
                wandb.log({"step": global_step, "loss": loss.item(), "accuracy": acc, "buffer_size": len(replay_buffer)})
                
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.2%}", "Buf": len(replay_buffer)})
                pbar.update(1)
                
                if global_step % SAVE_INTERVAL == 0:
                    torch.save({'step': global_step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                               os.path.join(CHECKPOINT_DIR, f"chess_ai_sl_kd_step{global_step}.pt"))
                global_step += 1

    engine.quit()
    wandb.finish()

if __name__ == "__main__":
    train_sl()

    engine.quit()
    wandb.finish()

if __name__ == "__main__":
    train_sl()
