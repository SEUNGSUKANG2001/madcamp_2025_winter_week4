import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import chess.engine
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import os
import random
from tqdm import tqdm
import wandb
from collections import deque
import sys
import copy

# === 설정 (CONFIGURATION) ===
MODEL_NAME = "Maxlegrec/ChessBot"
STOCKFISH_PATH = "stockfish"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-5
NUM_EPISODES = 100000
STOCKFISH_DEPTH = 4
SAVE_INTERVAL = 100
CHECKPOINT_DIR = "../checkpoints"
WANDB_PROJECT = "chess-ai-rl"

# LoRA 설정
USE_LORA = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["query_proj", "key_proj", "value_proj", "out_proj", "ff1", "ff2", "policy_head"]
ENTROPY_COEFF = 0.01  # 엔트로피 보너스 계수
ADVANTAGE_EPS = 1e-8  # 수치 안정성용

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === 모델 로드 ===
print(f"학습 모델 로드 중 ({DEVICE})...")
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)

# 모델 내부의 은닉 함수 및 변수 접근
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

# === 상대방 모델(Self-Play용) 초기화 ===
# 학습에 영향을 주지 않도록 별도의 인스턴스 사용
opponent_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
if USE_LORA:
    opponent_model = get_peft_model(opponent_model, config)
opponent_model.eval() # 상대방은 항상 평가 모드

# === WandB 초기화 ===
wandb.init(project=WANDB_PROJECT, config={
    "lr": LR, "episodes": NUM_EPISODES, "depth": STOCKFISH_DEPTH, "device": DEVICE
})

def get_differentiable_policy(model, fens, action_indices):
    """FEN 목록과 액션 인덱스를 받아 미분 가능한 로그 확률과 엔트로피를 반환합니다."""
    tensors = []
    for fen in fens:
        arr = fen_to_tensor(fen)
        tensors.append(torch.from_numpy(arr).to(DEVICE).to(torch.float32).view(1, 8, 8, 19))
    
    input_tensor = torch.stack(tensors) # [batch, 1, 8, 8, 19]
    # 모델(ChessBot)은 (batch, seq_len, 8, 8, channels)의 5차원을 기대합니다.
    # tensors 내 각 요소가 (1, 8, 8, 19)이므로 stack 결과는 이미 5차원입니다.
        
    output = model(input_tensor)
    logits = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output
    logits = logits.view(-1, 1929)
    
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    action_tensor = torch.tensor(action_indices, device=DEVICE).unsqueeze(1)
    selected_log_probs = log_probs.gather(1, action_tensor).squeeze(1)
    
    # Entropy: -sum(p * log_p)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    return selected_log_probs, entropy

def play_rl_episode(model, opponent, ai_side=chess.WHITE, is_engine=True, depth=STOCKFISH_DEPTH):
    """게임을 진행하고 (FEN, Action) 리스트와 최종 보상을 반환합니다."""
    board = chess.Board()
    fens, actions = [], []
    
    while not board.is_game_over():
        if board.turn == ai_side:
            # AI의 차례 (학습 대상)
            fen = board.fen()
            
            try:
                # 모델로부터 수의 확률 분포 획득
                res = model.get_move_from_fen_no_thinking(fen, T=1.0, device=DEVICE, return_probs=True)
                
                if not res:
                    # 도저히 수를 낼 수 없는 경우 (드묾)
                    legal_moves = list(board.legal_moves)
                    move = random.choice(legal_moves)
                    board.push(move)
                    continue
                
                legal_moves_uci = [m.uci() for m in board.legal_moves]
                filtered = [(m, p) for m, p in res.items() if m in legal_moves_uci]
                
                if not filtered:
                    # 합법적 수가 하나도 없는 경우 (드묾)
                    legal_moves = list(board.legal_moves)
                    move = random.choice(legal_moves)
                    board.push(move)
                    continue
                
                moves, probs = zip(*filtered)
                probs_t = torch.tensor(probs)
                if probs_t.sum() <= 0: probs_t = torch.ones_like(probs_t)
                probs_t /= probs_t.sum()
                
                idx = torch.multinomial(probs_t, 1).item()
                selected_move_uci = moves[idx]
                
                # 모델의 전역 policy_index에서의 인덱스 찾기
                lookup = selected_move_uci[:-1] if (selected_move_uci.endswith('n') and len(selected_move_uci) == 5) else selected_move_uci
                try:
                    g_idx = policy_index.index(lookup)
                except ValueError:
                    g_idx = policy_index.index(selected_move_uci)
                
                # 인덱스 찾기에 성공한 경우에만 학습 데이터에 추가
                fens.append(fen)
                actions.append(g_idx)
                board.push_uci(selected_move_uci)
                
            except Exception as e:
                # 'b2c1n' is not in list 등 내부 인덱싱 오류 발생 시
                # 해당 수만 건너뛰고 게임은 랜덤하게 계속 진행합니다.
                legal_moves = list(board.legal_moves)
                if not legal_moves: break
                
                fallback_move = random.choice(legal_moves)
                # print(f"Warning: AI error ({e}). Using random move {fallback_move} and skipping this step for training.")
                board.push(fallback_move)
                continue
        else:
            # 상대방의 차례
            if is_engine:
                result = opponent.play(board, chess.engine.Limit(depth=depth))
                board.push(result.move)
            else:
                # Self-Play 모델 대결 (상대방도 합법적인 수만 두도록 필터링)
                try:
                    res = opponent.get_move_from_fen_no_thinking(board.fen(), T=1.0, device=DEVICE, return_probs=True)
                    if not res:
                        # 수를 낼 수 없는 경우 랜덤하게
                        move = random.choice(list(board.legal_moves))
                        board.push(move)
                    else:
                        legal_moves_uci = [m.uci() for m in board.legal_moves]
                        filtered = [(m, p) for m, p in res.items() if m in legal_moves_uci]
                        
                        if not filtered:
                            move = random.choice(list(board.legal_moves))
                            board.push(move)
                        else:
                            moves, probs = zip(*filtered)
                            probs_t = torch.tensor(probs)
                            if probs_t.sum() <= 0: probs_t = torch.ones_like(probs_t)
                            probs_t /= probs_t.sum()
                            idx = torch.multinomial(probs_t, 1).item()
                            board.push_uci(moves[idx])
                except Exception:
                    # 어떤 이유로든 실패하면 랜덤하게 한 수
                    if not board.is_game_over():
                        board.push(random.choice(list(board.legal_moves)))
                

    res_str = board.result()
    reward = -1.0
    if (ai_side == chess.WHITE and res_str == "1-0") or (ai_side == chess.BLACK and res_str == "0-1"):
        reward = 1.5
    elif res_str == "1/2-1/2":
        reward = -0.1
        
    return fens, actions, reward

def evaluate_against_ep0(model, opponent_model, n_games=5):
    """최초 버전(ep0)과 대결하여 승리 횟수를 확인합니다."""
    # ep0 체크포인트 로드
    ep0_path = os.path.join(CHECKPOINT_DIR, "chess_ai_rl_ep0.pt")
    if not os.path.exists(ep0_path):
        return None
    
    opponent_model.load_state_dict(torch.load(ep0_path, map_location=DEVICE)['model_state_dict'], strict=False)
    opponent_model.eval()
    
    wins, draws, losses = 0, 0, 0
    
    for i in range(n_games):
        # 번갈아가며 선공 수행
        ai_side = chess.WHITE if i % 2 == 0 else chess.BLACK
        # play_rl_episode와 유사하지만 학습 데이터를 수집하지 않는 평가 모드
        board = chess.Board()
        while not board.is_game_over():
            curr_model = model if board.turn == ai_side else opponent_model
            try:
                res = curr_model.get_move_from_fen_no_thinking(board.fen(), T=0.1, device=DEVICE, return_probs=True)
                if not res:
                    board.push(random.choice(list(board.legal_moves)))
                else:
                    legal_moves_uci = [m.uci() for m in board.legal_moves]
                    filtered = [(m, p) for m, p in res.items() if m in legal_moves_uci]
                    if not filtered:
                        board.push(random.choice(list(board.legal_moves)))
                    else:
                        moves, probs = zip(*filtered)
                        probs_t = torch.tensor(probs)
                        if probs_t.sum() <= 0: probs_t = torch.ones_like(probs_t)
                        probs_t /= probs_t.sum()
                        idx = torch.multinomial(probs_t, 1).item()
                        board.push_uci(moves[idx])
            except Exception:
                if not board.is_game_over():
                    board.push(random.choice(list(board.legal_moves)))
        
        res_str = board.result()
        if (ai_side == chess.WHITE and res_str == "1-0") or (ai_side == chess.BLACK and res_str == "0-1"):
            wins += 1
        elif res_str == "1/2-1/2":
            draws += 1
        else:
            losses += 1
            
    return wins, draws, losses

def train():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    start_episode = 1
    avg_reward = 0
    stockfish_loss_history = deque(maxlen=100) # Stockfish 상대 패배율
    overall_loss_history = deque(maxlen=100)   # 전체 패배율 (Self-Play 포함)
    
    # 마지막 체크포인트 불러오기
    if os.path.exists(CHECKPOINT_DIR):
        cps = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt') and 'ep' in f]
    else:
        cps = []
        
    if cps:
        # 파일명을 숫자 순으로 정렬하여 가장 큰 에피소드 번호를 찾음
        latest = max(cps, key=lambda x: int(x.split('ep')[-1].split('.pt')[0]))
        cp = torch.load(os.path.join(CHECKPOINT_DIR, latest), map_location=DEVICE)
        model.load_state_dict(cp['model_state_dict'], strict=False)
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        start_episode = cp['episode'] + 1
        avg_reward = cp.get('avg_reward', 0)
        print(f"체크포인트 로드: {latest} (start_episode={start_episode})")
    else:
        # 최초 실행 시 ep0 저장 (Self-Play가 즉시 작동하도록 함)
        save_path = os.path.join(CHECKPOINT_DIR, "chess_ai_rl_ep0.pt")
        torch.save({
            'episode': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_reward': 0
        }, save_path)
        print("최초 ep0 체크포인트를 저장했습니다.")

    pbar = tqdm(range(start_episode, NUM_EPISODES + 1), desc="학습")
    
    # 통계 추적용 Deque (최근 100경기)
    sf_stats = {"w": deque(maxlen=100), "d": deque(maxlen=100), "l": deque(maxlen=100)}
    self_stats = {"w": deque(maxlen=100), "d": deque(maxlen=100), "l": deque(maxlen=100)}
    
    # 커리큘럼 학습 변수
    curr_sf_depth = STOCKFISH_DEPTH
    
    BATCH_SIZE = 25 # 100 에피소드마다 실제 업데이트 수행 (Gradient Accumulation)
    optimizer.zero_grad() # 시작 전 초기화
    
    for ep in pbar:
        # 상대방 선택 (30% 엔진, 70% 셀프 플레이)
        is_engine = random.random() < 0.3
        opp_name = "Stockfish"
        
        if not is_engine:
            # 70% 상황: [가장 최근 체크포인트 2개, 현재 모델] 중에서 랜덤 선택
            cps = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')], 
                        key=lambda x: int(x.split('ep')[-1].split('.pt')[0]))
            
            # 선택 후보군 구성
            pool = []
            if cps:
                pool.extend(cps[-2:]) # 최신 2개
            pool.append("CURRENT")   # 현재 실시간 모델
            
            chosen = random.choice(pool)
            
            if chosen == "CURRENT":
                # 현재 모델의 가중치를 상대방에게 복사 (학습 중인 최신 버전)
                opponent_model.load_state_dict(model.state_dict())
                opp_name = "Self(Current)"
            else:
                # 선택된 과거 체크포인트 로드
                opp_path = os.path.join(CHECKPOINT_DIR, chosen)
                opponent_model.load_state_dict(torch.load(opp_path, map_location=DEVICE)['model_state_dict'], strict=False)
                opp_name = f"Self({chosen})"
        
        ai_side = chess.WHITE if ep % 2 == 0 else chess.BLACK
        # dynamic depth 적용
        fens, actions, reward = play_rl_episode(model, engine if is_engine else opponent_model, ai_side, is_engine, depth=curr_sf_depth)
        
        if fens:
            # 배치 사이즈로 나누어 평균적인 그래디언트 세기 유지
            log_probs, entropy = get_differentiable_policy(model, fens, actions)
            
            # Advantage calculation
            advantage = reward - avg_reward
            
            # Policy Gradient Loss with Entropy Regularization
            pg_loss = -(log_probs * advantage).mean()
            entropy_loss = -ENTROPY_COEFF * entropy.mean()
            loss = (pg_loss + entropy_loss) / BATCH_SIZE
            
            loss.backward()
            
            # 100 에피소드마다 가중치 업데이트
            if ep % BATCH_SIZE == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # 가중치 업데이트 후 ep0와 평가 수행
                eval_res = evaluate_against_ep0(model, opponent_model)
                if eval_res:
                    w, d, l = eval_res
                    print(f"\n[평가 ep{ep}] vs ep0: {w}승 {d}무 {l}패")
                    wandb.log({"eval/win_vs_ep0": w, "eval/draw_vs_ep0": d, "eval/loss_vs_ep0": l})
            
            # 기록 및 통계 업데이트
            avg_reward = 0.99 * avg_reward + 0.01 * reward
            
            # 승/무/패 판정 (수정된 보상 기준)
            is_w = 1 if reward == 1.5 else 0
            is_d = 1 if reward == -0.1 else 0
            is_l = 1 if reward == -1.0 else 0
            
            if is_engine:
                sf_stats["w"].append(is_w); sf_stats["d"].append(is_d); sf_stats["l"].append(is_l)
            else:
                self_stats["w"].append(is_w); self_stats["d"].append(is_d); self_stats["l"].append(is_l)
            
            # 비율 계산 함수
            def get_rate(dq): return sum(dq) / len(dq) if dq else 0
            
            sf_wr = get_rate(sf_stats["w"])
            sf_dr = get_rate(sf_stats["d"])
            sf_lr = get_rate(sf_stats["l"])
            self_wr = get_rate(self_stats["w"])
            self_dr = get_rate(self_stats["d"])
            self_lr = get_rate(self_stats["l"])
            
            # 커리큘럼 학습: Stockfish 상대 승률이 60%를 넘으면 Depth 증가
            # 최소 20판 이상의 데이터가 쌓였을 때 판단 (초반 노이즈 방지)
            if is_engine and len(sf_stats["w"]) >= 20 and sf_wr >= 0.60:
                curr_sf_depth += 1
                print(f"\n[커리큘럼] Stockfish 승률 {sf_wr:.1%} 달성! Depth를 {curr_sf_depth}로 올립니다.")
                # 새로운 난이도에서의 측정을 위해 통계 초기화
                for k in sf_stats: sf_stats[k].clear()
                sf_wr, sf_dr, sf_lr = 0, 0, 0
            
            wandb.log({
                "ep": ep, 
                "reward": reward, 
                "loss": loss.item() * BATCH_SIZE, 
                "pg_loss": pg_loss.item(),
                "entropy": entropy.mean().item(),
                "avg_reward": avg_reward, 
                "sf/win_rate": sf_wr, "sf/draw_rate": sf_dr, "sf/loss_rate": sf_lr,
                "sf/depth": curr_sf_depth,
                "self/win_rate": self_wr, "self/draw_rate": self_dr, "self/loss_rate": self_lr,
                "opponent": opp_name
            })
            
            pbar.set_postfix({
                "AvgR": f"{avg_reward:.2f}", 
                "SF_WR": f"{sf_wr:.1%}", 
                "D": curr_sf_depth,
                "Opp": opp_name
            })

        if ep % SAVE_INTERVAL == 0:
            torch.save({'episode': ep, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'avg_reward': avg_reward}, 
                       os.path.join(CHECKPOINT_DIR, f"chess_ai_rl_ep{ep}.pt"))

    engine.quit()
    wandb.finish()

if __name__ == "__main__":
    train()
