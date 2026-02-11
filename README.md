# Chess AI - 강화학습 체스 인공지능

## 프로젝트 소개

이 프로젝트는 Transformer 기반의 체스 인공지능을 강화학습(RL)으로 학습시키는 시스템입니다. [Maxlegrec/ChessBot](https://huggingface.co/Maxlegrec/ChessBot) 모델을 기반으로, LoRA(Low-Rank Adaptation)를 사용하여 효율적으로 파인튜닝하며, Self-Play와 Stockfish 엔진을 활용한 커리큘럼 학습을 통해 성능을 향상시킵니다.

## 주요 기능

### 🤖 AI 학습 시스템
- **강화학습 (Reinforcement Learning)**: REINFORCE 알고리즘 기반 Policy Gradient 학습
- **Self-Play**: 과거 버전의 모델과 대결하며 지속적으로 발전
- **커리큘럼 학습**: Stockfish 엔진의 난이도(depth)를 동적으로 조절하여 점진적 난이도 증가
- **LoRA 파인튜닝**: 적은 파라미터로 효율적인 학습 수행

### 🎮 웹 인터페이스
- **실시간 대국**: 학습된 AI와 대국할 수 있는 인터랙티브 체스판
- **색상 선택**: 백 또는 흑으로 플레이 가능
- **실시간 분석**: 
  - 현재 포지션의 승률 분석 (백 승/무승부/흑 승)
  - AI가 추천하는 상위 5개 수와 확률
- **보드 조작**: 보드 회전, 무르기, 게임 리셋 기능

## 프로젝트 구조

```
madcamp_2025_winter_week4/
├── chess-ai/                    # AI 학습 관련 코드
│   ├── scripts/
│   │   ├── train_rl.py         # 강화학습 메인 스크립트
│   │   ├── train_sl.py         # 지도학습 스크립트
│   │   └── evaluate.py         # 모델 평가 스크립트
│   └── utils/
│       ├── check_layers.py     # 모델 레이어 확인
│       └── inspect_model.py    # 모델 구조 검사
│
├── web/                         # 웹 인터페이스
│   ├── app.py                  # Flask 백엔드 서버
│   ├── templates/
│   │   └── index.html          # 프론트엔드 UI
│   └── requirements.txt        # 웹 서버 의존성
│
├── checkpoints/                 # 학습된 모델 체크포인트
├── data/                        # 학습 데이터
└── wandb/                       # WandB 실험 로그

```

## 설치 방법

### 필수 요구사항
- Python 3.8+
- CUDA 지원 GPU (권장)
- Stockfish 체스 엔진

### 1. Python 패키지 설치

```bash
pip install torch transformers peft chess wandb flask flask-cors
```

### 2. Stockfish 설치

**Ubuntu/WSL:**
```bash
sudo apt-get install stockfish
```

**Windows:**
- [Stockfish 공식 웹사이트](https://stockfishchess.org/download/)에서 다운로드
- PATH에 추가하거나 `train_rl.py`에서 경로 지정

## 사용 방법

### AI 학습하기

1. **강화학습 시작:**
```bash
cd chess-ai/scripts
python train_rl.py
```

학습 과정은 WandB에 자동으로 기록됩니다:
- 승률, 무승부율, 패배율
- Policy Loss 및 Entropy
- Stockfish 난이도 변화
- 평균 보상 추이

2. **학습 설정 커스터마이징:**

`train_rl.py` 파일에서 다음 파라미터를 조정할 수 있습니다:

```python
LR = 2e-5                    # 학습률
NUM_EPISODES = 100000        # 총 에피소드 수
STOCKFISH_DEPTH = 4          # 초기 Stockfish 난이도
SAVE_INTERVAL = 100          # 체크포인트 저장 간격
LORA_R = 8                   # LoRA rank
```

### 웹 인터페이스 실행

1. **Flask 서버 시작:**
```bash
cd web
python app.py
```

2. **브라우저에서 접속:**
```
http://localhost:5000
```

3. **게임 플레이:**
   - 색상 선택 (백/흑)
   - 드래그 앤 드롭으로 말 이동
   - AI의 수와 분석을 실시간으로 확인

## 주요 알고리즘

### 강화학습 (REINFORCE)

- **보상 체계:**
  - 승리: +1.5
  - 무승부: -0.1
  - 패배: -1.0

- **Policy Gradient Loss:**
```python
advantage = reward - avg_reward
pg_loss = -(log_probs * advantage).mean()
entropy_loss = -ENTROPY_COEFF * entropy.mean()
total_loss = pg_loss + entropy_loss
```

### 커리큘럼 학습

Stockfish 상대 승률이 60%를 넘으면 자동으로 난이도(depth) 증가:
```python
if sf_win_rate >= 0.60:
    curr_sf_depth += 1
```

### Self-Play 전략

각 에피소드마다:
- 30%: Stockfish와 대국
- 70%: 과거 버전 모델과 Self-Play
  - 가장 최근 2개 체크포인트 또는 현재 모델 중 랜덤 선택

## 모델 아키텍처

### 베이스 모델
- **ChessBot** (Maxlegrec/ChessBot)
- Transformer 기반 체스 전용 모델
- FEN(Forsyth-Edwards Notation)을 입력으로 받아 1929개의 가능한 수에 대한 정책 출력

### LoRA 적용
```python
TARGET_MODULES = [
    "query_proj", "key_proj", "value_proj", 
    "out_proj", "ff1", "ff2", "policy_head"
]
```

## 웹 인터페이스 특징

### 시각적 분석
- **승률 바**: 백/무승부/흑 승률을 시각적으로 표시
- **어드밴티지 스코어**: 현재 포지션의 유불리 수치화
- **Top 5 수**: AI가 추천하는 최선의 수들과 선택 확률

### 사용자 경험
- **다크 모드** 기본 적용
- **반응형 디자인**: 데스크톱/모바일 모두 지원
- **부드러운 애니메이션**: Glassmorphism 스타일
- **실시간 피드백**: AI 사고 중 로딩 표시

## 학습 모니터링

### WandB 대시보드
프로젝트는 다음 메트릭을 추적합니다:

- `reward`: 각 에피소드의 보상
- `sf/win_rate`: Stockfish 상대 승률
- `sf/depth`: 현재 Stockfish 난이도
- `self/win_rate`: Self-Play 승률
- `pg_loss`: Policy Gradient Loss
- `entropy`: 정책 엔트로피
- `eval/win_vs_ep0`: 초기 버전 대비 승률

### 체크포인트 관리
- 100 에피소드마다 자동 저장
- 파일명: `chess_ai_rl_ep{에피소드번호}.pt`
- 포함 정보: 모델 가중치, 옵티마이저 상태, 평균 보상

## 트러블슈팅

### GPU 메모리 부족
```python
# train_rl.py에서 배치 사이즈 줄이기
BATCH_SIZE = 10  # 기본값: 25
```

### Stockfish 경로 문제
```python
# train_rl.py 수정
STOCKFISH_PATH = "/path/to/your/stockfish"
```

### 웹 서버 포트 변경
```python
# web/app.py 수정
app.run(host="0.0.0.0", port=8080)
```

## 성능 향상 팁

1. **학습률 조정**: 너무 높으면 불안정, 너무 낮으면 느림
2. **엔트로피 계수**: 높이면 탐험 증가, 낮추면 수렴 빠름
3. **Self-Play 비율**: 초반엔 Stockfish 비중 높이고, 후반엔 Self-Play 늘리기
4. **배치 크기**: GPU 메모리 허용 범위 내에서 최대화

## 향후 개선 방향

- [ ] MCTS (Monte Carlo Tree Search) 통합
- [ ] 더 정교한 보상 함수 (포지션 평가 포함)
- [ ] Distributed 학습 지원
- [ ] 오프닝 북 학습
- [ ] 엔드게임 테이블베이스 활용

## 참고 자료

- [ChessBot 모델](https://huggingface.co/Maxlegrec/ChessBot)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [REINFORCE Algorithm](https://link.springer.com/article/10.1007/BF00992696)
- [Stockfish Chess Engine](https://stockfishchess.org/)

---

**개발 환경**: KAIST 몰입캠프 2025 겨울 Week 4  
**개발 기간**: 2026년 2월
