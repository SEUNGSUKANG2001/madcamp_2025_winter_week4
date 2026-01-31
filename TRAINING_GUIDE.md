# 체스 AI 학습 가이드

## 📊 데이터 확인

현재 `data/processed/lichess-2500-180_dataset/` 디렉토리에 전처리된 데이터가 있습니다.

## 🚀 학습 시작하기

### 방법 1: 데이터셋 이름으로 학습 (권장)

```bash
cd /workspace/madcamp_2025_winter_week4

python3 chess-ai/scripts/train_supervised.py \
    --train-dataset-name lichess-2500-180_dataset \
    --val-dataset-name lichess-2500-180_dataset \
    --epochs 50 \
    --batch-size 256 \
    --learning-rate 1e-3 \
    --device auto \
    --checkpoint-dir checkpoints
```

### 방법 2: 직접 경로 지정

```bash
cd /workspace/madcamp_2025_winter_week4

python3 chess-ai/scripts/train_supervised.py \
    --train-data data/processed/lichess-2500-180_dataset \
    --val-data data/processed/lichess-2500-180_dataset \
    --epochs 50 \
    --batch-size 256 \
    --learning-rate 1e-3 \
    --device auto \
    --checkpoint-dir checkpoints
```

## 📝 주요 파라미터 설명

- `--train-dataset-name`: 학습용 데이터셋 이름 (data/processed/ 안의 폴더명)
- `--val-dataset-name`: 검증용 데이터셋 이름 (같은 데이터셋 사용 가능)
- `--epochs`: 학습 에포크 수 (기본값: 50)
- `--batch-size`: 배치 크기 (기본값: 256, GPU 메모리에 따라 조정)
- `--learning-rate`: 학습률 (기본값: 1e-3)
- `--device`: 사용할 디바이스 (auto/cuda/cpu, 기본값: auto)
- `--checkpoint-dir`: 체크포인트 저장 디렉토리 (기본값: checkpoints)
- `--min-elo`: 최소 ELO 필터링 (기본값: 2600)

## 💡 학습 팁

1. **GPU 메모리 부족 시**: `--batch-size`를 줄이세요 (예: 128, 64)
2. **학습 속도 향상**: `--batch-size`를 늘리세요 (예: 512, 1024)
3. **검증 데이터 분리**: 가능하면 별도의 검증 데이터셋을 사용하세요
4. **체크포인트**: 10 에포크마다 자동으로 저장됩니다

## 📈 학습 모니터링

학습 중 다음 정보가 출력됩니다:
- Loss (전체 손실, 정책 손실, 가치 손실)
- Accuracy (Top-1, Top-3, Top-5 정확도)
- Validation 메트릭

## 🔄 학습 재개

이전 체크포인트에서 학습을 재개하려면:
```python
# 코드에서 직접 체크포인트 로드
from models.utils import load_checkpoint
model, optimizer, epoch = load_checkpoint('checkpoints/checkpoint_epoch_10.pth')
```

## ⚠️ 주의사항

- 데이터셋이 큰 경우 첫 로딩에 시간이 걸릴 수 있습니다
- HuggingFace datasets 형식의 데이터는 자동으로 감지됩니다
- ELO 필터링은 데이터셋 형식에 따라 자동으로 적용됩니다
