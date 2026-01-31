# Chess AI (DQN)

강화 학습(Double Deep Q-Networks)을 이용한 체스 AI 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 체스 엔진과 DQN 알고리즘을 결합하여 스스로 학습하고 성장하는 체스 에이전트를 개발하는 것을 목표로 합니다. 보드 상태를 신경망의 입력으로 변환하고, 최적의 수를 예측하는 Q-함수를 학습합니다.

## 디렉토리 구조

- `data/`: 데이터 저장소
  - `raw/`: 원본 게임 데이터 (PGN 파일 등)
  - `processed/`: 학습용으로 가공된 데이터
- `chess-ai/`: 소스 코드
  - `agents/`: DQN 에이전트 및 학습 로직
  - `configs/`: 설정 파일 (하이퍼파라미터 등)
  - `env/`: 체스 환경 래퍼 (chess 라이브러리 활용)
  - `scripts/`: 데이터 다운로드 및 기타 유틸리티 스크립트
  - `training/`: 학습 루프 및 체크포인트 관리
  - `utils/`: 헬퍼 함수 및 로그 유틸리티
- `pyproject.toml`: 프로젝트 의존성 및 설정

## 시작하기

### 환경 설정

이 프로젝트는 `poetry` 또는 `pip`을 사용하여 의존성을 관리합니다.

#### 방법 1: 가상환경 사용 (권장)

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
# 또는 pyproject.toml 기반 설치
pip install -e .
```

#### 방법 2: Poetry 사용

```bash
# Poetry 설치 (없는 경우)
curl -sSL https://install.python-poetry.org | python3 -

# 의존성 설치
poetry install
```

#### 방법 3: 직접 설치 (시스템 패키지 관리자 사용 시)

```bash
# 필요한 패키지 직접 설치
pip install --break-system-packages datasets chess torch numpy pyyaml h5py tqdm zstandard
```

**주의**: macOS에서 `externally-managed-environment` 오류가 발생하면 가상환경 사용을 권장합니다.

### 데이터 다운로드

강화학습의 사전 학습(Pre-training)을 위한 데이터를 다운로드하려면 다음 스크립트를 실행하세요.

```bash
python chess-ai/scripts/download_data.py
```

## 주요 기능

- **Double DQN**: Q-값 과대평가 문제를 해결하기 위한 Double DQN 알고리즘 적용
- **Experience Replay**: 학습 효율을 높이기 위한 경험 재현 버퍼 사용
- **Custom Environment**: 체스 보드 상태를 8x8x12 텐서로 변환하여 신경망 입력 최적화
