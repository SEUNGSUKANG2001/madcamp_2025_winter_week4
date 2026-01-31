# 설치 가이드

## 빠른 시작

### 1. 가상환경 생성 및 활성화

```bash
# 프로젝트 루트에서
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows
```

### 2. 의존성 설치

```bash
# 방법 A: pip로 직접 설치
pip install datasets chess torch numpy pyyaml h5py tqdm zstandard python-chess

# 방법 B: pyproject.toml 기반 (가상환경에서)
pip install -e .

# 방법 C: Poetry 사용 (Poetry가 설치된 경우)
poetry install
```

### 3. 데이터셋 확인

```bash
# 가상환경 활성화 후
python3 chess-ai/scripts/load_and_check_dataset.py
```

## 문제 해결

### `externally-managed-environment` 오류

macOS에서 시스템 Python을 사용하는 경우 발생할 수 있습니다. 해결 방법:

1. **가상환경 사용 (권장)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install datasets
   ```

2. **--break-system-packages 플래그 사용 (비권장)**
   ```bash
   pip install --break-system-packages datasets
   ```

### `poetry: command not found`

Poetry가 설치되어 있지 않은 경우:

```bash
# Poetry 설치
curl -sSL https://install.python-poetry.org | python3 -

# 또는 pip로 설치
pip install poetry
```

### `datasets` 라이브러리 설치 실패

```bash
# 가상환경에서
pip install --upgrade pip
pip install datasets
```

## 필수 패키지 목록

- `datasets`: HuggingFace datasets 라이브러리
- `chess` 또는 `python-chess`: 체스 게임 로직
- `torch`: PyTorch (딥러닝 프레임워크)
- `numpy`: 수치 계산
- `h5py`: HDF5 파일 처리
- `tqdm`: 진행 표시줄
- `zstandard`: 압축 파일 처리
- `pyyaml`: YAML 설정 파일

## 확인

설치가 완료되었는지 확인:

```bash
python3 -c "import datasets; import chess; import torch; print('All packages installed!')"
```
