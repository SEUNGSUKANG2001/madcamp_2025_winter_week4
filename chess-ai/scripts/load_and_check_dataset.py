#!/usr/bin/env python3
"""
Load and check dataset from data/processed folder.

This script demonstrates how to load datasets using the project structure.
"""

import os
import sys

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, os.path.join(project_root, "chess-ai"))

try:
    from datasets import load_from_disk
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("=" * 60)
    print("ERROR: 'datasets' library not found!")
    print("=" * 60)
    print("\n다음 명령어로 설치하세요:")
    print("  pip install datasets")
    print("\n또는 poetry를 사용하는 경우:")
    print("  poetry add datasets")
    print("\n또는 pyproject.toml에 이미 포함되어 있다면:")
    print("  poetry install")
    print("=" * 60)
    sys.exit(1)

from data.preprocessing import get_project_root, get_dataset_path

# 스크립트 위치 기준 프로젝트 루트 계산
script_file = os.path.abspath(__file__)
project_root = get_project_root(script_file)

# 데이터셋 경로 (data/processed 폴더 안)
dataset_name = "chess_dataset"  # 또는 다른 데이터셋 이름
dataset_path = get_dataset_path(dataset_name, script_path=script_file)

# 여러 가능한 데이터셋 이름 시도
possible_names = ["chess_dataset", "test_twic_dataset", "twic_dataset"]

dataset = None
for name in possible_names:
    try_path = get_dataset_path(name, script_path=script_file)
    if os.path.exists(try_path):
        dataset_path = try_path
        dataset_name = name
        print(f"Found dataset: {name} at {dataset_path}")
        break

if os.path.exists(dataset_path):
    # 데이터셋 로드 (이미 위에서 DATASETS_AVAILABLE 체크 완료)
    dataset = load_from_disk(dataset_path)

    print(f"\n--- 데이터셋 정보 ---")
    print(f"경로: {dataset_path}")
    print(f"총 경기 수: {len(dataset)}")
    print(f"컬럼 구성: {dataset.column_names}")
    print(f"--------------------\n")

    # 첫 번째 경기 확인
    print("--- 첫 번째 경기 샘플 ---")
    if len(dataset) > 0:
        first_game = dataset[0]
        for key, value in first_game.items():
            if key == 'moves':
                # moves는 길 수 있으므로 일부만 표시
                moves_preview = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                print(f"{key}: {moves_preview}")
            else:
                print(f"{key}: {value}")
    else:
        print("데이터셋이 비어있습니다.")
    
    # 몇 가지 통계
    print(f"\n--- 통계 ---")
    if 'WhiteElo' in dataset.column_names and 'BlackElo' in dataset.column_names:
        white_elos = [int(g.get('WhiteElo', 0)) if str(g.get('WhiteElo', '')).isdigit() else 0 
                     for g in dataset if g.get('WhiteElo')]
        black_elos = [int(g.get('BlackElo', 0)) if str(g.get('BlackElo', '')).isdigit() else 0 
                     for g in dataset if g.get('BlackElo')]
        
        if white_elos:
            print(f"평균 White ELO: {sum(white_elos) / len(white_elos):.0f}")
            print(f"최소 White ELO: {min(white_elos)}")
            print(f"최대 White ELO: {max(white_elos)}")
        
        if black_elos:
            print(f"평균 Black ELO: {sum(black_elos) / len(black_elos):.0f}")
            print(f"최소 Black ELO: {min(black_elos)}")
            print(f"최대 Black ELO: {max(black_elos)}")
    
    print(f"\n데이터셋을 사용하려면:")
    print(f"  from chess_ai.data.dataset import ChessDataset")
    print(f"  dataset = ChessDataset(dataset_name='{dataset_name}')")
    
else:
    print(f"에러: {dataset_path} 경로를 찾을 수 없습니다.")
    print(f"\n가능한 데이터셋 이름을 확인하세요:")
    processed_dir = os.path.join(project_root, "data", "processed")
    if os.path.exists(processed_dir):
        dirs = [d for d in os.listdir(processed_dir) 
                if os.path.isdir(os.path.join(processed_dir, d))]
        if dirs:
            print(f"  발견된 데이터셋:")
            for d in dirs:
                print(f"    - {d}")
        else:
            print(f"  {processed_dir} 폴더가 비어있습니다.")
    else:
        print(f"  {processed_dir} 폴더가 존재하지 않습니다.")
