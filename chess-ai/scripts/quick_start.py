#!/usr/bin/env python3
"""
Quick start guide: 데이터셋 확인 및 사용 예제
"""

import os
import sys

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, os.path.join(project_root, "chess-ai"))

from datasets import load_from_disk
from data.preprocessing import get_dataset_path

print("=" * 60)
print("Chess AI 데이터셋 사용 가이드")
print("=" * 60)

# 데이터셋 경로
dataset_name = "test_twic_dataset"
dataset_path = get_dataset_path(dataset_name, script_path=__file__)

if os.path.exists(dataset_path):
    print(f"\n✅ 데이터셋 발견: {dataset_name}")
    print(f"   경로: {dataset_path}\n")
    
    # 데이터셋 로드
    dataset = load_from_disk(dataset_path)
    
    print(f"📊 데이터셋 정보:")
    print(f"   - 총 경기 수: {len(dataset)}")
    print(f"   - 컬럼: {', '.join(dataset.column_names)}")
    
    # 샘플 확인
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n📝 첫 번째 경기 샘플:")
        print(f"   - White: {sample.get('White', 'N/A')}")
        print(f"   - Black: {sample.get('Black', 'N/A')}")
        print(f"   - Result: {sample.get('Result', 'N/A')}")
        print(f"   - Moves: {str(sample.get('moves', ''))[:100]}...")
    
    print(f"\n🚀 다음 단계:")
    print(f"\n1. 데이터셋을 HDF5 형식으로 변환 (학습용):")
    print(f"   python3 chess-ai/scripts/process_datasets_to_hdf5.py \\")
    print(f"       --dataset-name {dataset_name} \\")
    print(f"       --output-name train.h5")
    
    print(f"\n2. 또는 직접 ChessDataset으로 사용:")
    print(f"   from chess_ai.data.dataset import ChessDataset")
    print(f"   dataset = ChessDataset(dataset_name='{dataset_name}', cache_in_memory=True)")
    
    print(f"\n3. 학습 시작:")
    print(f"   python3 chess-ai/scripts/train_supervised.py \\")
    print(f"       --train-dataset-name {dataset_name} \\")
    print(f"       --epochs 10 \\")
    print(f"       --batch-size 64")
    
    print(f"\n" + "=" * 60)
    
else:
    print(f"❌ 데이터셋을 찾을 수 없습니다: {dataset_path}")
    print(f"\n사용 가능한 데이터셋을 확인하세요:")
    processed_dir = os.path.join(project_root, "data", "processed")
    if os.path.exists(processed_dir):
        dirs = [d for d in os.listdir(processed_dir) 
                if os.path.isdir(os.path.join(processed_dir, d))]
        if dirs:
            for d in dirs:
                print(f"   - {d}")
        else:
            print(f"   {processed_dir} 폴더가 비어있습니다.")
