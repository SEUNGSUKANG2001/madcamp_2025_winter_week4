from datasets import load_from_disk
import os

# 스크립트 위치 기준 프로젝트 루트 계산
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

# 데이터셋 경로
dataset_path = os.path.join(project_root, "data2", "test_twic_dataset")

if os.path.exists(dataset_path):
    # 데이터셋 로드
    dataset = load_from_disk(dataset_path)

    print(f"--- 데이터셋 정보 ---")
    print(f"총 경기 수: {len(dataset)}")
    print(f"컬럼 구성: {dataset.column_names}")
    print(f"--------------------\n")

    # 첫 번째 경기 확인
    print("--- 첫 번째 경기 샘플 ---")
    print(dataset[0])
else:
    print(f"에러: {dataset_path} 경로를 찾을 수 없습니다.")