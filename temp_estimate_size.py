import os
import sys
import time
from datasets import load_from_disk
import shutil
import numpy as np

# 프로젝트 루트를 경로에 추가
# 프로젝트 루트를 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
# chess-ai 폴더를 path에 추가
sys.path.append(os.path.join(script_dir, "chess-ai"))

from scripts.preprocess_to_positions import process_batch

def main():
    # script_dir is the project root (madcamp_2025_winter_week4)
    input_path = os.path.join(script_dir, "data", "processed", "lichess-2500-180_dataset")
    output_path = os.path.join(script_dir, "data", "temp_position_sample")
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        
    print(f"Loading dataset from {input_path}...")
    ds = load_from_disk(input_path)
    
    # 100개만 샘플링 (빠른 확인용)
    sample_size = 100
    sample_ds = ds.select(range(sample_size))
    
    print(f"Processing {sample_size} games...")
    start_time = time.time()
    
    pos_ds = sample_ds.map(
        process_batch,
        batched=True,
        batch_size=10,
        num_proc=1,
        remove_columns=ds.column_names,
        desc="Processing sample",
        load_from_cache_file=False
    )
    
    duration = time.time() - start_time
    print(f"Processing took {duration:.2f} seconds.")
    print(f"Total positions extracted: {len(pos_ds)}")
    
    # 구조 출력 (사용자가 '어떻게 돼?'라고 물었으므로 구체적으로 보여줌)
    print("\n--- Processed Data Structure (Example) ---")
    if len(pos_ds) > 0:
        example = pos_ds[0]
        print("Keys:", example.keys())
        print(f"Position shape: {np.array(example['position']).shape} (Dtype: {np.array(example['position']).dtype})")
        print(f"Move index: {example['move']}")
        print(f"Value: {example['value']}")
        print(f"Move Number: {example['move_number']}")
        print(f"Player: {example['player']}")
        print("------------------------------------------\n")
    
    print(f"Saving to {output_path}...")
    pos_ds.save_to_disk(output_path)
    
    # 크기 측정
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(output_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
            
    print(f"Total size for {sample_size} games: {total_size / (1024*1024):.2f} MB")
    
    # 추정
    total_games = 7088787
    estimated_total_size_mb = (total_size / sample_size) * total_games / (1024*1024)
    estimated_total_size_gb = estimated_total_size_mb / 1024
    estimated_total_size_tb = estimated_total_size_gb / 1024
    
    print(f"--- Estimation Results ---")
    print(f"Average positions per game: {len(pos_ds) / sample_size:.1f}")
    print(f"Estimated total size for {total_games} games: {estimated_total_size_gb:.2f} GB ({estimated_total_size_tb:.2f} TB)")

if __name__ == "__main__":
    main()
