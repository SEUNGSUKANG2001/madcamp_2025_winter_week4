import os
import sys
import logging
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, Dataset

# 프로젝트 루트를 경로에 추가
# scripts/ 가 chess-ai/ 내부에 있으므로, 두 번 위로 올라가야 madcamp_2025_winter_week4 디렉토리에 도달합니다.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(os.path.join(project_root, "chess-ai"))

from data.preprocessing import process_game_from_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_batch(batch):
    """
    게임 배치(batch)를 받아 모든 포지션을 추출합니다.
    (One-to-many mapping)
    """
    all_positions = []
    all_moves = []
    all_values = []
    all_move_numbers = []
    all_players = []
    
    # 배치 내의 각 게임 처리
    # batch는 {'Site': [...], 'moves': [...], ...} 형태의 딕셔너리
    num_games = len(batch['moves'])
    for i in range(num_games):
        game_data = {k: batch[k][i] for k in batch.keys()}
        positions = process_game_from_datasets(game_data)
        for pos in positions:
            all_positions.append(pos["position"])
            all_moves.append(pos["move"])
            all_values.append(pos["value"])
            all_move_numbers.append(pos["move_number"])
            all_players.append(pos["player"])
            
    return {
        "position": all_positions,
        "move": all_moves,
        "value": all_values,
        "move_number": all_move_numbers,
        "player": all_players
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess chess games to positions.")
    parser.add_argument("--shard_index", type=int, default=0, help="Index of the shard to process (0-based).")
    parser.add_argument("--total_shards", type=int, default=1, help="Total number of shards.")
    args = parser.parse_args()

    # 경로 설정
    processed_dir = os.path.join(project_root, "data", "processed")
    output_base_dir = os.path.join(project_root, "data", "processed_position")
    os.makedirs(output_base_dir, exist_ok=True)

    # 처리할 대상 데이터셋 (예: lichess-2500-180_dataset)
    target_datasets = [d for d in os.listdir(processed_dir) if d.endswith("_dataset")]
    
    if not target_datasets:
        logger.error(f"'{processed_dir}'에서 데이터셋을 찾을 수 없습니다.")
        return

    for ds_name in target_datasets:
        input_path = os.path.join(processed_dir, ds_name)
        
        # 출력 경로 설정 (샤딩 지원)
        if args.total_shards > 1:
            output_ds_name = f"{ds_name}_part_{args.shard_index}_of_{args.total_shards}"
        else:
            output_ds_name = ds_name
            
        output_path = os.path.join(output_base_dir, output_ds_name)
        
        if os.path.exists(output_path):
            logger.info(f"이미 존재함: {output_path}, 건너뜜.")
            continue

        try:
            # 1. 게임 레벨 데이터셋 로드
            game_ds = load_from_disk(input_path)
            
            # 샤딩 적용
            if args.total_shards > 1:
                total_games = len(game_ds)
                logger.info(f"전체 게임 수: {total_games}. 샤드 {args.shard_index}/{args.total_shards} 처리 중.")
                
                # datasets 라이브러리의 shard 기능 사용 (contiguous=True로 연속된 데이터 선택)
                game_ds = game_ds.shard(num_shards=args.total_shards, index=args.shard_index, contiguous=True)
                logger.info(f"샤딩 후 게임 수: {len(game_ds)}")
                
            logger.info(f"'{ds_name}' 병렬 처리 시작 (Workers: 8)... Output: {output_ds_name}")
            
            # 2. dataset.map을 사용하여 병렬 처리
            # batched=True: 속도 향상을 위해 묶음 처리
            # num_proc=10: 10개의 워커 프로세스 사용 -> 8개로 조정됨
            # remove_columns: 기존 게임 수준의 컬럼들(Site, White 등) 제거
            pos_ds = game_ds.map(
                process_batch,
                batched=True,
                batch_size=100, # 메모리 상황에 따라 조절 가능
                num_proc=8,
                remove_columns=game_ds.column_names,
                desc=f"Processing {output_ds_name}"
            )
            
            # 3. 데이터셋 저장
            logger.info(f"포지션 데이터셋 저장 중: {output_path}")
            pos_ds.save_to_disk(output_path)
            logger.info(f"완료: {ds_name} -> {len(pos_ds)} 포지션 저장됨.")
            
        except Exception as e:
            logger.error(f"'{ds_name}' 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    main()

