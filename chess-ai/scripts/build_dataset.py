import os
import io
import zstandard as zstd
import chess.pgn
import time
import multiprocessing
import shutil
from datasets import Dataset, concatenate_datasets, load_from_disk

def process_single_file(args):
    """
    하나의 .pgn.zst 파일을 처리하여 임시 Dataset으로 저장합니다.
    """
    filename, raw_dir, temp_dir, target_headers = args
    input_path = os.path.join(raw_dir, filename)
    temp_output_path = os.path.join(temp_dir, filename.replace(".pgn.zst", ".tmp_ds"))
    
    print(f"[{filename}] 처리 시작...")
    
    games = []
    count_total = 0
    count_saved = 0
    
    try:
        with open(input_path, 'rb') as f_in:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f_in) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                while True:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break
                    
                    count_total += 1
                    
                    # 필터: 시간패, 기권, 규칙위반 등을 제외
                    termination = game.headers.get("Termination", "").lower()
                    is_unusual = any(x in termination for x in ["time", "abandoned", "rules", "illegal", "forfeit", "infraction"])
                    
                    if is_unusual:
                        continue
                    
                    # 데이터 추출
                    game_data = {k: game.headers.get(k, "") for k in target_headers}
                    
                    # 기보(Moves) 추출 (eval 및 주석 제거)
                    exporter = chess.pgn.StringExporter(columns=None, headers=False, comments=False, variations=False)
                    game_data["moves"] = game.accept(exporter)
                    
                    games.append(game_data)
                    count_saved += 1
                    
                    if count_saved % 10000 == 0:
                        print(f"  [{filename}] {count_total} 경기 읽음, {count_saved} 경기 저장 중...")
        
        if games:
            ds = Dataset.from_list(games)
            ds.save_to_disk(temp_output_path)
            print(f"[{filename}] 완료! ({count_saved} 경기 저장됨)")
            return temp_output_path
        else:
            print(f"[{filename}] 저장할 경기가 없습니다.")
            return None
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def build_dataset():
    """
    raw 데이터의 .zst 파일들을 병렬로 처리하여 datasets 형식으로 저장합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    dataset_output_path = os.path.join(processed_dir, "chess_dataset")
    temp_dir = os.path.join(processed_dir, "temp_chunks")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    files = [f for f in os.listdir(raw_dir) if f.endswith(".pgn.zst")]
    target_headers = ["Site", "Date", "White", "Black", "Result", "ECO", "UTCDate", "Termination", "Link"]
    
    print(f"병렬 처리 시작 (파일 수: {len(files)}, CPU 코어: {multiprocessing.cpu_count()})")
    start_time = time.time()
    
    # 병렬 작업 준비
    pool_args = [(f, raw_dir, temp_dir, target_headers) for f in files]
    
    with multiprocessing.Pool() as pool:
        temp_ds_paths = pool.map(process_single_file, pool_args)
    
    # 유효한 경로만 필터링
    valid_paths = [p for p in temp_ds_paths if p is not None]
    
    if not valid_paths:
        print("에러: 저장된 데이터가 없습니다.")
        return

    print("\n데이터셋 병합 중...")
    datasets_to_combine = [load_from_disk(p) for p in valid_paths]
    final_dataset = concatenate_datasets(datasets_to_combine)
    
    print(f"최종 결과를 {dataset_output_path}에 저장 중...")
    final_dataset.save_to_disk(dataset_output_path)
    
    # 임시 디렉토리 삭제
    print("임시 파일 정리 중...")
    shutil.rmtree(temp_dir)
    
    end_time = time.time()
    print(f"\n모든 작업 완료!")
    print(f"소요 시간: {end_time - start_time:.2f}초")
    print(f"총 {len(final_dataset)} 개의 경기가 저장되었습니다.")

if __name__ == "__main__":
    # 라이브러리 체크
    try:
        import chess.pgn
        import datasets
    except ImportError as e:
        print(f"Error: Required library not installed. {e}")
        print("Please install dependencies: pip install python-chess datasets zstandard")
        exit(1)
        
    build_dataset()
