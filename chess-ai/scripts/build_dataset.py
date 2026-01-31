import os
import io
import zstandard as zstd
import chess.pgn
import time
import multiprocessing
import shutil
from datasets import Dataset
from tqdm import tqdm
from functools import partial

def process_single_game(pgn_text, target_headers):
    """
    단일 PGN 문자열을 파싱하여 필터링 및 데이터 추출을 수행합니다.
    """
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return None
        
        # 필터: 시간패, 기권, 규칙위반 등을 제외
        termination = game.headers.get("Termination", "").lower()
        is_unusual = any(x in termination for x in ["time", "abandoned", "rules", "illegal", "forfeit", "infraction"])
        
        if is_unusual:
            return None
        
        # 데이터 추출
        game_data = {k: game.headers.get(k, "") for k in target_headers}
        
        # 기보(Moves) 추출 (eval 및 주석 제거)
        exporter = chess.pgn.StringExporter(columns=None, headers=False, comments=False, variations=False)
        game_data["moves"] = game.accept(exporter)
        
        return game_data
    except Exception:
        return None

def get_raw_games_iterator(text_stream):
    """
    PGN 스트림에서 게임 하나하나의 raw text를 추출하는 이터레이터입니다.
    """
    current_game = []
    for line in text_stream:
        # 새로운 게임의 시작을 알리는 [Event "..." ] 헤더 감지
        if line.startswith("[Event ") and current_game:
            yield "".join(current_game)
            current_game = [line]
        else:
            current_game.append(line)
    if current_game:
        yield "".join(current_game)

def build_dataset():
    """
    raw 데이터의 .zst 파일들을 하나씩 순차적으로 처리합니다.
    파일 내부의 경기들은 병렬로 처리하여 실시간 진행률을 표시합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # 특정 파일(lichess-2500-180.pgn.zst)만 처리하도록 수정
    target_file = "lichess-2500-180.pgn.zst"
    if target_file in os.listdir(raw_dir):
        files = [target_file]
    else:
        print(f"Error: {target_file}을 {raw_dir}에서 찾을 수 없습니다.")
        return
    
    target_headers = ["Site", "Date", "White", "Black", "Result", "ECO", "UTCDate", "Termination", "Link"]
    
    print(f"데이터셋 빌드 시작 (대상 파일: {target_file}, CPU 코어: {multiprocessing.cpu_count()})")

    for filename in files:
        input_path = os.path.join(raw_dir, filename)
        # 파일별로 개별 데이터셋 저장 경로 설정
        dataset_name = filename.replace(".pgn.zst", "_dataset")
        output_path = os.path.join(processed_dir, dataset_name)
        
        print(f"\n[{filename}] 처리 중...")
        file_size = os.path.getsize(input_path)
        start_time = time.time()
        
        # 중간 파일 (JSONL) 경로
        temp_jsonl = os.path.join(processed_dir, f"{filename}.jsonl")
        game_count = 0
        
        try:
            with open(input_path, 'rb') as f_in, open(temp_jsonl, 'w', encoding='utf-8') as f_out:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f_in) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                    raw_games = get_raw_games_iterator(text_stream)
                    
                    with multiprocessing.Pool() as pool:
                        func = partial(process_single_game, target_headers=target_headers)
                        pbar = tqdm(total=file_size, desc=f"  {filename}", unit="B", unit_scale=True, unit_divisor=1024)
                        
                        last_pos = 0
                        import json
                        for game_data in pool.imap(func, raw_games, chunksize=1000):
                            if game_data:
                                f_out.write(json.dumps(game_data) + "\n")
                                game_count += 1
                            
                            curr_pos = f_in.tell()
                            pbar.update(curr_pos - last_pos)
                            last_pos = curr_pos
                            pbar.set_postfix(games=game_count, refresh=False)
                        
                        pbar.close()
            
            if game_count > 0:
                print(f"  [{filename}] JSONL을 Dataset으로 변환 중... (이 과정은 시간이 다소 소요될 수 있습니다)")
                ds = Dataset.from_json(temp_jsonl)
                ds.save_to_disk(output_path)
                
                # 중간 파일 삭제
                if os.path.exists(temp_jsonl):
                    os.remove(temp_jsonl)
                    
                elapsed = time.time() - start_time
                print(f"  [{filename}] 완료! {game_count} 경기 저장됨 (소요 시간: {elapsed:.2f}초)")
            else:
                print(f"  [{filename}] 저장할 데이터가 없습니다.")
                if os.path.exists(temp_jsonl):
                    os.remove(temp_jsonl)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\n모든 파일 작업 완료!")

if __name__ == "__main__":
    # 필요한 라이브러리 체크
    try:
        import chess.pgn
        import datasets
        from tqdm import tqdm
    except ImportError as e:
        print(f"Error: Required library not installed. {e}")
        print("Please install dependencies: pip install python-chess datasets zstandard tqdm")
        exit(1)
        
    build_dataset()

