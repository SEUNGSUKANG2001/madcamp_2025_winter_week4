import os
import io
import zstandard as zstd
import chess.pgn
import time
from datasets import Dataset

def process_single_zst(input_path, output_path):
    """
    하나의 .zst 파일을 처리하여 지정된 경로에 Dataset으로 저장합니다.
    """
    target_headers = ["Site", "Date", "White", "Black", "Result", "ECO", "UTCDate", "Termination", "Link"]
    games = []
    
    print(f"[{os.path.basename(input_path)}] 처리 시작...")
    start_time = time.time()
    
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
                    
                    if count_saved >= 1000:
                        print(f"  [목표 달성] {count_saved} 경기 도달. 파싱을 중단합니다.")
                        break
                    
                    if count_saved % 500 == 0:
                        print(f"  {count_total} 경기 읽음, {count_saved} 경기 저장 중...")
        
        if games:
            ds = Dataset.from_list(games)
            ds.save_to_disk(output_path)
            
            end_time = time.time()
            print(f"\n--- 처리 결과 ---")
            print(f"입력 파일: {input_path}")
            print(f"출력 경로: {output_path}")
            print(f"총 {count_total} 경기 중 {count_saved} 경기 저장 완료")
            print(f"소요 시간: {end_time - start_time:.2f}초")
            print(f"-----------------\n")
        else:
            print("저장할 경기가 없습니다.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    # 설정 (필요에 따라 수정하세요)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    # 기본 입력 파일: data/raw/twic.pgn.zst (복사하지 않고 직접 사용)
    raw_dir = os.path.join(project_root, "data", "raw")
    input_zst = os.path.join(raw_dir, "twic.pgn.zst")
    
    if not os.path.exists(input_zst):
        print(f"Error: {input_zst} 파일이 없습니다.")
        sys.exit(1)
    
    # 출력 경로: data2 폴더 내부
    output_ds = os.path.join(project_root, "data2", "test_twic_dataset")
    
    print(f"테스트 실행 준비 완료.")
    print(f"입력: {input_zst}")
    print(f"출력: {output_ds}")
    
    response = input("계속하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        process_single_zst(input_zst, output_ds)
    else:
        print("취소되었습니다.")
