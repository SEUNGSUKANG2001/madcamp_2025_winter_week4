import os
import io
import zstandard as zstd
import chess.pgn

def debug_headers(file_path, num_raw_lines=30, num_games=3):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Debugging: {file_path} ---")
    
    try:
        with open(file_path, 'rb') as f_in:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f_in) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                # 1. Raw 30 lines 출력
                print(f"\n[RAW DATA - First {num_raw_lines} lines]")
                print("=" * 60)
                raw_lines = []
                for _ in range(num_raw_lines):
                    line = text_stream.readline()
                    if not line:
                        break
                    print(line.strip())
                    raw_lines.append(line)
                print("=" * 60)
                
                # 2. Parsing 결과 출력
                print(f"\n[PARSED GAMES - First {num_games} games]")
                print("-" * 60)
        
        # 지정된 헤더 목록
        target_headers = ["Site", "Date", "White", "Black", "Result", "ECO", "UTCDate", "Termination", "Link"]
        
        with open(file_path, 'rb') as f_in:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f_in) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for i in range(num_games):
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break
                    
                    print(f"Game {i+1}:")
                    for header in target_headers:
                        val = game.headers.get(header, "N/A")
                        print(f"  {header}: {val}")
                    
                    # 기보(Moves) 출력 (eval 제거된 깨끗한 기보)
                    exporter = chess.pgn.StringExporter(columns=None, headers=False, comments=False)
                    moves = game.accept(exporter)
                    print(f"  Moves: {moves[:200]}..." if len(moves) > 200 else f"  Moves: {moves}")
                    print("-" * 30)
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    sample_file = os.path.join(project_root, "data", "raw", "chesscom-2400-180.pgn.zst")
    debug_headers(sample_file)
