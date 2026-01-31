import os
import io
try:
    import zstandard as zstd
except ImportError:
    print("Error: 'zstandard' library is not installed.")
    print("Please install it using: pip install zstandard")
    exit(1)

def read_first_few_lines(file_path, num_lines=100):
    """
    .zst 압축 파일을 풀지 않고 스트리밍 방식으로 읽어서 상위 N줄을 출력합니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Reading from: {file_path}\n" + "-"*40)
    
    try:
        with open(file_path, 'rb') as f:
            # Zstd 압축 해제기 생성
            dctx = zstd.ZstdDecompressor()
            
            # 스트림 리더를 통해 압축 해제된 데이터를 읽음
            with dctx.stream_reader(f) as reader:
                # 텍스트 스트림으로 래핑 (UTF-8 인코딩)
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                for i, line in enumerate(text_stream):
                    if i >= num_lines:
                        break
                    print(f"[{i+1}] {line.strip()}")
                    
    except Exception as e:
        print(f"Error during reading: {e}")

if __name__ == "__main__":
    # 프로젝트 루트 기준 샘플 파일 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    # 예시로 lichess-2400-eval.pgn.zst 파일을 읽어봅니다.
    sample_file = os.path.join(project_root, "data", "raw", "lichess-2400-eval.pgn.zst")
    
    read_first_few_lines(sample_file, num_lines=100)
