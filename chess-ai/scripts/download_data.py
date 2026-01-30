import kagglehub
import os
import shutil

def download_dataset():
    """ Kaggle에서 체스 데이터셋을 다운로드하고 프로젝트의 data/raw 폴더로 이동합니다. """
    
    # 프로젝트 루트 및 대상 디렉토리 경로 정의
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    target_dir = os.path.join(project_root, "data", "raw")
    
    # 대상 디렉토리가 없으면 생성
    os.makedirs(target_dir, exist_ok=True)
    
    print("Kaggle에서 chessmont-big-dataset 다운로드를 시작합니다...")
    # Kagglehub 캐시 폴더로 최신 버전 다운로드
    cache_path = kagglehub.dataset_download("chessmontdb/chessmont-big-dataset")
    print(f"캐시 위치에 다운로드됨: {cache_path}")
    
    print(f"파일을 프로젝트 폴더로 이동 중: {target_dir}")
    
    # 캐시의 파일들을 확인하며 대상 폴더로 복사 (이미 존재하는 경우 건너뜀)
    for item in os.listdir(cache_path):
        source_item = os.path.join(cache_path, item)
        dest_item = os.path.join(target_dir, item)
        
        # 이미 파일이나 폴더가 존재하는지 확인
        if os.path.exists(dest_item):
            print(f"가 존재함: {item} - 건너뜁니다.")
            continue
            
        # 파일/폴더 타입에 맞춰 복사 수행
        if os.path.isdir(source_item):
            shutil.copytree(source_item, dest_item)
            print(f"폴더 복사 완료: {item}")
        else:
            shutil.copy2(source_item, dest_item)
            print(f"파일 복사 완료: {item}")
            
    print(f"데이터셋 준비가 완료되었습니다: {target_dir}")
    return target_dir

if __name__ == "__main__":
    download_dataset()
