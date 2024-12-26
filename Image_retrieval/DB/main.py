from process_dataset import process_dataset
from tap import Tap

# HVT Config 클래스
class Config(Tap):
    model: str = "dino_vits16"  # 모델 이름
    ds: str = "SOP"  # 데이터셋 이름
    
if __name__ == "__main__":
    # 데이터셋 경로와 데이터베이스 경로 설정
    dataset_path = "/home/work/ModuInterior/dataset/bonn-custom/houzz"
    db_path = "/home/work/ModuInterior/dataset/bonn-custom/houzz/hvt_db/furniture_database.sqlite3"
    
    # HVT Config 설정
    cfg = Config().parse_args()
    
    # 데이터 처리 시작
    process_dataset(dataset_path, db_path, cfg)
    
