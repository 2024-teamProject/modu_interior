import os
import torch
import json
import sqlite3
import numpy as np
from tqdm import tqdm
from feature_extractor import load_vgg16_model, load_vit_model, load_hvt_model, get_transform, extract_features
from db_manager import DBManager

def process_dataset(dataset_path, db_path, cfg):
    """
    데이터셋을 처리하고 각 클래스별로 별도의 데이터베이스에 저장합니다.
    VGG16(or ViT or HViT)으로 추출한 피처를 데이터베이스에 저장합니다.
    :param dataset_path: 데이터셋 디렉토리 경로
    :param db_path: 하나의 데이터베이스 파일 경로
    :param cfg: HVT 모델 설정을 위한 Config 객체
    """ 
    # device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = load_vit_model()  # ViT 모델 로드
    #model = load_vgg16_model()  # VGG16 모델 로드
    model = load_hvt_model(cfg) # HVT 모델 로드
    model = model.to(device) # 모델을 해당 장치로 이동
    
    transform = get_transform()  # 이미지 전처리 로드 VGG16에서만 사용
    categories = ['beds', 'chairs', 'dressers', 'lamps', 'sofas', 'tables']

    db_manager = DBManager(db_path=db_path)
        
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            print(f"폴더를 찾을 수 없습니다: {category_path}")
            continue
        
        # 카테고리 추가 (category 테이블에 카테고리 추가)
        db_manager.insert_category(category)

        styles = os.listdir(category_path)
        for style in styles:
            style_path = os.path.join(category_path, style)
            if not os.path.isdir(style_path):
                continue

            # 디렉토리 내 이미지 처리
            image_data = []

            # 스타일별 이미지 파일 리스트 생성
            image_files = [img_name for img_name in os.listdir(style_path) if img_name.endswith(('.jpg', '.jpeg', '.png'))]
            
            # tqdm을 사용해 진행 상황 표시
            #tqdm(image_files, desc=f"Processing images in style '{style}'", unit="image")
            for img_name in image_files:   
                img_path = os.path.join(style_path, img_name)
                try:
                    # VGG16 or ViT or HVT 으로 피처 추출
                    feature = extract_features(img_path, model, transform, use_model='hvt',device=device)

                    # 예제 데이터로 제품명과 가격 설정
                    productname = os.path.splitext(img_name)[0]  # 파일 이름에서 제품명 추출
                    price = np.random.uniform(100, 1000)  # 랜덤 가격 설정
                    
                    # category_id 추출
                    category_id = db_manager.get_category_id(category)
                    
                    # 가구 데이터 삽입 (furniture 테이블)
                    db_manager.insert_furniture(productname, category_id, style, price, img_name)
                    
                    # 가구 ID 추출
                    furniture_id = db_manager.get_furniture_id(img_name)
                    
                    # feature 삽입 (feature 테이블)
                    feature_str = json.dumps(feature.tolist())  # 벡터를 문자열로 변환
                    db_manager.insert_feature(furniture_id, feature_str)

                    # 메타데이터 삽입 (metadata 테이블)
                    feature_id = db_manager.get_feature_id(furniture_id)  # feature_id 조회
                    db_manager.insert_metadata(furniture_id, feature_id)

                    
                    # image_data 리스트에 데이터 추가
                    image_data.append({
                        "productname": productname,
                        "filename": img_name,
                        "style": style,
                        "price": price,
                        "features": feature,
                    })

                #except Exception as e:
                #    print(f"이미지 처리 오류: {img_path}, {e}")
                finally:
                    print(' ')
        print(f"db에 저장 완료: {db_path}")   

