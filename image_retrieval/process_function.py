import cv2
import sqlite3
import torch
import torch.nn as nn
import glob as glob
import numpy as np
from tap import Tap
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
# from transformers import ViTModel, ViTFeatureExtractor
import sys
sys.path.append('/home/work/ModuInterior/myoungjin/hvt')
from model import init_model  # HVT 모델 초기화 코드
import hyptorch.pmath as pmath
#from simsiam.builder import SimSiam
from torchvision.models import resnet50
 
# HVT Config 클래스
class Config(Tap):
    model: str = "dino_vits16"  # 모델 이름
    ds: str = "SOP"  # 데이터셋 이름
    
# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(cropped_image, model_name='vgg16', cfg=None):
    # Load Model
    #ImageNet 데이터셋에서 훈련된 VGG16의 가중치 로드
    #Dense 4 layer까지 사용(classification용 layer 제거)
    if model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])  # 마지막 레이어 조정
        model.eval()
    
        # Transform
        transform = transforms.Compose([
            #이미 reisize (224,224)되어있어서 추가안함
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with torch.no_grad(): #grad 계산 비활성화
            img_tensor = transform(cropped_image).unsqueeze(0) #(1, 3, 224, 224)
            features = model(img_tensor).flatten().numpy() #(4096, 0)
            
    elif model_name == 'vit':
        # ViT 모델 로드
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k") #이미지를 전처리하고, 모델에 입력할 수 있도록 변환
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        model.eval()

        # Transform 및 특징 추출
        # feature_extractor: transformers 라이브러리에서 제공하는 이미지 전처리 도구
        inputs = feature_extractor(images=cropped_image, return_tensors="pt") #이미지를 텐서로 변환하고 PyTorch 텐서로 반환
        
        # 특징 추출
        with torch.no_grad():
            outputs = model(**inputs) #(1, 1(cls_token)+196(patch), 768)
            features = outputs.last_hidden_state[:, 0, :].squeeze().numpy() # CLS 토큰의 출력 사용
            #(768,)
            
    elif model_name == 'hvt':
        # 분류를 위한 head layer 제거해 특징 추출만
        cfg_dict = vars(cfg)  # Config 객체를 딕셔너리로 변환
        model = init_model(cfg_dict)  # 모델 초기화
        model.head = torch.nn.Identity()  # 헤드 제거
        
        # 이미지를 전처리하여 모델에 입력
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = transform(cropped_image).unsqueeze(0).to(device) # (1, 3, 224, 224)
        
        with torch.no_grad():
            features = model(img_tensor).squeeze().cpu().numpy()  # HVT는 Identity 헤드를 사용하고 있으므로 특징 벡터만 반환
            #(384, ) 

    else:
        raise ValueError("지원되지 않는 모델입니다. 'vgg16' 또는 'vit'을 선택하세요.")
    
    return features
        

def cosine_similarity(x, y):
    xy = np.dot(x, y)
    xx = np.dot(x, x)
    yy = np.dot(y, y)
    dist = xy / np.sqrt(xx * yy)
    return dist

# L2 유사도 계산 함수
def l2_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# 하이퍼볼릭 유사도 계산
def calculate_hyperbolic_similarity(x, y, c=1.0):
    # x, y가 numpy.ndarray일 경우, torch.Tensor로 변환
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)
    # 하이퍼볼릭 거리 계산
    distance = pmath.dist(x, y, c=c)
    similarity = 1 / (1 + distance)  # 거리 기반 유사도를 0~1 범위로 변환
    return similarity.item()


def get_db_files(query, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results


def load_feature_and_product_info_from_db(db_path, category_name=None):
    """
    category_name에 해당하는 제품 데이터 가져옴
    
    Args:
        db_path (str): SQLite 데이터베이스 경로.
        category_name (str): 조회해야할 category_name.

    Returns:
        list: features 리스트.
        list: ID 리스트.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 데이터 읽기
    if category_name:
        query = f"SELECT category_id FROM category WHERE category_name = ?"
        cursor.execute(query, (category_name,))
        category_result = cursor.fetchone()
        if category_result:
            category_id = category_result[0]
        else:
            print(f"카테고리 '{category_name}' 가 데이터베이스에 없습니다.")
            return [], []
    else:
        return [], []

    # 해당 category_id에 맞는 furniture와 feature 데이터를 조인하여 가져오기
    query = """
        SELECT f.furniture_id, f.productname, f.style, f.price, f.filename, GROUP_CONCAT(fe.feature) AS features
        FROM furniture f
        JOIN metadata m ON f.furniture_id = m.furniture_id
        JOIN feature fe ON m.feature_id = fe.feature_id
        JOIN category c ON f.category_id = c.category_id
        WHERE c.category_name = ?
        GROUP BY f.furniture_id
    """
    cursor.execute(query, (category_name,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    ids = []
    features = []
    product_info = []
    for row in rows:
        ids.append(row[0]) # furniture_id
        feature_str = row[5]  # features 컬럼에 있는 문자열로 된 벡터 추출
        feature = np.array([float(x) for x in feature_str.strip('[]').split(', ')]) # features 처리
        features.append(feature)
        
    # 제품 정보 추가
        product_info.append({
            'productname': row[1],
            'style': row[2],
            'price': row[3],
            'filename': row[4]
        })

    return np.array(features), ids, product_info


def main():
    # HVT Config 설정
    cfg = Config().parse_args()
    # 이미지를 직접 로드
    #a bed_1.png, a dresser_1.png, a sofa_1.png, a chair_6.png, a lamp_6.png, a table_6.png
    image_path = '/home/work/ModuInterior/junyeong/ImageGeneration/ig_test_img/a bed_1.png'
    
    # 카테고리 이름(클래스 이름)
    category_name = 'beds'
    
    # 사용하려는 모델 선택
    model_name = 'hvt'#'vgg16', 'vit', 'hvt'
    
    # PIL로 이미지 로드
    cropped_image = Image.open(image_path).convert('RGB')
    
    # 데이터 베이스 경로
    db_path = f"/home/work/ModuInterior/dataset/bonn-custom/houzz/hvt_db/furniture_database.sqlite3" 
    
    # DB에서 feature, product_info 가져오기
    feature_db, ids, product_info = load_feature_and_product_info_from_db(db_path, category_name)
    
    # Feature Extraction
    query_vector = extract_features(cropped_image, model_name=model_name,cfg=cfg)
    
    # 데이터 가져오기 (유사도 기준으로 가장 유사한 데이터만 사용)
    catalogs = {}
    for i, y_tensor in enumerate(feature_db):
        catalogs[ids[i]] = list(y_tensor)  # feature 데이터 추가
        
        #모델에 따라 다른 유사도 계산
        if model_name in ['vgg16', 'vit']:
            similarity_score = cosine_similarity(query_vector, y_tensor).item()
            #similarity_score = l2_distance(query_vector, y_tensor.numpy())
            catalogs[ids[i]].append(similarity_score)
        elif model_name == 'hvt':
            similarity_score = calculate_hyperbolic_similarity(query_vector, y_tensor, c=1.0)
            catalogs[ids[i]].append(similarity_score)
    
    # 유사도를 기준으로 상위 5개 항목 추출
    top_5_items = sorted(catalogs.items(), key=lambda x: x[1][-1], reverse=True)[:5]
    #top_5_items = sorted(catalogs.items(), key=lambda x: x[1][-1], reverse=False)[:5]  # L2 유사도는 거리가 작을수록 유사도가 높음

    # Print
    plt.figure(figsize=(10, 10))
    print(f"{'Category':<10}\t{'Style':<15}\t{'Product':<45}\t{'Price':<5}\t{'Similarity':<5}")
    
    for j, item in enumerate(top_5_items):
        # 이미지 경로 처리
        id = item[0]
        feature_data = item[1]
        product = product_info[ids.index(id)]  # 해당 제품 정보 가져오기
        
        print(f'{category_name:<10}\t{product["style"]:<15}\t{product["productname"]:<45}\t{int(product["price"]):<5}\t{round(feature_data[-1], 5):<5}')
        plt.subplot(1, 5, j + 1)
        product_image_path = f'/home/work/ModuInterior/dataset/bonn-custom/houzz/{category_name}/{product["style"]}/{product["filename"]}'
            
        product = cv2.imread(product_image_path, cv2.IMREAD_UNCHANGED)
        # 이미지가 정상적으로 로드되지 않으면 경로를 출력하고 오류 처리
        if product is None:
            print(f"Error: Unable to load image from {image_path}")
            continue  # 이미지가 없으면 다음 항목으로 넘어갑니다.
        
        product = cv2.cvtColor(product, cv2.COLOR_BGR2RGB)
        
        plt.imshow(product)
        plt.title(f'Rank {j + 1}\n{round(item[1][-1], 5):<5}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'/home/work/ModuInterior/myoungjin/similarity/similarity_image_test/0_final_1_beds_similar_product_{model_name}_test', bbox_inches='tight', pad_inches=0)
    plt.cla()


if __name__ == "__main__":
    main()
