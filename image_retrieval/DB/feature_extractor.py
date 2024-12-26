import torch
import torch.nn as nn
from torchvision import models, transforms
#from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
from tap import Tap
import sys
sys.path.append('/home/work/ModuInterior/myoungjin/hvt')
from model import init_model  # HVT 모델 초기화 코드
import hyptorch.pmath as pmath

# VGG16 모델 로드
def load_vgg16_model():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])  # 마지막 레이어 조정
    model.eval()
    return model

# ViT 모델 로드
def load_vit_model():
    # Pretrained ViT 모델 및 Feature Extractor 로드
    model_name = 'google/vit-base-patch16-224-in21k'
    model = ViTModel.from_pretrained(model_name)  # Transformer만 사용
    model.eval()
    return model

#HVT 모델 로드 
def load_hvt_model(cfg):
    cfg_dict = vars(cfg)  # Config 객체를 딕셔너리로 변환
    model = init_model(cfg_dict)  # 모델 초기화
    model.head = torch.nn.Identity()  # 헤드 제거
    return model

# 이미지 전처리 VGG16, HVT
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    

# 이미지 피처 추출
def extract_features(image_path, model, transform, use_model='vgg16',device='cpu'):
    img = Image.open(image_path).convert('RGB')
    
    if use_model == 'vit':
        '''
        Futurewarning 해결
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # ViT 모델에 맞게 전처리된 이미지를 입력
        inputs = processor(images=img, return_tensors="pt")
        '''
        # ViT 모델은 feature extractor에서 전처리를 내장하고 있기 때문에 별도의 transform을 하지 않음
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # ViT 모델에 맞게 전처리된 이미지를 입력
        inputs = feature_extractor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # 첫 번째 토큰이 Class Token
            
    elif use_model == 'hvt':
        
        img_tensor = transform(img).unsqueeze(0).to(device)  # 배치 차원 추가
        
        with torch.no_grad():
            features = model(img_tensor).squeeze().cpu().numpy()  # HVT는 Identity 헤드를 사용하고 있으므로 특징 벡터만 반환
        
    elif use_model == 'vgg16':
        # 모델을 device로 이동
        model = model.to(device)
        # VGG16 모델에 대한 일반적인 전처리
        img_tensor = transform(img).unsqueeze(0)  # 배치 차원 추가

        with torch.no_grad():
            features = model(img_tensor).flatten().numpy()
    else:
        raise ValueError("Unsupported model type. Please choose from 'vit', 'hvt', or 'vgg16'.")
    
    return features
