import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from simsiam.builder import SimSiam
from torchvision.models import resnet50
import matplotlib.pyplot as plt

# CUDA 설정
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Pre-trained SimSiam 모델 로드
pretrained_path = "/home/work/ModuInterior/yunsu/simsiam/SimSiam/pretrain_ckpt.pth.tar"
encoder = SimSiam(resnet50, dim=2048)
state_dict = torch.load(pretrained_path, map_location="cpu")["state_dict"]
encoder.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
encoder = encoder.to(device)
encoder.eval()

# 전처리 함수 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 비교할 기준 이미지 로드 및 전처리
img1_path = "/home/work/ModuInterior/junyeong/ImageGeneration/ig_test_img/a sofa_1.png"
img1 = Image.open(img1_path).convert("RGB")
img1_tensor = transform(img1).unsqueeze(0).to(device)

# 기준 폴더 경로 (tables 내부 모든 하위 폴더 포함)
root_folder_path = "/home/work/ModuInterior/dataset/bonn-custom/houzz/sofas"

# 모든 하위 디렉토리 포함 순회
similarities = []  # 유사도 저장
for root, dirs, files in os.walk(root_folder_path):  # 재귀적으로 모든 디렉토리 탐색
    for img_file in files:
        img_file_path = os.path.join(root, img_file)
        
        # 이미지 파일만 처리 (확장자 확인)
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 자기 자신은 제외
        if img_file_path == img1_path:
            continue

        try:
            # 이미지 로드 및 전처리
            img2 = Image.open(img_file_path).convert("RGB")
            img2_tensor = transform(img2).unsqueeze(0).to(device)

            # 특징 벡터 추출
            with torch.no_grad():
                z1 = encoder.encoder(img1_tensor)
                z2 = encoder.encoder(img2_tensor)

            # 코사인 유사도 계산
            cosine_similarity = torch.nn.functional.cosine_similarity(z1, z2).item()
            similarities.append((img_file_path, cosine_similarity))
        except Exception as e:
            print(f"Error processing file {img_file_path}: {e}")

# 유사도가 높은 순으로 정렬
similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

# 상위 5개 출력
top_5_similar = similarities[:5]

# 저장 경로 설정
result_path = "/home/work/ModuInterior/results/top_5_similar_images.png"
result_dir = os.path.dirname(result_path)
os.makedirs(result_dir, exist_ok=True)  # 디렉토리 생성 (존재하지 않을 경우)

# 상위 5개 이미지 시각화 및 저장
fig, axes = plt.subplots(1, 6, figsize=(20, 8))
axes[0].imshow(img1)
axes[0].set_title("Reference Image")
axes[0].axis('off')

for j, (img_path, similarity) in enumerate(top_5_similar):
    try:
        product = Image.open(img_path).convert("RGB")
        axes[j + 1].imshow(product)
        axes[j + 1].set_title(f"Rank {j + 1}\nSim: {similarity:.4f}")
        axes[j + 1].axis('off')
    except Exception as e:
        print(f"Error displaying file {img_path}: {e}")

# 결과 이미지 저장
plt.tight_layout()
plt.savefig(result_path)
print(f"Top-5 유사도 결과가 {result_path}에 저장되었습니다.")