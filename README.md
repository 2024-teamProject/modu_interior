[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2F2024-teamProject%2Fmodu_interior&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub downloads](https://img.shields.io/github/downloads/2024-teamProject/modu_interior/total.svg?logo=github)](https://github.com/2024-teamProject/modu_interior/releases)

<h2 align="center">🎖️Tech Stack🎖️</h2>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C?style=flat-square&logo=PyTorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/HuggingFace-%23FFC107?style=flat-square&logo=HuggingFace&logoColor=black" alt="Hugging Face" />
    <img src="https://img.shields.io/badge/StableDiffusion-%2346ABF2?style=flat-square&logoColor=white" alt="Stable Diffusion" /> 
    <img src="https://img.shields.io/badge/Transformer-%230072C6?style=flat-square&logoColor=white" alt="Transformer" /> 
    <img src="https://img.shields.io/badge/SQLite-%23003B57?style=flat-square&logo=SQLite&logoColor=white" alt="SQLite" />
    <img src="https://img.shields.io/badge/Streamlit-%23FF4B4B?style=flat-square&logo=Streamlit&logoColor=white" alt="Streamlit" />
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/16414f34-5092-4993-8d3e-f96a5efe877d" width="500">
</p>


# 모두의 인테리어 (Modu Interior)
- 기간 : 2024년 11월 25일 ~ 2024년 12월 27일 (약 5주)
<table>
  <tbody>
    <tr>
      <td align="center">
        <a href="https://github.com/팀원1깃허브주소">
          <img src="https://via.placeholder.com/100" width="100px;" alt=""/>
          <br /><sub><b>팀원1 이름</b></sub>
        </a>
        <br />
      </td>
      <td align="center">
        <a href="https://github.com/팀원2깃허브주소">
          <img src="https://via.placeholder.com/100" width="100px;" alt=""/>
          <br /><sub><b>팀원2 이름</b></sub>
        </a>
        <br />
      </td>
      <td align="center">
        <a href="https://github.com/팀원3깃허브주소">
          <img src="https://via.placeholder.com/100" width="100px;" alt=""/>
          <br /><sub><b>팀원3 이름</b></sub>
        </a>
        <br />
      </td>
      <td align="center">
        <a href="https://github.com/팀원4깃허브주소">
          <img src="https://via.placeholder.com/100" width="100px;" alt=""/>
          <br /><sub><b>팀원4 이름</b></sub>
        </a>
        <br />
      </td>
    </tr>
  </tbody>
</table>

# Shortest

생성형 모델 기반 가구 생성 및 추천 시스템

https://github.com/user-attachments/assets/210aab65-5609-4bed-9512-6c07b6a90d5e


## 프로젝트 개요
**생성형 모델 기반 가구 생성 및 추천 시스템**

이 프로젝트는 소비자의 방 분위기에 적합한 가구를 생성하고 원하는 스타일의 가구를 추천하는 서비스를 제공하는 것을 목적으로 시작했습니다.

<img width="400" src="https://github.com/2024-teamProject/modu_interior/blob/main/img/modu_interior_flowchart.png"/>

사용자가 실제 자신의 방과 간단한 가구 배치, 원하는 스타일을 입력하면, 해당 스타일의 가구가 방에 배치된 모습을 시뮬레이션으로 보여준 후, 생성된 가구와 유사한 실제 제품을 추천해주는 기능을 제공하고자 합니다. 이 서비스를 통해 소비자들은 맞춤형 가구를 찾는 데 걸리는 시간을 절약하고, 구매 실패로 인한 시간적·경제적 손실을 방지할 수 있습니다. 가구 브랜드 및 플랫폼 입장에서는 소비자의 신뢰도와 만족도를 향상시켜 매출 증가를 기대할 수 있으며, 동시에 소비자에게 인테리어 참여 경험을 제공함으로써 브랜드 마케팅에도 활용할 수 있습니다.


## 주요 기능
#### 1. 인테리어 가구 생성 (Image Generation)
<img width="600" src="https://github.com/2024-teamProject/modu_interior/blob/main/img/GLIGEN%20Input-Output.png"/>

사용자로부터 방 이미지, Class 정보가 포함된 Bounding box, Style Keyword를 입력 받아 GLIGEN 모델을 활용해 방 분위기와 스타일 키워드에 알맞는 가구 이미지를 생성함
- Input
  - Grounded Image : 빈 방의 배경 이미지
  - Condition 1. Layout : Bounding Box, 객체가 위치할 공간
  - Condition 2. Phrase : Class Name, 가구 객체의 이름
  - Condition 3. Prompt : Text Prompt, 스타일 키워드 프롬프트
- Output : 빈 방의 Bounding Box 위치에 스타일 키워드 프롬프트가 적용된 가구 이미지가 생성됨
  
#### 2. 유사한 제품 추천 (Image Retrieval)
<img width="600" src="https://github.com/2024-teamProject/modu_interior/blob/main/img/HViT%20Input-Output.png"/>

판매 가능한 제품 정보와 HViT 모델로 추출된 특징맵을 데이터 베이스에 저장. 이후 HViT 모델로 생성된 가구의 특징맵을 추출하고, DB에 저장된 특징맵과 Hyperbolic Distance를 계산하여 유사한 제품 추천
- Input
  - GLIGEN 기반으로 Generated Image
  - Product DB에 저장된 Image
- Output : 생성된 이미지와 가장 유사한 k개의 제품 정보



## Project Environment

- OS: Ubuntu 22.04 (Linux)
- CPU: Intel Xeon Gold 5218 @ 2.30GHz (4 Cores, x86_64)
- GPU: Tesla V100
- Python: 3.12.8

### Dependencies

- torch==2.5.1, torchvision==0.20.1, torchaudio==2.5.1
- numpy==2.2.1, scipy==1.14.1, scikit-learn==1.6.0
- opencv-python==4.10.0.84, matplotlib==3.10.0
- timm==1.0.12, pytorch-metric-learning==2.8.1
- 기타 라이브러리는 requirements.txt에 포함

Requirements 설치:

```bash

pip install -r requirements.txt

```

## Project Structure

```
.
├── Image_generation                  # 이미지 생성 모듈
│   └── gligen-hp-test.ipynb          # GLIGEN 가구이미지 생성 테스트 파일
├── Image_retrieval                   # 이미지 검색 모듈
│   ├── DB                            # DB
│   │   ├── database_check.ipynb      # Feature Shape 확인 파일
│   │   ├── db_manager.py             # DB 생성 및 관리 파일
│   │   ├── feature_extractor.py      # 이미지 특징 추출기
│   │   ├── main.py                   # DB 모듈 실행 파일
│   │   └── process_dataset.py        # 이미지 데이터 처리 파일
│   ├── hvt                           # Hyperbolic Vision Transformers
│   │   ├── hyptorch                  # HViT
│   │   ├── poincare                  # Poincaré 시각화 파일
│   │   ├── model.py                  # HViT model
│   ├── SimSiam                       # Simple Siamese
│   │   ├── simsiam                   # SimSiam
│   │   ├── main_lincls.py            # SimSiam Pipeline 파일
│   │   ├── main_simsiam.py           # SimSiam 실행 파일
│   │   └── similarity.py             # Simsiam 유사도계산 파일
│   └── process_function.py           # 이미지 검색 및 추천 시스템 파일
├── initializer.py                    # 가구 이미지 생성 및 DB Feature 검색 파일
├── main.py                           # 가구 이미지 생성 및 추천 시스템 Streamlit 파일
└── requirements.txt                  # requirements
```
