[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2F2024-teamProject%2Fmodu_interior&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub downloads](https://img.shields.io/github/downloads/2024-teamProject/modu_interior/total.svg?logo=github)](https://github.com/2024-teamProject/modu_interior/releases)

# 모두의 인테리어 (Modu Interior)
- 기간 : 2024년 11월 25일 ~ 2024년 12월 27일 (약 5주)
- 참여 인원 : 노혜정, 박윤수, 손명진, 차준영 (팀장)


Sample video

https://github.com/user-attachments/assets/0d7d3603-50b8-4dae-84b1-78e43aca6aa1



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



## 환경



## 파일 설명

```
.
├── Image_generation
│   ├── README.md
│   ├── bg-for-hp-test.jpg
│   ├── bg-for-prompt-test.jpeg
│   ├── gligen-hp-test.ipynb
│   ├── gligen-prompt-test-1.ipynb
│   └── gligen-prompt-test-2.ipynb
├── Image_retrieval
│   ├── DB
│   │   ├── database_check.ipynb
│   │   ├── db_manager.py
│   │   ├── feature_extractor.py
│   │   ├── main.py
│   │   └── process_dataset.py
│   ├── hvt
│   │   ├── hyptorch
│   │   ├── poincare
│   │   ├── model.py
│   │   └── README.md
├── SimSiam
│   └── process_function.py
├── img
├── initializer.py
└── main.py

```
