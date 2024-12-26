#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import ImageFilter
import random

class TwoCropsTransform:
    # 같은 이미지로 다른 2개의 무작위 변환 생성
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        # query : 첫 번째 무작위 변환
        q = self.base_transform(x)
        # key : 두 번째 무작위 변환
        k = self.base_transform(x)
        return [q, k]
    

class GaussianBlur(object):
    # 이미지 블러 처리
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x): # 기본값 0.1 ~ 2.0 사이 랜덤하게 선택
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x