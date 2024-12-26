from PIL import Image
import numpy as np
import sqlite3

import torch
from torchvision import models, transforms
from diffusers import StableDiffusionGLIGENPipeline
from compel import Compel

import os
from tap import Tap
import sys
sys.path.append('/home/work/ModuInterior/hyejeong/demo/hvt')
from Image_retrieval.hvt.model import init_model  # HVT 모델 초기화 코드
import hyptorch.pmath as pmath




# HVT Config 클래스
class Config(Tap):
    model: str = "dino_vits16"  # 모델 이름
    ds: str = "SOP"  # 데이터셋 이름
# 초기 상태 저장
init_flag = False
model = None
cfg = Config().parse_args()


def get_initializer():
    global init_flag, model
    if not init_flag:
        model = Initializer()  # 싱글톤 인스턴스 생성
        init_flag = True
    return model


class Initializer(object):
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.set_device()
            self.set_ig_model()
            self.set_is_model()
            self.set_variable()
            print("Set Device and Model")

    def reset(self):
        self.set_variable()

    def set_variable(self):
        self.bg_image = None
        self.ig_image = None
        self.total_product_list = None

    def set_device(self):
        if torch.backends.mps.is_available():
            self._device_type = torch.device("mps")
        # elif torch.cuda.is_available():
        #     self._device_type = torch.device("cuda")
        #     print(f"Using CUDA device: {self._device_type}")
        else:
            self._device_type = torch.device("cpu")
        print("Device using:: ", self._device_type)


    def get_device(self):
        return self._device_type


    # Initialize Pre-trained Generation Model
    def set_ig_model(self):
        self._gligen_ig = StableDiffusionGLIGENPipeline.from_pretrained(
            "masterful/gligen-1-4-inpainting-text-box", 
            variant="fp16", 
            torch_dtype=torch.float16
        ).to("cuda:1")
        self._compel = Compel(
            tokenizer=self._gligen_ig.tokenizer, 
            text_encoder=self._gligen_ig.text_encoder
        )


    def get_ig_model(self):
        return self._gligen_ig, self._compel


    # Initialize Pre-trained Feature Extractor Model
    def set_is_model(self):
        cfg_dict = vars(cfg)
        self._hvt_is = init_model(cfg_dict)
        self._hvt_is.head = torch.nn.Identity()

    def get_is_model(self):
        return self._hvt_is



    # ───── 이미지 생성 ! ─────
    # 1. 인테리어 가구 생성에 사용할 배경 이미지 불러와서 초기화
    # 2. 이미지 생성에 필요한 bbox, class name 정보 저장
    # → width (350), height (350 * aspect_ratio) 상태로 정보가 넘어옴
    # → 따라서 원본 이미지를 너비가 350으로 scaling한 이미지에서 normalization
    # 3. 생성 모델로 이미지 생성
    def set_bg_image(self, image_dir):
        def _target_center_crop_and_resize(image):
            width, height = image.size
            crop_size = min(width, height)
            x1, y1 = (width - crop_size) // 2, (height - crop_size) // 2
            x2, y2 = (width + crop_size) // 2, (height + crop_size) // 2
            image = image.crop((x1, y1, x2, y2))
            image = image.resize((512, 512), Image.LANCZOS)
            return image
        
        bg_image = Image.open(image_dir).convert('RGB')
        self.bg_image = _target_center_crop_and_resize(bg_image)
        self.resize = 512


    def get_bg_image(self, resize=None):
        if self.bg_image is None: 
            return None
        bg_image = self.bg_image
        if resize is not None:
            bg_image = bg_image.resize((resize, resize), Image.LANCZOS)
            self.resize=resize
        return bg_image


    def set_bbox_cls_info(self, info):
        canvas_to_class = {
            "#FF0000": "a bed",
            "#FF7F00": "a chair",
            "#FFFF00": "a dresser",
            "#00FF00": "a lamp",
            "#0000FF": "a sofa",
            "#9400D3": "a table"
        }
        
        self.boxes = []
        self.phrases = []
        for left, top, width, height, stroke in zip(
            info['left'], info['top'], info['width'], info['height'], info['stroke']
        ):
            x1, y1 = left, top
            x2, y2 = x1 + width, y1 + height
            self.boxes.append([
                x1/self.resize, 
                y1/self.resize, 
                x2/self.resize, 
                y2/self.resize
            ])
            self.phrases.append(canvas_to_class[stroke])
    

    def get_bbox_cls_info(self):
        return self.boxes, self.phrases
    

    def write_prompt_style(self, style):
        if   style == "Contemporary":
            return "Contemporary furniture design, Clean lines, Sleek surfaces, Neutral tones, Bold color accents, Minimalistic elegance, Subtle textures, Modern comfort"
        elif style == "Industry":
            return "Industrial furniture design, Exposed metal, Industrial metal frame, Raw wood, Visible framework, Urban aesthetics, Neutral tones, Functional and raw charm"
        elif style == "Mid-century":
            return "Mid-century modern furniture design, Warm wood tones, Smooth contours, Rounded edges, Retro aesthetics, Organic shapes, Functional design, Minimalistic flair"
        elif style == "Modern":
            return "Modern minimalist furniture design, Sleek lines, Clear structure, Achromatic tones, Contemporary aesthetics, Functional elegance, Subtle reflection"
        elif style == "Rustic":
            return "Rustic farmhouse furniture design, Natural wood grain, Unpolished texture, Vintage charm, Cozy aesthetics, Earthy tones"
        elif style == "Scandinavian":
            return "Scandinavian furniture design, Natural wood, Bright and airy, Muted colors, Soft pastels, Simple line, Minimalist aesthetics, Functional simplicity"


    def generation_run(self, style, step=50):
        if self.total_product_list is not None: # 추천 제품 리스트 초기화
            self.total_product_list = None  

        prompt_default   = "Best quality, Ultra high quality, Highly detailed, Intricate details, Photo-realistic, Cinematic lighting, Hyper-detailed textures, Well-balanced layout, Spatial harmony, 8k resolution"
        prompt_scene_fit = "Harmony with ambient lightning, Blending seamlessly with surroundings, Accurate perspective, Scale with the scene"
        prompt_style     = self.write_prompt_style(style)
        prompt           = f'("{prompt_default}", "{prompt_scene_fit}", "({prompt_style})+++").and()'

        positive = self._compel.build_conditioning_tensor(prompt)
        negative = self._compel.build_conditioning_tensor("")
        [positive, negative] = self._compel.pad_conditioning_tensors_to_same_length([positive, negative])

        self.ig_image = self._gligen_ig(
            prompt_embeds          = positive,
            negative_prompt_embeds = negative,
            gligen_inpaint_image   = self.bg_image.copy(),
            gligen_phrases         = self.phrases,
            gligen_boxes           = self.boxes,
            gligen_scheduled_sampling_beta = 1.0,
            guidance_scale                 = 8.0,
            output_type         = "pil",
            num_inference_steps = step,
        ).images[0]
        self.set_ig_crop_image()


    def get_ig_image(self, resize=None):
        if self.ig_image is None: 
            return None
        ig_image = self.ig_image
        if resize is not None:
            ig_image = ig_image.resize((resize, resize), Image.LANCZOS)
            self.resize=resize
        return ig_image

    
    
    # ───── 이미지 검색 ! ─────
    # 1. 저장된 bbox 정보를 바탕으로 이미지 crop and resize
    # 2. 외부로부터 검색하길 원하는 제품 이미지만 가져와 유사 제품 탐색
    def set_ig_crop_image(self):
        self.cropped_images = []
        for box in self.boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1 * 512)
            y1 = int(y1 * 512)
            x2 = int(x2 * 512)
            y2 = int(y2 * 512)
            cropped = self.ig_image.copy().crop((x1, y1, x2, y2))
            w, h = cropped.size
            resized = cropped.resize((w, h), Image.LANCZOS)
            self.cropped_images.append(resized)


    def get_ig_crop_image(self):
        return self.cropped_images


    def get_product_info_from_db(self, db_name):
        db_dir = f'/home/work/ModuInterior/dataset/bonn-custom/houzz/hvt_db/{db_name}'
        conn = sqlite3.connect(db_dir)
        cursor = conn.cursor()
        query = "SELECT * FROM products"
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows


    def extract_features(self, cropped_image, cfg=cfg):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(cropped_image).unsqueeze(0).to(self._device_type)
        with torch.no_grad():
            feature = self._hvt_is(img_tensor).squeeze().numpy()
        return feature


    def cosine_similarity(self, x, y):
        xy = np.dot(x, y)
        xx = np.dot(x, x)
        yy = np.dot(y, y)
        dist = xy / np.sqrt(xx * yy)
        return dist


    def calculate_hyperbolic_similarity(self, x, y, c=1.0):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y)
        distance = pmath.dist(x, y, c=c)
        similarity = 1 / (1 + distance)
        return similarity.item()
    

    def search_run(self, idx, top_k=3):        
        cropped_image  = self.cropped_images[idx]
        output_feature = self.extract_features(cropped_image)
        class_name = self.phrases[idx]
        db_name    = class_name[2:] + 's_db.sqlite3'
        samples    = self.get_product_info_from_db(db_name)

        catalogs = dict()
        for sample in samples:
            catalogs[sample[0]] = list(sample[1:-1])
            catalogs[sample[0]].append(class_name[2:] + 's')
            target_tensor = np.array([float(x) for x in sample[-1].strip('[]').split(', ')])            
            similarity_score= self.calculate_hyperbolic_similarity(output_feature, target_tensor, c=1.0)
            catalogs[sample[0]].append(similarity_score)
        top_k_items = sorted(catalogs.items(), key=lambda x: x[1][-1], reverse=True)[:top_k]
        return top_k_items

