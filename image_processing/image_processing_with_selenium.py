# Copyright 2024 kairess

 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at

 #     http://www.apache.org/licenses/LICENSE-2.0

 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.

# https://github.com/Redcof/vit-gpt2-image-captioning
# https://github.com/huggingface

from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# 모델 및 토크나이저 로드
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def image_captioning(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

import os
import requests

def download_image(image_url, save_folder):
    try:
        response = requests.get(image_url)
        img_name = image_url.split('/')[-1]
        img_path = os.path.join(save_folder, img_name)
        
        with open(img_path, 'wb') as f:
            f.write(response.content)
        
        print(f"{img_name} 이미지가 {img_path} 경로에 저장되었습니다.")
        
        return img_path
    
    except Exception as e:
        print(f"이미지 다운로드 및 저장 중 오류 발생: {str(e)}")
        return None

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import os

# Chrome WebDriver 설정
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 웹페이지 접속
url = "https://www.wolyo.co.kr/news/articleView.html?idxno=242164"
#url= "https://papago.naver.com"
driver.get(url)

# 특정 이미지 URL 설정
target_image_url = "https://cdn.wolyo.co.kr/news/photo/202406/242164_131304_5811.jpg"
#target_image_url = "https://papago.naver.com/static/img/papago_og.png"
# 저장할 폴더 경로 설정 및 폴더 생성
save_folder = "captured_images"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 이미지 다운로드 및 저장
img_path = download_image(target_image_url, save_folder)


# 이미지 캡셔닝 함수 호출
if img_path:
    image_captions = image_captioning([img_path])
    print(f"Image Captions: {image_captions}")

# WebDriver 종료
driver.quit()

