# Copyright 2024 kairess
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright 2024 [EunseoBack]
# - 로컬에 저장된 모델, 특징 추출기, 토크나이저를 로드하는 코드 수정
# - 테스트 이미지 캡셔닝 결과 print문으로 출력하도록 코드 추가
#
# https://github.com/Redcof/vit-gpt2-image-captioning
# https://github.com/huggingface

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# 로컬에서 모델 로드
model = VisionEncoderDecoderModel.from_pretrained("./local_model")
feature_extractor = ViTImageProcessor.from_pretrained("./local_feature_extractor")
tokenizer = AutoTokenizer.from_pretrained("./local_tokenizer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
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
    print("Final Caption is", preds) # 추가된 코드 입니다
    return preds

# 테스트 이미지로 예측
predict_step(['test.png'])  # ['a woman holding up a camera to take a picture']
