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
# - 모델, 특징 추출기, 토크나이저를 로컬에 저장하는 코드 추가
#
# https://github.com/Redcof/vit-gpt2-image-captioning
# https://github.com/huggingface

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# 다운로드 및 로컬에 저장
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.save_pretrained("./local_model")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor.save_pretrained("./local_feature_extractor")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer.save_pretrained("./local_tokenizer")
