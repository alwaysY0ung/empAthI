# empAthI
[emphAthI]는 사회적 약자와 공감하고 돕는 것을 목표로 시작한 대학생 팀으로, 시각장애인을 위한 웹 접근성 서비스인 Accesight(Access+sight)를 제공하는 것을 목표로 합니다.

### [서비스 목적 및 배경]
시각장애인의 정보 접근성을 보장합니다.
AI 기술을 통해 기존 '스크린 리더' 서비스를 보완하고 강화합니다.
이전에는 단순히 게시된 웹사이트를 읽어주는 것에 국한되었지만, Accesight는 HTML과 이미지 비전 분석을 통해 웹의 특정 기능과 외관을 안내하고, 키보드와 마우스를 통한 입력이 아닌 카메라와 마이크(손동작과 음성)를 통한 입력으로 받아 웹을 조작 제어할 수 있습니다.
인터넷으로 인한 정보와 접근성의 확산은 전 세계적으로 지식과 기술의 확장을 가져왔습니다.
이러한 유용성과 폭발적 잠재력을 가진 웹은 시각적 정보 중심으로 제공되어 시각장애인이 접근하기 어렵습니다.
AceSight 서비스를 통해 시각장애인도 매우 중요한 정보 접근권을 보장받을 수 있습니다.

# Files
GPT_API: GPT API를 사용하는 예제 코드(GPT_APl_usage_example.py)가 있는 폴더입니다.
STT_Whisper: Whisper 오픈소스 모델을 사용하는 예제 코드(whisper_usage_example.py)가 있는 폴더입니다.
gTTS: gTTS를 사용하는 예제 코드(gTTS_usage_example.py)가 있는 폴더입니다.

hand_detect_MediaPipe_model: MediaPipe에 8가지 손동작을 학습시켜 Accesight 서비스에 사용했습니다. 학습에 사용한 자료와 코드가 있는 폴더입니다.
 dataset: SoyunJung이 직접 촬영하여 만든 동작별 dataset이 있는 폴더입니다. 각 동작에 대한 설명은 (pose_guide.txt)에 있습니다. repository에 지정해둔 라이센스 하에 자유롭게 사용하셔도 좋습니다.
 models: 8종류 손동작을 학습 완료한 모델(mediapipe_hand_detect_model.keras)이 있는 폴더입니다.
 MediaPipe_model_usage_example.py: 학습한 MediaPipe model을 활용하는 예제 코드입니다. 각 동작을 인식하면 각 동작을 인식했음을 print하는 예제 코드입니다.
 create_dataset.py: dataset 제작에 사용한 코드입니다.
 test.py: 학습된 8가지 동작을 웹캠에서 인식하고, 화면에 표기해 모델을 테스트해볼 수 있는 코드입니다.
 train.ipynb: Mediapipe 학습에 활용한 코드입니다.

 image_captioning: 이미지 캡셔닝을 활용하는 예제 코드(caption.py)가 있는 폴더입니다.
  image_captioning_updated: 모델을 로컬에 저장하여 사용할 수 있는 예제 코드가 있는 폴더입니다.

image_processing: Selenium을 이용해 이미지를 로드하고 이미지 캡셔닝을 수행하는 예제 코드(image_processing_with_selenium.py)가 있는 폴더입니다.

voice: MeloTTS AI로 음성 생성한 코드(MeloTTS_generations.ipynb)와 생성한 음성을 모아둔 폴더입니다.
 generations: MeloTTS AI로 생성한 음성을 모아둔 폴더
 MeloTTS_generations.ipynb: MeloTTS로 음성 생성한 코드

LICENSE, README.md, requirements.txt, test.png

**AcceSight.py**:  AcceSight를 실행하고 사용해볼 수 있는 메인 코드입니다.
**openai_ssh_key.py**: AcceSight.py를 정상 실행하려면 이 코드에 OpenAI API Key를 입력해야합니다. API key는 자신의 것을 사용해도 좋습니다. 시연을 위해, **최종 보고서 가장 밑에 API Key를 기재**해두었습니다.


# 사용 및 실행 Manual

## 환경 세팅
1. '<>Code'버튼을 클릭하고, 'Download ZIP'하여 코드를 다운로드한다.
2. 압축을 푼다.
3. 새로운 가상환경(Python version: 3.12.1)을 만든다.
4. 가상환경을 activate하고, 압축을 푼 폴더의 위치로 이동한 후 pip install -r requirements.txt 한다. 필요한 패키지가 모두 설치된다.
5. 

## 코드 테스트
0. openai_ssh_key.py를 실행한다. 자신의 OpenAI API Key를 NLP_API_KEY = ""의 "" 사이에 입력한다.
11조 emphAthI의 최종 보고서 최하단에, 시연용 OpenAI API Key를 기재해 제출했습니다.
이를 입력하여 사용해주시면 됩니다.
1. API key 입력을 완료했다면 AcceSight.py를 실행한다.
단, 실행하는 환경의 성능에 따라 whipser의 model_size를 적절하게 변경함이 좋다.
실행환경의 성능이 높지 않다면
AcceSight.py 코드 본문에서 
def transcribe_audio(file_path, model_size='large', language='Korean'):을
def transcribe_audio(file_path, model_size='small', language='Korean'):로 변경함이 좋다.

## 사용하는 패키지

**Hand Detect AI_ MediaPipe**
pip install opencv-python
pip install numpy
pip install mediapipe
pip install tensorflow
(tensorflow version == 2.16.1) 2024.06.19 기준 lastest version

**gTTS**
pip install gtts
pip install playsound
pip install pygame

**Image captioning**
pip install transformers
pip install torch
pip install pillow

**GPT API**
pip install openai

**whisper**
pip install openai-whisper

**Selenium**
pip install selenium
pip install webdriver_manager

**download image**
pip install requests

**html text extract**
pip install beautifulsoup4

**record**
pip install pyaudio

pip install -r requirements.txt


## 라이센스
**Hand Detect AI_ MediaPipe**
This project uses code from [gesture-recognition](https://github.com/kairess/gesture-recognition) licensed under the Apache-2.0 License.

**gTTS**
This project uses code from [gTTS]([https://github.com/kairess/gesture-recognition](https://github.com/pndurette/gTTS/tree/main)) licensed under the MIT License.

**MeloTTS**
This project uses code from [MeloTTS](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md) licensed under the MIT License.

**Image captioning**
This project uses code from [vit-gpt2-image-captioning](https://github.com/Redcof/vit-gpt2-image-captioning)licensed under the Apache-2.0 License.



## 환경 및 버전
python = 3.12.1
