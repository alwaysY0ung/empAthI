# empAthI
[emphAthI]는 사회적 약자와 공감하고 돕는 것을 목표로 시작한 대학생 팀으로, 시각장애인을 위한 웹 접근성 서비스인 Accesight(Access+sight)를 제공하는 것을 목표로 합니다.

### [서비스 목적 및 배경]
시각장애인의 정보 접근성을 보장합니다.

AI 기술을 통해 기존 '스크린 리더' 서비스를 보완하고 강화합니다.

단순 스크린 리더 기술은 단순히 게시된 웹사이트를 읽어주는 것에 국한되어있지만, Accesight는 HTML과 이미지 비전 분석을 통해 웹의 특정 기능과 외관을 안내하고, 키보드와 마우스를 통한 입력이 아닌 카메라와 마이크(손동작과 음성)를 통한 입력으로 받아 웹을 조작 제어할 수 있습니다.

인터넷으로 인한 정보와 접근성의 확산은 전 세계적으로 지식과 기술의 확장을 가져온 중요한 요소입니다.

이러한 유용성과 잠재력을 가진 웹은 시각적 정보 중심으로 제공되어 시각장애인이 접근하기 어렵습니다.

AceSight 서비스를 통해 시각장애인도 웹 정보 접근권을 보장받을 수 있습니다.

# Files
GPT_API: GPT API를 사용하는 예제 코드(GPT_APl_usage_example.py)가 있는 폴더입니다.

STT_Whisper: Whisper 오픈소스 모델을 사용하는 예제 코드(whisper_usage_example.py)가 있는 폴더입니다.

gTTS: gTTS를 사용하는 예제 코드(gTTS_usage_example.py)가 있는 폴더입니다.

hand_detect_MediaPipe_model: MediaPipe에 8가지 손동작을 학습시켜 Accesight 서비스에 사용했습니다. 학습에 사용한 자료와 코드가 있는 폴더입니다.
* dataset: SoyunJung이 직접 촬영하여 만든 동작별 dataset이 있는 폴더입니다. 각 동작에 대한 설명은 (pose_guide.txt)에 있습니다. repository에 지정해둔 라이센스 하에 자유롭게 사용하셔도 좋습니다.
* models: 8종류 손동작을 학습 완료한 모델(mediapipe_hand_detect_model.keras)이 있는 폴더입니다.
* MediaPipe_model_usage_example.py: 학습한 MediaPipe model을 활용하는 예제 코드입니다. 각 동작을 인식하면 각 동작을 인식했음을 print하는 예제 코드입니다.
* create_dataset.py: dataset 제작에 사용한 코드입니다.
* test.py: 학습된 8가지 동작을 웹캠에서 인식하고, 화면에 표기해 모델을 테스트해볼 수 있는 코드입니다.
* train.ipynb: Mediapipe 학습에 활용한 코드입니다.

image_captioning: 이미지 캡셔닝을 활용하는 예제 코드(caption.py)가 있는 폴더입니다.
 * image_captioning_updated: 모델을 로컬에 저장하여 사용할 수 있는 예제 코드가 있는 폴더입니다.

image_processing: Selenium을 이용해 이미지를 로드하고 이미지 캡셔닝을 수행하는 예제 코드(image_processing_with_selenium.py)가 있는 폴더입니다.

voice: MeloTTS AI로 음성 생성한 코드(MeloTTS_generations.ipynb)와 생성한 음성을 모아둔 폴더입니다.
* generations: MeloTTS AI로 생성한 음성을 모아둔 폴더
* MeloTTS_generations.ipynb: MeloTTS로 음성 생성한 코드

LICENSE, README.md, requirements.txt
test.png, output.mp3, temp.mp3는 시연에 필요할 수 있는 파일

### **AcceSight.py**:  AcceSight를 실행하고 사용해볼 수 있는 메인 코드입니다.

### **openai_ssh_key.py**: AcceSight.py를 정상 실행하려면 이 코드에 OpenAI API Key를 입력해야합니다.
API key는 자신의 것을 사용해도 좋습니다. 시연을 위해, **최종 보고서 가장 밑에 API Key를 기재**해두었습니다.


# 사용 및 실행 Manual

## 환경 세팅
1. '<>Code'버튼을 클릭하고, 'Download ZIP'하여 코드를 다운로드한다.
2. 압축을 푼다.
3. 새로운 가상환경(Python version: 3.12.1)을 만든다.
4. 가상환경을 activate하고, 압축을 푼 폴더의 위치로 이동한 후 pip install -r requirements.txt 한다. 필요한 패키지가 모두 설치된다.
5. openai_ssh_key.py를 열어 편집한다. 자신의 OpenAI API Key를 NLP_API_KEY = ""의 "" 사이에 입력한다.
**11조 emphAthI의 최종 보고서 최하단에, 시연용 OpenAI API Key를 기재해 제출했습니다.**
이를 입력하여 사용해주시면 됩니다.
6. 실행하는 환경의 성능에 따라 whipser의 model_size를 적절하게 변경함이 좋다. 
API key 입력을 완료했고 실행하는 환경의 성능이 whisper의 large size 모델을 사용할 수 있는 환경이라면 바로 AcceSight.py를 실행하면 된다.
실행환경의 성능이 높지 않다면, AcceSight.py 코드 본문에서 

def transcribe_audio(file_path, model_size='large', language='Korean'):을

def transcribe_audio(file_path, model_size='small', language='Korean'):로 변경함이 좋다.

7. 첫 실행 사이트는 "https://papago.naver.com"로 지정되어있다.
이를 변경하고싶다면, AcceSight.py를 열어 initial_url = "https://papago.naver.com" 부분의 주소를 수정하면 된다. 

8. 컴퓨터에 마이크와 웹캠을 연결한다.


## 코드 테스트
0. AcceSight.py를 실행한다.
1. initial_url에 따른 웹이 실행된다. https://papago.naver.com이 실행된다.
2. 기다린다. 자동으로 다음의 동작들이 수행되고있기에 시간이 걸리는 것이지 오류가 아니니 기다려야한다. 웹의 HTML을 추출하여, GPT-4o에 HTML을 전달하고 웹을 묘사해줄 것을 요청한다. GPT-4o가 웹을 묘사한 글을 답하고, 이를 gTTS로 TTS하여 재생한다. 따라서 기다린다. GPT-4o LLM 호출과 TTS에 시간이 소요된다. 오류가 아니므로 기다리면, 곧 웹을 설명하는 TTS 음성이 재생된다.
2. TTS 음성 재생이 끝나면 "손동작 인식을 시작합니다"라는 음성과 함께 웹캠이 켜진다.
3. 학습시켜둔 8종류의 손동작 중 어떤 동작을 인식하느냐에 따라 다음 단계와 기능이 달라진다.
다음 7가지 손동작 중 원하는 기능을 상황에 맞게 선택하여, 웹캠에 해당 손동작을 보여주면 된다.
손동작 종류에 따라 다음과 같은 단계가 이어진다.

(1) spin 동작(검지 손가락만 펼친 포즈) 인식 = 새로고침 트리거
1. 동작을 인식한 직후, 자동으로 웹에서 새로고침 동작을 수행한다.

(2) back 동작 (다섯 손가락 모두 펼친 포즈) 인식 = 뒤로가기 트리거
1. 동작을 인식한 직후, 자동으로 웹에서 '뒤로가기' 동작을 수행한다.

(3) good 동작 (주먹에서 엄지만 펼친 포즈) 인식 = 스크린리더 트리거
1. 동작을 인식한 직후, 자동으로 웹에서 HTML을 추출하고 파싱하여 스크린 상에 표시되는 텍스트만 골라 저장한다.
2. 이후 자동으로 이를 TTS하고 음성파일로 저장한다.
3. 이를 사용자에게 재생하여 스크린리더 기능을 수행한다.

(4) okay 동작 (검지와 엄지를 동그랗게 붙인 포즈) 인식 = 텍스트 입력 트리거
1. 동작을 인식한 직후, 자동으로 안내 음성을 재생한다. "지금부터 음성 인식을 시작합니다. 음성인식을 종료하려면 손을 주먹 쥐어주세요."
2. 자동으로 녹음을 시작하고, 손동작 인식을 위해 한 번 더 웹캠을 실행한다.
3. 원하는 동작을 음성으로 지시한다. 예를 들어, '왼쪽 번역 입력 텍스트 박스에, 인공지능공학부라고 적어줘.'라고 말한다. 말이 끝났으면 웹캠에 주먹을 보여준다.
4. stop(주먹) 동작을 인식한 직후, 자동으로 음성 녹음을 종료하고 저장한다.
5. 저장된 음성명령 녹음은 자동으로 Whipser을 통해 STT 된다.
6. STT된 텍스트 명령은 text 입력용 프롬프트와 자동으로 조합되어 LLM(GPT-4o)에 전달된다.
7. LLM은 "사용자가 원하는 텍스트 필드에 해당하는 CSS Selector, 사용자가 입력되길 원하는 텍스트"형태로 답을 return한다. 이를 쉼표(,) 기준으로 파싱하여, 각각 Selenium의 웹 제어 동작에 사용되는 변수로 사용한다.
8. 4~7의 자동 동작이 끝나면, "사용자 입력에 따른 텍스트 입력을 수행합니다."라는 음성 안내와 함께, 사용자가 원하던 해당 텍스트 필드에 원하는 텍스트를 입력한다.

(5) click 동작(검지와 중지만 펼친 포즈) 인식 = 클릭 트리거
1. 동작을 인식한 직후, 자동으로 안내 음성을 재생한다. "지금부터 음성 인식을 시작합니다. 음성인식을 종료하려면 손을 주먹 쥐어주세요."
2. 자동으로 녹음을 시작하고, 손동작 인식을 위해 한 번 더 웹캠을 실행한다.
3. 원하는 동작을 음성으로 지시한다. 예를 들어, '로그인 버튼.'이라고 말한다. 말이 끝났으면 웹캠에 주먹을 보여준다.
4. stop(주먹) 동작을 인식한 직후, 자동으로 음성 녹음을 종료하고 저장한다.
5. 저장된 음성명령 녹음은 자동으로 Whipser을 통해 STT 된다.
6. STT된 텍스트 명령은 text 입력용 프롬프트와 자동으로 조합되어 LLM(GPT-4o)에 전달된다.
7. LLM은 사용자가 원하는 버튼에 해당하는 "CSS Selector"형태로 답을 return한다. 이를 Selenium의 클릭 웹 제어 동작에 사용되는 변수로 사용한다.
8. 4~7의 자동 동작이 끝나면, "사용자 입력에 따른 클릭을 수행합니다."라는 음성 안내와 함께, 사용자가 원하던 버튼을 클릭한다.

(6) capture 동작(검지와 엄지만 펼친 포즈) 인식 = 이미지 캡셔닝 트리거
1. 동작을 인식한 직후, 자동으로 안내 음성을 재생한다. "지금부터 음성 인식을 시작합니다. 음성인식을 종료하려면 손을 주먹 쥐어주세요."
2. 자동으로 녹음을 시작하고, 손동작 인식을 위해 한 번 더 웹캠을 실행한다.
3. 원하는 동작을 음성으로 지시한다. 예를 들어, 월드컵 게임 사이트를 켜두고있었다면 '왼쪽의 그림' 혹은 '오른쪽의 그림.'이라고 말한다. 말이 끝났으면 웹캠에 주먹을 보여준다.
4. stop(주먹) 동작을 인식한 직후, 자동으로 음성 녹음을 종료하고 저장한다.
5. 저장된 음성명령 녹음은 자동으로 Whipser을 통해 STT 된다.
6. STT된 텍스트 명령은 text 입력용 프롬프트와 자동으로 조합되어 LLM(GPT-4o)에 전달된다.
7. LLM은 사용자가 원하는 이미지에 해당하는 "URL"형태로 답을 return한다. 
8. 이 URL로 자동으로 이미지를 다운받는다.
9. 다운받은 이미지로 Image Captioning을 진행한다
10. Image Captioning 결과를 LLM에게 전달해 자연스러운 한국어로 다듬는다
11. 4~10의 자동 동작이 끝나면, "해당 이미지에 대한 설명입니다."라는 음성 안내와 함께, 사용자가 원하던 이미지의 한국어 설명을 TTS한다.

(7) away 동작(다섯 손가락을 늘어뜨린 포즈) 인식 = 웹 종료 트리거
1. 동작을 인식한 직후, "사용자 입력에 따라 프로그램을 종료합니다"라는 음성 안내와 함께, 웹과 프로그램이 종료된다.

(1)~(6) 중 어떤 동작을 선택했어도 해당 동작이 완료되고 나면,
다시 2번으로 돌아가 새로운 손동작을 인식하고 (1)~(7) 중 하나의 동작을 수행함을 반복한다.
(7)이 동작하여 프로그램이 종료될 때 까지 무한히 반복된다.

원하는 행동에 따라, 웹캠에 손동작을 인식시켜서 7가지 중 원하는 동작을 자유롭게 실행하면 된다.
예를 들어 https://papago.naver.com 시연 시에는, (okay)왼쪽 텍스트박스에 텍스트 입력 요청, (spin)새로고침, (click)로그인 버튼 클릭 요청, (back)뒤로가기, (away)종료 순으로 시연해볼 수 있다. 
이 과정에서 (stop)음성인식 중지도 당연히 시연된다.


**시연해볼만한 페이지**
https://papago.naver.com
https://chatgpt.com
https://www.piku.co.kr/w/3g4445

**시연이 어려운 페이지**
HTML이, GPT-4o API로 처리하기에는 지나치게 많은 분량의 텍스트일 경우.
이러한 페이지는 지금의 LLM 기술로는 처리하기 어렵습니다.
단, 추후 LLM 기술이 발전한다면 이 프로그램의 역량도 함께 성장합니다.


## 사용하는 패키지
환경설정 4번대로 **pip install -r requirements.txt** 를 실행하면 다음의 모든 패키지가 설치됩니다.

**Hand Detect AI_ MediaPipe**
* pip install opencv-python
* pip install numpy
* pip install mediapipe
* pip install tensorflow
(tensorflow version == 2.16.1) 2024.06.19 기준 lastest version

**gTTS**
* pip install gtts
* pip install playsound
* pip install pygame

**Image captioning**
* pip install transformers
* pip install torch
* pip install pillow

**GPT API**
* pip install openai

**whisper**
* pip install openai-whisper

**Selenium**
* pip install selenium
* pip install webdriver_manager

**download image**
* pip install requests

**html text extract**
* pip install beautifulsoup4

**record**
* pip install pyaudio



## 사용한 오픈소스의 라이센스
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
