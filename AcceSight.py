"""
MediaPipe
Copyright 2024 kairess
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
https://github.com/kairess/gesture-recognition
위 repository의 test.py 코드와 robot.py를 일부 수정했습니다. 수정 사항은 주석으로 표시했습니다.
I made some modifications to the code in test.py. The changes are indicated with comments.
"""
"""
gTTS
MIT License

Copyright (c) 2024 Pierre Nicolas Durette

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

https://github.com/pndurette/gTTS/tree/main의 tts.py를 참고해 구현한 코드입니다. gTTS 사용법 예제 코드.
"""
"""
image_captioning
Copyright 2024 kairess
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

https://github.com/Redcof/vit-gpt2-image-captioning
https://github.com/huggingface
"""

"""
BeautifulSoup
 MIT License

 Copyright (c) 2024 Pierre Nicolas Durette

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 https://github.com/akalongman/python-beautifulsoup
"""

"""
BeautifulSoup
 MIT License

 Copyright (c) 2024 Pierre Nicolas Durette

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 https://github.com/akalongman/python-beautifulsoup
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from gtts import gTTS
from playsound import playsound
import os
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
import time

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

import whisper
import re

from openai import OpenAI

# image captioning 관련 추가부분
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

import pyaudio
import wave
import threading

from bs4 import BeautifulSoup, NavigableString, Tag


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# 모든 프롬프트는 임시로 작성했습니다. 추후 적절한 프롬프트로 변경합시다.
# mediapipe와 whisper과 nlp과 image captioning 관련 코드는 임시로 비워두었습니다. 추후 조립합시다.
# pseudocode를 기반으로 임시로 작성한 코드이므로 오류가 있을 수 있습니다. 

# Selenium 설정
def start_browser(url):
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    return driver

def get_current_url(driver):
    return driver.current_url

def refresh_page(driver):
    driver.refresh()

def go_back_page(driver):
    driver.back()

def input_text(driver, field_id, text):
    input_field = driver.find_element(By.ID, field_id)
    input_field.clear()
    input_field.send_keys(text)
    # input_field.send_keys(Keys.RETURN) #임시로 엔터까지

def click_element(driver, element_id):
    element = driver.find_element(By.ID, element_id)
    element.click()

# def capture_image(driver, element_id):
#     image_element = driver.find_element(By.ID, element_id)
#     image_src = image_element.get_attribute('src')
#     # 이미지 다운로드 로직 추가 필요함
#     return image_src

def play_wav_file(file_path): # 연이어서 같은 동작을 수행할 때 등, 동일한 .wav가 열려있을 때 I/O 오류를 방지하기 위한 함수
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # 오디오 파일 열기
            audio = pyaudio.PyAudio()
            stream = audio.open(format=audio.get_format_from_width(wav_file.getsampwidth()),
                                channels=wav_file.getnchannels(),
                                rate=wav_file.getframerate(),
                                output=True)
            
            # 오디오 데이터 읽기 및 재생
            data = wav_file.readframes(1024)
            while data:
                stream.write(data)
                data = wav_file.readframes(1024)
            
            # 파일 닫기
            stream.stop_stream()
            stream.close()
            audio.terminate()
    except wave.Error as e:
        print(f"Failed to open the file: {file_path}")
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An error occurred while playing the file: {file_path}")
        print(f"Error: {str(e)}")

# TTS_입력받은 text를 음성으로 재생
def play_tts(text):
    tts = gTTS(text=text, lang='ko')
    audio_file = "temp.mp3"
    tts.save(audio_file)
    
    # 파일이 저장될 때까지 대기
    while not os.path.exists(audio_file):
        time.sleep(0.1)
    
    playsound(audio_file)


# GPT-4o NLP API 호출 함수_ 입력받은 prompt에 대한 답변 string return
def NLP_call(prompt):
    NLP_API_KEY = "API KEY 입력"
    client = OpenAI(api_key = NLP_API_KEY)

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": prompt}
    ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

# # 웹페이지 코드 기반으로 NLP 설명을 받는 함수_ 입력받은 html에 대한 설명 string return
# def describe_page_with_nlp(html_code):
#     page_description_prompt = f"다음 페이지를, 시각장애인에게 설명해주듯이 세세하고 길게 묘사해서 설명하라. html 개발적 요소를 설명하지 말고, 기능과 UI를 위주로 설명하라. 마크다운 요소 없이 대답하라. html: \n {html_code}"
#     description = NLP_call(page_description_prompt)
#     return description

# Hand pose 인식 MediaPipe_ 웹캠을 실행해서 포즈를 인식하고, 인식한 포즈 이름 string return
def hand_recognize():
    # actions 내용 수정
    actions = ['capture', 'good', 'okay', 'back', 'spin', 'stop', 'click', 'away']
    seq_length = 30

    # model명 수정
    model = load_model('hand_detect_MediaPipe_model/models/mediapipe_hand_detect_model.keras')

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    seq = []
    action_seq = []

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action


                # 인식할 경우 수행할 동작 수정. 예시로 print문을 삽입해둠. 이곳을 수정하면 원하는 동작을 사용할 수 있습니다.
                if this_action == 'capture':
                    print("[*]capture 인식")
                    return "capture"
                elif this_action == 'good':
                    print("[*]good 인식")
                    return "good"
                elif this_action == 'okay':
                    print("[*]okay 인식")
                    return "okay"
                elif this_action == 'back':
                    print("[*]back 인식")
                    return "back"
                elif this_action == 'spin':
                    print("[*]spin 인식")
                    return "spin"
                elif this_action == 'stop':
                    print("[*]stop 인식")
                    return "stop"
                elif this_action == 'click':
                    print("[*]click 인식")
                    return "click"
                elif this_action == 'away':
                    print("[*]away 인식")
                    return "away"
                
                cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break

# STT_whisper 모델 사용
def transcribe_audio(file_path, model_size='large', language='Korean'): # 모델 size는 small로 할지 medium으로 할지 large로 할지 고민 중
    # 모델 로드
    model = whisper.load_model(model_size)

    # 오디오 파일에서 음성 인식
    result = model.transcribe(file_path, language=language)
    return result['text']

# STT_whisper 결과 파싱하여 후처리
def remove_timestamps(text):
    # [시간 --> 시간] 패턴을 제거하는 정규 표현식
    pattern = r"\[\d{2}:\d{2}:\d{2} --> \d{2}:\d{2}:\d{2}\]"
    return re.sub(pattern, "", text)

# STT_whisper_ 입력받은 오디오파일(경로)에 대한 STT 결과 string return
def stt(audio_file): # 녹음된 오디오 파일 경로 입력    
    transcription = transcribe_audio(audio_file)
    cleaned_transcription = remove_timestamps(transcription)

    # # 결과를 텍스트 파일에 저장
    # with open("temp/result.txt", "w", encoding="utf-8") as file: # result.txt에 임시 저장
    #     file.write(cleaned_transcription)

    # print("[*]녹음본이 result.txt에 저장됐습니다.")

    return cleaned_transcription # 파일이 아닌 text를 바로 return하게 했습니다!

# image captioning 함수 추가하였습니다.
#이미지 캡셔닝 함수 호출 방법은 다음과 같습니다.
#image_captioning([이미지파일명])
#이미지 캡셔닝 함수 리턴값은 문자열 리스트 입니다.
def image_captioning(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")
    images.append(i_image)

  pixel_values = feature_extractor(
    images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds
    
# 이미지 다운로드 함수 추가하였습니다.
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

# 이미지 설명함수 추가하였습니다.
def describe_image(target_image_url): # 특정 이미지 URL 입력
    # Chrome WebDriver 설정
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # # 웹페이지 접속
    # url = "https://www.wolyo.co.kr/news/articleView.html?idxno=242164"
    # driver.get(url)

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
        # TTS로 이미지 설명문장 출력
        caption_text = image_captions[0]
    return caption_text



# 사용자 음성 녹음하는 로직 추가
# 녹음 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1  # 채널 수를 2에서 1로 변경
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "temp/recorded.wav"

audio = pyaudio.PyAudio()
frames = []
stream = None
recording = False

def start_recording():
    global frames, stream, recording
    frames = []
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    recording = True
    threading.Thread(target=record).start()

def record():
    global frames, recording
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)

def stop_recording():
    global recording, stream
    recording = False
    stream.stop_stream()
    stream.close()
    
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def record_and_save():
    # play_tts("입력할 텍스트를 말해주세요.")
    start_recording()
    while True:
        detected_gesture = hand_recognize()
        if detected_gesture == "stop":
            stop_recording()
            audio_file = "temp/recorded.wav"
            return audio_file

# BeautifulSoup html에서 눈에 보이는 text 추출
def is_visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True

def html_to_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    body, text = soup.body, []
    for element in body.descendants:
        if type(element) == NavigableString:
            parent_tags = (t for t in element.parents if type(t) == Tag)
            hidden = False
            for parent_tag in parent_tags:
                if (parent_tag.name in ('area', 'base', 'basefont', 'datalist', 'head', 'link',
                                        'meta', 'noembed', 'noframes', 'param', 'rp', 'script',
                                        'source', 'style', 'template', 'track', 'title', 'noscript',
                                        'button', 'option', 'fieldset', 'legend', 'label', 'footer',
                                        'figcaption', 'figure', 'header', 'nav', 'aside', 'object',
                                        'embed', 'applet', 'iframe', 'select', 'optgroup', 'datalist') or
                    parent_tag.has_attr('hidden') or
                    (parent_tag.name == 'input' and parent_tag.get('type') == 'hidden') or
                    parent_tag.has_attr('aria-hidden') and parent_tag['aria-hidden'] == 'true' or
                    parent_tag.has_attr('type') and parent_tag['type'] == 'hidden' or
                    parent_tag.has_attr('style') and 'display:none' in parent_tag['style'] or
                    parent_tag.has_attr('class') and any(c in parent_tag['class'] for c in ('hidden', 'sr-only', 'dropdown')) or
                    parent_tag.has_attr('style') and 'visibility:hidden' in parent_tag['style'] or
                    parent_tag.has_attr('style') and 'opacity:0' in parent_tag['style'] or
                    parent_tag.has_attr('aria-expanded') and parent_tag['aria-expanded'] == 'false'):
                    hidden = True
                    break
            if hidden or not is_visible(element):
                continue
            
            string = ' '.join(element.string.split())
            if string:
                if element.parent.name == 'a':
                    a_tag = element.parent
                    if 'href' in a_tag.attrs:
                        string = a_tag['href']
                        if (    type(a_tag.previous_sibling) == NavigableString and
                                a_tag.previous_sibling.string.strip() ):
                            text[-1] = text[-1] + ' ' + string
                            continue
                elif element.previous_sibling and element.previous_sibling.name == 'a':
                    text[-1] = text[-1] + ' ' + string
                    continue
                elif element.parent.name == 'p':
                    string = '\n' + string
                text += [string]
    doc = '\n'.join(text)
    return doc

# 메인 함수
def main():
    initial_url = "https://papago.naver.com/"
    driver = start_browser(initial_url)
    current_url = get_current_url(driver)

    html = driver.page_source
    # page_description = describe_page_with_nlp(html_code)
    page_description_prompt = f"다음 페이지를, 시각장애인에게 설명해주듯이 세세하고 길게 묘사해서 설명하라. html 개발적 요소를 설명하지 말고, 기능과 UI를 위주로 설명하라. 마크다운 요소 없이 대답하라. html: \n {html}"
    page_description = NLP_call(page_description_prompt)
    # play_tts(page_description)
    print(page_description)
    initial_url = current_url
    
    while True:
        if current_url != initial_url:
            html = driver.page_source
            # page_description = describe_page_with_nlp(html_code)
            page_description_prompt = f"다음 페이지를, 시각장애인에게 설명해주듯이 세세하고 길게 묘사해서 설명하라. html 개발적 요소를 설명하지 말고, 기능과 UI를 위주로 설명하라. 마크다운 요소 없이 대답하라. html: \n {html}"
            page_description = NLP_call(page_description_prompt)
            # play_tts(page_description)
            print(page_description)
            initial_url = current_url
        
        # Mediapipe 손동작 인식
        detected_gesture = hand_recognize()

        # play_tts("손동작 인식을 시작합니다.") # 인식하지 못하는 문제가 간헐적으로 발생
        
        if detected_gesture == "spin": # 작동 문제 없음
            refresh_page(driver)
            play_wav_file("voice/generations/re.wav")
            current_url = get_current_url(driver)
        elif detected_gesture == "back":
            go_back_page(driver)
            play_wav_file("voice/generations/back.wav")
            current_url = get_current_url(driver)
        elif detected_gesture == "okay":
            play_wav_file("voice/generations/text.wav")
            audio_file = record_and_save()
            input_text_audio = stt(audio_file)  # Whisper를 사용한 음성 인식 결과
            print("인식된 텍스트는 '"+input_text_audio+"'입니다.")
            html = driver.page_source
            input_text_prompt = f"제공한 HTML 코드에서 다음 텍스트박스를 찾아서 알려줘. '텍스트박스ID, 입력할내용string' 의 형식으로 대답해. 다른 말은 필요없어. 인삿말과 설명과 같은 다른 말을 덧붙이는 것은 엄격히 금지한다, 오로지 텍스트박스의 ID, 입력할내용string 만을 답하시오. 찾아야하는 텍스트박스와 입력해야할 내용= '{input_text_audio}' html: \n {html}"
            input_text_info = NLP_call(input_text_prompt)
            field_id, text = input_text_info.split(',')
            # field_id = "txtSource"
            # text = "숙명여대"
            input_text(driver, field_id, text)
            play_wav_file("voice/generations/announce_text.wav")
        elif detected_gesture == "click":
            play_wav_file("voice/generations/click.wav")
            audio_file = record_and_save()          
            click_element_audio = stt(audio_file)  # Whisper를 사용한 음성 인식 결과
            html = driver.page_source
            click_element_prompt = f"제공한 HTML 코드에서 다음 버튼 id를 찾아서 알려줘. 다른 말은 필요없어. 인삿말과 설명과 같은 다른 말을 덧붙이는 것은 엄격히 금지한다, 오로지 버튼의 ID만을 답하시오. 찾아야하는 버튼= '{click_element_audio}' html:{html}"
            click_element_id = NLP_call(click_element_prompt)

            # click_element_id = "btnTranslate"
            click_element(driver, click_element_id)
            # play_tts("요소를 클릭했습니다.")
        elif detected_gesture == "good":
            html = driver.page_source
            page_text = html_to_text(html) # 추출한 텍스트
            print(page_text)
            #여기서 바로 읽지 말고, nlp 한 번 거쳐서 다듬기.
            # "이걸 TTS할건데, 여기서 듣기에 방해되는 더미 문자열은 지우고 의미있는 문자열만 남겨봐 원본 문자열을 최대한 유지해" 이런 프롬프트로
            # play_tts(page_text)
            # play_tts("페이지의 내용 모두 읽어드렸습니다.")
        elif detected_gesture == "capture":
            play_wav_file("voice/generations/image.wav")
            audio_file = record_and_save()       
            image_caption_audio = stt(audio_file)  # Whisper를 사용한 음성 인식 결과
            html = driver.page_source
            image_find_prompt = f"제공한 HTML 코드에서 다음 이미지의 URL을 찾아서 알려줘. 다른 말은 필요없어. 인삿말과 설명과 같은 다른 말을 덧붙이는 것은 엄격히 금지한다, 오로지 이미지의 URL만을 답하시오. 찾아야하는 이미지 = '{image_caption_audio}' html: \n {html}" # 이미지 URL 받기
            target_image_url = NLP_call(image_find_prompt)
            image_description_en = describe_image(target_image_url)
            print(image_description_en)
            # image_description_kr = f""
            play_wav_file("voice/generations/announce_image.wav") # "해당 이미지에 대한 설명입니다."
            #play_tts(image_description)
        elif detected_gesture == "away":
            play_wav_file("voice/generations/exit.wav")
            break
        
        current_url = get_current_url(driver)
    
    driver.quit()


if __name__ == "__main__":
    main()
