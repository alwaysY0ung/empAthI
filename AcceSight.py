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


from selenium import webdriver
from selenium.webdriver.common.by import By
from gtts import gTTS
from playsound import playsound
import os
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

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

def click_element(driver, element_id):
    element = driver.find_element(By.ID, element_id)
    element.click()

def capture_image(driver, element_id):
    image_element = driver.find_element(By.ID, element_id)
    image_src = image_element.get_attribute('src')
    # 이미지 다운로드 로직 추가 필요함
    return image_src

# TTS
def play_tts(text):
    tts = gTTS(text=text, lang='ko')
    audio_file = "temp.mp3"
    tts.save(audio_file)
    playsound(audio_file)
    os.remove(audio_file)

# NLP 호출 함수_추후 추가하겠습니다
def NLP_call(prompt):
    # 여기에 NLP 호출 코드를 작성
    # 예: NLP API에 prompt를 전달하고 결과를 반환
    return "NLP 응답 결과"

# 웹페이지 코드 기반으로 NLP 설명을 받는 함수
def describe_page_with_nlp(html_code):
    page_description_prompt = f"다음 html코드를 기반으로 웹페이지를 설명해줘(임시프롬프트임).\n코드: {html_code}"
    description = NLP_call(page_description_prompt)
    return description

def hand_recognize():
    # actions 내용 수정
    actions = ['capture', 'good', 'okay', 'back', 'spin', 'stop', 'click', 'away']
    seq_length = 30

    # model명 수정
    model = load_model('models/mediapipe_hand_detect_model.keras')

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


# whisper 함수 추후 추가하겠습니다. stt()라고 표시해둠.

# image captioning 함수 추가 필요합니다. 주석으로 표기해둠.

# 메인 함수
def main():
    initial_url = "https://google.com"
    driver = start_browser(initial_url)
    current_url = get_current_url(driver)
    
    while True:
        if current_url != initial_url:
            html_code = driver.page_source
            page_description = describe_page_with_nlp(html_code)
            play_tts(page_description)
            initial_url = current_url
        
        play_tts("손동작 인식을 시작합니다.")
        
        # Mediapipe 손동작 인식
        detected_gesture = hand_recognize()
        
        if detected_gesture == "spin":
            refresh_page(driver)
            play_tts("페이지를 새로고침했습니다.")
            current_url = get_current_url(driver)
        elif detected_gesture == "back":
            go_back_page(driver)
            play_tts("이전 페이지로 이동했습니다.")
            current_url = get_current_url(driver)
        elif detected_gesture == "okay":
            play_tts("입력할 텍스트를 말해주세요.")
            input_text_audio = stt()  # Whisper를 사용한 음성 인식 결과
            input_text_prompt = f"아까 네가 설명해준 거 : {page_description}\n 텍스트 ID와 입력되길 원하는 텍스트: {input_text_audio} 'ID', '입력할 텍스트' 형식으로 대답하시오. 이외의 답변은 엄격히 금지"
            input_text_info = NLP_call(input_text_prompt)
            field_id, text = input_text_info.split(',')
            input_text(driver, field_id, text)
            play_tts("텍스트 입력 완료했습니다.")
        elif detected_gesture == "click":
            play_tts("클릭하고 싶은 요소를 말해주세요.")
            click_element_audio = stt()  # Whisper를 사용한 음성 인식 결과
            click_element_prompt = f"아까 네가 설명해준 거 : {page_description}\n 내가 설명 원하는 거: {click_element_audio} 'ID'만 대답하시오. 이외의 답변은 엄격히 금지"
            click_element_id = NLP_call(click_element_prompt)
            click_element(driver, click_element_id)
            play_tts("요소를 클릭했습니다.")
        elif detected_gesture == "good":
            page_text = "여기서 BeautifulSoup를 이용해 HTML 텍스트를 추출하고 TTS로 읽음"
            play_tts(page_text)
            play_tts("페이지의 내용 모두 읽어드렸습니다.")
        elif detected_gesture == "capture":
            play_tts("듣고 싶은 이미지를 말씀해주세요.")
            image_caption_audio = stt()  # Whisper를 사용한 음성 인식 결과
            image_description_prompt = f"아까 네가 설명해준 거 : {page_description}\n 내가 설명 원하는 거: {image_caption_audio} '이미지 ID'만 대답하시오. 이외의 답변은 엄격히 금지"
            image_id = NLP_call(image_description_prompt)
            image_src = capture_image(driver, image_id)
            image_description = "" #여기서 이미지 캡셔닝 모델로부터 설명을 받음
            play_tts(image_description)
            play_tts("이미지 설명 완료했습니다.")
        elif detected_gesture == "away":
            play_tts("웹 브라우징을 종료합니다.")
            break
        
        current_url = get_current_url(driver)
    
    driver.quit()

if __name__ == "__main__":
    main()
