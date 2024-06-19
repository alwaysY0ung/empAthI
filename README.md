# empAthI

[emphAthI] is a university student team that started with the aim of empathizing with and helping the socially disadvantaged, and aims to provide Accesight (Access+sight), a service for the blind to access the web.

### [Service Purpose and Background]
It guarantees access to information for the visually impaired.
It will reinforce and strengthen existing 'screen reader' services through AI technology.
Previously, it was limited to simply reading the website that was posted, but AceSight guides the specific functions and appearance of the web through HTML and image vision analysis.

### [Expected user]
a blind person
a person who is unable to move his or her body for reasons such as nerve paralysis
the digital underprivileged

### [Core Value and Differences of Services]
The spread of information and access caused by the Internet has led to the expansion of knowledge and technology worldwide.
The web, which has such usefulness and explosive potential, is difficult for the visually impaired to access because it is provided around visual information.
Through the AceSight service, visually impaired people can also be guaranteed a very important right to access information.

### [Planning major service functions]
It controls the movements of web browsers such as 'backward' and 'refresh' with the movement of the hand.
STT controls specific commands that are different for each page, and gestures control universal commands that are possible for all pages.
Expandability: Although it is now a finger gesture for the visually impaired, it can be useful to more people by allowing them to select options such as detecting gaze changes or detecting foot movements depending on the type of disability in the future.
Even if the gesture is intended to implement the degree of backward refresh for now, various gestures may be implemented and provided in the future, or a user may customize and register an operation or an operation for each operation.

### [Understanding the contents of the web]
The website is outlined immediately when the website is moved.
It analyzes the contents of the website with vision and nlp (which also performs html code analysis) and voiceizes it with TTS.
I will take the stt command again and explain in more detail what the user is curious about.

### [User-Web Interaction]
Click: Performs the user's desired click instead
Enter text: Enter text instead of text desired by the user
NLP controls the web based on html and user commands.



## 라이브러리

**Hand Detect AI_ MediaPipe**
pip install opencv-python
pip install numpy
pip install mediapipe
pip install tensorflow
(tensorflow version == 2.16.1)

**gTTS**
pip install gtts
pip install playsound

**Image captioning**
pip install transformers
pip install torch
pip install pillow

**AcceSight.py**
pip install selenium
pip install webdriver-manager
pip install beautifulsoup4

**GPT API**
pip install openai

**whisper**
pip install openai-whisper

**Selenium**
pip install selenium
pip install webdriver_manager



## 라이센스
**Hand Detect AI_ MediaPipe**
This project uses code from [gesture-recognition](https://github.com/kairess/gesture-recognition) licensed under the Apache-2.0 License.

**gTTS**
This project uses code from [gTTS]([https://github.com/kairess/gesture-recognition](https://github.com/pndurette/gTTS/tree/main)) licensed under the MIT License.

**Image captioning**
This project uses code from [vit-gpt2-image-captioning](https://github.com/Redcof/vit-gpt2-image-captioning)licensed under the Apache-2.0 License.



## 환경 및 버전
python = 3.12.1
