import whisper
import re

def transcribe_audio(file_path, model_size='small', language='Korean'): # 모델 size는 small로 할지 medium으로 할지 large로 할지 고민 중
    # 모델 로드
    model = whisper.load_model(model_size)

    # 오디오 파일에서 음성 인식
    result = model.transcribe(file_path, language=language)
    return result['text']

def remove_timestamps(text):
    # [시간 --> 시간] 패턴을 제거하는 정규 표현식
    pattern = r"\[\d{2}:\d{2}:\d{2} --> \d{2}:\d{2}:\d{2}\]"
    return re.sub(pattern, "", text)

# 사용 예제
audio_file = "mp3 경로" # 경로 입력해야함
transcription = transcribe_audio(audio_file)
cleaned_transcription = remove_timestamps(transcription)

# 결과를 텍스트 파일에 저장
with open("result.txt 저장할 경로", "w", encoding="utf-8") as file: # result 저장할 경로 넣어줘야함
    file.write(cleaned_transcription)

print("Transcription saved to result.txt")
