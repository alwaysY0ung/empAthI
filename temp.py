def main():
	# 웹을 열고 첫페이지를 사용자에게 설명한다
	초기주소 = https://google.com
	selenium(초기주소)
		#셀레니움으로 웹을 연다. 
		
		
	
	# 플로우차트 상에서, 모든 손동작이 끝나고 돌아가는 동일한 한 지
	주소1 = 초기주소
	while(true):
		if 주소1 = null:
			break
		else 주소1 = 반복할동작함수(주소1)
		
	웹브라우징 종료 함수 (셀레니움)
	

## 메서드 모음
	
	현재페이지 주소를 받는 함수() <- 셀레니움 찾기.
	
웹크롤링함수():	웹 크롤링을 해서 string을 return
	return html코드(string)
tts(설명): 
  #웹 페이지 설명에 대한 안내 음성 메세지
	return 알아서 (음성재생. 리턴없이. 또는 아
	
	
	tts끝나면 model 환경 on(): (			return string (뭔가있다. 공부하기)
	# 사용자의 입력을 기다린다 (순간 detection될때까지)
	# 손동작이 감지된다
		return string (뭔가있다. 공부하기)
		

def 반복할동작():
		string1 = 웹크롤링함수()
		
		if ### 텍스트입력, 이미지캡셔닝의 경우 주소가 변하지 않으므로. 웹페이지설명 단계를 조건문으로 생략할 수 있는 알고리즘 생각나면 하기
	  
	  아까웹페이지설명저장된string = nlp(html코드(string1))
		tts(string2)
		# tts 확실히 끝나면 아래 반복문이 실행될 수 있도록.
		
		next_action = 'string 뭔가 있다 선언하기' (지금은 null)
		while(not next_action):
				next_action = model 환경 on()
				TTS 안내멘트
			
		switch case:
			case1:
				next_action = '새로고침'
					do TTS 안내멘트
					do 새로고침 함수
			case2:
				next_action = '뒤로가기'
					do TTS 안내멘트
					do 뒤로가기 함수
			case3:
				next_action = '텍스트입력'
					do TTS 안내멘트
					do 텍스트입력 함수
			case4:
				next_action = '클릭'
					do TTS 안내멘트
					do 클릭 함수
			case5:
				next_action = '추가 설명'
					do TTS 안내멘트
					do 추가설명 함수
			case6:
				next_action = '이미지 설명'
					do TTS 안내멘트
					do 이미지설명 함수(아까웹페이지설명저장된string, STT된 string)
			case7:
				next_action = '종료'
					do TTS 안내멘트
					do 현재주소 = null
					
		if 현재주소 != null:
			현재주소 = 현재 페이지 주소 받는 함수()
		return 현재주소
		


def 텍스트입력 함수():
	MP3 = 음성입력받기()
	STT된 string = whisper(MP3) # 이 텍스트필드에 요러요러한 텍스트 적어줘
	텍스트입력프롬프트string = "아까 네가 설명해준 거 :" + 아까웹페이지설명저장된string + "\n 텍스트 ID와 입력되길 원하는 텍스트:" + STT된string + "'ID', '입력할 텍스트' 형식으로 대답하시오. 이외의 답변은 엄격히 금지"

	텍스트ID와텍스트string = NLP호출(텍스트입력프롬프트string) # (id, 입력할 string)
	id, 입력할 string = 텍스트필드ID #알아서 잘 포맷팅
	셀레니움으로 잘 ~ 텍스트 입력하기


def 클릭 함수():
	MP3 = 음성입력받기()
	STT된 string = whisper(MP3) # 이 버튼을 클릭하고싶어 ㅎㅎ
	버튼클릭프롬프트string = "아까 네가 설명해준 거 :" + 아까웹페이지설명저장된string + "\n 내가 설명 원하는 거:" + STT된string + "ID만 대답하시오. 이외의 답변은 엄격히 금지"

	버튼ID = NLP호출(버튼클릭프롬프트string)
	셀레니움으로 잘 ~ 버튼 클릭하기

def 추가 설명():
	추가설명프롬프트string = "아까 네가 설명해준 거 :" + 아까웹페이지설명저장된string + "\n 여기서 더 자세히 설명해줘"
	추가설명string = NLP호출(추가설명프롬프트string)
	TTS(string) # 음성 내보내기

def 이미지 설명(아까웹페이지설명저장된string, STT된 string):
	MP3 = 음성입력받기()
	STT된 string = whisper(MP3) # 오른쪽에 있는 사진 더 설명해줘
	이미지설명프롬프트string = "아까 네가 설명해준 거 :" + 아까웹페이지설명저장된string + "\n 내가 설명 원하는 거:" + STT된string + "이미지 ID만 대답하시오. 이외의 답변은 엄격히 금지"

	이미지ID = NLP호출(이미지설명프롬프트string)
	image = # 셀레니움 동작 잘 짜기 ^.^
	이미지설명string = 이미지캡셔닝
	TTS(이미지설명string) # 음성 출력
	
		
def 음성입력 받기():
	return MP3
	
def whisper(MP3):
	return string

def 이미지캡셔닝(image):
	string = AI실행
	return string

def NLP호출(string<- 각자 프롬프트 작성):
	return string
	
def 웹페이지_코드기반_nlp의_설명(html코드string)
	웹페이지설명프롬프트string = "다음 html코드를 기반으로 웹페이지를 설명해줘. 'n\코드: " + html코드string
	코드기반설명string = NLP호출(웹페이지설명프롬프트string)


