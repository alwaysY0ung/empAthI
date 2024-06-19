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

# WebDriver 종료
driver.quit()
