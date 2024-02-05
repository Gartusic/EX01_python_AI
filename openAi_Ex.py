from openai import OpenAI # openai 모델 사용
import requests
from flask import Flask, render_template, request


# 플라스크 : 파이썬 동적 html 코드 생성 웹서버 프레임워크(마이크로프레임워크)임
app = Flask(__name__)

# OpenAI API 엔드포인트 URL
url = "https://api.openai.com/v1/chat/completions"

# OpenAI API 인증 토큰 (API 키)ㅎ
api_key = "" # 자신의 고유 키를 입력해요

# API 요청 보내기
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def html_run():
    #body = input("body : ")
    #qtn = input("Question : ")
    # htmls = "<h1 style='background:skyblue'>나의 친 우 OpenAI 의 도 래 . </h1>"
    # htmls += "<div>질문: "+question+"</div>"
    # htmls += "<div>대답: "+aiRun(message)+"</div>"
    # htmls += "<br><br><br><br><br><br>"
    # htmls += "<input type='text' value='' id='schVal' /><button onclick='javascript:goSearch()'>전송</button>"
    # htmls += "<marquee>사랑해</marquee>"
    return render_template('openAI_Ex.html')

@app.route('/answer')
def ai_run():
    # question : openai를 통해 보낼 질문
    question = request.form['qVal']
    message = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": question}]
    }
    response = requests.post(url, json=message, headers=headers)

    rtn=""

    # API 응답 처리
    if response.status_code == 200:
        result = response.json()
        assistant_response = result["choices"][0]["message"]["content"]
        rtn=assistant_response
        print("대답:", assistant_response)
    else:
        print("API 요청 실패:", response.text)
        rtn=response.text
    return rtn



@app.route('/')
def home():
    # return html_run()
    return render_template('openAI_Ex.html')

if __name__ == '__main__':
    app.run(debug=True) #http://127.0.0.1:5000 로컬서버로 run함