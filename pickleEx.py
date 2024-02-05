import pickle # .pickle 형식으로 저장된 모델  파일을 불러옴
from flask import Flask
import requests

#pickle을 통해 모델을 저장 및 불러오기 할 수 있다.

with open('modelSaved/mnist.pickle', 'rb') as f: # 해당 모델을 불러옴
    model = pickle.load(f)

# 플라스크 : 파이썬 동적 html 코드 생성 웹서버 프레임워크(마이크로프레임워크)임
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

if __name__ == '__main__':
    app.run(debug=True) #http://127.0.0.1:5000 로컬서버로 run함

