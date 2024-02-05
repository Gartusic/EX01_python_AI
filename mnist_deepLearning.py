from tensorflow.keras.models import Sequential # Sequential : 순차적인 신경망을 구성할 때 사용할 수 있는 함수
from tensorflow.keras.layers import Dense, Activation # Dense : 레이어의 뉴런 갯수를 설정, Activation : 활성화 함수
from tensorflow.keras.utils import to_categorical # to_categorical : 원-핫 인코딩을 구현할 수 있는 함수
from tensorflow.keras.datasets import mnist # mnist : 딥러닝 모델 개발을 연습할 수 있는 데이터셋 load_data()
import numpy as np
import matplotlib.pyplot as plt # 시각화 라이브러리
import pickle

#####숫자 예측 인공지능 - 다중분류 , 심층신경망 #####

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)

X_train = x_train.reshape(60000,784)
X_test = x_test.reshape(10000,784)
X_train = X_train.astype("float32") # 정규화하기 위해 데이터를 0 - 1 사이의 값으로 바꿔줌 : 실수형으로 형변환
X_test = X_test.astype("float32")
X_train /= 255 # 255로 나눔으로써 0-1 사이의 값이 됨
X_test /= 255
print("X Training matrix shape",X_train.shape)
print("X Testing matrix shape",X_test.shape)

# 수치형 데이터를 범주형 데이터로 바꾸는 작업 필요. 원핫인코딩 작업 ㄱ
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)
print("Y Training matrix shape",Y_train.shape)
print("Y Testing matrix shape",Y_test.shape)

# 인공지능 만들 떄 데이터 전처리 작업이 복잡함... 
#입력층(784개 - 입력층의 뉴런의 수 . 28x28) 은닉층1(512 - reLu Activation함수 사용) 은닉층2(256개 - ) 출력층(10개- 분류되는 결과값)

#인공지능 설계 - 심층신경망 CNN
model = Sequential() # Sequential 모델 정의
model.add(Dense(512, input_shape=(784,))) # 입력층 추가. 512개의 노드로 연결됨 , 총 401920(784*512+512)개 파라미터로 이루어짐
model.add(Activation('relu'))
model.add(Dense(256)) # 은닉층 추가... 256개의 노드로 연결됨 , 총 131328(512*256+256)개 파라미터로 이루어짐
# 이 다음 은닉층을 따로 설정해줄 필요가 없다. Keras가 자동으로 해줌.
model.add(Activation('relu'))
model.add(Dense(10)) #출력층 추가
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=128,epochs=10,verbose=1) # epochs: 반복학습 몇번할지 , verbose : 로그레벨(?)정하기??

#predicted_classes :예측된 값(softmax함수 사용으로 전체 도합 1이 됨) 중 가로 행에서 제일 큰 값을 노출
predicted_classes = np.argmax(model.predict(X_test),axis=1)
#correct_indices : 정답
correct_indices = np.nonzero(predicted_classes == y_test)[0] #np.nonzero() : 실제 값과 예측 값 비교하는 함수
#incorrect_indices: 오답
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

# 인공지능 학습 결과값 시각화
plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1) #3*3 모양의 표 만듦
    correct = correct_indices[i]
    incorrect = incorrect_indices[i]
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray')
    plt.title("Predicted {}, Class {} NotCorrect{}".format(predicted_classes[correct],y_test[correct],predicted_classes[incorrect]))
plt.tight_layout()
plt.show()

with open('modelSaved/mnist.pickle', 'wb') as fw: # 완성된 모델을 pickle을 이용해 해당 경로에 저장
    pickle.dump(model, fw)