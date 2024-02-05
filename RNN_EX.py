from keras.models import Sequential
from keras.layers import SimpleRNN, Dense #SimpleRNN : 가장 기본적인 순환신경망 모습...(LSTM,GRU는 더 발전된 RNN)
from sklearn.preprocessing import MinMaxScaler # 데이터 정규화...에 사용
from sklearn.metrics import mean_squared_error #결과 정확도 계산
from sklearn.model_selection import train_test_split # 데이터를 훈련/검증 데이터 별로 나눔
import math # 수학계산
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv # csv파일을 읽어주는 라이브러리 (csv : comma separated values. 파이선 인공지능 학습에 주로 사용되는 파일포맷)

#####전염병 예측 인공지능 - 순환신경망(RNN) #####
# 연속된 데이터를 활용하여 이후의 값을 예측

# terminal에서 ... git clone https://github.com/yhlee1627/deeplearning.git 으로 학습 및 검증데이터를 가져오기
dataframe = read_csv('deeplearning/corona_daily.csv', usecols=[3],
                     engine='python', skipfooter=3) # csv파일을 dataframe으로 정의
                        # usecols=[3] 사용하여 확진자(Confirmed) 수만 불러올 수 있음
print(dataframe)

dataset = dataframe.values # value 값만 가져옴
dataset= dataset.astype('float32') # 정규화 시 나눗셈을 사용하므로 실수형으로 데이터값을 바꿔주어야함
#데이터 정규화는 보통 1이상의 수들을 1 미만의 소수점자리 수로 다 바꿈

scaler = MinMaxScaler(feature_range=(0,1)) # 0,1사이의 수로 바꿀 것임
Dataset = scaler.fit_transform(dataset)
train_data, test_data = train_test_split(Dataset, test_size=0.2, shuffle=False) # train_test_split함수로 훈련/검증 데이터 분리
#train_test_split(arrays, test_size, train_size, random_state, shuffle, stratify) parameter 별 설명
# arrays : 분할시킬 데이터를 입력 (Python list, Numpy array, Pandas dataframe 등..)
# test_size : 테스트 데이터셋의 비율(float)이나 갯수(int) (default = 0.25)
# train_size : 학습 데이터셋의 비율(float)이나 갯수(int) (default = test_size의 나머지)
# random_state : 데이터 분할시 셔플이 이루어지는데 이를 위한 시드값 (int나 RandomState로 입력)
# shuffle : 셔플여부설정 (default = True)
# stratify : 지정한 Data의 비율을 유지한다. 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진 Binary Set일 때, stratify=Y로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할된다.


print(len(train_data), len(test_data))
print(type(test_data))
#데이터 가공 함수 정의
def create_dataset(dataset, look_back):
    x_data = []
    y_data = []
    for i in range(len(dataset)-look_back): # 전체 일수에서 날의 간격만큼을 뺀 만큼을 for 문 돌림. 왜냐면 3일 이후 데이터를 계산 시 1,2,3일 째 되는 날의 이후부터 싀작하기 때 문 이 당!
        data = dataset[i:(i+look_back),0] # 3일 치를 뽑아 data에 저장
        x_data.append(data) #x_data에 data 3일 치를 저장
        y_data.append(dataset[i+look_back,0]) #y_data에는 3일 이후인 4일째 되는 날의 데이터를 저장
    return np.array(x_data), np.array(y_data)


look_back = 3
x_train, y_train = create_dataset(train_data, look_back)
x_test, y_test = create_dataset(test_data, look_back)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# 현재 데이터의 모습은 3개의 데이터가 85층으로 이루어진 모습
# 1*3의 형태로 85개를 넣어야한당
X_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
X_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
print(X_train.shape, X_test.shape) # 3차원 배열로 변환됨ㅇ

model = Sequential() #각각의 레이어가 선형으로 연결되어 있으므로 Sequential로 설정
model.add(SimpleRNN(3, input_shape=(1,look_back))) #RNN의 기초 기법 사용
model.add(Dense(1,activation="linear")) #최종 예측 값 - 하나의 값(1), linear
model.compile(loss='mse',optimizer='adam') #loss는 오차값 계산(mse가 평균)
model.summary()

model.fit(X_train, y_train, epochs=100,batch_size=1, verbose=1) # 모델 학습 실행

# 완성된 모델의 값(0~1 사이로 즉,범위가 제한된 값으로 정규화Normalization된 상태)은 실제 값으로 변환이 필요하다.
# 이러한 변환은 scaler 를 통해 가능합니 다 ~ !

#예측 ㄱ (정규화된 값으로 예측 될 것임)
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

TrainPredict = scaler.inverse_transform(trainPredict) # 역정규화를 통해 실제 값으로 변환함!!!
Y_train = scaler.inverse_transform([y_train])
TestPredict = scaler.inverse_transform(testPredict)
Y_test = scaler.inverse_transform([y_test])

print(TrainPredict)
print(TestPredict)
