import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #egnore some log error messages
import tensorflow as tf  # terminal에서 pip install tensorflow
import numpy as np
import matplotlib.pyplot as plt # 그래프를 그리기 위한 라이브러리 (pip install matplotlib)

print(tf.__version__) #tensorflow 버전 출력


#Initialization of Tensors
x = tf.constant(4, shape=(1,1), dtype=tf.float32) # specify
x = tf.constant([[1,2,3],[4,5,6]]) # matrix

x= tf.ones((3,3)) # 3 by 3 matrix
x= tf.zeros((3,3))
x = tf.eye(3) # I for the identity matrix # one-Hot coding???
x = tf.random.normal((3,3), mean=0, stddev=1)
x = tf.random.uniform((1,3),minval=0,maxval=1)
x= tf.range(9)
x= tf.range(start=1,limit=10,delta=2)
x= tf.cast(x, dtype=tf.float64) # convert between 2 types
#tf.float(23,4,352), .... tf.int() tf.bool()

#Mathematical Operations
x= tf.constant([1,2,3])
y = tf.constant([9,8,7])

# 덧셈하는 법.
z= tf.add(x,y)
z = x + y

# 뺄셈하는 법.
z = tf.subtract(x,y)
z= x - y

# 나눗셈
z = tf.divide(x,y)
z = x / y

#곱셈
z= tf.multiply(x,y)
z = x * y

#
z = tf.tensordot(x,y,axes=1)

arr = [1,2,3,4,5]
len(arr) # 길이
#type(arr)
arr[1:3] # 배열 slice 하는 법.
arr[1:] #마지막 요소까지 갖고옴

#numpy 배열
narr = np.array([1,2,3,45,6]) # 1차원 배열
darr = np.array([[1,3,5,7,9],[1,4,5,6,7]]) # 2차원 배열
print(darr.shape) #배열의 모양

rshpDarr = darr.reshape(5,2) # 모양이 (2,5) 인 2차원 배열을 (5,2)로 변경
rshpDarr2 = darr.reshape(10,) # 2차원 배열을 1차원으로 변경

r1000 = np.random.rand(1000) # 균일한 무작위값 1000개를 생성
#plt.hist(r1000) # 히스토그램으로 표시
#plt.grid() # 히스토그램을 격자무늬 형태로 표시

rn = np.random.normal(0,1,1000) # 평균이 0, 표준편차가 1인 정규 분포로 무작위 값 3개 생성
plt.hist(rn) # 히스토그램으로 표시
plt.grid() # 히스토그램을 격자무늬 형태로 표시
#plt.show() # 표 보여주기

np.random.seed(0) # 랜덤한 수를 뽑을 때 같아질 수 있게 함??? 기준을 정해줌/???
print(np.random.rand(3)) #똑같이 나옴

#파이썬 반복문 - 배열 사용
five = [1,2,3,4,5]
for i in five :
    print(i)

#배열 없이 range() 함수 사용하여 반복문
for i in range(5) :
    if i > 3 :
        print("삼삼허이")
    else :
        print(i)