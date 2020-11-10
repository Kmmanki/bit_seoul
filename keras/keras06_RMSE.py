from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np



#1. 정제된 데이터 세팅
#학습
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
#평가
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
#예측 x값
x_pred = np.array([16,17,18])

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. 데이터 모델 훈련
model.compile(loss='mse', optimizer='adam'
, metrics=['mae','acc']
)
model.fit(x_train, y_train, epochs=100, batch_size=1 ) 

#loss, acc = model.evaluate(x, y)
#4.평가
loss = model.evaluate(x_test, y_test, batch_size=1)

#print('acc:',acc)
print('loss: ',loss)

#새로운 평가지표를 만들기 위해 
y_predict = model.predict(x_test)
print(y_predict)


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# predict의 값과 y의 갑과 비교 한 것을 평가 지표료 사용 
# 즉 훈련을 통해 얻은 결과 값과 원래으 값을 비교(mse로 )
print("RMSE: ",RMSE(y_test,y_predict))

