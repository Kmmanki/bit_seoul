#1.데이터
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x= np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])#(4,3)
y= np.array([4,5,6,7])#(4,)


print("x.shape: ",x.shape)

# x s= x.reshape(x.shape[0], x.shape[1], 1 )
# x = x.reshape(4,3)
x = x.reshape(4,3,1)
print("x.shape: ",x.shape)


#2. 모델 구성 
#몇개씩 작업을 할 것인가 즉 1개씩 작업 요소 3개에 1개씩 잘라서 작업 행무시하고 2차원을 받는다. 반드시 3차원으로 존재햐야한다.
#(행, 열, 자르는 크기)

model= Sequential()
model.add(LSTM(32, activation='relu', input_shape=(3,1))) 
# model.add(LSTM(10, input_dim=1, input_length=3)) 
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', metrics=['mse'], optimizer='adam')
model.fit(x,y, batch_size=1, verbose=1, epochs=100, validation_split=0.2)

x_input = np.array([5,6,7])
x_input = x_input.reshape(1,3,1) #왜 131? 1개의 데이터는 3개의 요소를 가지며 1개씩 자른다.
# x_input = x_input.reshape(1,3) #왜 131? 1개의 데이터는 3개의 요소를 가지며 1개씩 자른다.

y_predict = model.predict(x_input)
print(y_predict)


def R2(y_test, y_precit):
    return r2_score(y_test, y_precit)

# print(R2(y_test, y_predict))