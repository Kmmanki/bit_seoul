from sklearn.datasets import load_boston,load_diabetes
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import mean_squared_error, r2_score

# import shutil
# shutil.rmtree(r"D:\Study\graph")

import numpy as np

dataset = load_boston()

x = dataset.data
y = dataset.target

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print(x.shape) #506,13
print(y.shape) #506

x = x.reshape(506,13,1,1)
print(x.shape) #506,13,1,1

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

print(x_train.shape) #404,13,1,1
print(y_train.shape) #404

model = Sequential() #r2 =
model.add(Conv2D(30, (2,2), padding='same',  input_shape=(13,1,1)))     # 4, 1, 30
# valid 의 경우 
# 자르게 되면 행으로 4번 열로 1번 자르게 됨 -> (4,1,30(노드수))
model.add(Conv2D(100, (2,2), padding='same', activation='relu'))
#4행 1열을 2,2로 자르려고 하니 padding 없이 자를 수 없다!!!
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=[])
ealystopping = EarlyStopping(monitor='loss', patience=30, mode='min')

to_hist = TensorBoard(log_dir='graph', write_graph=True, write_images=True, histogram_freq=0)
model.fit(x_train, y_train, epochs=500, callbacks=[ealystopping, to_hist], verbose=1, validation_split=0.2, batch_size=4)

#평가
loss = model.evaluate(x_train, y_train, batch_size=4)

#예측
x_predict = x_test[30:40]
y_answer = y_test[30:40]

y_predict = model.predict(x_predict)


def RMSE(y_answer, y_predict):
    return np.sqrt(mean_squared_error(y_answer, y_predict))

def R2(y_answer, y_predict):
    return r2_score(y_answer, y_predict)

#결과
print("loss: ",loss)
print("RMSE", RMSE(y_answer, y_predict))
print("R2: ", R2(y_answer, y_predict) )
print("정답: ", y_answer.reshape(10,))
print("예측: ", y_predict)
print("keras43 cnn")

'''
loss:  9.247981071472168
RMSE 2.4538836028479074
R2:  0.92965559081358
정답:  [24.8 19.3 19.9  8.7 19.2 23.9 12.8 13.2 44.  16.1]
예측:  [[23.457638]
 [19.322706]
 [19.161924]
 [ 7.818993]
 [22.493671]
 [23.926142]
 [11.77245 ]
 [ 8.968629]
 [39.86415 ]
 [19.29022 ]]
 keras43 cnn
'''
