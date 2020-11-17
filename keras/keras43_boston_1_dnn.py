from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import mean_squared_error, r2_score

import shutil

shutil.rmtree(r"D:\Study\graph")



import numpy as np

dataset = load_boston()

x = dataset.data
y = dataset.target

print(x.shape) #506,13
print(y.shape) #506

#데이터 전처리 scaler 
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

print(x_train.shape, x_test.shape)#(404, 13) (102, 13)
print(y_train.shape, y_test.shape)#(404,) (102,)


#모델링

model = Sequential() #r2 =0.8
model.add(Dense(30, activation='relu', input_shape=(13,)))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
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
'''
r2 0.8이상

loss:  13.658388137817383
RMSE 2.6497556936268225
R2:  0.8747736192967522
정답:  [10.5 16.  21.5 31.7 17.5  8.5  8.4 20.4 22.2  7. ]
예측:  [[ 9.7567425]
 [15.355091 ]
 [17.999798 ]
 [32.273556 ]
 [15.05445  ]
 [ 8.083621 ]
 [ 9.621383 ]
 [16.73284  ]
 [20.664963 ]
 [12.763078 ]]
'''