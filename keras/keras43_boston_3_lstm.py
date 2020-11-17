from sklearn.datasets import load_boston,load_diabetes
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D, Flatten, LSTM
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
x = x.reshape(506,13,1)
print(x.shape) #442,10


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

print(x_train.shape) #353,10,,1
print(y_train.shape) #442


model = Sequential() #r2 =
model.add(LSTM(30, input_shape=(13,1)))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(500))
model.add(Dropout(0.2))
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
print("keras43 lstm")

'''
r2 0.8이상
loss:  23.120969772338867
RMSE 5.777079742293692
R2:  0.8066005652896088
정답:  [23.7 30.8 50.  50.  34.9 18.2 12.7 15.6 22.8 42.8]
예측:  [[27.132082]
 [29.678228]
 [44.111652]
 [40.36204 ]
 [33.17927 ]
 [23.030607]
 [13.887516]
 [19.933538]
 [25.182308]
 [30.92628 ]]
'''