#DNN 선형회귀 모델(linear)
from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense,Dropout, Conv1D, Flatten
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score

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

x = x.reshape(506,13,1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

print(x_train.shape, x_test.shape)#(404, 13) (102, 13)
print(y_train.shape, y_test.shape)#(404,) (102,)


#모델링

model = Sequential() #r2 =0.8
model.add(Conv1D(30, 3, activation='relu', input_shape=(13,1)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
ealystopping = EarlyStopping(monitor='loss', patience=30, mode='min')

to_hist = TensorBoard(log_dir='graph', write_graph=True, write_images=True, histogram_freq=0)


hist=model.fit(x_train, y_train, epochs=100, callbacks=[ealystopping
], verbose=1, validation_split=0.2, batch_size=4)

#평가
loss = model.evaluate(x_test, y_test, batch_size=4)

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

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red')
plt.plot(hist.history['val_loss'], marker='.', c='blue')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.legend(['loss','val_loss'])

plt.subplot(2,1,2)
plt.plot(hist.history['mae'], marker='.', c='red')
plt.plot(hist.history['val_mae'], marker='.', c='blue')
plt.grid()
plt.title('mae')
plt.ylabel('mae')
plt.legend(['loss','val_mae'])

plt.show()
'''
Conv1D
loss:  [15.370065689086914, 2.698878765106201]
RMSE 4.021386230371479
R2:  0.7578399168944582

'''