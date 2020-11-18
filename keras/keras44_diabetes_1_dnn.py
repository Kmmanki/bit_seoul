from sklearn.datasets import load_boston,load_diabetes
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


import numpy as np

dataset = load_diabetes()

x = dataset.data
y = dataset.target

print(x.shape) #442,10
print(y.shape) #442


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)


model = Sequential() #r2 = 
model.add(Dense(10, activation='relu', input_shape=(10,)))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=[])
ealystopping = EarlyStopping(monitor='loss', patience=100, mode='min')

to_hist = TensorBoard(log_dir='graph', write_graph=True, write_images=True, histogram_freq=0)
hist=model.fit(x_train, y_train, epochs=1000, callbacks=[ealystopping, to_hist], verbose=1, validation_split=0.2, batch_size=2)

#평가
loss = model.evaluate(x_test, y_test, batch_size=2)

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
print("keras44_dnn")

print(hist)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('loss & va_loss')
plt.ylabel("loss, val_loss")
plt.xlabel('epoches')
plt.legend(['loss', 'val_loss'])



plt.show()

import shutil
shutil.rmtree(r"D:\Study\graph")
'''
r2 0.8이상
loss:  1467.665283203125
RMSE 33.018933909118225
R2:  0.8381602000275039
정답:  [ 77.  64.  90. 118.  88. 265.  67. 252. 258.  71.]
예측:  [[ 64.03085]
 [ 84.54816]
 [ 85.9801 ]
 [ 81.85154]
 [ 88.46721]
 [179.01033]
 [100.3627 ]
 [237.30905]
 [274.30518]
 [ 71.09758]]

'''