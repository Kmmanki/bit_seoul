import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#2.model

model = Sequential()
model.add(LSTM(100, input_shape=(4,1)))
model.add(Dense(50, name='quen1'))
model.add(Dense(10, name='quen2'))

model.add(Dense(1, name='quen3'))

model.summary()
#VSCODE 파일경로의 디폴트는 열고 있는 프로젝트의 최상단이다. \n 은 개행이기 때문에 ]\\
model.save("./save/keras28.h5")
# model.save(".\save\keras28_2.h5")
# model.save(".//save//keras28_3.h5")
# model.save(".\\save\\keras28._4.h5")