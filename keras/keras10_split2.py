#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))

#f(x) = x+ 100 #w = 1 b =100

x_train = np.array(x[:60])
y_train = np.array(y[:60])
x_val = np.array(x[60:80])
y_val = np.array(y[60:80])
x_test = np.array(x[80:])
y_test = np.array(y[80:])



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델작성
model = Sequential()
model.add(Dense(1, input_dim =1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3.컴파일
model.compile(loss='mse', metrics=['acc'], optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.2)
loss =model.evaluate(x_test, y_test, batch_size=1)

# 여긴 y_predic이 필요 
from sklearn.metrics import mean_squared_error
def RMS(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score
def R2(y_test, y_predict):
    return r2_score(y_test,y_predict)