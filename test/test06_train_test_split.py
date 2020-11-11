import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#1. 데이터 셋
x = np.array(range(1, 101))
y = np.array(range(1, 101))
x_train, x_test, y_train, y_test = train_test_split(x, y,  shuffle = False, train_size=0.8)
x_train, x_val, y_train, y_val=train_test_split(x_train,  y_train,  train_size=0.8, shuffle = False)

#2.모델링
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(1))

#3.컴파일
model.compile(loss='mse', optimizer='adam', metrics=['acc','mae'])

#4.학습
# model.fit(x_train, y_train, validation_split=0.2, batch_size=1)
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), batch_size=1)

#5 평가
loss, acc, mae = model.evaluate(x_test, y_test, batch_size=1)
#5-1 RMSE
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

#5-2 R2
def R2(y_test, y_predict):
    return r2_score(y_test, y_predict)

print("R2: ",R2(y_test, y_predict))
print("RMSE",RMSE(y_test, y_predict))
print(y_predict)