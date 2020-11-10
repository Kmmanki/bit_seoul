from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))

#f(x) = x+ 100 #w = 1 b =100
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size= 0.7, shuffle=False)
    
print("x_train: ",x_train)
print("x_test: ", x_test)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size = 0.8, shuffle=False)

print("x_train: ",x_train)
print("y_val: ",x_val)

x_predict = np.array(range(201,231))


# exit()

#2. 모델작성
model = Sequential()
model.add(Dense(100, input_dim =1))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(1))

#3.컴파일
model.compile(loss='mse', metrics=['acc'], optimizer='adam')

model.fit(x_train, y_train, batch_size=1, epochs=100, validation_data=(x_val,y_val))

loss =model.evaluate(x_test, y_test, batch_size=1)

y_predict = model.predict(x_test)

# 여긴 y_predic이 필요 

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


def R2(y_test, y_predict):
    return r2_score(y_test,y_predict)

rmse = RMSE(y_test, y_predict)
r2 = R2(y_test, y_predict)

print("loss: ",loss)
print("RMSE:",rmse)
print("R2:",r2)


print(x_test)
print(y_predict)