from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
y_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
# x_val = np.array([11,12,13,14,15])
# y_val = np.array([11,12,13,14,15])
# x_pred = np.array([16,17,18])
x_test = np.array([16,17,18,19,20])
y_test = np.array([16,17,18,19,20])

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam'
, metrics=['mae','acc']
)
model.fit(x_train, y_train, epochs=100, batch_size=1
# , validation_data=(x_val,y_val) ) 
, validation_split=0.2) # train의 20%를 검증용으로 사용한다.

loss = model.evaluate(x_test, y_test, batch_size=1)


y_predict = model.predict(x_test)
print(y_predict)


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE: ",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2: ",r2)