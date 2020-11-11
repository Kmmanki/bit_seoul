import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
#1. data
x = np.array(range(1,101))
y = np.array((range(101, 201), range(711, 811), range(100)))

x = x.T
y= y.T

x_train, x_test, y_train, y_test = train_test_split( x,y, train_size= 0.6, shuffle=True)

#2.모델 
model = Sequential()
model.add(Dense(100 ,input_dim =1 ))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(3))

model.compile(loss='mse', metrics=['acc'], optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.2)

loss, acc =model.evaluate(x_test, y_test, batch_size= 1)

y_predict = model.predict(x_test)

#-------------------------------------------------------------------------------------------
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

def R2(y_test, y_predict):
    return r2_score(y_test,y_predict)

rmse = RMSE(y_test, y_predict)
r2 = R2(y_test, y_predict)

print("loss: ",loss)
print("RMSE:",rmse)
print("R2:",r2)
