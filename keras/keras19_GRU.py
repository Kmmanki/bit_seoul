#1.데이터
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x= np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])#(4,3)
y= np.array([4,5,6,7])#(4,)


print("x.shape: ",x.shape)

x = x.reshape(4,3,1)
print("x.shape: ",x.shape)


#2. 모델 구성 

model= Sequential()
model.add(GRU(16, activation='relu', input_shape=(3,1))) 
model.add(Dense(16))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', metrics=['mse'], optimizer='adam')
model.fit(x,y, batch_size=1, verbose=1, epochs=100)

x_input = np.array([5,6,7])
x_input = x_input.reshape(1,3,1) 

y_predict = model.predict(x_input)
print(y_predict)

