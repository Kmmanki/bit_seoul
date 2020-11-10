from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np


x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

model = Sequential()
model.add(Dense(30, input_dim=1, activation='relu'))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam'
, metrics=['mae','acc']
)
hist = model.fit(x, y, epochs=100 )

loss, acc,mse  = model.evaluate(x, y)

print('acc:',acc)
print('loss: ',loss)

y_predict = model.predict(x)

print(y_predict)


