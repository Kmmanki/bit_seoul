from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np

#1. 정제된 데이터 세팅

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])


#2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim=1, activation='relu'))
model.add(Dense(500))
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1))

#3. 데이터 모델 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x, y, epochs=100, batch_size=2)

#4평가
loss, acc = model.evaluate(x, y, batch_size=2)

print('acc:',acc)
print('loss: ',loss)

y_predict = model.predict(x)

print(y_predict)
for y in y_predict:
    print(round(float(y)))

