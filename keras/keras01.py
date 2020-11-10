from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 정제된 데이터 세팅
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_predic =np.arange(21,26)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 데이터 모델 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x, y, epochs=500)

#4.평가
loss, acc = model.evaluate(x, y, batch_size=2)

print('acc:',acc)
print('loss: ',loss)

print(hist.history['acc'])

y_predict = model.predict(x_predic)
print(y_predict)
for y in y_predict:
    print(round(float(y)))