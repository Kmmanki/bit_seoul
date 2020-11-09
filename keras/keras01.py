from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 정제된 데이터 세팅
# x = np.array([1.0/10, 2.0/10, 3.0/10, 4.0/10, 5.0/10])
# y = np.array([1.0/10, 2.0/10, 3.0/10, 4.0/10, 5.0/10])

x = np.array([1,2,3,4,5], dtype=np.float32)
y = np.array([1,2,3,4,5], dtype=np.float32)


#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 데이터 모델 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x, y, epochs=500, batch_size=2)

loss, acc = model.evaluate(x, y, batch_size=2)

print('acc:',acc)
print('loss: ',loss)

#print(hist.history['acc'])
#무엇 때문에 acc 가 0.2 인가

