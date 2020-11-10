from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np

#1. 정제된 데이터 세팅
# x = np.array([1.0/10, 2.0/10, 3.0/10, 4.0/10, 5.0/10])
# y = np.array([1.0/10, 2.0/10, 3.0/10, 4.0/10, 5.0/10])

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.arange(11,21)
y_test = np.arange(11,21)

x_predic =np.arange(21,26)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 데이터 모델 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x, y, epochs=500, batch_size=2)

loss, acc = model.evaluate(x_test, y_test, batch_size=2)

print('acc:',acc)
print('loss: ',loss)

print(hist.history['acc'])
#무엇 때문에 acc 가 0.2 인가

# plt.scatter(hist.history['acc'], hist.history['loss']) 
# plt.title("..")
# plt.xlabel("acc")
# plt.ylabel("loss")
# plt.axis([-1, 2, -1, 2])
# plt.show() 

y_predict = model.predict(x_predic)
print(y_predict)
for y in y_predict:
    print(round(float(y)))