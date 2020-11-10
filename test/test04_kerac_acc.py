from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

x_predic =np.arange(21,26)
# print(x_train)
# print(x_test)

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', metrics=['accuracy'], optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=2)

loss, acc = model.evaluate(x_test, y_test, batch_size=2)
print("acc: ",acc)
print("loss: ",loss)

y_predict = model.predict(x_predic)
print(y_predict)
for y in y_predict:
    print(round(float(y)))

    