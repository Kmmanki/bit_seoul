from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test)  = cifar10.load_data()

row = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
x_train = x_train.reshape(x_train.shape[0], row, 1).astype('float32')/255.
x_test  =x_test.reshape(x_test.shape[0], row, 1).astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(LSTM((10), input_shape=(row,1)))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
ealystopping = EarlyStopping(monitor='loss',patience=20, mode='auto')
to_hist = TensorBoard(log_dir='grahp',write_graph=True, write_images=True, histogram_freq=0)
model.fit(x_train, y_train, epochs=300, batch_size=512, validation_split=0.2, callbacks=[ealystopping, to_hist])

loss, acc=model.evaluate(x_test, y_test, batch_size=512)

x_predict = x_test[20:30]
y_answer = y_test[20:30]


y_predict = model.predict(x_predict)

y_predict = np.argmax(y_predict, axis=1)
y_answer = np.argmax(y_answer, axis=1)

print("acc",acc)
print("loss",loss)
print("정답",y_answer)
print("예상값",y_predict)

'''
DNN(Dense)
acc 0.398499995470047
loss 1.7589677572250366
적중률: 30%
정답   [7 0 4 9 5 2 4 0 9 6]
예상값 [4 0 0 1 4 6 6 0 1 6]

RNN(LSTM)
acc 0.1995999962091446
loss 2.1863977909088135
적중률: 10%
정답   [7 0 4 9 5 2 4 0 9 6]
예상값 [4 9 8 4 7 4 6 6 7 6]

CNN(Conv2D)
acc 0.3303999900817871
loss 2.2403793334960938
20%
정답   [7 0 4 9 5 2 4 0 9 6]
예상값 [4 0 0 1 2 4 3 2 1 6]
'''