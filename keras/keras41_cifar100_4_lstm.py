from tensorflow.keras.datasets import cifar100, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test)  = cifar100.load_data()

# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape) #(50000,1) (10000, 1)

#1.데이터 전처리 스케일링, 라벨링

color = 1
if x_train.shape[3] == 3:
    color = 3
    
cols = x_train.shape[1]*x_train.shape[2]*color
x_train = x_train.reshape(x_train.shape[0], cols,1 ).astype('float32')/255.
x_test  = x_test.reshape(x_test.shape[0], cols,1 ).astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델링 Conv2D
model = Sequential()
model.add(LSTM(10, input_shape=(cols,1 )))
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu' ))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
ealystopping = EarlyStopping(monitor='loss',patience=10, mode='auto')
to_hist = TensorBoard(log_dir='grahp',write_graph=True, write_images=True, histogram_freq=0)
model.fit(x_train, y_train, epochs=200, batch_size=64, validation_split=0.2, callbacks=[ealystopping, to_hist])

loss, acc=model.evaluate(x_test, y_test, batch_size=64)


x_predict = x_test[20:30]
y_answer = y_test[20:30]


y_predict = model.predict(x_predict)

y_predict = np.argmax(y_predict, axis=1)
y_answer = np.argmax(y_answer, axis=1)

print("acc",acc)
print("loss",loss)
print("정답:   ",y_answer)
print("예상값: ",y_predict)

score = 0
for i in range(10):
    if y_answer[i] == y_predict[i]:
        score +=1

print(score)

'''
cnn(Conv2D)


DNN(Dense)


RNN(LSTM)

'''