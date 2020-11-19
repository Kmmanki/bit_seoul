#CNN레이어, 다중분류 모델(softmax)
from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import numpy as np

(x_train, y_train), (x_test, y_test)  = fashion_mnist.load_data()

#1.데이터 전처리 스케일링, 라벨링
color = 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2], color ).astype('float32')/255.
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], color ).astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델링 Conv2D
model = Sequential()
model.add(Conv1D(100, 2, input_shape=(28,28,color)))
model.add(Dropout(0.1))
model.add(Conv1D(100, 2, padding='same'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
ealystopping = EarlyStopping(monitor='loss',patience=10, mode='auto')
to_hist = TensorBoard(log_dir='grahp',write_graph=True, write_images=True, histogram_freq=0)


hist=model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[ealystopping])

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

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6) ) # 단위 찾아보기

plt.subplot(2, 1, 1) #2행 1열의 첫 번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss' )
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss' )
plt.grid()

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')


plt.subplot(2, 1, 2) #2행 2열의 첫 번째
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()

plt.title('accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])
plt.show()

'''

conv1D
acc 0.8367000222206116
loss 0.4879576563835144
정답:    [2 5 7 9 1 4 6 0 9 3]
예상값:  [2 5 7 9 1 2 6 0 9 4]



'''