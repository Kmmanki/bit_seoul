#최적의 튠으로 구성하시오
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test)  = cifar10.load_data()

x_train = x_train.astype('float32')/255. ## 10000,32,32,3
x_test  =x_test.astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3,3), input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (3,3), input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(126))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
ealystopping = EarlyStopping(monitor='loss',patience=20, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3,
                             factor=0.5, verbose=1)

model.fit(x_train, y_train, epochs=300, batch_size=512, validation_split=0.2, callbacks=[ealystopping, reduce_lr])

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