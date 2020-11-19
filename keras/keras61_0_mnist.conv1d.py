from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Dropout
import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape) 





from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)
print(y_train[0])

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. 
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
print(x_train[0])

x_pred = x_test[-10:]
y_pred = y_test[-10:]


#2. 모델

model = Sequential()
model.add(Conv1D(50, 3, padding="same", input_shape=(28,28,1)))
model.add(Dropout(0.2)) 
model.add(Conv1D(30, 3, padding="same"))
model.add(Dropout(0.2))
model.add(Conv1D(20, 3, padding="same"))
model.add(Dropout(0.2))
model.add(Conv1D(15, 3, padding="same"))
model.add(Dropout(0.2))
model.add(Conv1D(10, 3, padding="same"))
model.add(Dropout(0.2))
model.add(Conv1D(5, 3, padding="same"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()


#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(patience=3,mode='auto',monitor='loss')

model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])


#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

'''
Conv1D
loss :  0.15969528257846832
acc :  0.9550999999046326
'''