from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D,Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test)  = cifar10.load_data()

x_train = x_train.astype('float32')/255. ## 10000,32,32,3
x_test  =x_test.astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


path = './save/cifar10/modelSave'
path2 = './save/cifar10/'

####1. loadmodel
model1 = load_model(path+'.h5')
loss, acc=model1.evaluate(x_test, y_test, batch_size=64)
print("model1 loss: ", loss)
print("model2 acc: ", acc)

####2. loadCheckPoint
model2 = load_model(path2+"cp_cifar10-04--1.5607.hdf5")
loss, acc=model2.evaluate(x_test, y_test, batch_size=64)
print("model2 loss: ", loss)
print("model2 acc: ", acc)

####3. loadweight
model3 = Sequential()
model3.add(Conv2D(100, (2,2), input_shape=(32,32,3)))
model3.add(Flatten())
model3.add(Dense(100, activation='relu'))
model3.add(Dense(10, activation='softmax'))

model3.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

model3.load_weights(path+"_weight.h5")

loss, acc = model3.evaluate(x_test, y_test, batch_size=64)
print("model3 loss: ", loss)
print("model3 acc: ", acc)