
from tensorflow.keras.datasets import cifar100, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import numpy as np

(x_train, y_train), (x_test, y_test)  = cifar100.load_data()

color = 1
if x_train.shape[3] == 3:
    color = 3
    
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], color ).astype('float32')/255.
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], color ).astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델링 Conv2D



path = './save/cifar100/modelSave'
path2 = './save/cifar100/'

####1. loadmodel
model1 = load_model(path+'.h5')
loss, acc=model1.evaluate(x_test, y_test, batch_size=64)
print("model1 loss: ", loss)
print("model2 acc: ", acc)

####2. loadCheckPoint
model2 = load_model(path2+"keras49_cp_3_cifar100-cnn03--2.8933.hdf5") # 계속 수정
loss, acc=model2.evaluate(x_test, y_test, batch_size=64)
print("model2 loss: ", loss)
print("model2 acc: ", acc)

####3. loadweight

model3 = Sequential()
model3.add(Conv2D(100, (2,2), input_shape=(x_test.shape[1],x_test.shape[2],color)))
model3.add(Conv2D(150, (2,2)))
model3.add(MaxPooling2D())
model3.add(Conv2D(150, (2,2)))
model3.add(Conv2D(100, (2,2)))
model3.add(Flatten())
model3.add(Dense(50))
model3.add(Dense(80))
model3.add(Dense(100, activation='softmax'))

model3.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')


model3.load_weights(path+"_weight.h5")

loss, acc = model3.evaluate(x_test, y_test, batch_size=64)
print("model3 loss: ", loss)
print("model3 acc: ", acc)