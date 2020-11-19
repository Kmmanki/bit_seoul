from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dropout
import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. 
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

#predict 만들기
x_pred = x_test[-10:]
y_pred = y_test[-10:]

path = './save/mnist/modelSave'
path2 = './save/mnist/'

####1. loadmodel
model1 = load_model(path+'.h5')
loss, mae=model1.evaluate(x_test, y_test, batch_size=64)
print("model1 loss: ", loss)
print("model2 acc: ", mae)

####2. loadCheckPoint
model2 = load_model(path2+"keras49-mnist-13--0.1365.hdf5") # 계속 수정
loss, mae=model2.evaluate(x_test, y_test, batch_size=64)
print("model2 loss: ", loss)
print("model2 mae: ", mae)

####3. loadweight
model3 = Sequential()
model3.add(Conv2D(50, (2,2), padding="same", input_shape=(28,28,1)))
model3.add(Dropout(0.2)) 
model3.add(Conv2D(30, (2,2), padding="same"))
model3.add(Dropout(0.2))
model3.add(Conv2D(20, (2,2), padding="same"))
model3.add(Dropout(0.2))
model3.add(Conv2D(15, (2,2), padding="same"))
model3.add(Dropout(0.2))
model3.add(Conv2D(10, (2,2), padding="same"))
model3.add(Dropout(0.2))
model3.add(Conv2D(5, (2,2), padding="same"))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Dropout(0.2))
model3.add(Flatten())
model3.add(Dense(20, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(10, activation='softmax'))

model3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])



model3.load_weights(path+"_weight.h5")

loss, mae = model3.evaluate(x_test, y_test, batch_size=64)
print("model3 loss: ", loss)
print("model3 acc: ", mae)  