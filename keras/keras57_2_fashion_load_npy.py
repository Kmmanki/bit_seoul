from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import numpy as np



path = './data/'

xta = 'x_train.npy'
yta = 'y_train.npy'
xt = 'x_test.npy'
yt = 'y_test.npy'
name = 'fashion'

x_train =np.load(path+name+xta)
y_train =np.load(path+name+yta)
x_test = np.load(path+name+xt)
y_test = np.load(path+name+yt)


#1.데이터 전처리 스케일링, 라벨링
color = 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2], color ).astype('float32')/255.
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], color ).astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델링 Conv2D








path = './save/fashion/modelSave'
path2 = './save/fashion/'

####1. loadmodel
model1 = load_model(path+'.h5')
loss, acc=model1.evaluate(x_test, y_test, batch_size=64)
print("model1 loss: ", loss)
print("model2 acc: ", acc)

####2. loadCheckPoint
model2 = load_model(path2+"keras49_cp_2_fashion_cnn05--0.4472.hdf5") # 계속 수정
loss, acc=model2.evaluate(x_test, y_test, batch_size=64)
print("model2 loss: ", loss)
print("model2 acc: ", acc)

####3. loadweight
model3 = Sequential()
model3.add(Conv2D(100, (2,2), input_shape=(28,28,color)))
model3.add(Dropout(0.1))
model3.add(Flatten())
model3.add(Dense(200))
model3.add(Dense(100))
model3.add(Dense(10, activation='softmax'))

model3.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

model3.load_weights(path+"_weight.h5")

loss, acc = model3.evaluate(x_test, y_test, batch_size=64)
print("model3 loss: ", loss)
print("model3 acc: ", acc)