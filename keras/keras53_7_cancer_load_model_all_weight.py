from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Flatten
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print(x.shape) #569,30
print(y.shape) #569
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)


path = './save/cancer/modelSave'
path2 = './save/cancer/'


####1. loadmodel
model1 = load_model(path+'.h5')
loss, acc=model1.evaluate(x_test, y_test, batch_size=64)
print("model1 loss: ", loss)
print("model2 acc: ", acc)

####2. loadCheckPoint
model2 = load_model(path2+"36keras49_cp_7_caner_dnn_bi--0.1419.hdf5") # 계속 수정
loss, acc=model2.evaluate(x_test, y_test, batch_size=64)
print("model2 loss: ", loss)
print("model2 acc: ", acc)

####3. loadweight

model3 = Sequential()
model3.add(Dense(10, activation='relu', input_shape=(30,)))
model3.add(Dense(20, activation='relu'))
model3.add(Dense(30, activation='relu'))
model3.add(Dense(20, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))

model3.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')

model3.load_weights(path+"_weight.h5")

loss, acc = model3.evaluate(x_test, y_test, batch_size=64)
print("model3 loss: ", loss)
print("model3 acc: ", acc)



#