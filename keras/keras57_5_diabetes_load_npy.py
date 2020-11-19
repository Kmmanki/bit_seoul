from sklearn.datasets import load_boston,load_diabetes
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential,load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import numpy as np

path = './data/'
name = 'diabetes'

x =np.load(path+name+'x.npy')
y =np.load(path+name+'y.npy')


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)



path = './save/diabetes/modelSave'
path2 = './save/diabetes/'

####1. loadmodel
model1 = load_model(path+'.h5')
loss, acc=model1.evaluate(x_test, y_test, batch_size=64)
print("model1 loss: ", loss)
print("model2 acc: ", acc)

####2. loadCheckPoint
model2 = load_model(path2+"keras49_cp_5_diabetes_dnn48--3292.1221.hdf5") # 계속 수정
loss, acc=model2.evaluate(x_test, y_test, batch_size=64)
print("model2 loss: ", loss)
print("model2 acc: ", acc)

####3. loadweight

model3 = Sequential() #r2 = 
model3.add(Dense(10, activation='relu', input_shape=(10,)))
model3.add(Dense(300, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(200, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(30, activation='relu'))
model3.add(Dense(1))

model3.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

model3.load_weights(path+"_weight.h5")

loss, acc = model3.evaluate(x_test, y_test, batch_size=64)
print("model3 loss: ", loss)
print("model3 acc: ", acc)

















