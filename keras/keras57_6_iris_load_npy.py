import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

#라벨링 1개 x컬럼3
path = './data/'
name = 'iris'

x =np.load(path+name+'x.npy')
y =np.load(path+name+'y.npy')

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x= x.reshape(x.shape[0], x.shape[1], 1,1  )

print(x.shape) #150,4,1,1
print(y.shape) #150,3 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
print(x_train.shape,x_test.shape) #120,4 /30,4
print(y_train.shape,y_test.shape) #120,3 /30,3

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)




path = './save/iris/modelSave'
path2 = './save/iris/'

####1. loadmodel
model1 = load_model(path+'.h5')
loss, acc=model1.evaluate(x_test, y_test, batch_size=64)
print("model1 loss: ", loss)
print("model2 acc: ", acc)

####2. loadCheckPoint
model2 = load_model(path2+"keras49_cp_6_iris_cnn114--0.1269.hdf5") # 계속 수정
loss, acc=model2.evaluate(x_test, y_test, batch_size=64)
print("model2 loss: ", loss)
print("model2 acc: ", acc)

####3. loadweight

model3 = Sequential() #r2 =0.8
model3.add(Conv2D(20, (2,2), padding='same', activation='relu', input_shape=(4,1,1)))
model3.add(Dropout(0.5))
model3.add(Flatten())
model3.add(Dense(300, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(100, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(3, activation='softmax'))

model3.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

model3.load_weights(path+"_weight.h5")

loss, acc = model3.evaluate(x_test, y_test, batch_size=64)
print("model3 loss: ", loss)
print("model3 acc: ", acc)