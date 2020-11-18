from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

dataset = load_boston()

x = dataset.data
y = dataset.target

print(x.shape) #506,13
print(y.shape) #506

#데이터 전처리 scaler 
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

print(x_train.shape, x_test.shape)#(404, 13) (102, 13)
print(y_train.shape, y_test.shape)#(404,) (102,)


#모델링


path = './save/boston/modelSave'
path2 = './save/boston/'

####1. loadmodel
model1 = load_model(path+'.h5')
loss, mae=model1.evaluate(x_test, y_test, batch_size=64)
print("model1 loss: ", loss)
print("model2 acc: ", mae)

####2. loadCheckPoint
model2 = load_model(path2+"keras49_cp_4_boston94--7.3563.hdf5") # 계속 수정
loss, mae=model2.evaluate(x_test, y_test, batch_size=64)
print("model2 loss: ", loss)
print("model2 mae: ", mae)

####3. loadweight

model3 = Sequential() #r2 =0.8
model3.add(Dense(30, activation='relu', input_shape=(13,)))
model3.add(Dense(100, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(400, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(300, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(100, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(1))

model3.compile(loss='mse', optimizer='adam', metrics=['mae'])


model3.load_weights(path+"_weight.h5")

loss, mae = model3.evaluate(x_test, y_test, batch_size=64)
print("model3 loss: ", loss)
print("model3 acc: ", mae)  