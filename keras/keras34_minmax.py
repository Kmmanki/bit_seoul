import numpy as np
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split

#전처리 연산은 부동소수점 시 빨라진다.
#y, 라벨, 타겟은 바꾸지 않는다.
#x를 최댓값으로 나누면 0~1사이의 값이 된다. x의 값이 바뀌어도 0.0001,0.0002 는 3과 매칭이 되기에 문제가 없다. 

x = np.array([
    [1,2,3],[2,3,4],[3,4,5],[4,5,6],
    [5,6,7],[6,7,8],[7,8,9],[8,9,10],
    [9,10,11],[10,11,12]
    ,[2000,3000,4000],[3000,4000,5000],[4000,5000,6000]
    ,[100,200,300]
])
y = np.array([
    4,5,6,7
    ,8,9,10,11,
    12,13,5000,6000,7000,400
])

x_predict = np.array([55,65,75])
x_predict = x_predict.reshape(1,3)
x_predict2 = np.array([6600,6700,6800])
x_predict2 = x_predict.reshape(1,3)

#여기
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
print(scaler.data_max_)
scaler.data_max_
x = scaler.transform(x)
print(x)

x_predict = scaler.transform(x_predict)
print(x_predict)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9)

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(3,)))
model.add(Dense(300, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(1))

model.compile(metrics='mse', loss='mse', optimizer='adam')

elaystopping = EarlyStopping(monitor='loss', patience=125, mode='min')

model.fit(x_train,y_train, batch_size=2, verbose=1, epochs=10000, validation_split=0.1, callbacks=[elaystopping])

loss, mse = model.evaluate(x_test, y_test,batch_size=2)


y_pred = model.predict(x_predict)
y_pred2 = model.predict(x_predict2)

print(x_predict)
print("loss: ", loss)
print("mse: ", mse)
print("y_pred: ", y_pred)
print("y_pred2: ", y_pred2)

