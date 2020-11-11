import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential ,Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split


x = np.array([range(1, 101), range(311, 411), range(100)])
y = np.array([range(101, 201)])

x = x.T
y= y.T

x_train, x_test, y_train, y_test = train_test_split( x,y, train_size= 0.6, shuffle=True)



input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1) 
dense2 = Dense(4, activation='relu')(dense1) 
dense3 = Dense(3, activation='relu')(dense2) 
output1 = Dense(1)(dense3) # 선형 회귀에서 마지막에는 linear . 
model = Model(inputs= input1, outputs= output1)
# 활성화 함수 각레이어마다 가중치가 있다. 너무 크거나 너무 작은 값을 정렬해준다. Dense에서는 default  =linear 

# model = Sequential()
# model.add(Dense(5, input_shape=(3, ), activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(1))

model.summary()

# model.compile(loss='mse', metrics=['mse'], optimizer='adam')
# model.fit(x_train, y_train, batch_size=8, epochs=200, validation_split=0.2, verbose=2)

# loss, acc =model.evaluate(x_test, y_test, batch_size= 8)

# y_predict = model.predict(x_test)



# #-------------------------------------------------------------------------------------------
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# def R2(y_test, y_predict):
#     return r2_score(y_test,y_predict)

# rmse = RMSE(y_test, y_predict)
# r2 = R2(y_test, y_predict)

# print("\n")
# print("loss: ",loss)
# print("RMSE:",rmse)
# print("R2:",r2)
