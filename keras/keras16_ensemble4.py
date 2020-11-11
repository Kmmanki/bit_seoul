import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, concatenate
from sklearn.model_selection import train_test_split

x1 = np.array([range(1, 101), range(311, 411), range(100)])


y1 = np.array((range(101, 201), range(711, 811), range(100)))
y2 = np.array((range(501, 601), range(431, 531), range(100, 200)))
y3 = np.array((range(601, 701), range(531, 631), range(300, 400)))


x1 = x1.T
y1= y1.T
y2 = y2.T
y3 = y3.T


x1_train, x1_test, y1_train, y1_test = train_test_split( x1,y1, train_size= 0.6, shuffle=True)
y2_train, y2_test, y3_train, y3_test = train_test_split( y2,y3, train_size= 0.6, shuffle=True)



m1_input = Input(shape=(3,),name='model1_input')
m1_dense1 = Dense(300, activation='relu', name='m1_d1')(m1_input) 
m1_dense2 = Dense(200, activation='relu', name='m1_d2')(m1_dense1) 
m1_dense3 = Dense(100, activation='relu', name='m1_d3')(m1_dense2) 
m1_output = Dense(80, name='m1_out')(m1_dense3) 

#output 모델
output1 = Dense(60, name='out1_d1')(m1_output)
output1 = Dense(30, name='out1_d2')(output1)
output1 = Dense(10, name='out1_d3')(output1)
output1 = Dense(3, name='out1_d4')(output1)

output2 = Dense(60, name='out2_d1')(m1_output)
output2 = Dense(30, name='out2_d2')(output2)
output2 = Dense(10, name='out2_d3')(output2)
output2 = Dense(3, name='out2_d4')(output2)

output3 = Dense(60, name='out3_d1')(m1_output)
output3 = Dense(30, name='out3_d2')(output3)
output3 = Dense(10, name='out3_d3')(output3)
output3 = Dense(3, name='out3_d4')(output3)

#모델 정의
model = Model(inputs=[m1_input], outputs = [output1, output2, output3])
model.summary()

# #y3_train 추가
model.compile(loss='mse', optimizer= 'adam', metrics=['mse'])
model.fit([x1_train],[y1_train, y2_train, y3_train],
epochs=100, batch_size=8, validation_split=0.25, verbose=2)

result = model.evaluate([x1_test],[y1_test, y2_test, y3_test], batch_size = 8)

y1_predict, y2_predict, y3_predict,  = model.predict([x1_test])

print()
print("total_loss: ",result[0])
print("out1_loss: ",result[1])
print("out1_metrics: ",result[2])
print("out2_loss: ",result[3])
print("out2_metrics: ",result[4])
print("out3_loss: ",result[5])
print("out3_metrics: ",result[6])

print('==========================================================')

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

def R2(y_test,y_predict):
    return r2_score(y_test,y_predict)

print("y1_test RMSE: ", RMSE(y1_test, y1_predict))
print("y1_test R2: ", R2(y1_test, y1_predict))
print('==========================================================')
print("y2_test RMSE: ", RMSE(y2_test, y2_predict))
print("y2_test R2: ", R2(y2_test, y2_predict))
print('==========================================================')
print("y3_test RMSE: ", RMSE(y3_test, y3_predict))
print("y3_test R2: ", R2(y3_test, y3_predict))