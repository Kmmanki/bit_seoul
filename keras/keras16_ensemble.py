import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, concatenate
from sklearn.model_selection import train_test_split

x1 = np.array([range(1, 101), range(311, 411), range(100)])
y1 = np.array((range(101, 201), range(711, 811), range(100)))

print(x1.shape)
print(y1.shape)

x1 = x1.T#(100,3)
y1= y1.T#(100,3)

print(x1.shape)
print(y1.shape)

x2 = np.array([range(4, 104), range(761, 861), range(100)])
y2 = np.array((range(501, 601), range(431, 531), range(100, 200)))



print('-----------------------')

print(x2.shape)
print(y2.shape)

x2 = x2.T
y2 = y2.T

print(x2.shape)
print(y2.shape)

x1_train, x1_test, y1_train, y1_test = train_test_split( x1,y1, train_size= 0.6, shuffle=True)
x2_train, x2_test, y2_train, y2_test = train_test_split( x2,y2, train_size= 0.6, shuffle=True)


m1_input = Input(shape=(3,),name='model1_input')
m1_dense1 = Dense(5, activation='relu', name='m1_d1')(m1_input) 
m1_dense2 = Dense(4, activation='relu', name='m1_d2')(m1_dense1) 
m1_dense3 = Dense(3, activation='relu', name='m1_d3')(m1_dense2) 
m1_output = Dense(3, name='m1_out')(m1_dense3) 

m2_input = Input(shape=(3,))
m2_dense1 = Dense(5, activation='relu', name='m2_d1')(m2_input) 
m2_dense2 = Dense(4, activation='relu', name='m2_d2')(m2_dense1) 
m2_dense3 = Dense(3, activation='relu', name='m2_d3')(m2_dense2) 
m2_output = Dense(3, name='m2_out')(m2_dense3) 



#model의 병합
merge1 =  Concatenate()([m1_output, m2_output])
# merge1 =  Concatenate(axis=1)([m1_output, m2_output])
# merge1 =  Concatenate(axis=0)([m1_output, m2_output])

middle1 = Dense(30, name='middle_d1')(merge1)
middle2 = Dense(7, name='middle_d2')(middle1)
middle3 = Dense(11, name='middle_d3')(middle2)

#output 모델
    #두 output 모델 모두 middle3에서 나오기 때문에 2개의 outputmodel 필요
output1 = Dense(30, name='out1_d1')(middle3)
output1 = Dense(7, name='out1_d2')(output1)
output1 = Dense(3, name='out1_d3')(output1)


output2 = Dense(30, name='out2_d1')(middle3)
output2 = Dense(7, name='out2_d2')(output2)
output2 = Dense(3, name='out2_d3')(output2)

#모델 정의
model = Model(inputs=[m1_input, m2_input], outputs = [output1, output2])


model.summary()

model.compile(loss='mse', optimizer= 'adam', metrics=['mse'])
model.fit([x1_train, x2_train],[y1_train, y2_train],
epochs=100, batch_size=8, validation_split=0.25, verbose=1)

#첫번 째 전체 로스 output1 loss metrics, output2의 loss metrics
result = model.evaluate([x1_test, x2_test],[y1_test, y2_test], batch_size = 8)

print(result)