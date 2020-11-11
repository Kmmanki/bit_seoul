import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, concatenate
from sklearn.model_selection import train_test_split

x1 = np.array([range(1, 101), range(311, 411), range(100)])
x2 = np.array([range(4, 104), range(761, 861), range(100)])

y1 = np.array((range(101, 201), range(711, 811), range(100)))

x1 = x1.T
x2 = x2.T
y1= y1.T

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split( x1,x2,y1, train_size= 0.8, shuffle=True)

print(x2_train)
print(x2_test)

###################################################################모델1 시작
m1_input = Input(shape=(3,),name='model1_input')
m1_dense1 = Dense(32, activation='relu', name='m1_d1')(m1_input) 
# m1_output = Dense(32, name='m1_out2')(m1_dense1) 

m2_input = Input(shape=(3,))
m2_dense1 = Dense(64, activation='relu', name='m2_d1')(m2_input) 
# m2_output = Dense(16, name='m2_out2')(m2_dense1) 

# #model의 병합
merge1 =  Concatenate()([m1_dense1, m2_dense1])
# merge1 =  Concatenate(axis=1)([m1_output, m2_output])
# merge1 =  Concatenate(axis=0)([m1_output, m2_output])

# #output 모델
output1 = Dense(32, name='out1_d1', activation='relu')(merge1)
output1 = Dense(16, name='out1_d1', activation='relu')(merge1)
output1 = Dense(3, name='out1_d4', activation='linear')(output1)
model1 = Model(inputs=[m1_input, m2_input], outputs = [output1])

# #모델 정의############################################################모델1 끝 모델2 시작
model = Model(inputs=[m1_input, m2_input], outputs = [output1])

m1_input = Input(shape=(3,),name='model1_input')
m1_dense1 = Dense(32, activation='relu', name='m1_d1')(m1_input) 
# m1_output = Dense(32, name='m1_out2')(m1_dense1) 

m2_input = Input(shape=(3,))
m2_dense1 = Dense(64, activation='relu', name='m2_d1')(m2_input) 
# m2_output = Dense(16, name='m2_out2')(m2_dense1) 

# #model의 병합
merge1 =  Concatenate()([m1_dense1, m2_dense1])
# merge1 =  Concatenate(axis=1)([m1_output, m2_output])
# merge1 =  Concatenate(axis=0)([m1_output, m2_output])

# #output 모델
output1 = Dense(32, name='out1_d1', activation='relu')(merge1)
output1 = Dense(8, name='out1_d1', activation='relu')(merge1)
output1 = Dense(3, name='out1_d4', activation='linear')(output1)
model2 = Model(inputs=[m1_input, m2_input], outputs = [output1])
###############################################################################모델2 끝
# #모델 정의



model1_r2s = []
model1_rmses = []
model2_r2s = []
model2_rmses = []

model1.summary()
model2.summary()

# #y3_train 추가
model1.compile(loss='mse', optimizer= 'adam', metrics=['mse'])
model2.compile(loss='mse', optimizer= 'adam', metrics=['mse'])

for i in range(100):
    model1.fit([x1_train, x2_train],[y1_train], epochs=100, batch_size=1, validation_split=0.2, verbose=0)
    model2.fit([x1_train, x2_train],[y1_train], epochs=100, batch_size=1, validation_split=0.2, verbose=0)

    result = model1.evaluate([x1_test, x2_test],[y1_test], batch_size =1)
    result = model2.evaluate([x1_test, x2_test],[y1_test], batch_size =1)

    y_predict_model1 = model1.predict([x1_test,x2_test])
    y_predict_model2 = model2.predict([x1_test,x2_test])

    def RMSE(y_test,y_predict):
        return np.sqrt(mean_squared_error(y_test,y_predict))

    def R2(y_test,y_predict):
        return r2_score(y_test,y_predict)

    print('==============================================================='+str(i)+"번 학습========================================")
    
    model1_r2s.append(R2(y1_test, y_predict_model1))
    model1_rmses.append(RMSE(y1_test, y_predict_model1))

    model2_r2s.append(R2(y1_test, y_predict_model2))
    model2_rmses.append(RMSE(y1_test, y_predict_model2))

print("model1=======================================")
print("model1_r2s",np.mean(model1_r2s))
print("model1_rmses: ",np.mean(model1_rmses))
print("model1_min_r2:", np.min(model1_r2s))
print("\nmodel2=======================================")
print("model2_r2s",np.mean(model2_r2s))
print("model2_rmses: ",np.mean(model2_rmses))
print("model2_min_r2:", np.min(model2_r2s))


