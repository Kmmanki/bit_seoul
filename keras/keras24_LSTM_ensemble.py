import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, LSTM, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping

#1.data
x1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]
            ,[5,6,7],[6,7,8],[7,8,9,],[8,9,10]
            ,[9,10,11],[10,11,12]
            ,[20,30,40],[30,40,50],[40,50,60]
])
x2 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60]
            ,[50,60,70],[60,70,80],[70,80,90,],[80,90,100]
            ,[90,100,110],[100,110,120]
            ,[2,3,4],[3,4,5],[4,5,6]
])
x1_input = np.array([55,65,75])
x2_input = np.array([65,75,85])

y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1=x1.reshape(13,3)
x2=x2.reshape(13,3)

# #predict
x1_input = x1_input.reshape(1,3)
x2_input = x2_input.reshape(1,3)

# #2. model
model1 = Sequential()
model1.add(LSTM(100, activation='relu', input_shape=(3,1)))
model1.add(Dense(50, activation='relu'))
model1.add(Dense(30, activation='relu'))
model1.add(Dense(30, activation='relu'))
model1.add(Dense(10, activation='relu'))


model2 = Sequential()
model2.add(LSTM(60, activation='relu',input_shape =(3,1)))
model2.add(Dense(10, activation='relu'))
model2.add(Dense(50, activation='relu'))
model2.add(Dense(50, activation='relu'))
model2.add(Dense(10, activation='relu'))
model2.add(Dense(1, activation='relu'))

concat_model = Concatenate()([model1.output, model2.output])

model_concat = Dense(30, activation='relu')(concat_model)
model_concat = Dense(50, activation='relu')(model_concat)
model_concat = Dense(100, activation='relu')(model_concat)
model_concat = Dense(100, activation='relu')(model_concat)
model_concat = Dense(60, activation='relu')(model_concat)
model_concat = Dense(1, activation='linear')(model_concat)

model = Model(inputs=[model1.input, model2.input], outputs=[model_concat])

model.summary()
#Compile
model.compile(loss= 'mse', metrics=['mse'], optimizer='adam')
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min')
model.fit([x1, x2], y, batch_size=3, epochs=1000, verbose=1
, callbacks=[earlyStopping]
)


y_predict = model.predict([x1_input, x2_input])
print(y_predict)

# loss = model.evaluate(x_input, np.array([80]), batch_size=1)
# print(loss)