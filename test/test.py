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

x1=x1.reshape(13,3,1)
x2=x2.reshape(13,3,1)

# #2. model
input1 = Input(shape=(3,1))
lstm_d = LSTM(5)(input1)
input1_d = LSTM(5)(lstm_d)

input2 = Input(shape=(3,1))
lstm_d2 = LSTM(5)(input2)
input2_d = LSTM(5)(lstm_d2)

concat_model = Concatenate()([input1_d, input2_d])
model_concat = Dense(1)(concat_model)

model = Model(inputs=[input1, input2], outputs=[model_concat])


model.summary()
#Compile
model.compile(loss= 'mse', metrics=['mse'], optimizer='adam')
earlyStopping = EarlyStopping(monitor='loss', patience=125, mode='min')
model.fit([x1, x2], y, batch_size=3, epochs=10000, verbose=1, callbacks=[earlyStopping])

# #predict
x1_input = x1_input.reshape(1,3)
x2_input = x2_input.reshape(1,3)

y_predict = model.predict([x1_input, x2_input])
print(y_predict)

# loss = model.evaluate(x_input, np.array([80]), batch_size=1)
# print(loss)