import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU ,Input

#1.data
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]
            ,[5,6,7],[6,7,8],[7,8,9,],[8,9,10]
            ,[9,10,11],[10,11,12]
            ,[20,30,40],[30,40,50],[40,50,60]
])

y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("shape: ",x.shape)
x = x.reshape(13,3)



# #2. model
input = Input(shape=(3,1))
lstm = LSTM(200)(input)
dense = Dense(180)(lstm)
dense = Dense(150)(dense)
dense = Dense(110)(dense)
dense = Dense(60)(dense)
dense = Dense(10)(dense)
output = Dense(1)(dense)

model = Model(inputs=[input], outputs=[output])
model.summary()

# #Compile
model.compile(
    loss= 'mse',
     metrics=[],
      optimizer='adam'
      )

from tensorflow.keras.callbacks import EarlyStopping

predicts = []

for i in range(10):


    earlyStopping = EarlyStopping(monitor='loss', patience=125, mode='auto')
    # earlyStopping = EarlyStopping(monitor='loss', patience=50, mode='min')

    loss = model.fit(
        x, y,
        batch_size=1,
        epochs=10000,
        verbose=1,
        callbacks=[earlyStopping]
        )

    # #predict

    x_input = np.array([50,60,70])

    print("x_input: ",x_input.shape)

    x_input = x_input.reshape(1,3)

    loss = model.evaluate(
        x_input, 
        np.array([80]), 
        batch_size=1
        )

    y_predict = model.predict(x_input)

    predicts.append(y_predict[0][0])
    print(y_predict[0][0])
    print(loss)


print(predicts)

# #LSTM  80 = 75.82542 , loss = 17.427146911621094 param = 342,601
# #simpleRNN 80 = 74.5879 loss = 29.2908 parm = 1,151
# #GRU 80 = 80.43947 loss = 0.1931324601173401 , param = 302,801