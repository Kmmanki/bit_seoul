import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

#1.data
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]
            ,[5,6,7],[6,7,8],[7,8,9,],[8,9,10]
            ,[9,10,11],[10,11,12]
            ,[20,30,40],[30,40,50],[40,50,60]
])

y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("shape: ",x.shape)
x = x.reshape(13,3,1)



# #2. model
model = Sequential()
model.add(LSTM(300, input_shape=(3,1), return_sequences=True))
model.add(LSTM(100, input_shape=(3,1)))
model.add(Dense(130))
model.add(Dense(50))
model.add(Dense(1))

model.summary()
#Compile
model.compile(loss= 'mse', metrics=['mse'], optimizer='adam')
earlyStopping = EarlyStopping(monitor='loss', patience=1000, mode='auto')
model.fit(x, y, batch_size=1, epochs=10000 , verbose=1, callbacks=[earlyStopping])

# #predict


x_input = np.array([50,60,70])
x_input = x_input.reshape(1,3,1)

y_predict = model.predict(x_input)

loss = model.evaluate(x_input, np.array([80]), batch_size=1)
print(loss)
print(y_predict)