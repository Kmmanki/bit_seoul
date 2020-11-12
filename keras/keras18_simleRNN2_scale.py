import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

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
model.add(LSTM(200, input_shape=(3,1)))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

#Compile
model.compile(loss= 'mse', metrics=[], optimizer='adam')
loss = model.fit(x, y, batch_size=1, epochs=100 , verbose=0)

#predict

x_input = np.array([50,60,70])
print("x_input: ",x_input.shape)
x_input = x_input.reshape(1,3,1)
loss = model.evaluate(x_input, np.array([80]), batch_size=1)
y_predict = model.predict(x_input)

model.summary()

print(y_predict)
print(loss)
