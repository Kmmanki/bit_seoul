#이진분류
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x = x.reshape(569,30,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(30,1)))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
ealystopping = EarlyStopping(monitor='loss',patience=20, mode='auto')
hist=model.fit(x_train, y_train, epochs=300, batch_size=4, validation_split=0.2, callbacks=[ealystopping])

loss, acc=model.evaluate(x_test, y_test, batch_size=4)

x_predict = x_test[20:30]
y_answer = y_test[20:30]


y_predict = model.predict(x_predict)

print("acc",acc)
print("loss",loss)
print("정답",y_answer)
print("예상값",y_predict)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('loss & va_loss')
plt.ylabel("loss, val_loss")
plt.xlabel('epoches')
plt.legend(['loss', 'val_loss'])



plt.show()

'''
acc 0.9649122953414917
loss 0.13466119766235352
정답 [0 0 1 0 0 1 1 1 0 1]
'''