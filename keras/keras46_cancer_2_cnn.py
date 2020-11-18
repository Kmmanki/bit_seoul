#이진분류
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x = x.reshape(569,30,1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

model = Sequential()
model.add(Conv2D(10, (2,2), padding='same' ,activation='relu', input_shape=(30,1,1)))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30, activation='relu'))
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
print("예상값",np.round(y_predict).reshape(10,))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & va_loss')
plt.ylabel("loss, val_loss")
plt.xlabel('epoches')
plt.legend(['loss', 'val_loss', 'acc','var_acc'])



plt.show()

'''
acc 0.9561403393745422
loss 0.20207355916500092
정답 [1 0 1 0 1 1 1 1 1 1]
'''