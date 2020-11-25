#pca로 축소하여 모델 완성
#0.95 1 모델만들어 dnn과 비교
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test)  = mnist.load_data()

x = np.append(x_train, x_test, axis=0)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x = x.reshape(x.shape[0], x.shape[1] *  x.shape[2])

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)

d95 = np.argmax(cumsum > 0.95) +1 
d1 = np.argmax(cumsum >= 1.0) +1 

x_d95 = PCA(n_components=d95)
x_d1 = PCA(n_components=d1)

x_d95 = x_d95.fit_transform(x)
x_d1 = x_d1.fit_transform(x)

x_d95_train, x_d95_test = x_d95[:60000], x_d95[60000:]
x_d1_train, x_d1_test = x_d1[:60000], x_d1[60000:]

print(x_d95_train.shape)
print(x_d95_test.shape)

model95 = Sequential()
model95.add(Dense(10, input_shape=(x_d95_train.shape[1], )))
model95.add(Dense(64, activation='relu'))
model95.add(Dense(64, activation='relu'))
model95.add(Dense(10, activation='softmax'))

model1 = Sequential()
model1.add(Dense(10, input_shape=(x_d1.shape[1], )))
model1.add(Dense(64, activation='relu'))
model1.add(Dense(64, activation='relu'))
model1.add(Dense(10, activation='softmax'))

model95.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['Accuracy'] ) 
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['Accuracy'] ) 
earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min')

hist = model1.fit(x_d1_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1, callbacks=[earlyStopping])
hist = model95.fit(x_d95_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1, callbacks=[earlyStopping])

loss1, acc1 = model1.evaluate(x_d1_test, y_test, batch_size=32)
loss95, acc95 = model95.evaluate(x_d95_test, y_test, batch_size=32)

print("loss95",loss95)
print("acc95",acc95)
print("loss1",loss1)
print("acc1",acc1)


'''
No PCA 
 DNN
loss:  0.32157716155052185
acc:  0.13564999401569366

PCA 95
loss95 0.2272239774465561
acc95 0.08601000159978867

PCA 1 
loss1 0.4165453612804413
acc1 0.21830999851226807
'''