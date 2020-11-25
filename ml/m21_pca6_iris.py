import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_iris

x,y =load_iris(return_X_y=True)

print(x.shape)


pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)

d95 = np.argmax(cumsum > 0.95) +1 
d1 = np.argmax(cumsum >= 1.0) +1 

x_d95 = PCA(n_components=d95)
x_d1 = PCA(n_components=d1)

x_d95 = x_d95.fit_transform(x)
x_d1 = x_d1.fit_transform(x)

x_d95_train, x_d95_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_d1_train, x_d1_test  = train_test_split(x,  train_size=0.8)


print(x_d95_train.shape)
print(x_d95_test.shape)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model95 = Sequential()
model95.add(Dense(10, activation='relu', input_shape=(x_d95_train.shape[1],)))
model95.add(Dense(20, activation='relu'))
model95.add(Dense(30, activation='relu'))
model95.add(Dense(20, activation='relu'))
model95.add(Dense(3, activation='softmax'))

model1 = Sequential()
model1.add(Dense(10, activation='relu', input_shape=(x_d1_train.shape[1],)))
model1.add(Dense(20, activation='relu'))
model1.add(Dense(30, activation='relu'))
model1.add(Dense(20, activation='relu'))
model1.add(Dense(3, activation='softmax'))

ealystopping = EarlyStopping(monitor='loss',patience=20, mode='auto')

model95.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
model1.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

hist =model95.fit(x_d95_train, y_train, epochs=300, batch_size=512, validation_split=0.2, callbacks=[ealystopping])
hist =model1.fit(x_d1_train, y_train, epochs=300, batch_size=512, validation_split=0.2, callbacks=[ealystopping])

loss1, acc1 = model1.evaluate(x_d1_test, y_test, batch_size=512)
loss95, acc95 = model95.evaluate(x_d95_test, y_test, batch_size=512)

print("loss95",loss95)
print("acc95",acc95)
print("loss1",loss1)
print("acc1",acc1)

'''
no pca
acc 0.9666666388511658
loss 0.06045258417725563
정답 [1 1 0 2 0 0 1 1 2 2]
예상값 [1 2 0 2 0 0 1 1 2 2]

loss95 0.10723720490932465
acc95 0.9666666388511658


loss1 1.1785309314727783
acc1 0.2666666805744171

'''
