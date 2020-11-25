from sklearn.datasets import load_breast_cancer
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


x, y = load_breast_cancer(return_X_y=True)


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


model1 = Sequential()
model1.add(Dense(10, activation='relu', input_shape=(x_d1_train.shape[1],)))
model1.add(Dense(20, activation='relu'))
model1.add(Dense(30, activation='relu'))
model1.add(Dense(20, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))

model95 = Sequential()
model95.add(Dense(10, activation='relu', input_shape=(x_d95_train.shape[1],)))
model95.add(Dense(20, activation='relu'))
model95.add(Dense(30, activation='relu'))
model95.add(Dense(20, activation='relu'))
model95.add(Dense(1, activation='sigmoid'))

ealystopping = EarlyStopping(monitor='loss',patience=20, mode='auto')

model1.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
model95.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')

hist=model1.fit(x_d1_train, y_train, epochs=300, batch_size=4, validation_split=0.2, callbacks=[ealystopping])
hist=model95.fit(x_d95_train, y_train, epochs=300, batch_size=4, validation_split=0.2, callbacks=[ealystopping])


loss1, acc1 = model1.evaluate(x_d1_test, y_test, batch_size=512)
loss95, acc95 = model95.evaluate(x_d95_test, y_test, batch_size=512)

print("loss95",loss95)
print("acc95",acc95)
print("loss1",loss1)
print("acc1",acc1)


'''
no pca
acc 0.9649122953414917
loss 0.10879302769899368
정답 [0 1 1 1 1 0 1 1 1 0]

loss95 0.15082110464572906
acc95 0.9561403393745422

loss1 0.6565091609954834
acc1 0.6315789222717285
'''
