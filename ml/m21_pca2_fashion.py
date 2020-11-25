import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


(x_train, y_train), (x_test, y_test)  = fashion_mnist.load_data()

x = np.append(x_train, x_test, axis=0)


print(x.shape)

x = x.reshape(x.shape[0], x.shape[1] *  x.shape[2]* 1)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)

d95 = np.argmax(cumsum > 0.95) +1 
d1 = np.argmax(cumsum >= 1.0) +1 

x_d95 = PCA(n_components=d95)
x_d1 = PCA(n_components=d1)

x_d95 = x_d95.fit_transform(x)
x_d1 = x_d1.fit_transform(x)

x_d95_train, x_d95_test = x_d95[:x_train.shape[0]], x_d95[x_train.shape[0]:]
x_d1_train, x_d1_test = x_d1[:x_train.shape[0]], x_d1[x_train.shape[0]:]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model95 = Sequential()
model95.add(Dense(10, input_shape=(x_d95_train.shape[1], )))
model95.add(Dense(100 ))
model95.add(Dense(200 ))
model95.add(Dense(100 ))
model95.add(Dense(10))
model95.add(Dense(10, activation='softmax'))

model1 = Sequential()
model1.add(Dense(10, input_shape=(x_d1_train.shape[1], )))
model1.add(Dense(100 ))
model1.add(Dense(200 ))
model1.add(Dense(100 ))
model1.add(Dense(10))
model1.add(Dense(10, activation='softmax'))

ealystopping = EarlyStopping(monitor='loss',patience=10, mode='auto')

model1.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
model95.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

print('----------------------------')
print(x_d1_train.shape)
print(x_d1_train.shape)
print(y_train.shape)

model95.fit(x_d95_train, y_train, epochs=200, batch_size=512, validation_split=0.2, callbacks=[ealystopping])
model1.fit(x_d1_train, y_train, epochs=200, batch_size=512, validation_split=0.2, callbacks=[ealystopping])


loss1, acc1 = model1.evaluate(x_d1_test, y_test, batch_size=512)
loss95, acc95 = model95.evaluate(x_d95_test, y_test, batch_size=512)

print("loss95",loss95)
print("acc95",acc95)
print("loss1",loss1)
print("acc1",acc1)

'''
noPCA
acc 0.8385999798774719
loss 0.49222782254219055

loss95 0.4693557620048523
acc95 0.8353999853134155


loss1 0.5224361419677734
acc1 0.8334000110626221
'''


