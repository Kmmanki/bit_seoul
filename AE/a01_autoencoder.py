import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float32')/255.
x_test = x_test.reshape(10000,784).astype('float32')/255.

# print(x_train[0])
# print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input = Input(shape=(784,))
encode = Dense(64, activation='relu')(input)
decode = Dense(784, activation='relu')(encode)

autoencoder = Model(input, decode)

autoencoder.summary()

# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, 
                validation_split=0.2, verbose=1)

y_predict = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
n=10
for i in range(n):
    ax = plt.subplot(2,n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(y_predict[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()