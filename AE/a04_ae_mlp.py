import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float32')/255.
x_test = x_test.reshape(10000,784).astype('float32')/255.

# print(x_train[0])
# print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder (hidden_layer_size):
    model = Sequential()
    model.add(Dense(units= hidden_layer_size, input_shape=(784,), activation='relu' ) )
    model.add(Dense(units=128))
    model.add(Dense(units=64))
    model.add(Dense(units=128))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

#pca 시 가장손실이 적은 값이 154였음
model = autoencoder(hidden_layer_size=154)

model.compile(optimizer='adam', loss='mse', metrics=['acc'])

model.fit(x_train, x_train, epochs=10
)
outputs = model.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10)) = plt.subplots(2,5,figsize=(20,7))
random_images = random.sample(range(outputs.shape[0]), 5)
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(outputs[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("input", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])


for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(outputs[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("output", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.show()

    