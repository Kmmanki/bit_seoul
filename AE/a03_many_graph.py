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
    model.add(Dense(units=784, activation='sigmoid'))
    return model

#pca 시 가장손실이 적은 값이 154였음
model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=2)
model_04 = autoencoder(hidden_layer_size=4)
model_08 = autoencoder(hidden_layer_size=8)
model_016 = autoencoder(hidden_layer_size=16)
model_032 = autoencoder(hidden_layer_size=32)

model_01.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_01.fit(x_train, x_train, epochs=10)

model_02.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_02.fit(x_train, x_train, epochs=10)

model_04.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_04.fit(x_train, x_train, epochs=10)

model_08.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_08.fit(x_train, x_train, epochs=10)

model_016.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_016.fit(x_train, x_train, epochs=10)

model_032.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_032.fit(x_train, x_train, epochs=10)

outputs_01 = model_01.predict(x_test)
outputs_02 = model_02.predict(x_test)
outputs_04 = model_04.predict(x_test)
outputs_08 = model_08.predict(x_test)
outputs_016 = model_016.predict(x_test)
outputs_032 = model_032.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, axes = plt.subplots(7,5, figsize=(15,15))
random_images = random.sample(range(outputs_01.shape[0]), 5 )

outputs = [x_test, outputs_01, outputs_02, outputs_04
            , outputs_08, outputs_016, outputs_032]

for row_num, row in enumerate(axes):
    for col_num, ax  in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()