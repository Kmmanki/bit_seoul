import numpy as np
from tensorflow.keras.datasets import cifar10
(x_train, _), (x_test, _) = cifar10.load_data()

x_train = x_train.astype('float32')/255.

x_test = x_test.astype('float32')/255.

output_units =x_train.shape[1]*x_train.shape[2]*x_train.shape[3]

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size= x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

# print(x_train[0])
# print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten

def autoencoder (hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(154, (3,3), strides=(1,1),
                    padding='valid', input_shape=(x_train.shape[1],
                                                x_train.shape[2],x_train.shape[3])
                     ))
    model.add(Conv2D(128, padding='same',kernel_size= (3,3) ) )
    model.add(Conv2D(128, padding='same',kernel_size= (3,3) ) )
    model.add(Conv2D(64, padding='same',kernel_size= (3,3) ) )
    model.add(Conv2D(64, padding='valid',kernel_size= (3,3) ) )
    model.add(Flatten())
    model.add(Dense(units=output_units, activation='sigmoid'))
    return model

#pca 시 가장손실이 적은 값이 154였음
model = autoencoder(hidden_layer_size=154)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train_noised, 
        x_train.reshape(x_train.shape[0], output_units ), epochs=10, batch_size=512)
outputs = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10), (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3,5,figsize=(20,7))


random_images = random.sample(range(outputs.shape[0]), 5)
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(x_train.shape[1],x_train.shape[2],x_train.shape[3]), cmap='gray')
    if i == 0:
        ax.set_ylabel("input", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(x_train.shape[1],x_train.shape[2],x_train.shape[3]), cmap='gray')
    if i == 0:
        ax.set_ylabel("noised", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(outputs[random_images[i]].reshape(x_train.shape[1],x_train.shape[2],x_train.shape[3]), cmap='gray')
    if i == 0:
        ax.set_ylabel("output", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

    