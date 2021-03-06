from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test)  = cifar10.load_data()

print(x_train[0])
print("ytrain: ", y_train[0])

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

plt.imshow(x_train[0])
plt.show()