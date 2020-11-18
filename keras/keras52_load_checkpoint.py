import numpy as np
import matplotlib.pylab as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

(x_train, y_train), (x_test, y_test)  = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. 
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255. 

model = load_model('./model/185--8.9836.hdf5')

result = model.evaluate(x_test, y_test, batch_size=32)
print(result)


'''
'''