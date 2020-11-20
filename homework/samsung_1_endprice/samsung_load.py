from tensorflow.keras.datasets import cifar100, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D,Concatenate,Input, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

x_samsung = np.load('./homework/samsung_1_endprice/x_samsung.npy', allow_pickle=True)
y_samsung = np.load('./homework/samsung_1_endprice/y_samsung.npy', allow_pickle=True)
x_bit = np.load('./stock/x_bit.npy', allow_pickle=True)

# print(x_samsung.shape) #(620, 5, 7)
# print(y_samsung.shape) #(620,)
# print(x_bit.shape) #(620, 5, 6)



samsung_x_predict = x_samsung[-6].reshape(1,5,7)
bit_x_predict = x_bit[-6].reshape(1,5,6)
print(samsung_x_predict.shape)
print(bit_x_predict.shape)
model = load_model('./homework/samsung_1_endprice/samsung_model-565862.1875.hdf5')

price = model.predict([samsung_x_predict, bit_x_predict])
print(price)