from tensorflow.keras.datasets import cifar100, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D,Concatenate,Input, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

x_samsung = np.load('./homework/samsung_2_startprice/x_samsung.npy', allow_pickle=True)
y_samsung = np.load('./homework/samsung_2_startprice/y_samsung.npy', allow_pickle=True)
x_bit = np.load('./homework/samsung_2_startprice/x_bit.npy', allow_pickle=True)
x_kosdak = np.load('./homework/samsung_2_startprice/x_kosdak.npy', allow_pickle=True)
x_gold = np.load('./homework/samsung_2_startprice/x_gold.npy', allow_pickle=True)

# print(x_samsung.shape) #(620, 5, 7)
# print(y_samsung.shape) #(620,)
# print(x_bit.shape) #(620, 5, 6)


x_predict_samsung = x_samsung[-6]
x_predict_bit = x_bit[-6]
x_predict_kosdak = x_kosdak[-6]
x_predict_gold = x_gold[-6]


x_predict_samsung = x_predict_samsung.reshape(1, x_predict_samsung.shape[0], x_predict_samsung.shape[1])
x_predict_bit = x_predict_bit.reshape(1,x_predict_bit.shape[0], x_predict_bit.shape[1])
x_predict_kosdak = x_predict_kosdak.reshape(1,x_predict_kosdak.shape[0], x_predict_kosdak.shape[1])
x_predict_gold = x_predict_gold.reshape(1,x_predict_gold.shape[0], x_predict_gold.shape[1])


model = load_model('./homework/samsung_2_startprice/models/samsung_model-373808.6250.hdf5')

end_price = model.predict([x_predict_samsung,x_predict_bit,x_predict_kosdak, x_predict_gold ])
print(end_price)
