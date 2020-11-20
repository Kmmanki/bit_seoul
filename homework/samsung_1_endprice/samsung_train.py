from tensorflow.keras.datasets import cifar100, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D,Concatenate,Input, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

x_samsung = np.load('./stock/x_samsung.npy', allow_pickle=True)
y_samsung = np.load('./stock/y_samsung.npy', allow_pickle=True)
x_bit = np.load('./stock/x_bit.npy', allow_pickle=True)

print(x_samsung.shape) #(620, 5, 7)
print(y_samsung.shape) #(620,)
print(x_bit.shape) #(620, 5, 6)

samsung_x_predict = x_samsung[-6].reshape(1,5,7)
bit_x_predict = x_bit[-6].reshape(1,5,6)

x_samsung_train, x_samsung_test, y_samsung_train, y_samsung_test = train_test_split(
    x_samsung, y_samsung, train_size=0.8)
    
x_bit_train, x_bit_test = train_test_split(
    x_bit, train_size=0.8)

##########################2 모델링#####################

sam_input = Input(shape=(5,7))
sam_lstm = LSTM(100,return_sequences=True )(sam_input)
output = Dropout(0.15)(sam_lstm)
sam_lstm = LSTM(50,return_sequences=False)(sam_lstm)
sam_dense = Dense(100, activation='relu')(sam_lstm)
sam_dense = Dense(50, activation='relu')(sam_dense)
sam_dense = Dense(50, activation='relu')(sam_dense)
sam_dense = Dense(10, activation='relu')(sam_dense)

bit_input = Input(shape=(5,6))
bit_lstm = LSTM(50,return_sequences=True )(bit_input)
output = Dropout(0.1)(bit_lstm)
bit_lstm = LSTM(10,return_sequences=False)(bit_lstm)
bit_dense = Dense(100, activation='relu')(bit_lstm)
bit_dense = Dense(50, activation='relu')(bit_dense)

merge1 =  Concatenate()([sam_dense, bit_dense])

output = Dense(300, activation='relu')(merge1) #드롭아웃 추가
output = Dropout(0.2)(output)
output = Dense(200, activation='relu')(output)
output = Dense(100, activation='relu')(output)
output = Dense(30, activation='relu')(output)
output = Dense(1)(output)

model = Model(inputs=[sam_input, bit_input], outputs = [output])


model.compile(loss='mse', optimizer='adam', metrics=[])
ealystopping = EarlyStopping(monitor='loss', patience=60, mode='min')

modelPath = './stock/model/samsung_model-{val_loss:.4f}.hdf5'
checkPoint = ModelCheckpoint(filepath=modelPath, monitor='val_loss', save_best_only=True, mode='auto')
for i in range(5):
    hist = model.fit([x_samsung_train, x_bit_train]
                    , [y_samsung_train], epochs=1000, callbacks=[ealystopping, checkPoint],
                    verbose=1, validation_split=0.2, batch_size=4)

# #평가
loss = model.evaluate([x_samsung_test, x_bit_test], [y_samsung_test], batch_size=4)

print("loss",loss)

samsung_x_predict = x_samsung[-5]
bit_x_predict = x_bit[-5]
end_price = model.predict([samsung_x_predict,bit_x_predict])

plt.figure(figsize=(10,6) ) # 단위 찾아보기

plt.subplot(2, 1, 1) #2행 1열의 첫 번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss' )
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss' )
plt.grid()

plt.show()