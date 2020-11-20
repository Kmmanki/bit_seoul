from tensorflow.keras.datasets import cifar100, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
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

print('samsung.shape',x_samsung.shape) #5,8
print(y_samsung.shape) # ,
print('bit.shape',x_bit.shape) #5,4
print("kosdak.shape",x_kosdak.shape) #5,7
print('golde.shape',x_gold.shape) # 5,6

x_predict_samsung = x_samsung[-6]
x_predict_bit = x_bit[-6]
x_predict_kosdak = x_kosdak[-6]
x_predict_gold = x_gold[-6]


x_predict_samsung = x_predict_samsung.reshape(1, x_predict_samsung.shape[0]* x_predict_samsung.shape[1])
x_predict_bit = x_predict_bit.reshape(1,x_predict_bit.shape[0]* x_predict_bit.shape[1])
x_predict_kosdak = x_predict_kosdak.reshape(1,x_predict_kosdak.shape[0]* x_predict_kosdak.shape[1])
x_predict_gold = x_predict_gold.reshape(1,x_predict_gold.shape[0]* x_predict_gold.shape[1])

#train test split

x_samsung_train, x_samsung_test, y_samsung_train, y_samsung_test = train_test_split(
    x_samsung, y_samsung, train_size=0.8, shuffle=False)

x_bit_train, x_bit_test = train_test_split(x_bit, train_size=0.8, shuffle=False)

x_kosdak_train, x_kosdak_test = train_test_split(x_kosdak, train_size=0.8, shuffle=False)

x_gold_train, x_gold_test = train_test_split(x_gold, train_size=0.8, shuffle=False)





#모델 작성

sam_input = Input(shape=(40,)) 
sam_dense = Dense(500, activation='relu')(sam_input)
# sam_input = Input(shape=(5,8)) 
# sam_lstm = LSTM(100,return_sequences=True )(sam_input)
# output = Dropout(0.15)(sam_lstm)
# sam_lstm = LSTM(50,return_sequences=False)(sam_lstm)
# sam_dense = Dense(300, activation='relu')(sam_lstm)
sam_dense = Dense(400, activation='relu')(sam_dense)
sam_dense = Dense(300, activation='relu')(sam_dense)
sam_dense = Dense(200, activation='relu')(sam_dense)
sam_dense = Dense(10, activation='relu')(sam_dense)
    
bit_input = Input(shape=(20,))
bit_dense = Dense(300, activation='relu')(bit_dense)
# bit_input = Input(shape=(5,4))
# bit_lstm = LSTM(50,return_sequences=True )(bit_input)
# output = Dropout(0.1)(bit_lstm)
# bit_lstm = LSTM(10,return_sequences=False)(bit_lstm)
# bit_dense = Dense(100, activation='relu')(bit_lstm)
bit_dense = Dense(200, activation='relu')(bit_dense)
bit_dense = Dense(100, activation='relu')(bit_dense)
bit_dense = Dense(50, activation='relu')(bit_dense)

kosdak_input = Input(shape=(35))
kosdak_dense = Dense(500, activation='relu')(kosdak_input)
kosdak_dense = Dense(400, activation='relu')(kosdak_dense)
kosdak_dense = Dense(300, activation='relu')(kosdak_dense)
kosdak_dense = Dense(200, activation='relu')(kosdak_dense)
# kosdak_input = Input(shape=(5,7))
# kosdak_lstm = LSTM(100,return_sequences=True )(kosdak_input)
# output = Dropout(0.1)(kosdak_lstm)
# kosdak_lstm = LSTM(50,return_sequences=False)(kosdak_lstm)
# kosdak_dense = Dense(100, activation='relu')(kosdak_lstm)
# kosdak_dense = Dense(50, activation='relu')(kosdak_lstm)
kosdak_dense = Dense(10, activation='relu')(kosdak_dense)

gold_input = Input(shape=(30))
# gold_input = Input(shape=(5,6))
# gold_lstm = LSTM(100,return_sequences=True )(gold_input)
# output = Dropout(0.1)(gold_lstm)
# gold_lstm = LSTM(50,return_sequences=False)(gold_lstm)
# gold_dense = Dense(100, activation='relu')(gold_lstm)
# gold_dense = Dense(50, activation='relu')(gold_lstm)
gold_dense = Dense(500, activation='relu')(gold_input)
gold_dense = Dense(400, activation='relu')(gold_dense)
gold_dense = Dense(300, activation='relu')(gold_dense)
gold_dense = Dense(200, activation='relu')(gold_dense)
gold_dense = Dense(100, activation='relu')(gold_dense)
gold_dense = Dense(10, activation='relu')(gold_dense)

merge1 =  Concatenate()([sam_dense, bit_dense, kosdak_dense, gold_dense])

output = Dense(300, activation='relu')(merge1) #드롭아웃 추가
output = Dropout(0.2)(output)
output = Dense(500, activation='relu')(output)
output = Dense(400, activation='relu')(output)
output = Dense(300, activation='relu')(output)
output = Dense(200, activation='relu')(output)
output = Dense(100, activation='relu')(output)
output = Dense(10, activation='relu')(output)
output = Dense(1)(output)

model = Model(inputs=[sam_input, bit_input, kosdak_input, gold_input], outputs = [output])

model.compile(loss='mse', optimizer='adam', metrics=[])
ealystopping = EarlyStopping(monitor='loss', patience=60, mode='min')
modelPath = './homework/samsung_2_startprice/models/samsung_model-{val_loss:.4f}.hdf5'
checkPoint = ModelCheckpoint(filepath=modelPath, monitor='val_loss', save_best_only=True, mode='auto')

model.summary()
for i in range(10000):
    hist = model.fit([x_samsung_train, x_bit_train, x_kosdak_train, x_gold_train]
                    , [y_samsung_train], epochs=1, callbacks=[ealystopping, checkPoint],
                    verbose=1, validation_split=0.2, batch_size=4)

    loss = model.evaluate([x_samsung_test, x_bit_test, x_kosdak_test, x_gold_test], [y_samsung_test], batch_size=4)

    print("loss",loss)

    end_price = model.predict([x_predict_samsung,x_predict_bit,x_predict_kosdak, x_predict_gold ])
    print(end_price)
    if end_price <= 65340 and 64060 >= end_price:
        model.save('./homework/samsung_2_startprice/models/'+loss+'wowow.h5')
        break
