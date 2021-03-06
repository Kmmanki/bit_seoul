from tensorflow.keras.datasets import cifar100, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, MaxPooling2D,Concatenate,Input, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

plt.figure(figsize=(15,15))
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


x_predict_samsung = x_predict_samsung.reshape(1, x_predict_samsung.shape[0], x_predict_samsung.shape[1])
x_predict_bit = x_predict_bit.reshape(1,x_predict_bit.shape[0], x_predict_bit.shape[1])
x_predict_kosdak = x_predict_kosdak.reshape(1,x_predict_kosdak.shape[0], x_predict_kosdak.shape[1])
x_predict_gold = x_predict_gold.reshape(1,x_predict_gold.shape[0], x_predict_gold.shape[1])
 
#train test split

x_samsung_train, x_samsung_test, y_samsung_train, y_samsung_test = train_test_split(
    x_samsung, y_samsung, train_size=0.8, shuffle=False)

x_bit_train, x_bit_test = train_test_split(x_bit, train_size=0.8, shuffle=False)

x_kosdak_train, x_kosdak_test = train_test_split(x_kosdak, train_size=0.8, shuffle=False)

x_gold_train, x_gold_test = train_test_split(x_gold, train_size=0.8, shuffle=False)





#모델 작성

sam_input = Input(shape=(5,8)) 
sam_model = Conv1D(32, 4,  padding = 'same',activation='relu')(sam_input)
sam_model = Dropout(0.2)(sam_model)
sam_model = Conv1D(64, 4,  padding = 'same', activation='relu')(sam_model)
sam_model = Dropout(0.2)(sam_model)
sam_model = Conv1D(128, 4,  padding = 'same', activation='relu')(sam_model)
sam_model = Dropout(0.2)(sam_model)
sam_model = Flatten()(sam_model)

    
bit_input = Input(shape=(5,4))
bit_model = Conv1D(32, 2, padding = 'same',activation='relu')(bit_input)
bit_model = Flatten()(bit_model)
bit_model = Dense(10, activation='relu')(bit_model)

kosdak_input = Input(shape=(5,7))
kosdak_model = Conv1D(16, 2,padding = 'same', activation='relu')(kosdak_input)
kosdak_model = Flatten()(kosdak_model)
kosdak_model = Dense(64, activation='relu')(kosdak_model)

gold_input = Input(shape=(5,6))
gold_model = Conv1D(16, 2, padding = 'same',activation='relu')(gold_input)
gold_model = Flatten()(gold_model)
gold_model = Dense(64, activation='relu')(gold_model)

merge1 =  Concatenate()([sam_model, bit_model, kosdak_model, gold_model])

output = Dense(64, activation='relu')(merge1) #드롭아웃 추가
output = Dense(128, activation='relu')(merge1) #드롭아웃 추가
output = Dense(216, activation='relu')(merge1) #드롭아웃 추가
output = Dropout(0.2)(output)
output = Dense(1)(output)

for i    in range(1,2):
    model = Model(inputs=[sam_input, bit_input, kosdak_input, gold_input], outputs = [output])

    model.compile(loss='mse', optimizer='adam', metrics=[])
    ealystopping = EarlyStopping(monitor='loss', patience=100, mode='min')
    modelPath = './homework/samsung_2_startprice/models/samsung_model-Conv1D-{val_loss:.4f}.hdf5'
    checkPoint = ModelCheckpoint(filepath=modelPath, monitor='val_loss', save_best_only=True, mode='auto')


    model.summary()
    hist = model.fit([x_samsung_train, x_bit_train, x_kosdak_train, x_gold_train]
                    , [y_samsung_train], epochs=1000, callbacks=[ealystopping, checkPoint],
                    verbose=1, validation_split=0.2, batch_size=4)

    loss = model.evaluate([x_samsung_test, x_bit_test, x_kosdak_test, x_gold_test], [y_samsung_test], batch_size=4)


    print("loss",loss)
    y_predict = model.predict([x_samsung_test, x_bit_test, x_kosdak_test, x_gold_test])


    r2 = r2_score(y_samsung_test, y_predict)
    r2 = r2.reshape(1,)[0]
    r2str = '%.3f' % r2
    
    end_price = model.predict([x_predict_samsung,x_predict_bit,x_predict_kosdak, x_predict_gold ])
    end_price=end_price.reshape(1,)[0]
    
    print(r2)
    print(end_price)

    if end_price <= 66000 and 63000 >= end_price and r2 > 0.99 :
        model.save('./homework/samsung_2_startprice/models/savemodel/Conv1D99/r2__'+r2str+str(loss)+'wowow.h5')
    elif end_price <= 66000 and 63000 >= end_price and r2 > 0.98:
        model.save('./homework/samsung_2_startprice/models/savemodel/Conv1D98/r2__'+r2str+str(loss)+'wowow.h5')
    elif end_price <= 66000 and 63000 >= end_price and r2 > 0.97:
        model.save('./homework/samsung_2_startprice/models/savemodel/Conv1D97/r2__'+r2str+str(loss)+'wowow.h5')
    elif end_price <= 66000 and 63000 >= end_price and r2 > 0.96:
        model.save('./homework/samsung_2_startprice/models/savemodel/Conv1D96/r2__'+r2str+str(loss)+'wowow.h5')
    else:
        model.save('./homework/samsung_2_startprice/models/savemodel/Conv1D_trash/r2__'+r2str+str(loss)+'wowow.h5')

    plt.subplot(1,2,i)
    plt.plot(y_predict, marker='.', c='red')
    plt.plot(y_samsung_test, marker='.', c='blue')
    plt.grid()
    plt.title(str(i)+"times"+r2str)
    plt.ylabel('price')
    plt.legend(['predict','real_price'])

    plt.subplot(1,2,i+1)
    plt.plot(hist.history['loss'], marker='.', c='red')
    plt.plot(hist.history['val_loss'], marker='.', c='blue')
    plt.grid()
    plt.title(str(i)+"times_loss")
    plt.ylabel('loss')
    plt.legend(['loss','val_loss'])
 
 
plt.show()