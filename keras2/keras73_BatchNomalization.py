import numpy as np
import matplotlib.pylab as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM,BatchNormalization,Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.regularizers import l1,l2

(x_train, y_train), (x_test, y_test)  = mnist.load_data()


# 데이터의 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. 
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255. 
print(x_train[0])
LSTM
#2. 모델
model = Sequential()
model.add(Conv2D(32, (2,2), kernel_initializer='he_normal',input_shape=(28,28,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (2,2),kernel_regularizer=l1(0.01), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=2)) 
model.add(Flatten()) 

model.add(Dense(64) )
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10, activation='softmax')) 

model.summary()


#3. 컴파일 , 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'], ) # 모든 acc를 합치면 1 
earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
hist = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=128, verbose=1, callbacks=[earlyStopping, to_hist])

#4. 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print('loss: ', loss)
print('acc: ', acc)


x_predict = x_test[30:40]
y_answer = y_test[30:40]


answer_list = np.argmax(y_answer, axis=1)

y_predict_list = []
y_predict = model.predict(x_predict)



y_predict_list = np.argmax(y_predict, axis=1)
print("y 실제값", answer_list)
print("y 예측값", y_predict_list)


'''
loss:  0.12854883074760437
acc:  0.9896000027656555
'''