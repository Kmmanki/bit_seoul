import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

#1. 데이터 전처리

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]* x_train.shape[2], 1 ).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]* x_test.shape[2], 1 ).astype('float32')/255.



#2 모델링
#2-1 LSTM의 모델링의 경우 60000만개, 28*28데이터의 수 

model = Sequential()
model.add(LSTM(30, input_shape=(28*28,1))) 
model.add(Dense(60, activation='relu' ))
model.add(Dense(90, activation='relu' ))
model.add(Dense(120, activation='relu' ))
model.add(Dense(10, activation='softmax' ))

#3. 컴파일 및 훈련
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam' )
ealystopping = EarlyStopping(monitor='loss', mode='min', patience=10)
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
hist = model.fit(x_train, y_train, batch_size=512, epochs=300,  validation_split=0.2, callbacks=[ealystopping, to_hist])

#4.예측 및 y_predict 구하기

loss, acc = model.evaluate(x_test, y_test, batch_size=512)

x_predict = x_test[20:30]
y_answer = y_test[20:30]

y_predict = model.predict(x_predict)
#디코딩
y_predict = np.argmax(y_predict, axis=1)
y_answer = np.argmax(y_answer, axis=1)

print("정답은",y_answer)
print("예상은",y_predict)

'''
acc: 0.4368
loss: 1.4874
정답률 60%
정답은 [9 6 6 5 4 0 7 4 0 1]
예상은 [9 6 2 4 4 0 7 1 8 1]
'''
