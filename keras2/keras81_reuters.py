from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, Flatten, Dense, LSTM, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2 #단어사전의 개수 
)
print(x_train.shape, x_test.shape) #(8982,) (2246,) 8982개의 문장
print(y_train.shape, y_test.shape)# (8982,) (2246,)

print(x_train[0])
print(y_train[0])

print(len(x_train[0]))
print(len(x_train[11]))

#y 의 카테고리 개수 

category = np.max(y_train)+1
print("카테고리의 종류 : ",category)

#y의 유니크 값 구하기 
y_bunpo = np.unique(y_train)
print(y_bunpo)
'''
토큰화가 끝났지만 길이가 다르다 0으로 모두 같은 길이로 만들어주어야 한다.
[1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 3
69, 186, 90, 67, 7, 89, 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 2
2, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 1
1, 15, 22, 155, 11, 15, 7, 48, 9, 4579, 1005, 504, 6, 258, 6, 272,
 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 90, 67, 52, 29, 20
 9, 30, 32, 132, 6, 109, 15, 17, 12]

'''





from tensorflow.keras.preprocessing.sequence import pad_sequences
x_test = pad_sequences(x_test,maxlen=100, padding='pre')
x_train = pad_sequences(x_train, maxlen=100, padding='pre')
# print(x_train[0])

print(x_train.shape) #(8982, 2376)
model = Sequential()
model.add(Embedding(10000, 30, input_length=2376))
model.add(LSTM(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(46, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')

reduce_lr = ReduceLROnPlateau(monitor='loss', patience=3,
                             factor=0.5, verbose=1)

ealystopping = EarlyStopping(monitor='loss',patience=20, mode='auto')

model.fit(x_train, y_train, epochs=100, callbacks=[reduce_lr, ealystopping])

acc = model.evaluate(x_test, y_test)[1]
print(acc)


