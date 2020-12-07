from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, LSTM, BatchNormalization, Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)
print(x_train.shape, x_test.shape) #(25000,) (25000,) # 25000의 문장
print(y_train.shape, y_test.shape)  #(25000,) (25000,) # 25000

print(x_train[0])
print(y_train[0])

print(len(x_train[0]))
print(len(x_train[11]))

#y 의 카테고리 개수 
category = np.max(y_train)+1
print("카테고리의 종류 : ",category) #2 

#y의 유니크 값 구하기 
y_bunpo = np.unique(y_train)
print(y_bunpo)# 2진분류

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test,maxlen=100, padding='pre')

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
