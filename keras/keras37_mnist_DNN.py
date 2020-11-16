import numpy as np
import matplotlib.pylab as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
'''
DNN을 사용한 분류 
x_train x_test , y_train, y_test 를 가져온다.
분류를 위해 y값들을 OneHotEndoding 한다.
가져온 x값들은 (60000, 28,28 ) 하지만 DNN은 2차원만 받음 -> (60000,(28*28) )로 reshape ->input_shape(28*28,)
분류를 하기 위해 out레이어의 노드 수 = 분류 종류 수 -> 0~9임으로 10 그리고 분류의 반환은 softmax =>model.add(Dense(10, activation='softmax'))
x_predict는 sacleing해야하지만 test 값에서 가져오기 때문에 이미 스케일링이 끝난 상태 그냥 슬라이싱으로 가져옴
y_predict는 OneHotEncoding한 결과로 반환됨 Decode 시키기위해 np.argmax(y) -> argmax는 배열에 들은 값중 가장 큰 값의 인덱스를 반환 
'''
(x_train, y_train), (x_test, y_test)  = mnist.load_data()

# 데이터의 전처리 1.OneHotEncoding
row = x_train.shape[1] * x_train.shape[2] 
x_train = x_train.reshape(x_train.shape[0], row).astype('float32')/255.
x_test  =x_test.reshape(x_test.shape[0], row).astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(y_train.shape)

#2. 모델
model = Sequential()
model.add(Dense(10, input_shape=(row, )))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


#3. 컴파일 , 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['Accuracy'] ) 
earlyStopping = EarlyStopping(monitor='loss', patience=30, mode='min')
# # to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
hist = model.fit(x_train, y_train, validation_split=0.2, epochs=500, batch_size=32, verbose=1, callbacks=[earlyStopping])

# #4. 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss: ', loss)
print('acc: ', acc)

# #실습 test 데이터 10개를 가져와서 predict  y 디코딩 np argmax
# #earlystopping, tensorboard
x_predict = x_test[30:40]
y_answer = y_test[30:40]

y_predict = model.predict(x_predict)
answer_list = []
for y in y_answer:
    answer_list.append(np.argmax(y))

y_predict_list = []

for y in y_predict:
    y_predict_list.append(np.argmax(y))


print("y 실제값", answer_list)
print("y 예측값", y_predict_list)


# '''
# Conv2d
# loss:  0.12854883074760437
# acc:  0.9896000027656555
# DNN
# loss:  0.33511531352996826
# acc:  0.33511531352996826
# '''