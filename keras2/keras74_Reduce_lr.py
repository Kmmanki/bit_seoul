import numpy as np
import matplotlib.pylab as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau

(x_train, y_train), (x_test, y_test)  = mnist.load_data()


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) #(6만장,), #(1만장,)
# print(y_train.shape[0]) #5

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. #전체 이미지, 가로, 세로, 채널 
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255. # -> minmaxScaling
print(x_train[0])
LSTM
#2. 모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28,28,1), padding='same'))
model.add(Conv2D(12, (2,2), padding='valid'))# 28,28,10을 받음 why -> padding='same'이라 rows, cols가 줄어들지 않음
model.add(Conv2D(13, (3,3))) #27,27,20 받음
model.add(Conv2D(15, (3,3), strides=2)) 
model.add(MaxPooling2D(pool_size=2)) 
model.add(Flatten()) 
model.add(Dense(20, activation='relu')) 
model.add(Dense(10, activation='softmax')) #여기 중요 분류는 무조건 softmax!!!! 반드시 원핫 인코딩이 필요하다.
model.summary()

#3. 컴파일 , 훈련

model.compile(loss='categorical_crossentropy', 
            optimizer='adam', metrics=['acc'], ) # 모든 acc를 합치면 1 

earlyStopping = EarlyStopping(monitor='loss', 
                            patience=5, mode='min')

ck = ModelCheckpoint('./model', save_best_only=True,
                     save_weights_only=True, monitor='val_loss')

#3번동안 개선이 없으면 lr을 50% 감축, 더 지나서 5번 없으면 Es에 의해 종료
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3,
                             factor=0.5, verbose=1)

hist = model.fit(x_train, y_train, validation_split=0.2, epochs=30, batch_size=128, verbose=1, callbacks=[earlyStopping, ck])

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