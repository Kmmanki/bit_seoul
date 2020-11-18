import numpy as np
import matplotlib.pylab as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

(x_train, y_train), (x_test, y_test)  = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. 
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255. 
print(x_train[0])

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28,28,1), padding='same'))
model.add(Conv2D(12, (2,2), padding='valid'))
model.add(Conv2D(13, (3,3))) 
model.add(Conv2D(15, (3,3), strides=2)) 
model.add(MaxPooling2D(pool_size=2)) 
model.add(Flatten()) 
model.add(Dense(20, activation='relu')) 
model.add(Dense(10, activation='softmax')) 
model.summary()

# model.save('./save/model_test01_1.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'], ) 

######################################################
#02d 에포크의 2자리 수 정수, val_loss 4자리 실수 
modelPath = './model/keras50_1{epoch:02d}--{val_loss:.4f}.hdf5'
checkPoint = ModelCheckpoint(filepath=modelPath, monitor='val_loss', save_best_only=True, mode='auto' )
######################################################

earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min')
# to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
hist = hist = model.fit(x_train, y_train, validation_split=0.2, epochs=50, batch_size=128, verbose=1, callbacks=[earlyStopping, checkPoint])

#모델 + 가중치
model.save('./save/weight_test02.h5')
#가중치 정장
model.save_weights('./save/weight_test02.h5')

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']


result = model.evaluate(x_test, y_test, batch_size=128)
print('loss: ', result[0])
print('acc: ', result[1])

x_predict = x_test[30:40]
y_answer = y_test[30:40]


answer_list = np.argmax(y_answer, axis=1)

y_predict_list = []
y_predict = model.predict(x_predict)


y_predict_list = np.argmax(y_predict, axis=1)
print("y 실제값", answer_list)
print("y 예측값", y_predict_list)

########################################시각화
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10,6) ) # 단위 찾아보기

# plt.subplot(2, 1, 1) #2행 1열의 첫 번째
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss' )
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss' )
# plt.grid()

# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')


# plt.subplot(2, 1, 2) #2행 2열의 첫 번째
# plt.plot(hist.history['acc'], marker='.', c='red')
# plt.plot(hist.history['val_acc'], marker='.', c='blue')
# plt.grid()

# plt.title('accuracy')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc','val_acc'])
# plt.show()
# #########################################


'''
'''