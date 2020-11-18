#CNN레이어, 다중분류 모델(softmax)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D,Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test)  = cifar10.load_data()

x_train = x_train.astype('float32')/255. ## 10000,32,32,3
x_test  =x_test.astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model = Sequential()
# model.add(Conv2D(100, (2,2), input_shape=(32,32,3)))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='softmax'))

model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=(32,32,3)))
model.add(Conv2D(32, (3,3), padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), padding="same"))
model.add(Conv2D(64, (3,3), padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
#######
modelPath = './save/cifar10/cp_cifar10-{epoch:02d}--{val_loss:.4f}.hdf5'
checkPoint = ModelCheckpoint(filepath=modelPath, monitor='val_loss', save_best_only=True, mode='auto' )
#######

# ealystopping = EarlyStopping(monitor='loss',patience=10, mode='auto')
# hist=model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[checkPoint,ealystopping])

es = EarlyStopping(patience=30,mode='auto',monitor='loss')
hist = model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=2, validation_split=0.2, callbacks=[es])


path = "./save/cifar10/modelSave"
model.save(path+'.h5')
model.save_weights(path+'_wegiht.h5')

loss, acc=model.evaluate(x_test, y_test, batch_size=64)

x_predict = x_test[20:30]
y_answer = y_test[20:30]


y_predict = model.predict(x_predict)

y_predict = np.argmax(y_predict, axis=1)
y_answer = np.argmax(y_answer, axis=1)

print("acc",acc)
print("loss",loss)
print("정답",y_answer)
print("예상값",y_predict)
print("keras49_cp_1_cifar10")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6) ) # 단위 찾아보기

plt.subplot(2, 1, 1) #2행 1열의 첫 번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss' )
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss' )
plt.grid()

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')


plt.subplot(2, 1, 2) #2행 2열의 첫 번째
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()

plt.title('accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])
plt.show()

'''
끝

acc 0.33219999074935913
loss 2.181807041168213
정답 [7 0 4 9 5 2 4 0 9 6]
예상값 [4 0 0 5 2 6 3 0 9 6]
keras49_cp_1_cifar10

'''