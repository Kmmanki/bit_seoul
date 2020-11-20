
#CNN레이어, 다중분류 모델(softmax)
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,Conv1D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

#라벨링 1개 x컬럼3
dataset = load_iris()

x = dataset.data
y = dataset.target

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x= x.reshape(x.shape[0], x.shape[1], 1,1  )

print(x.shape) #150,4,1,1
print(y.shape) #150,3 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
print(x_train.shape,x_test.shape) #120,4 /30,4
print(y_train.shape,y_test.shape) #120,3 /30,3

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential() #r2 =0.8
model.add(Conv1D(20, 2, padding='same', activation='relu', input_shape=(4,1,1)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
ealystopping = EarlyStopping(monitor='loss',patience=20, mode='auto')
to_hist = TensorBoard(log_dir='grahp',write_graph=True, write_images=True, histogram_freq=0)



hist = model.fit(x_train, y_train, epochs=300, batch_size=64, validation_split=0.2, callbacks=[ealystopping
#  to_hist
 ])

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
conv1D
acc 0.9333333373069763
loss 0.1552175134420395
정답 [0 0 0 2 0 0 2 0 1 0]
예상값 [0 0 0 2 0 0 2 0 1 0]

'''