from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test)  = fashion_mnist.load_data()

# print(x_train[0])
# print("ytrain: ", y_train[0])

# print(x_train.shape, x_test.shape) #(60000, 28,28) (10000,28,28)
# print(y_train.shape, y_test.shape) #(60000,) (10000,)

#1.데이터 전처리 스케일링, 라벨링

color = 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]* color,1 ).astype('float32')/255.
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]* color,1 ).astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델링 Conv2D
model = Sequential()
model.add(LSTM(10, input_shape=(28*28*color, 1 )))
model.add(Dense(100 ))
model.add(Dense(200 ))
model.add(Dense(100 ))
model.add(Dense(10))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
ealystopping = EarlyStopping(monitor='loss',patience=10, mode='auto')
to_hist = TensorBoard(log_dir='grahp',write_graph=True, write_images=True, histogram_freq=0)
model.fit(x_train, y_train, epochs=200, batch_size=512, validation_split=0.2, callbacks=[ealystopping, to_hist])

loss, acc=model.evaluate(x_test, y_test, batch_size=512)


x_predict = x_test[20:30]
y_answer = y_test[20:30]


y_predict = model.predict(x_predict)

y_predict = np.argmax(y_predict, axis=1)
y_answer = np.argmax(y_answer, axis=1)

y_predict_list = []
y_answer_list = []

def lableToName(y, list):
    for i in y:
        if i ==0:
            list.append("티셔츠")
        elif i==1:
            list.append("바지")
        elif i==2:
            list.append("풀오버")
        elif i==3:
            list.append("드레스")
        elif i==4:
            list.append("코트")
        elif i==5:
            list.append("샌들")
        elif i==6:
            list.append("셔츠")
        elif i==7:
            list.append("스니커즈")
        elif i==8:
            list.append("가방")
        elif i==9:
            list.append("앵글부츠")

lableToName(y_answer, y_answer_list)
lableToName(y_predict, y_predict_list)
print("acc",acc)
print("loss",loss)
print("정답:   ",y_answer_list)
print("예상값: ",y_predict_list)

score = 0
for i in range(10):
    if y_answer_list[i] == y_predict_list[i]:
        score +=1

print(score)

'''
cnn
acc 0.830299973487854
loss 0.499494343996048
정답:    ['풀오버', '샌들', '스니커즈', '앵글부츠', '바지', '코트', '셔츠', '티셔츠', '앵글부츠', '드레스']
예상값:  ['풀오버', '샌들', '스니커즈', '앵글부츠', '바지', '풀오버', '셔츠', '티셔츠', '앵글부츠', '코트']
8

'''