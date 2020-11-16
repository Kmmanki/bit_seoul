import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

#1. 데이터 전처리
#1-1 mnist에서 데이터 불러오기, train, test
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#1-2 분류 모델이기 때문에 y를 oneHotEncoding 하기
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#1-4 Conv2d의 reshapesms 이미지 수, 가로, 세로, 채널로 reshape필요
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1 ).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1 ).astype('float32')/255.
#1-3 빠르고 정확하 연산을 위해 minmaxScaling
# scaler = MinMaxScaler() 2차원만 가능한가?
# scaler.fit(x_train) # 원래는 x 전체 넣어야함
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2 모델링
#2-1 CNN 중 Conv2d를 이용한 모델링
#2-2 Conv2d의 input_shape는 가로 세로 채널로 채널은 흑백은 1 컬러는 3을 가진다.
#2-3 Conv2d에서 Dense로 연결 시 Conv2d는 4치원을 반환하므로 Flatten을 사용하여 스칼라로 output으로 반환
#2-4 최종 레이어에서의 노드 수는 분류의 총 종류 수, activation은 반드시 softmax
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28,28,1))) #x.shape = 600000,28,28 ->28,28,흑백은 채널1
model.add(Conv2D(20, (2,2)))
model.add(Conv2D(30, (2,2)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10, activation='softmax' ))# 노드 수는 반환되는 값들의 종류 수 0~ 9 10개 분류이기에 activation 은 softmax

#3. 컴파일 및 훈련
#3-1 분류의 loss는 반드시 categorical_crossentropy
#3-2 분류의 메트릭스는 acc가 좋음
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam' )
ealystopping = EarlyStopping(monitor='loss', mode='min', patience=30)
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
hist = model.fit(x_train, y_train, batch_size=32, epochs=300, verbose=1, validation_split=0.2, callbacks=[ealystopping, to_hist])

#4.예측 및 y_predict 구하기
#4-1 y_predict는 OneHotEncoding된 상태로 반환되기 때문에 np.argmax를 사용하여 디코딩 필요
loss, acc = model.evaluate(x_test, y_test, batch_size=32)

x_predict = x_test[20:30] #본래 새로운 값을 받아서 스케일링필요함
y_answer = y_test[20:30]

y_predict = model.predict(x_predict)
#디코딩
y_predict = np.argmax(y_predict, axis=1)
y_answer = np.argmax(y_answer, axis=1)

print("정답은",y_answer)
print("예상은",y_predict)


