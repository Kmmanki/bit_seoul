#최적의 튠으로 구성하시오
from tensorflow.keras.datasets import cifar10

from tensorflow.keras.applications import ResNet101
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
#전처리
(x_train, y_train), (x_test, y_test)  = cifar10.load_data()

x_train = x_train.astype('float32')/255. ## 10000,32,32,3
x_test  =x_test.astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#모델링
##여기만 수정
name = 'ResNet101'
t_model = ResNet101(include_top=False, weights='imagenet', input_shape=(32,32,3))
##
t_model.trainable=False
model = Sequential()
model.add(t_model)
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#컴파일 및 훈련
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')


ealystopping = EarlyStopping(monitor='loss',patience=10, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3,
                             factor=0.5, verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[ealystopping, reduce_lr])

loss, acc=model.evaluate(x_test, y_test, batch_size=32)

print(name," acc : ",acc)
print(name," loss : ",loss)
'''
ResNet101  acc :  0.3402999937534332
ResNet101  loss :  1.8770614862442017
'''