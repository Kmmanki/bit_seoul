from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential

# model = VGG16()
vgg = VGG16(weights='imagenet', input_shape=(32,32,3), include_top=False)
vgg.trainable=False
vgg.summary()
print(len(vgg.trainable_weights)) #동결하기 전 가중치 32 = 바이어스 16 + 가중치 16, 동결 후 0 학습 하지 않는다 

model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(256))
# model.add(BatchNormalization()) # 가중치 + 바이어스 연산함
# model.add(Dropout(0.2)) # 가중치 + 바이어스 연산 안함
model.add(Activation('relu')) # 가중치 + 바이어스 연산 안함
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))
model.summary()

print(len(model.trainable_weights)) #동결하기 전 가중치 32 = 바이어스 16 + 가중치 16, 동결 후 0 학습 하지 않는다 

import pandas as pd

pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]

aaa = pd.DataFrame(layers, columns=['Layer Type',
                                     'Layer name', 'Layer Trainable'])
                                     
print(aaa)