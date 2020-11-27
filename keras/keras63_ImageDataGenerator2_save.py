from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(33)

#이미지에 대한 생성 옵션 지정
train_datagen = ImageDataGenerator(rescale=1./255,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    rotation_range=5,
                                    zoom_range=1.2,
                                    shear_range=0.7,
                                    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

#flow 또는 flow_from_directory
#실제 데이터르 알려주고 이미지 불러오기

# 0= ad, 1= nomal로 라벨링됨
xy_train = train_datagen.flow_from_directory(
    './data/data1/train',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
    # , save_to_dir='./data/data1/train_1'

)

xy_test = test_datagen.flow_from_directory(
    './data/data1/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
    # , save_to_dir='../data/data1/test_1'
)



print('===================================================')
print(type(xy_train))
# print(xy_train[0].shape) #ERROR
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(xy_train[0][0].shape) #(5, 150, 150, 3) #배치사이즈 5 가로 5 세로 5 채널 3 X 총 160개의 데이터를 5개 씩 나누어서 작업
print(xy_train[0][1].shape) #(5,) y

# print(xy_train[1][0].shape) #(5, 150, 150, 3) #배치사이즈 5 가로 5 세로 5 채널 3 X
# print(xy_train[1][1].shape) #(5,) y
print('===================================================')

# print(xy_train[0][0][1].shape) #
# print(xy_train[0][0][:5]) #

# # np.save('./data/keras63_train_x.npy', arr=xy_train[0][0])
# # np.save('./data/keras63_train_y.npy', arr=xy_train[0][1])
# # np.save('./data/keras63_test_x.npy', arr=xy_test[0][0])
# # np.save('./data/keras63_test_t.npy', arr=xy_test[0][1])

model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=(150,150,3), activation='relu'))
model.add(Conv2D(32, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

#numpy로 저장 하자 배치를 최대 크기로 주어서 

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

# ealystopping = EarlyStopping(monitor='loss',patience=10, mode='auto')
hist = model.fit_generator(
    xy_train, 
    steps_per_epoch=100,
    epochs=20,
    validation_data = xy_test,
    validation_steps=4
    # callbacks=[ealystopping]
)

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
val_loss = hist.history['val_loss']
loss = hist.history['loss']
print(acc)
print(val_acc)
print(val_loss)
print(loss)
plt.subplot(2,1,1)
plt.plot(acc)
plt.plot(val_acc)
plt.subplot(2,1,2)
plt.plot(loss)
plt.plot(val_loss)
plt.show()