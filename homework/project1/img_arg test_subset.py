from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras import applications
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

train_size = 0
val_size = 0

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    global train_size
    global val_size
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        './homework/project1/data/',
       target_size=(178,218),
        batch_size=16,
        class_mode=None,
        shuffle=False,
        subset='training'
        )

    # print(generator[0].shape) # (16, 178, 218, 3) 16개를 가져와서  predict????

    bottleneck_features_train = model.predict_generator(
        generator, 20000 )
    train_size = bottleneck_features_train.shape[0]

    # print(bottleneck_features_train.shape)

    np.save('./bottleneck_features_train.npy',  arr= bottleneck_features_train)

    generator = datagen.flow_from_directory(
        './homework/project1/data/',
       target_size=(178,218),
        batch_size=16,
        class_mode=None,
        shuffle=False, #대머리 다가져오고 축구공 다가져옴
        subset='validation'
        )

    bottleneck_features_validation = model.predict_generator(
        generator, 20000 )
    val_size = bottleneck_features_validation.shape[0]
    np.save('./bottleneck_features_validation.npy', arr=            bottleneck_features_validation)



def train_top_model():
    train_data = np.load('./bottleneck_features_train.npy')
    validation_data = np.load('./bottleneck_features_validation.npy')
    global train_size
    global val_size
    # print(train_data.shape)
    # print(train_size/2)

    train_labels = np.array( [0] * int((train_size / 2)) + [1] * int((train_size / 2))) #대머리는 전체 사이즈의 2/1 (이진분류니까) 축구공은 전체 사이즈의 2/1
    validation_labels = np.array(  [0] * int((val_size / 2)) + [1] * int((val_size / 2)))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    ealyStopping = EarlyStopping(monitor='val_loss', mode='max', patience=5)
    model.fit(train_data, train_labels,
                callbacks=[ealyStopping],
              epochs=50,
              batch_size=16,
              validation_data=(validation_data, validation_labels))
    model.save('./moooooodel.h5')


def load_():
    model = applications.VGG16(include_top=False, weights='imagenet')
    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
    './homework/project1/x_predict', 
    target_size=(178,218),
    batch_size=20000,
    class_mode='binary'
        )
    y = generator[0][1]
    x_predict = model.predict_generator(  generator, 20000 )

    print(y)
    model = load_model('./moooooodel.h5')

    y_predict = model.predict(x_predict)
    y_predict = np.round(y_predict).reshape(y_predict.shape[0],)
    acc = accuracy_score(y,y_predict)
    print(acc)


# save_bottlebeck_features()
# train_top_model()
load_()
'''
no subset
(12633, 178, 218, 3)
subset'training'
'''



'''
TODO:
이미지 부풀리기
이미지 데이터가 부족한 경우 기존의 이미지를 회전 확대 등을 사용하여 재사용
https://wikidocs.net/72393
'''