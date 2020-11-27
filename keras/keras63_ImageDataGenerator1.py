from tensorflow.keras.preprocessing.image import ImageDataGenerator 

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
    #, save_to_dir='./data/data1_2/train'

)

xy_test = test_datagen.flow_from_directory(
    './data/data1/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
    #, save_to_dir='./data/data1_2/test'
    
)


# model.fit_generator(
#     xy_train, 
#     steps_per_epoch=100,
#     epochs=20,
#     validation_data = test_datagen,
#     validation_steps=4,
#     validation_split=0.2
# )