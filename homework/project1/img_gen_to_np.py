from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

# test_datagen = ImageDataGenerator(rescale=1./255)
xy=ImageDataGenerator(rescale=1./255).flow_from_directory(
    './homework/project1/data/', #실제 이미지가 있는 폴더는 라벨이 됨. (ad/normal=0/1)
    target_size=(178,218),
    batch_size=20000,
    class_mode='binary'
    # ,save_to_dir = './argumentation'
    
)

np.save('./homework/project1/proejct1_x.npy', arr=xy[0][0])
np.save('./homework/project1/proejct1_y.npy', arr=xy[0][1])



'''
TODO:
이미지 부풀리기
이미지 데이터가 부족한 경우 기존의 이미지를 회전 확대 등을 사용하여 재사용
https://wikidocs.net/72393
'''