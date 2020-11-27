import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tkinter import messagebox

xy=ImageDataGenerator(rescale=1./255).flow_from_directory(
    './homework/project1/x_predict', #실제 이미지가 있는 폴더는 라벨이 됨. (ad/normal=0/1)
    target_size=(178,218),
    batch_size=20000,
    class_mode='binary'
)

x=xy[0][0]



model = load_model('./homework/project1/models/model_Conv2D_train1_acc0.9861496090888977.h5')

model.summary()

y_predict = model.predict(x).reshape(x.shape[0],)
y_predict = np.round(y_predict).reshape(y_predict.shape[0],)
s = sum(y_predict)
print(y_predict.shape)
for i in range(20):
    if y_predict[i] ==  0:
        plt.imshow(x[i])
        plt.title('bald')
        plt.show()
    else:
        plt.imshow(x[i])
        plt.title('ball')
        plt.show()

    #대머리 11
    #축구공 9
print(s)

plt.show()

