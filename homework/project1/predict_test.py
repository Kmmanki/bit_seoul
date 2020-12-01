import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tkinter import messagebox
import autokeras as ak
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

xy=ImageDataGenerator(rescale=1./255).flow_from_directory(
    './homework/project1/x_predict', 
    target_size=(178,218),
    batch_size=20000,
    class_mode='binary'
)

x=xy[0][0]
y=xy[0][1]
print(x)


# model = load_model('./homework/project1/models/OKKK_model_conv2d_train1_acc0.9861496090888977.h5') #conv2d model.save
# model = load_model('./homework/project1/models/checkpoints/PJ-Conv2D-0.0446.hdf5') # conv2d #checkpoint
model = load_model("./homework/project1/models/autokeras1.h5", custom_objects=ak.CUSTOM_OBJECTS) # autokeras

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
print('acc', accuracy_score(y,y_predict))
plt.show()

