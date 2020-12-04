from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

img_dog  = load_img('./dog_cat/dog1.jpg', target_size=(224,224))
img_cat  = load_img('./dog_cat/cat1.jpg', target_size=(224,224))
img_s  = load_img('./dog_cat/s.jpg', target_size=(224,224))
img_ry  = load_img('./dog_cat/ryan.jpg', target_size=(224,224))
img_dog2  = load_img('./dog_cat/dog2.jpg', target_size=(224,224))


def to_bgr(img):
    img = img_to_array(img)
    img = preprocess_input(img)
    return img
# plt.imshow(img_dog)
# plt.show()

# print(arr_dog)
# print(type(arr_dog))# <class 'numpy.ndarray'>
# print(arr_dog.shape)# (720, 1280, 3)

# RGB to BGR

# print(type(arr_dog))
# print(arr_dog.shape) #(720, 1280, 3)

arr_dog = to_bgr(img_dog)
arr_cat = to_bgr(img_cat)
arr_s = to_bgr(img_s)
arr_ry = to_bgr(img_ry)
arr_dog2 = to_bgr(img_dog2)

# print(arr_cat.shape) #(224, 224, 3)

arr_input = np.stack([arr_dog, arr_cat,arr_s ,arr_ry,arr_dog2 ])
print(arr_input.shape) #(2, 224, 224, 3)

model = VGG16()
probs = model.predict(arr_input)

print(probs)
print("probs.shape : ",probs.shape) #probs.shape :  (2, 1000)

#이미지 결과 확인 
from tensorflow.keras.applications.vgg16 import decode_predictions

results = decode_predictions(probs)

print("===================================================")
print("result[0] : ",results[0])
print("===================================================")
print("reuslt[1] : ",results[1])
print("===================================================")
print("reuslt[2] : ",results[2])
print("===================================================")
print("reuslt[3] : ",results[3])
print("===================================================")
print("reuslt[4] : ",results[4])
print("===================================================")