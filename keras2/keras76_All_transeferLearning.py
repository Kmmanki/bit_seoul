from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import DenseNet121 , DenseNet169 , DenseNet201
from tensorflow.keras.applications import MobileNetV2, MobileNet
from tensorflow.keras.applications import InceptionV3 , InceptionResNetV2
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import Xception,ResNet101, ResNet101V2, ResNet152 ,ResNet152V2 , ResNet50 , ResNet50V2
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential



vgg16 = VGG16()
# vgg16.summary()
print("VGG16",len(vgg16.trainable_weights)/2) 
print('----------------------------------------------------------------------------')
vgg16 = VGG19()
# vgg16.summary()
print("VGG19레이어 수 ",len(vgg16.trainable_weights)/2) 
print('----------------------------------------------------------------------------')
vgg16 = Xception()
# vgg16.summary()
print("Xception",len(vgg16.trainable_weights)/2) 
print('----------------------------------------------------------------------------')
vgg16 = ResNet101()
# vgg16.summary()
print("ResNet101",len(vgg16.trainable_weights)/2) 
print('----------------------------------------------------------------------------')
vgg16 = ResNet101V2()
# vgg16.summary()
print("ResNet101V2",len(vgg16.trainable_weights)/2) 
print('----------------------------------------------------------------------------')
vgg16 = ResNet152()
# vgg16.summary()
print("ResNet152",len(vgg16.trainable_weights)/2) 
print('----------------------------------------------------------------------------')
vgg16 = ResNet50()
# vgg16.summary()
print("ResNet50",len(vgg16.trainable_weights)/2) 
print('----------------------------------------------------------------------------')
vgg16 = ResNet50V2()
# vgg16.summary()
print("ResNet50V2",len(vgg16.trainable_weights)/2) 

print('----------------------------------------------------------------------------')
vgg16 = NASNetLarge()
# vgg16.summary()
print("NASNetLarge",len(vgg16.trainable_weights)/2) 

print('----------------------------------------------------------------------------')
vgg16 = NASNetMobile()
# vgg16.summary()
print("NASNetMobile",len(vgg16.trainable_weights)/2) 

print('----------------------------------------------------------------------------')
vgg16 = DenseNet121()
# vgg16.summary()
print("DenseNet121",len(vgg16.trainable_weights)/2) 

print('----------------------------------------------------------------------------')
vgg16 = DenseNet169()
# vgg16.summary()
print("DenseNet169",len(vgg16.trainable_weights)/2) 

print('----------------------------------------------------------------------------')
vgg16 = DenseNet201()
# vgg16.summary()
print("DenseNet201",len(vgg16.trainable_weights)/2) 

print('----------------------------------------------------------------------------')
vgg16 = MobileNetV2()
# vgg16.summary()
print("MobileNetV2",len(vgg16.trainable_weights)/2) 


print('----------------------------------------------------------------------------')
vgg16 = MobileNet()
# vgg16.summary()
print("MobileNet",len(vgg16.trainable_weights)/2) 



print('----------------------------------------------------------------------------')
vgg16 = InceptionV3()
# vgg16.summary()
print("InceptionV3",len(vgg16.trainable_weights)/2) 

print('----------------------------------------------------------------------------')
vgg16 = InceptionResNetV2()
# vgg16.summary()
print("InceptionResNetV2",len(vgg16.trainable_weights)/2) 



'''
레이어의 수 



'''