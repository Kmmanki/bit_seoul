import numpy as np
import matplotlib.pylab as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

(x_train, y_train), (x_test, y_test)  = mnist.load_data()

print(x_train.shape, x_test.shape) #(6만장, 28, 28), #(1만장, 28, 28) 28 * 28 픽셀 이미지가 6만장있다!
print(y_train.shape, y_test.shape) #(6만장, 28, 28), #(1만장, 28, 28)
print(x_train[1])
print(y_train[1])

plt.imshow(x_train[-1], 'gray')
plt.show()

#분류를 하기 위해 각각의 이미지가 동등하다는 것을 표현하기위해 하나의 값만 True 나머지가 False인 상태로 바꾸는 것