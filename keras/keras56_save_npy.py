#나머지 6개 저장
import numpy as np

path = './data/'

xta = 'x_train.npy'
yta = 'y_train.npy'
xt = 'x_test.npy'
yt = 'y_test.npy'

sx = 'x.npy'
sy = 'y.npy'

#1.cifar10
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test)  = cifar10.load_data()
name = 'cifar10'
np.save(path+name+xta, arr=x_train)
np.save(path+name+yta, arr=y_train)
np.save(path+name+xt, arr=x_test)
np.save(path+name+yt, arr=y_test)



#2.fashion
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test)  = fashion_mnist.load_data()
name = 'fashion'
np.save(path+name+xta, arr=x_train)
np.save(path+name+yta, arr=y_train)
np.save(path+name+xt, arr=x_test)
np.save(path+name+yt, arr=y_test)


#3.cifar100
from tensorflow.keras.datasets import cifar100
(x_train, y_train), (x_test, y_test)  = cifar100.load_data()
name = 'cifar100'
np.save(path+name+xta, arr=x_train)
np.save(path+name+yta, arr=y_train)
np.save(path+name+xt, arr=x_test)
np.save(path+name+yt, arr=y_test)


#4.boston
from sklearn.datasets import load_boston
dataset = load_boston()
name = 'boston'
x = dataset.data
y = dataset.target
np.save(path+name+sx, arr=x)
np.save(path+name+sy, arr=y)


#5.diabetes
from sklearn.datasets import load_boston,load_diabetes
dataset = load_diabetes()
name = 'diabetes'
x = dataset.data
y = dataset.target
np.save(path+name+sx, arr=x)
np.save(path+name+sy, arr=y)


#6.iris
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target
name = 'iris'
np.save(path+name+sx, arr=x)
np.save(path+name+sy, arr=y)


#7.cnaser
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
name = 'cancer'
np.save(path+name+sx, arr=x)
np.save(path+name+sy, arr=y)
