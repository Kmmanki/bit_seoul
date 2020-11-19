from sklearn.datasets import load_iris
import numpy as np


iris = load_iris()
print(iris)

x_data = iris.data
y_data = iris.target

print(type(iris))
print(type(x_data))

np.save('./data/iris_x.npy', arr=x_data)
np.save('./data/iris_x.npy', arr=y_data)