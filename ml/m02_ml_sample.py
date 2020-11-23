import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.svm import LinearSVC

dataset = load_iris()

x = dataset.data
y = dataset.target

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

model = LinearSVC()

model.fit(x_train, y_train)

x_predict = x_test[20:30]
y_answer = y_test[20:30]



y_predict = model.predict(x_predict)

score = model.score(x_test, y_test)

print(score)
print(y_predict)
print(y_answer)



plt.show()

'''
acc 0.9666666388511658
loss 0.06045258417725563
정답 [1 1 0 2 0 0 1 1 2 2]
예상값 [1 2 0 2 0 0 1 1 2 2]
'''