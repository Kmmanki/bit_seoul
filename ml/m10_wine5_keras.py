import numpy as np
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

#1.데이터
datasets = pd.read_csv('./data/csv/winequality-white.csv', header=0, sep=";")

print(datasets)
# x = datasets[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
# y = datasets[['quality']]

x = datasets.iloc[:,[1,2,3,4,5,6,7,8,9,10]]
y = datasets.iloc[:,11]
x = x.to_numpy()
y = y.to_numpy()

y = to_categorical(y)
print(x.shape)#4898,10

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# #2. 모델
model = Sequential()
model.add(Dense(500, input_shape=(10,), activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(10, activation='softmax'))

# #3. 훈련 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'] )

es = EarlyStopping(monitor='val_loss', mode='auto',patience=50)
model.fit(x_train, y_train, verbose=1, epochs=300, validation_split=0.2, batch_size=16, callbacks=[es])

loss, acc = model.evaluate(x_test,y_test, batch_size=16)

# #4. 평가
# ###분류일 땐 acc, 선형회귀일 땐 R2와 비교하기 

y_predict = model.predict(x_test)
y_predict= np.argmax(y_predict, axis=1)
y_an = np.argmax(y_test[:10], axis=1)
print("acc: ", acc)
print("예상값: ",y_predict[:10])
print("정답값", y_an)



'''


# '''