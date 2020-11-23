from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data 
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

#2. model
model = Sequential()
model.add(Dense(4, input_shape=(2,)))
model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. elvaluate, predict
y_predict = model.predict(x_data)

print(x_data,"x_data의 예측결과",y_predict)

# acc = accuracy_score(y_data, y_predict)
# print("acc: ", acc)

score = model.evaluate(x_data, y_data)
print("evaluates: ", score)