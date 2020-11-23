from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
#1. data 
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,1]

#2. model
model  = LinearSVC()

#3. training
model.fit(x_data, y_data)

#4. elvaluate, predict
y_predict = model.predict(x_data)

print(x_data,"x_data의 예측결과",y_predict)

acc = accuracy_score(y_data, y_predict)
print("acc: ", acc)

print(np.array(y_data).shape)
print(y_predict.shape)
# score = model.score(np.array(y_data), y_predict)
# print("score: ", score)