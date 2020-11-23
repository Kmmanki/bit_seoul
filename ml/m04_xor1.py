from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
#1. data 
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

#2. model
model  = SVC()

#3. training
model.fit(x_data, y_data)

#4. elvaluate, predict
y_predict = model.predict(x_data)

print(x_data,"x_data의 예측결과",y_predict)

acc = accuracy_score(y_data, y_predict)
print("acc: ", acc)

score = model.score(x_data, y_data)
print("score: ", score)