import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터 전ㄴ처리
iris  = pd.read_csv('./data/csv/iris_ys.csv', header=0)

x = iris.iloc[:, :4]
y = iris.iloc[:, -1]

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

#2. 모델구성
kfold = KFold(n_splits=5, shuffle=True) #N이 의마하는 것으 무엇인가???
model  = SVC()
model1 = LinearSVC()
model3 = KNeighborsClassifier()
model4 = KNeighborsRegressor()
model5 = RandomForestClassifier()
model6 = RandomForestRegressor()

svcScore = cross_val_score(model,x_train,y_train, cv=kfold) #5번 훈련함 각 훈련마다의 score
linearSVC = cross_val_score(model1,x_train,y_train, cv=kfold) #5번 훈련함 각 훈련마다의 score
kNeighborsClassifier = cross_val_score(model3,x_train,y_train, cv=kfold) #5번 훈련함 각 훈련마다의 score
kNeighborsRegressor = cross_val_score(model4,x_train,y_train, cv=kfold) #5번 훈련함 각 훈련마다의 score
RandomForestClassifier = cross_val_score(model5,x_train,y_train, cv=kfold) #5번 훈련함 각 훈련마다의 score
RandomForestRegressor = cross_val_score(model6,x_train,y_train, cv=kfold) #5번 훈련함 각 훈련마다의 score

print("svcScore: ",svcScore)
print("linearSVC: ",linearSVC)
print("kNeighborsClassifier: ",kNeighborsClassifier)
print("kNeighborsRegressor: ",kNeighborsRegressor)
print("RandomForestClassifier: ",RandomForestClassifier)
print("RandomForestRegressor: ",RandomForestRegressor)

'''
svcScore:                           [1.         1.         0.95833333 1.         1.        ]
linearSVC:                          [0.91666667 0.83333333 0.875      0.83333333 0.70833333]
kNeighborsClassifier:               [1.         1.         0.91666667 0.95833333 1.        ]
kNeighborsRegressor:                [0.97142857 0.99551402 0.98456592 0.99799582 0.96957746]
RandomForestClassifier:             [1. 1. 1. 1. 1.]
RandomForestRegressor:              [0.999755   0.99550078 0.98314706 0.96230435 0.99858462]
'''
