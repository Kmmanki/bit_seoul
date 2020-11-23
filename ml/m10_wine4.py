import numpy as np
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd

#1.데이터
datasets = pd.read_csv('./data/csv/winequality-white.csv', header=0, sep=";")

y = datasets['quality']
x = datasets.drop('quality', axis=1)

print(x.shape) #(4898, 11)
print(y.shape) #(4898,)

newlist= []
for i in list(y):
    if i <= 4:
        newlist += [0] # newlist.append(0) , []는 인덱스가 아니라 리스트에 0들어간거
    elif i <= 7:
        newlist += [1]
    else :
        newlist += [2]

y = newlist

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) #142,13

#2. 모델
model = RandomForestClassifier()

#3. 훈련 
model.fit(x_train, y_train)

#4. 평가
score = model.score(x_test, y_test)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print("acc: ", acc)
# r2 = r2_score(y_test, y_predict)
# print("r2: ", r2)

print("예상값: ",y_predict[:10])
print("정답값", y_test[:10])
print("score: ", score)


'''
y 재분류 전
예상값:  [6 6 6 6 6 5 5 7 5 6]
정답값 [6 6 6 6 6 5 5 7 5 6]
acc:  0.7153061224489796
score:  0.7153061224489796

y 재분류 후 
예상값:  [1 1 1 1 1 1 1 1 1 1]       
정답값 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
acc:  0.9469387755102041
score:  0.9469387755102041


'''


