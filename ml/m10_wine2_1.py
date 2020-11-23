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

print(datasets)
# x = datasets[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
# y = datasets[['quality']]

x = datasets.iloc[:,[1,2,3,4,5,6,7,8,9,10]]
y = datasets.iloc[:,11]
x = x.to_numpy()
y = y.to_numpy()

print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape) #142,13

# #2. 모델
# model = LinearSVC()
model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor()

# #3. 훈련 
model.fit(x_train, y_train)

# #4. 평가
# ###분류일 땐 acc, 선형회귀일 땐 R2와 비교하기 
score = model.score(x_test, y_test)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print("acc: ", acc)
# r2 = r2_score(y_test, y_predict)
# print("r2: ", r2)

print("예상값: ",y_predict[:10].reshape(10,))
print("정답값", y_test[:10].reshape(10,))
print("score: ", score)


'''
RandomForestCalssfier 

예상값:  [6 6 6 6 6 5 5 7 5 6]
정답값 [6 6 6 6 6 5 5 7 5 6]
acc:  0.7153061224489796
score:  0.7153061224489796
# ===========================================
# KNeighborsClassifier 

예상값:  [6 6 6 6 5 5 5 7 5 6]
정답값 [6 6 6 6 6 5 5 7 5 6]
acc:  0.5540816326530612
score:  0.5540816326530612

# ==========================================
# linearSVC

예상값:  [6 6 6 6 6 6 6 6 5 6]
정답값 [6 6 6 6 6 5 5 7 5 6]
acc:  0.5326530612244897
score:  0.5326530612244897
# ==========================================
# svc 
예상값:  [6 6 6 6 6 5 6 6 6 6]
정답값 [6 6 6 6 6 5 5 7 5 6]
acc:  0.5540816326530612
score:  0.5540816326530612

# ==========================================
# KNeighborsRegressor 

# ==========================================
# RandomForestRegressor

# '''