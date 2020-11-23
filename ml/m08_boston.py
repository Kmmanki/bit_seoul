import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

#1.데이터
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
model = RandomForestRegressor()

#3. 훈련 
model.fit(x_train, y_train)

#4. 평가
###분류일 땐 acc, 선형회귀일 땐 R2와 비교하기 
score = model.score(x_test, y_test)

y_predict = model.predict(x_test)

# acc = accuracy_score(y_test, y_predict)
# print("acc: ", acc)
r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

print("예상값: ",y_predict[:10])
print("정답값", y_test[:10])
print("score: ", score)


'''
RandomForestCalssfier 

예상값:  [1 1 1 1 1 0 0 1 1 1]
정답값 [1 1 1 1 1 0 0 1 1 1]
acc:  0.9649122807017544
score:  0.9649122807017544
===========================================
KNeighborsClassifier 

예상값:  [1 1 1 1 1 0 0 1 1 1]
정답값 [1 1 1 1 1 0 0 1 1 1]
acc:  0.956140350877193
score:  0.956140350877193
==========================================
linearSVC
예상값:  [1 1 1 1 1 0 0 1 1 1]
정답값 [1 1 1 1 1 0 0 1 1 1]
acc:  0.9736842105263158
score:  0.9736842105263158

==========================================
svc 
예상값:  [1 1 1 1 1 0 0 1 1 1]
정답값 [1 1 1 1 1 0 0 1 1 1]
acc:  0.9649122807017544
score:  0.9649122807017544

==========================================
KNeighborsRegressor 
예상값:  [1. 1. 1. 1. 1. 0. 0. 1. 1. 1.]
정답값 [1 1 1 1 1 0 0 1 1 1]
r2:  0.8095556298028733
score:  0.8095556298028733
==========================================
RandomForestRegressor
r2:  0.8471417307049782
예상값:  [1.   0.99 1.   1.   0.97 0.18 0.   0.45 0.95 0.99]
정답값 [1 1 1 1 1 0 0 1 1 1]
score:  0.8471417307049782
'''