import numpy as np
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

#1.데이터
x,y = load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) #142,13

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
예상값:  [2 1 1 0 1 1 2 0 0 1]
정답값 [2 1 1 0 1 1 2 0 0 1]
acc:  1.0
score:  1.0

===========================================
KNeighborsClassifier 
예상값:  [2 1 1 0 1 1 2 0 0 1]
정답값 [2 1 1 0 1 1 2 0 0 1]
acc:  1.0
score:  1.0

==========================================
linearSVC
예상값:  [2 1 1 0 1 1 2 0 0 0]
정답값 [2 1 1 0 1 1 2 0 0 1]
acc:  0.9722222222222222
score:  0.9722222222222222

==========================================
svc 
예상값:  [2 1 1 0 1 1 2 0 0 1]
정답값 [2 1 1 0 1 1 2 0 0 1]
acc:  1.0
score:  1.0

==========================================
KNeighborsRegressor 
예상값:  [2.  1.  1.  0.  1.  1.  2.  0.  0.  1.4]
정답값 [2 1 1 0 1 1 2 0 0 1]
r2:  0.9775954738330976
score:  0.9775954738330976
==========================================
RandomForestRegressor
예상값:  [1.87 1.   1.   0.   0.99 1.   1.89 0.   0.   1.06]
정답값 [2 1 1 0 1 1 2 0 0 1]
r2:  0.9836701555869872
score:  0.9836701555869872
'''