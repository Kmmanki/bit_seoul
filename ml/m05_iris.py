import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

#1.데이터
x,y = load_iris(return_X_y=True)

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
r2 = r2_score(y_test, y_predict)

print("예상값: ",y_predict[:10])
print("정답값", y_test[:10])
# print("acc: ", acc)
print("r2: ", r2)
print("score: ", score)


'''
RandomForestCalssfier
예상값:  [1 1 1 0 1 1 0 0 0 1]
정답값 [1 1 1 0 1 1 0 0 0 2]
acc:  0.9333333333333333
score:  0.9333333333333333

===========================================
KNeighborsClassifier
예상값:  [1 1 2 0 1 1 0 0 0 2]
정답값 [1 1 1 0 1 1 0 0 0 2]
acc:  0.9
score:  0.9

==========================================
linearSVC
예상값:  [1 1 1 0 1 1 0 0 0 2]
정답값 [1 1 1 0 1 1 0 0 0 2]  
acc:  0.9333333333333333      
score:  0.9333333333333333    

==========================================
svc
예상값:  [1 1 1 0 1 1 0 0 0 2]
정답값 [1 1 1 0 1 1 0 0 0 2]  
acc:  0.9333333333333333      
score:  0.9333333333333333   

==========================================
KNeighborsRegressor 기존의 분류들 돌린거랑 acc 계산에서 오류
예상값:  [1.  1.2 1.6 0.  1.2 1.  0.  0.  0.  1.6]
정답값 [1 1 1 0 1 1 0 0 0 2]
r2:  0.9073825503355705
score:  0.9073825503355705
==========================================
RandomForestRegressor
예상값:  [1.   1.13 1.   0.   1.   1.   0.   0.   0.   1.29]
정답값 [1 1 1 0 1 1 0 0 0 2]
r2:  0.9503439597315436
score:  0.9503439597315436

'''