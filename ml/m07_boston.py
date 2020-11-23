import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

#1.데이터
x,y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
# model = LinearSVC()
model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor()

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
RandomForestCalssfier Unknown label type: 'continuous'


===========================================
KNeighborsClassifier Unknown label type: 'continuous'


==========================================
linearSVC Unknown label type: 'continuous'


==========================================
svc Unknown label type: 'continuous'


==========================================
KNeighborsRegressor 기존의 분류들 돌린거랑 acc 계산에서 오류
예상값:  [12.56 38.66 23.44 42.88 19.8  20.36 22.8  20.68 42.48 18.44]
정답값 [16.3 43.8 24.  50.  20.5 19.9 17.4 21.8 41.7 13.1]
r2:  0.8404010032786686
score:  0.8404010032786686
==========================================
RandomForestRegressor
예상값:  [15.35  46.811 27.8   46.914 21.297 20.827 19.493 20.364 45.331 16.104]
정답값 [16.3 43.8 24.  50.  20.5 19.9 17.4 21.8 41.7 13.1]
r2:  0.9188206895774511
score:  0.9188206895774511

'''