import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

#1.데이터
x,y = load_diabetes(return_X_y=True)

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
model = RandomForestClassifier()
# model = RandomForestRegressor()

#3. 훈련 
model.fit(x_train, y_train)

#4. 평가
###분류일 땐 acc, 선형회귀일 땐 R2와 비교하기 
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
RandomForestCalssfier
예상값:  [ 91. 208. 156.  64.  53. 138.  53.  87. 229. 101.]
정답값 [235. 150. 124.  78. 168. 253.  97. 102. 249. 142.]
acc:  0.011235955056179775
score:  0.011235955056179775

===========================================
KNeighborsClassifier
예상값:  [ 91. 129.  77.  64.  60.  63.  42.  53.  67.  74.]
정답값 [235. 150. 124.  78. 168. 253.  97. 102. 249. 142.]
acc:  0.0
score:  0.0

==========================================
linearSVC
예상값:  [ 91. 232. 281. 134.  97.  69.  53. 109. 230. 101.]
정답값 [235. 150. 124.  78. 168. 253.  97. 102. 249. 142.]
acc:  0.011235955056179775
score:  0.011235955056179775

==========================================
svc
예상값:  [ 91. 220.  91.  90.  90.  91.  53.  90. 220. 200.]
정답값 [235. 150. 124.  78. 168. 253.  97. 102. 249. 142.]
r2:  -0.0833765078750015
score:  0.0

==========================================
KNeighborsRegressor 기존의 분류들 돌린거랑 acc 계산에서 오류
예상값:  [166.4 190.6 124.  132.6 120.4 120.2 101.4 108.6 145.4 131.6]
정답값 [235. 150. 124.  78. 168. 253.  97. 102. 249. 142.]
r2:  0.38626977834604637

==========================================
RandomForestRegressor
예상값:  [162.87 207.24 156.88 129.55  96.16 119.83  92.65 143.24 150.12 123.08]
정답값 [235. 150. 124.  78. 168. 253.  97. 102. 249. 142.]
r2:  0.37875334896526014
score:  0.37875334896526014

'''