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
kfold = KFold(n_splits=5, shuffle=True)
model  = SVC()

score = cross_val_score(model,x_train,y_train, cv=kfold) #5번 훈련함 각 훈련마다의 score

print("score: ",score)


# model.fit(x_train, y_train)



# y_predict = model.predict(x_test)


# print("예상값: ",y_predict[:10])
# print("정답값", y_test[:10])

# acc = accuracy_score(y_test, y_predict)
# print("acc: ", acc)
# # r2 = r2_score(y_test, y_predict)
# # print("r2: ", r2)
# print("score: ", score)
