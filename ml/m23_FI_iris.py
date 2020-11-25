#기준 xg
#FI 0 제거
#하위 30% 제거
#디폴트와 성능비교

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, shuffle=True, train_size=0.8)


model1 = XGBClassifier(max_depth=4)
model1.fit(x_train, y_train)

print('디폴트 score', model1.score(x_test, y_test))


model = XGBClassifier(max_depth=4)
model.fit(x_train, y_train)
print('min',np.argmin(model.feature_importances_)) 

model2 = XGBClassifier(max_depth=4)

# print(x_train.shape)
x_train = x_train[:, 1:]
x_test = x_test[:, 1:]
# print(x_train.shape)

model2.fit(x_train, y_train)
print('최하위 1개 재거 score', model2.score(x_test, y_test))

'''
디폴트 score 1.0
min 0
최하위 1개 재거  1.0
'''


