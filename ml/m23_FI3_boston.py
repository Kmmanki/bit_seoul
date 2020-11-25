#기준 xg
#FI 0 제거
#하위 30% 제거
#디폴트와 성능비교

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston

x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, shuffle=True, train_size=0.8)


model1 = XGBRFRegressor()
model1.fit(x_train, y_train)

default_score =model1.score(x_test, y_test)


model = XGBRFRegressor()
model.fit(x_train, y_train)
print(model.feature_importances_) 

index7 =np.sort(model.feature_importances_)[::-1][int(0.7 *len(model.feature_importances_) )]

delete_list = []
for i in model.feature_importances_:
    if i < index7:
        print(i,"제거 ")
        delete_list.append(model.feature_importances_.tolist().index(i))



# print(delete_list)
model2 = XGBRFRegressor(max_depth=4)

# print(x_train.shape)
x_train  = np.delete(x_train, delete_list, axis=1)
x_test  = np.delete(x_test, delete_list, axis=1)
# print(x_train.shape)


model2.fit(x_train, y_train)
remove_score = model2.score(x_test, y_test)
print('디폴트 score', default_score)
print('최하위 20%',delete_list,"제거 점수",remove_score )

'''
디폴트 score 0.8827496232473773
최하위 20% [1, 6, 11] 제거 점수 0.8584355901417511
'''


