from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()

x ,y = load_breast_cancer(return_X_y=True)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, shuffle=True, train_size=0.8)

model = XGBClassifier(max_depth=4)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)

model.save_model('./save/xgb_save/xgbmodel_cancer.pickle.model')
print('저장됨')

model2 = XGBClassifier()
model2.load_model('./save/xgb_save/xgbmodel_cancer.pickle.model')
acc2 = model2.score(x_test, y_test)
print(acc)
print(acc2)
'''

'''