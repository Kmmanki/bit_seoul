#cancer, 
import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_wine
warnings.filterwarnings('ignore')

x,y = load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

parameters = [
    {'n_estimators':[100,200], #생성할 트리의 개수
    'max_depth':[6,8,10,12],
    'min_samples_leaf':[5,7,10],
    'min_samples_split':[3,5,10],
    'n_jobs':[-1],
    'verbose':[1]} # 모든 코어를 사용
]

#모델
kfold = KFold(n_splits=5, shuffle=True)
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold)

model.fit(x_train, y_train)
#평가 예측

print('최적의 파라미터: ', model.best_estimator_)

y_predict = model.predict(x_test)

print('최종 정답률: ', accuracy_score(y_test, y_predict))

'''
최적의 파라미터:  RandomForestClassifier(max_depth=12, min_samples_leaf=7, min_samples_split=5,
                       n_jobs=-1)
최종 정답률:  1.0

'''



