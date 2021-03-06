#분류

import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.datasets import load_boston
warnings.filterwarnings('ignore')

x,y = load_boston(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

parameters = [
{
'randomforestregressor__n_estimators':[100,200], #생성할 트리의 개수
    'randomforestregressor__max_depth':[6,8,10,12],
    'randomforestregressor__min_samples_leaf':[5,7,10],
    'randomforestregressor__min_samples_split':[3,5,10],
    'randomforestregressor__n_jobs':[-1]} # 모든 코어를 사용
 
]

# kfold = KFold(n_splits=5, shuffle=True) cv=5로 대체

pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())
# pipe = Pipeline([
#     ("minmax",MinMaxScaler()), ('malddong',SVC())
# ])
model = RandomizedSearchCV(pipe, parameters , cv=5, verbose=2)

model.fit(x_train, y_train)

print('acc', model.score(x_test,y_test))
y_predict =model.predict(x_test)
print('최종 정답률', r2_score(y_predict, y_test))
print("최적 파라미터", model.best_estimator_)



'''
no pipeline
최적의 파라미터:  RandomForestRegressor(max_depth=10, min_samples_leaf=5, min_samples_split=3,
                      n_estimators=200, n_jobs=-1)
최종 정답률:  0.9023018235061205


apply pipeline
score 0.9012044618747906
최종 정답률 0.8833558172144357
최적 파라미터 Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestregressor',
                 RandomForestRegressor(max_depth=12, min_samples_leaf=5,
                                       min_samples_split=5, n_jobs=-1))])
'''






