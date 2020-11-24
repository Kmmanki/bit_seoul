#분류

import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
warnings.filterwarnings('ignore')


wine = pd.read_csv('./data/csv/winequality-white.csv', header=0, sep=';')
print(wine.shape)

x= wine.iloc[:, 0:11]
y = wine.iloc[:, 11]
print(x.iloc[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

parameters = [
{
'randomforestclassifier__n_estimators':[10,100,200,300], #생성할 트리의 개수
    'randomforestclassifier__max_depth':[6,8,10,12,15],
    'randomforestclassifier__min_samples_leaf':[5,7,10,12],
    'randomforestclassifier__min_samples_split':[3,5,10,12],
    'randomforestclassifier__n_jobs':[-1]} # 모든 코어를 사용
 
]

# kfold = KFold(n_splits=5, shuffle=True) cv=5로 대체

pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier(), )
# pipe = Pipeline([
#     ("minmax",MinMaxScaler()), ('malddong',RandomForestClassifier())
# ])
model = RandomizedSearchCV(pipe, parameters , cv=5, verbose=2)

model.fit(x_train, y_train)

print('acc', model.score(x_test,y_test))
y_predict =model.predict(x_test)
print('최종 정답률', accuracy_score(y_predict, y_test))
print('최적의 파라미터', model.best_estimator_)

'''
GridSearchCv
최적의 파라미터:  RandomForestClassifier(max_depth=12, min_samples_leaf=5, min_samples_split=10,
                       n_estimators=200, n_jobs=-1, verbose=1)
최종 정답률:  0.6469387755102041

RandomizedSearchCV
최적의 파라미터:  RandomForestClassifier(max_depth=10, min_samples_leaf=5, min_samples_split=3,
                       n_jobs=-1, verbose=1)
                       최종 정답률:  0.6428571428571429

pipeline RandomizedSearchCV
최적의 파라미터 Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestclassifier',
                 RandomForestClassifier(max_depth=12, min_samples_leaf=5,
                                        min_samples_split=10, n_estimators=200,
                                        n_jobs=-1))])
acc 0.6612244897959184
최종 정답률 0.6612244897959184
'''










