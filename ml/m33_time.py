from xgboost import XGBRFClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score 
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=66)


model = XGBRegressor(n_jobs=-1, verbose=1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2', score)
thresholds = np.sort(model.feature_importances_) #피처를 소팅 
print(thresholds)

invaild_feature =[]
import time
start1 = time.time()
for thresh in thresholds:
    selection = SelectFromModel(model, threshold = thresh, prefit=True) # 피처의 개수를 하나씩 제거
    
    select_x_train = selection.transform(x_train) # 피쳐의 개수를 줄인 트레인을 반환

    selection_model = XGBRegressor(n_jobs=-1) # 모델 생성 
    selection_model.fit(select_x_train, y_train) #모델의 핏

    select_x_test = selection.transform(x_test) # 피쳐의 개수를 줄인 테스트 반환
    y_predict = selection_model.predict(select_x_test) # 프레딕트 

    score = r2_score(y_test, y_predict)

    invaild_feature.append(thresh)
    if select_x_train.shape[1] ==9:
        break

    # print("Thresh%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

start2 = time.time()

for thresh in thresholds:
    selection = SelectFromModel(model, threshold = thresh, prefit=True) # 피처의 개수를 하나씩 제거
    
    select_x_train = selection.transform(x_train) # 피쳐의 개수를 줄인 트레인을 반환

    selection_model = XGBRegressor(n_jobs=8) # 모델 생성 
    selection_model.fit(select_x_train, y_train) #모델의 핏

    select_x_test = selection.transform(x_test) # 피쳐의 개수를 줄인 테스트 반환
    y_predict = selection_model.predict(select_x_test) # 프레딕트 

    score = r2_score(y_test, y_predict)

    invaild_feature.append(thresh)
    if select_x_train.shape[1] ==9:
        break

    # print("Thresh%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
    
end = start2 - start1
print(end)
end2 = time.time() - start2
print('jobs 걸린시간', end2)

# print(model.feature_importances_)


#반복문을 통해 인덱스 반환
indexs = []
for i in invaild_feature:
    indexs.append( model.feature_importances_.tolist().index(i))

print(indexs)

# print(x_train)
x_train  = np.delete(x_train, indexs, axis=1)
x_test  = np.delete(x_test, indexs, axis=1)
# print(x_train)

print(x_train.shape)
# print(x_test.shape)

parameters = [
    {'n_estimators':[100,200,300], 'learning_rate':[0.1, 0.3, 0.001, 0.01], 'max_depth':[4,5,6] },
    {'n_estimators':[90,100,110], 'learning_rate':[0.1, 0.001 ,0.01], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90,110], 'learning_rate':[0.1,0.001, 0.5], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1 ], 'colsample_bylevel':[0.6,0.7,0.9]},
]

model = RandomizedSearchCV(XGBRegressor(), parameters, cv=5 ,verbose=1, n_jobs=-1)
model.fit(x_train, y_train)  #6

y_predict = model.predict(x_test)

print("RandomizedSearchCV R2:", r2_score(y_test, y_predict))