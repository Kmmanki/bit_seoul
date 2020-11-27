from xgboost import XGBRFClassifier, XGBRFRegressor
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




model = XGBRFRegressor(n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2', score)

thresholds = np.sort(model.feature_importances_) #피처를 소팅 
print(thresholds)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold = thresh, prefit=True) # 피처의 개수를 하나씩 제거
    
    select_x_train = selection.transform(x_train) # 피쳐의 개수를 줄인 트레인을 반환

    selection_model = XGBRFRegressor(n_jobs=-1) # 모델 생성 
    selection_model.fit(select_x_train, y_train) #모델의 핏

    select_x_test = selection.transform(x_test) # 피쳐의 개수를 줄인 테스트 반환
    y_predict = selection_model.predict(select_x_test) # 프레딕트 

    score = r2_score(y_test, y_predict)

    print("Thresh%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

score = model.score(x_test, y_test)
print('R2', score)