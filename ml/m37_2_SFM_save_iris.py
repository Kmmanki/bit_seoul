from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score 
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=66)


print(x_test.shape)
print(x_train.shape)

#모델
model = XGBClassifier(n_estimators=1000, learning_rate=0.01,
# model = XGBClassifier(nlearning_rate=0.01,
                        
                        )


model.fit(x_train, y_train, verbose=True,
            eval_metric=['merror', 'mlogloss'], #rmse, mse ,mae, logloss, error, auc
            eval_set=[(x_train, y_train),(x_test, y_test)],
            early_stopping_rounds=20
            )


thresholds = np.sort(model.feature_importances_) #피처를 소팅 
print(thresholds)

max_score = 0
invaild_feature =[]
for thresh in thresholds:
    selection = SelectFromModel(model, threshold = thresh, prefit=True) # 피처의 개수를 하나씩 제거
    
    select_x_train = selection.transform(x_train) # 피쳐의 개수를 줄인 트레인을 반환

    selection_model = XGBClassifier(n_jobs=-1) # 모델 생성 
    selection_model.fit(select_x_train, y_train) #모델의 핏

    select_x_test = selection.transform(x_test) # 피쳐의 개수를 줄인 테스트 반환
    y_predict = selection_model.predict(select_x_test) # 프레딕트 

    score = accuracy_score(y_test, y_predict)

    invaild_feature.append(thresh)
    if max_score <= score:
        import pickle
        pickle.dump(model, open('./save/SFM_save/m37_2_'+str(score*100.0)+'.dat', 'wb'))
        max_score = score        

    print("Thresh%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
 

#평가 예측
result = model.evals_result()
print('evlas result ',result['validation_1']['merror'][-1])

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(acc)