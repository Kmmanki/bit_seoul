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


print(x_test.shape)
print(x_train.shape)

#모델
# model = XGBRegressor(n_estimators=10, learning_rate=0.1,
model = XGBRegressor(nlearning_rate=0.01,
                        
                        )


model.fit(x_train, y_train, verbose=True,
            eval_metric=['rmse', ' mae'], #rmse, mse ,mae, logloss, error, auc
            eval_set=[(x_train, y_train),(x_test, y_test)]
            )


#평가 예측
result = model.evals_result()
print('evlas result ',result['validation_0']['rmse'][-1])
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(r2)