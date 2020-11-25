#pipeline 까지
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import numpy as np

import warnings

def return_x_train_y_train_without02(model, x_train, x_test):
    index_value =np.sort(model.feature_importances_)[::-1][int(0.7 *len(fi_model.feature_importances_) )]

    delete_list = []
    for i in model.feature_importances_:
        if i < index_value:
            print(i,"제거 ")
            delete_list.append(model.feature_importances_.tolist().index(i))

    x_train  = np.delete(x_train, delete_list, axis=1)
    x_test  = np.delete(x_test, delete_list, axis=1)
    
    return x_train, x_test


x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8)

parameters = [
    {'n_estimators':[100,200,300], 'learning_rate':[0.1, 0.3, 0.001, 0.01], 'max_depth':[4,5,6] },
    {'n_estimators':[90,100,110], 'learning_rate':[0.1, 0.001 ,0.01], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90,110], 'learning_rate':[0.1,0.001, 0.5], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1 ], 'colsample_bylevel':[0.6,0.7,0.9]},
]

fi_model = XGBRegressor()
fi_model.fit(x_train, y_train)

#하위 20% 제거 
x_train, x_test = return_x_train_y_train_without02(fi_model, x_train, x_test)



model = RandomizedSearchCV(XGBRegressor(), parameters, cv=5 ,verbose=1, n_jobs=-1)
model = make_pipeline(MinMaxScaler(),model )
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

score = model.score(x_test, y_test)
r2 = r2_score( y_test, y_predict)

print("r2", r2)
print('score: ', score)

'''
0.02731018 제거 
0.030616526 제거 
Fitting 5 folds for each of 10 candidates, totalling 50 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
[Parallel(n_jobs=-1)]: Done  26 tasks      | elapsed:    2.9s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:    2.9s finished
r2 0.2588420809073958
score:  0.2588420809073958
'''