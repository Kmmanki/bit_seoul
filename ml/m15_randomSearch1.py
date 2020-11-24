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


iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x= iris.iloc[:,0:4]
y = iris.iloc[:,4]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)


pipe = make_pipeline(MinMaxScaler(), SVC())
pipe.fit(x_train, y_train)

print('acc', pipe.score(x_test,y_test))


parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"]  },
    {"C":[1,10,100,1000], "kernel":["rbf", 'sigmoid'], 'gamma':[0.001, 0.0001]  }
    
]

kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(SVC(), parameters , cv=kfold, verbose=2)

pipe = make_pipeline(MinMaxScaler(), model)
pipe.fit(x_train, y_train)

print('acc', pipe.score(x_test,y_test))
y_predict =pipe.predict(x_test)
print('최종 정답률', accuracy_score(y_predict, y_test))










