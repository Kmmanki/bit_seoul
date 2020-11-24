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

# parameters = [
#     {"svc__C":[1,10,100,1000], "svc__kernel":["linear"]  },
#     {"svc__C":[1,10,100,1000], "svc__kernel":["rbf", 'sigmoid'], 'svc__gamma':[0.001, 0.0001]  }
    
# ]
parameters = [
    {"malddong__C":[1,10,100,1000], "malddong__kernel":["linear"]  },
    {"malddong__C":[1,10,100,1000], "malddong__kernel":["rbf", 'sigmoid'], 'malddong__gamma':[0.001, 0.0001]  }
    
]
# kfold = KFold(n_splits=5, shuffle=True) cv=5로 대체

pipe = make_pipeline(MinMaxScaler(), LinearSVC())
pipe = Pipeline([
    ("minmax",MinMaxScaler()), ('malddong',SVC())
])
model = RandomizedSearchCV(pipe, parameters , cv=5, verbose=2)

model.fit(x_train, y_train)

print('acc', model.score(x_test,y_test))
y_predict =model.predict(x_test)
print('최종 정답률', accuracy_score(y_predict, y_test))










