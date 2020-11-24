#분류

import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
warnings.filterwarnings('ignore')


iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x= iris.iloc[:,0:4]
y = iris.iloc[:,4]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"]  },
    {"C":[1,10,100,1000], "kernel":["rbf", 'sigmoid'], 'gamma':[0.001, 0.0001]  }
    
]
# parameters = [
#     {"C":[1,10,100,1000]  }, #나머진 디폴트의 1 , 10 ,100,1000 을 실행 
#     {"kernel":["rbf",'linear'] },
#     {'gamma':[0.001, 0.0001]  }
# ]



#모델
kfold = KFold(n_splits=5, shuffle=True)
model = GridSearchCV(SVC(), parameters , cv=kfold)

model.fit(x_train, y_train)

#평가 예측
print('최적의 매개변수', model.best_estimator_)
y_predict = model.predict(x_test)
print('최종 정답률', accuracy_score(y_predict, y_test))

'''
최적의 매개변수 SVC(C=1, kernel='linear')
최종 정답률 0.9666666666666667
'''
