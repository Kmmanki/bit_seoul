import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x= iris.iloc[:,0:4]
y = iris.iloc[:,4]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms:
    model = algorithm()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name,"의 정답률",accuracy_score(y_test,y_pred))

import sklearn
print(sklearn.__version__) ## 0.22.1 버전에 문제있어서 안됨 낮춰야함

'''
AdaBoostClassifier 의 정답률 0.6333333333333333
BaggingClassifier 의 정답률 0.9333333333333333
BernoulliNB 의 정답률 0.3
CalibratedClassifierCV 의 정답률 0.9
CategoricalNB 의 정답률 0.9
CheckingClassifier 의 정답률 0.3
'''
