#분류

import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score
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
    try:
        kfold = KFold(n_splits=5, shuffle=True)

        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        crossval_Score = cross_val_score(model,x_train,y_train, cv=kfold) #5번 훈련함 각 훈련마다의 score
        print(name,"의 crossvalidation",crossval_Score)
        print(name,"의 정답률",accuracy_score(y_test,y_pred))
        print("=====================================================")
        print("\n")
    except:
        pass
import sklearn
print(sklearn.__version__) ## 0.22.1 버전에 문제있어서 안됨 낮춰야함

'''
AdaBoostClassifier 의 crossvalidation [0.95833333 1.         0.91666667 0.875      0.95833333]
AdaBoostClassifier 의 정답률 0.6333333333333333
=====================================================


BaggingClassifier 의 crossvalidation [1.         0.875      1.         0.95833333 0.875     ]
BaggingClassifier 의 정답률 0.9666666666666667
=====================================================


BernoulliNB 의 crossvalidation [0.29166667 0.20833333 0.25       0.20833333 0.29166667]
BernoulliNB 의 정답률 0.3
=====================================================


CalibratedClassifierCV 의 crossvalidation [1.         0.83333333 0.91666667 0.875      0.875     ]
CalibratedClassifierCV 의 정답률 0.9
=====================================================


CheckingClassifier 의 crossvalidation [0. 0. 0. 0. 0.]
CheckingClassifier 의 정답률 0.3
=====================================================


ComplementNB 의 crossvalidation [0.79166667 0.66666667 0.70833333 0.66666667 0.5       ]
ComplementNB 의 정답률 0.6666666666666666
=====================================================


DecisionTreeClassifier 의 crossvalidation [0.91666667 0.91666667 1.         0.91666667 0.95833333]
DecisionTreeClassifier 의 정답률 0.9333333333333333
=====================================================


DummyClassifier 의 crossvalidation [0.33333333 0.20833333 0.29166667 0.29166667 0.41666667]
DummyClassifier 의 정답률 0.26666666666666666
=====================================================


ExtraTreeClassifier 의 crossvalidation [0.91666667 0.91666667 0.875      0.95833333 0.91666667]
ExtraTreeClassifier 의 정답률 0.8
=====================================================


ExtraTreesClassifier 의 crossvalidation [0.91666667 0.875      1.         1.         1.        ]
ExtraTreesClassifier 의 정답률 0.9666666666666667
=====================================================


GaussianNB 의 crossvalidation [0.91666667 1.         0.95833333 0.91666667 0.91666667]
GaussianNB 의 정답률 0.9666666666666667
=====================================================


GaussianProcessClassifier 의 crossvalidation [1.         1.         0.91666667 0.875      1.        ]
GaussianProcessClassifier 의 정답률 0.9666666666666667
=====================================================


GradientBoostingClassifier 의 crossvalidation [0.95833333 0.875      0.95833333 0.95833333 0.95833333]
GradientBoostingClassifier 의 정답률 0.9666666666666667
=====================================================


HistGradientBoostingClassifier 의 crossvalidation [0.95833333 0.95833333 1.         0.91666667 0.95833333]
HistGradientBoostingClassifier 의 정답률 0.8666666666666667
=====================================================


KNeighborsClassifier 의 crossvalidation [1.         0.95833333 0.91666667 0.95833333 0.95833333]
KNeighborsClassifier 의 정답률 0.9666666666666667
=====================================================


LabelPropagation 의 crossvalidation [1.         0.95833333 1.         0.91666667 0.95833333]
LabelPropagation 의 정답률 0.9333333333333333
=====================================================


LabelSpreading 의 crossvalidation [0.91666667 0.95833333 0.95833333 1.         0.95833333]
LabelSpreading 의 정답률 0.9333333333333333
=====================================================


LinearDiscriminantAnalysis 의 crossvalidation [0.875      1.         0.95833333 1.         1.        ]
LinearDiscriminantAnalysis 의 정답률 1.0
=====================================================


LinearSVC 의 crossvalidation [1.         1.         0.95833333 0.91666667 0.91666667]
LinearSVC 의 정답률 0.9666666666666667
=====================================================


LogisticRegression 의 crossvalidation [1.         0.875      0.95833333 0.91666667 0.95833333]
LogisticRegression 의 정답률 1.0
=====================================================


LogisticRegressionCV 의 crossvalidation [1.         1.         1.         0.91666667 0.91666667]
LogisticRegressionCV 의 정답률 1.0
=====================================================


MLPClassifier 의 crossvalidation [1.         0.95833333 1.         0.95833333 0.95833333]
MLPClassifier 의 정답률 1.0
=====================================================


MultinomialNB 의 crossvalidation [0.875      0.875      0.95833333 1.         0.95833333]
MultinomialNB 의 정답률 0.9666666666666667
=====================================================


NearestCentroid 의 crossvalidation [0.95833333 1.         0.79166667 0.95833333 0.91666667]
NearestCentroid 의 정답률 0.9333333333333333
=====================================================


NuSVC 의 crossvalidation [0.95833333 0.95833333 0.95833333 1.         0.95833333]
NuSVC 의 정답률 0.9666666666666667
=====================================================


PassiveAggressiveClassifier 의 crossvalidation [0.91666667 0.75       0.58333333 0.875      0.83333333]
PassiveAggressiveClassifier 의 정답률 0.9333333333333333
=====================================================


Perceptron 의 crossvalidation [0.54166667 0.66666667 0.70833333 0.875      0.625     ]
Perceptron 의 정답률 0.9333333333333333
=====================================================


QuadraticDiscriminantAnalysis 의 crossvalidation [0.95833333 1.         0.95833333 0.95833333 0.95833333]
QuadraticDiscriminantAnalysis 의 정답률 1.0
=====================================================


RadiusNeighborsClassifier 의 crossvalidation [0.91666667 0.91666667 0.95833333 1.         0.91666667]
RadiusNeighborsClassifier 의 정답률 0.9666666666666667
=====================================================


RandomForestClassifier 의 crossvalidation [1.         0.95833333 0.95833333 0.875      0.95833333]
RandomForestClassifier 의 정답률 0.9333333333333333
=====================================================


RidgeClassifier 의 crossvalidation [0.79166667 0.83333333 0.79166667 0.875      0.83333333]
RidgeClassifier 의 정답률 0.8666666666666667
=====================================================


RidgeClassifierCV 의 crossvalidation [0.66666667 0.79166667 0.875      0.91666667 0.91666667]
RidgeClassifierCV 의 정답률 0.8666666666666667
=====================================================


SGDClassifier 의 crossvalidation [0.625      0.625      0.875      0.83333333 0.83333333]
SGDClassifier 의 정답률 0.9
=====================================================


SVC 의 crossvalidation [0.95833333 1.         1.         0.91666667 0.91666667]
SVC 의 정답률 0.9666666666666667
=====================================================
'''
