import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/boston_house_prices.csv', header=1, index_col=None)

print(iris)

x= iris.iloc[:,0:13] # 0~ 12까지
y = iris.iloc[:,13] #13번째

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

print(x)
print(y)
allAlgorithms = all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률:', r2_score(y_test, y_pred))
    except:
        pass
import sklearn
print(sklearn.__version__) ## 0.22.1 버전에 문제있어서 안됨 낮춰야함

'''
ARDRegression 의 정답률 0.8012569266997763
AdaBoostRegressor 의 정답률 0.9066159967371444
BaggingRegressor 의 정답률 0.9091724041144214
BayesianRidge 의 정답률 0.7937918622384752
CCA 의 정답률 0.7913477184424631
DecisionTreeRegressor 의 정답률 0.7896277640738545
DummyRegressor 의 정답률 -0.0005370164400797517
ElasticNet 의 정답률 0.7338335519267194
ElasticNetCV 의 정답률 0.7167760356856181
ExtraTreeRegressor 의 정답률 0.8250182314361569
ExtraTreesRegressor 의 정답률 0.939795067174114
GammaRegressor 의 정답률 -0.0005370164400797517
GaussianProcessRegressor 의 정답률 -6.073105259620457
GeneralizedLinearRegressor 의 정답률 0.7442833362029138
GradientBoostingRegressor 의 정답률 0.945244700166123
HistGradientBoostingRegressor 의 정답률 0.9323597806119726
HuberRegressor 의 정답률 0.7551817913064872
'''
