#회귀

import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x= iris.iloc[:,0:4]
y = iris.iloc[:,4]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms:
    try:
        kfold = KFold(n_splits=5, shuffle=True)

        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        crossval_Score = cross_val_score(model,x_train,y_train, cv=kfold) #5번 훈련함 각 훈련마다의 score
        print(name,"의 crossvalidation",crossval_Score)
        print(name,"의 정답률",r2_score(y_test,y_pred))
        print("=====================================================")
        print("\n")
    except:
        pass
import sklearn
print(sklearn.__version__) ## 0.22.1 버전에 문제있어서 안됨 낮춰야함

'''
ARDRegression 의 crossvalidation [0.92440998 0.96291434 0.9332913  0.90277336 0.87838371]
ARDRegression 의 정답률 0.9018396130429644
=====================================================


AdaBoostRegressor 의 crossvalidation [0.87368421 0.99934783 0.89301011 0.92492196 0.89173946]
AdaBoostRegressor 의 정답률 0.9259751854556025
=====================================================


BaggingRegressor 의 crossvalidation [0.97115385 0.93684211 0.8756546  0.93879195 0.89      ]
BaggingRegressor 의 정답률 0.964765100671141
=====================================================


BayesianRidge 의 crossvalidation [0.91750582 0.9135044  0.93273173 0.94534611 0.9158692 ]
BayesianRidge 의 정답률 0.9055436990297968
=====================================================


CCA 의 crossvalidation [0.80820383 0.81452107 0.68993711 0.8364815  0.86662459]
CCA 의 정답률 0.8187077323940748
=====================================================


DecisionTreeRegressor 의 crossvalidation [0.92282958 0.94725275 0.94444444 0.86629526 0.93314763]
DecisionTreeRegressor 의 정답률 0.8993288590604027
=====================================================


DummyRegressor 의 crossvalidation [-0.02710843 -0.05146011 -0.06328125 -0.00138206 -0.00840336]
DummyRegressor 의 정답률 -0.010486577181207712
=====================================================


ElasticNet 의 crossvalidation [0.69166895 0.69544034 0.72996356 0.75658412 0.70552795]
ElasticNet 의 정답률 0.7066108603068253
=====================================================


ElasticNetCV 의 crossvalidation [0.95785507 0.92379445 0.94832067 0.87773127 0.93067259]
ElasticNetCV 의 정답률 0.9054184007985565
=====================================================


ExtraTreeRegressor 의 crossvalidation [0.72413793 1.         0.64705882 1.         1.        ]
ExtraTreeRegressor 의 정답률 0.6979865771812082
=====================================================


ExtraTreesRegressor 의 crossvalidation [0.98192143 0.9732972  0.8613883  0.95008877 0.9988625 ]
ExtraTreesRegressor 의 정답률 0.9429597315436241
=====================================================


GaussianProcessRegressor 의 crossvalidation [ 0.66162878  0.88335035 -0.16132306  0.88426081 -0.20360221]
GaussianProcessRegressor 의 정답률 0.6236633942510499
=====================================================


GeneralizedLinearRegressor 의 crossvalidation [0.86801466 0.87919662 0.89921799 0.86125885 0.83234028]
GeneralizedLinearRegressor 의 정답률 0.8513555448837072
=====================================================


GradientBoostingRegressor 의 crossvalidation [0.92007527 0.95289773 0.8301659  0.8962613  0.93222344]
GradientBoostingRegressor 의 정답률 0.9357447668151044
=====================================================


HistGradientBoostingRegressor 의 crossvalidation [0.94192808 0.91674577 0.97914165 0.99430053 0.84848484]
HistGradientBoostingRegressor 의 정답률 0.928390605664707
=====================================================


HuberRegressor 의 crossvalidation [0.90446154 0.89002605 0.93106801 0.95869936 0.94076487]
HuberRegressor 의 정답률 0.9036707892477903
=====================================================


KNeighborsRegressor 의 crossvalidation [0.99551402 0.9035514  0.87368421 0.98571429 0.97142857]
KNeighborsRegressor 의 정답률 0.9375838926174497
=====================================================


KernelRidge 의 crossvalidation [0.90433196 0.93307783 0.95649672 0.93062506 0.90067856]
KernelRidge 의 정답률 0.9063477491227538
=====================================================


Lars 의 crossvalidation [0.94300965 0.90516688 0.92370658 0.92000495 0.89004986]
Lars 의 정답률 0.9044795932990093
=====================================================


LarsCV 의 crossvalidation [0.94286701 0.88144024 0.95677167 0.85869261 0.94093805]
LarsCV 의 정답률 0.9044795932990093
=====================================================


Lasso 의 crossvalidation [0.38847547 0.38980915 0.30621762 0.37445991 0.3969527 ]
Lasso 의 정답률 0.41069764012321464
=====================================================


LassoCV 의 crossvalidation [0.93738134 0.94572929 0.93461515 0.92033943 0.90523267]
LassoCV 의 정답률 0.9049931850279125
=====================================================


LassoLars 의 crossvalidation [-0.00156685 -0.00138206 -0.00853064 -0.00074405 -0.00123626]
LassoLars 의 정답률 -0.010486577181207712
=====================================================


LassoLarsCV 의 crossvalidation [0.9393443  0.90650624 0.92501353 0.93781449 0.91764998]
LassoLarsCV 의 정답률 0.9044795932990093
=====================================================


LassoLarsIC 의 crossvalidation [0.90254    0.94995292 0.92097339 0.92642746 0.85426335]
LassoLarsIC 의 정답률 0.8896320119539598
=====================================================


LinearRegression 의 crossvalidation [0.93816271 0.91542229 0.92672886 0.88032894 0.94684103]
LinearRegression 의 정답률 0.9044795932990096
=====================================================


LinearSVR 의 crossvalidation [0.94369841 0.86856347 0.96690954 0.91443989 0.92076824]
LinearSVR 의 정답률 0.9045666871328574
=====================================================


MLPRegressor 의 crossvalidation [0.75477911 0.26699701 0.8909267  0.93737139 0.94021524]
MLPRegressor 의 정답률 0.8859677872731387
=====================================================


NuSVR 의 crossvalidation [0.95398224 0.94823024 0.87183688 0.94345476 0.97298966]
NuSVR 의 정답률 0.9244673733335868
=====================================================


OrthogonalMatchingPursuit 의 crossvalidation [0.91471277 0.92837184 0.89924969 0.94806714 0.85839287]
OrthogonalMatchingPursuit 의 정답률 0.8837955385010984
=====================================================


OrthogonalMatchingPursuitCV 의 crossvalidation [0.96024561 0.89936175 0.92871895 0.91161268 0.95418853]
OrthogonalMatchingPursuitCV 의 정답률 0.9044795932990095
=====================================================


PLSCanonical 의 crossvalidation [ 0.49088967  0.35066689  0.2244397  -0.05614707  0.02011599]
PLSCanonical 의 정답률 0.25725041480585165
=====================================================


PLSRegression 의 crossvalidation [0.9270311  0.96233529 0.89718692 0.92259765 0.89418263]
PLSRegression 의 정답률 0.8894807187718152
=====================================================


PassiveAggressiveRegressor 의 crossvalidation [0.92696562 0.92870785 0.88052114 0.88519787 0.88438199]
PassiveAggressiveRegressor 의 정답률 0.76512320523653
=====================================================


PoissonRegressor 의 crossvalidation [0.71586407 0.72487699 0.71855667 0.65214093 0.73780426]
PoissonRegressor 의 정답률 0.8087002910556181
=====================================================


RANSACRegressor 의 crossvalidation [0.93066639 0.95507032 0.88421042 0.89788738 0.94845521]
RANSACRegressor 의 정답률 0.9044795932990096
=====================================================


RadiusNeighborsRegressor 의 crossvalidation [0.92807196 0.92425077 0.90036252 0.95949458 0.97091887]
RadiusNeighborsRegressor 의 정답률 0.9242164261428097
=====================================================


RandomForestRegressor 의 crossvalidation [0.9308425  0.95518217 0.89631566 0.83956769 0.99684423]
RandomForestRegressor 의 정답률 0.9589614093959732
=====================================================


Ridge 의 crossvalidation [0.87758102 0.94487541 0.93214622 0.94202592 0.94878433]
Ridge 의 정답률 0.905957755289071
=====================================================


RidgeCV 의 crossvalidation [0.86692166 0.93220554 0.9188429  0.90124887 0.96105079]
RidgeCV 의 정답률 0.9048357192277003
=====================================================


SGDRegressor 의 crossvalidation [0.93908966 0.92376151 0.91929594 0.87953702 0.89021109]
SGDRegressor 의 정답률 0.8868125647804795
=====================================================


SVR 의 crossvalidation [0.94330109 0.95729555 0.93197827 0.92183451 0.96020066]
SVR 의 정답률 0.9213943776064586
=====================================================


TheilSenRegressor 의 crossvalidation [0.92552562 0.93090006 0.90447469 0.96434789 0.89675947]
TheilSenRegressor 의 정답률 0.8895242316608729


TransformedTargetRegressor 의 crossvalidation [0.90874032 0.94177074 0.9575169  0.90733866 0.92966223]
TransformedTargetRegressor 의 정답률 0.9044795932990096
=====================================================


TweedieRegressor 의 crossvalidation [0.81237411 0.89637111 0.85378123 0.88567069 0.88460299]
TweedieRegressor 의 정답률 0.8513555448837072
=====================================================

'''
