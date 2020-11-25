from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8)

n_estimators = 300
leaning_rate = 1
colsample_bytree =1
colsample_bylevel = 1

max_depth = 5
n_jobs = -1

model = XGBRegressor(
    max_depth=max_depth, leaning_rate =leaning_rate,
    n_estimators = n_estimators, n_jobs=n_jobs,
    colsample_bylevel = colsample_bylevel,  colsample_bytree = colsample_bytree
)

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

score = model.score(x_test, y_test)
r2 = r2_score( y_test, y_predict)

print("r2", r2)
print('score: ', score)
#score:  0.9134791130170657


plot_importance(model)
plt.show()

'''
나무 깊이 – max_depth (★★★)
-1로 설정하면 제한없이 분기한다. 많은 feature가 있는 경우 더더욱 높게 설정하며, 파라미터 설정시 제일 먼저 설정한다.
default는 -1로 제한없이 분기한다.

훈련량 – learning_rate / eta (★★★)
0.05~0.1 정도로 맞추고 그 이상이면 다른 파라미터들을 튜닝할때 편. 미세한 정확도 조정을 원하면 더 작게 
한번 다른 파라미터를 고정하면 바꿀필요가 없다. 또한 쓸데없이 긴 소수점으로 튜닝하는것도 별 쓰잘데기가 없다.

n_estimators 
부스트 트리의 양, 트리의 개수 

n_jobs
병렬처리 스레드의 개수

colsample_bylevel
트리의 레벨별로 훈련 데이터의 
변수를 샘플링해주는 비율. 보통0.6~0.9

colsample_bytree
트리를 생성할때 훈련 데이터에서 
변수를 샘플링해주는 비율. 보통0.6~0.9




'''