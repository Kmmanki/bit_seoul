# 11-23

키워드 

머신러닝, svc, randomForest, KNeighbors

## 머신러닝의 모델

단일 퍼셉트론의 경우 1차 함수 그래프 = 레이어를 늘리면 다차원 함수가 됨  = 복잡한 그래프를 그릴 수 있음.

- 딥러닝의 은닉층이 없는 경우 = 1차함수

- classfier = 분류 , regressor  = 회귀 but logisticRegressor = 분류

### 머신러닝의 모델간 분류 

m05~ 08

- model = LinearSVC() :분류

* model = SVC(): 분류

- model = KNeighborsClassifier() :분류

- model = KNeighborsRegressor(): 회귀

- model = RandomForestClassifier():분류

- model = RandomForestRegressor(): 회귀
  - 회귀 모델로 분류 가능
  - 분류모델로 회귀 불가능 Unknown label type: 'continuous'

- 모델이 분류라면 score = acc 회귀라면 score = r2