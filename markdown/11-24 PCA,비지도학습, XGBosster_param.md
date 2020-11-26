# 11-24

키워드

PCA,비지도학습, XGBosster_param

## PCA(주성분 분석, 차원 축소)

데이터의 수 많은 feature중 유효하지 않은 feature를 제거

x의 feature30  => y 1개

A                                   B                    C

x의 feature30 => x의 feature5 => y 1개

A의 y = B , B의 y = c



```
print(x.shape) #442,10 
pca = PCA(n_components=9) ->축소 완료한 컬럼의 수 
x2d = pca.fit_transform(x)
print(x2d.shape) # 442, 9
```





<br>

------------------

<br>



## 비지도학습

- y의 값이 없는 것
- 군집방식

<img src='https://t4.daumcdn.net/thumb/R720x0/?fname=http://t1.daumcdn.net/brunch/service/user/Jr9/image/qExzc3o2GigQ5Cd8mPsxF9WJEXI.png'/>





```
#feature_importances 20% 제거 방법
index7 =np.sort(model.feature_importances_)[::-1][int(0.7 *len(model.feature_importances_) )] 하위 20%의 값 가져오기

#index7 보다 작은 값의 index 찾기
delete_list = []
for i in model.feature_importances_:
    if i < index7:
        print(i,"제거 ")
        delete_list.append(model.feature_importances_.tolist().index(i))
        
        #해당 인덱스의 열 삭제 
x_train  = np.delete(x_train, delete_list, axis=1)
x_test  = np.delete(x_test, delete_list, axis=1)
```





<br>

----

<br>



## XGBoost Parameter

```
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, shuffle=True, train_size=0.8)

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
score = model.score(x_test, y_test)

print('score: ', score)
#score:  0.9134791130170657
```



- max_depth

  - -1로 설정하면 제한없이 분기한다. 많은 feature가 있는 경우 더더욱 높게 설정하며, 파라미터 설정시 제일 먼저 설정한다.

    default는 -1로 제한없이 분기한다.

- learning_rate

  - 0.05~0.1 정도로 맞추고 그 이상이면 다른 파라미터들을 튜닝할때 편. 미세한 정확도 조정을 원하면 더 작게 

    한번 다른 파라미터를 고정하면 바꿀필요가 없다. 또한 쓸데없이 긴 소수점으로 튜닝하는것도 별 쓰잘데기가 없다.

- n_estimators

  - 부스트 트리의 양, 트리의 개수 

- n_jobs

  - 병렬처리 스레드의 개수

- colsample_bylevel

  - 트리의 레벨별로 훈련 데이터의 

    변수를 샘플링해주는 비율. 보통0.6~0.9

- colsample_bytree

  - 트리를 생성할때 훈련 데이터에서 

    변수를 샘플링해주는 비율. 보통0.6~0.9