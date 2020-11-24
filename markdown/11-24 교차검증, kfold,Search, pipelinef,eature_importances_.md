# 11-24

키워드 

교차검증, kfold,Search, pipelinef,eature_importances_

## 검증
m12

검증하는 대상이 모든 데이터를 대변할 수 없음.

validationSet을 계속해서 변경하여 사용(Cross Validation, CV)

### KFold 교차검증

**held-out validation** 은 전체의 데이터중 일부를 validation set으로 사용 하는 것으로 데이터 셋의 크기가 작으면 평가의 신뢰성이 낮아짐

이를 해결하기 위해 모든 데이터가 최소 한번은테스트 셋으로 쓰이도록  고안된방법이 **KFold**

- n = 전체의 데이터를 n개로 나누어 나눈 것 하나하나를 검증셋으로 사용

  - Ex 100개의 데이터 n=5 

  - 데이터는 20개씩 5개로 쪼개어 0은 validationset 1은 trainset

  - 01111, 10111,11011,11101,11110 총 5개의 score를 반환

    

### GridSearchCV
m14
하이퍼 파라미터를 경우에 수를 모두 확인하여 모델을 반환

```
parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"]  },
    {"C":[1,10,100,1000], "kernel":["rbf"], 'gamma':[0.001, 0.0001]  },
    {"C":[1,10,100,1000], "kernel":["sigmoid"], 'gamma':[0.001, 0.0001]  }
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


```

```
최적의 매개변수 SVC(C=1, kernel='linear')
최종 정답률 0.9666666666666667
```



파라미터가 너무 많다. -> randomizedSearchCV 

```
model = RandomizedSearchCV(SVC(), parameters , cv=kfold)
```



<br>

-----------

<br>



### Pipeline
m16

pipeline을 사용하게 되면 cross_val 사용 시 train만 fit을 하게 되며 validation은 transform만 진행됨



```
parameters = [
{
'randomforestclassifier__n_estimators':[10,100,200,300], #생성할 트리의 개수
    'randomforestclassifier__max_depth':[6,8,10,12,15],
    'randomforestclassifier__min_samples_leaf':[5,7,10,12],
    'randomforestclassifier__min_samples_split':[3,5,10,12],
    'randomforestclassifier__n_jobs':[-1]} # 모든 코어를 사용
 
]

# kfold = KFold(n_splits=5, shuffle=True) cv=5로 대체

pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier(), )
# pipe = Pipeline([
#     ("minmax",MinMaxScaler()), ('malddong',RandomForestClassifier())
# ])
model = RandomizedSearchCV(pipe, parameters , cv=5, verbose=2)

```

<br>

--------------

<br>



## feature_importances_
m18
어떠한 컬럼이 결과를 만드는데 가장 영향을 많이주는 컬럼인지 파악 가능



```
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
model = GradientBoostingClassifier(max_depth=4)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)
print(model.feature_importances_) # 중요한 컬럼이 무엇인지 알려줌

DecisionTreeClassifier
총합은 1
[0.         0.         0.         0.         0.         0.
 0.         0.70458252 0.         0.         0.         0.00639525
 0.         0.01221069 0.         0.         0.         0.0162341
 0.         0.0189077  0.05329492 0.05959094 0.05247428 0.
 0.00940897 0.         0.         0.06690062 0.         0.        ]

RandomForestClassifier
총합은 1
acc 0.9385964912280702
[0.         0.         0.         0.         0.         0.
 0.         0.70458252 0.         0.         0.         0.00639525
 0.         0.01221069 0.         0.         0.         0.0162341
 0.         0.0189077  0.05329492 0.05959094 0.05247428 0.
 0.00940897 0.         0.         0.06690062 0.         0.        ]

GradientBoostingClassifier
acc: 0.956140350877193
[7.76584550e-06 4.35018642e-02 1.25786490e-04 5.74942454e-05
 2.50855217e-04 5.47308200e-04 9.03921649e-04 6.71611663e-01
 8.27929223e-04 6.11656786e-05 5.85725865e-03 1.92017782e-03
 3.69955119e-06 6.28758964e-03 1.32754336e-03 3.48643223e-04
 7.88182170e-03 1.53824214e-02 5.42112298e-04 1.09307689e-02
 5.80947359e-02 2.18344681e-02 5.50936037e-02 3.63666418e-03
 8.66421877e-03 2.12310255e-03 2.42599678e-03 7.93267827e-02
 3.90570999e-04 3.20657111e-05]
 
 
```



