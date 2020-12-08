# 12-07

키워드

Bidirectional, xgboost, eval, outlier



<br>

--------

<br>

## Bidirectional

keras83

시계열의 데이터는 순방향 뿐만 아니라 역방향 또한 시계열로 판단할 수 있음

그렇다면 역방향으로 연산을 진행 한다면 좋은 결과를 얻을 수도 있음

```
model.add(Embedding(10000, 30, input_length=2376))
model.add(Bidirectional(LSTM(128))) #param 162816 #기존 모델을 아래에서 위로 한번 더 하기 때문에 2배
# model.add(LSTM(128)) # 81408
```

lstm의 파라미터의 2배 파라미터가 출력된다. 

<br>

---

<br>

## xgboost eval

xgboost의 훈련 도중의 경과 확인

```
model.fit(x_train, y_train, verbose=True,
            eval_metric='rmse',
            eval_set=[(x_test, y_test)]
            )
```

eval_metric

- rmse
- mae
- logloss
- error
  - 이진분류 
- merror
  - 다중분류
- auc

<br>
---------------------
<br>

## outlier
이상치를 제거 
ml32

