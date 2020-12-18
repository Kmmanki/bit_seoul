# 12-16

키워드



## 비지도학습

k-mean

주어진 데이터를 k개의 클러스터로 묶는 알고리즘으로, 각 클러스터와 거리 차이의 분산을 최소화하는 방식으로 동작한다. 이 알고리즘은 자율 학습의 일종으로, 레이블이 달려 있지 않은 입력 데이터에 레이블을 달아주는 역할을 수행	

```
from sklearn.cluster import KMeans
model = KMenas(n_clusters=k)
k = 분류할 종류의 수
```

## <br>

----------

<br>

## Auto Encoder

```
#mnist DATA
input = Input(shape=(784,))
encode = Dense(64, activation='relu')(input)
decode = Dense(784, activation='relu')(encode)

autoencoder = Model(input, decode)

autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='mse')
```

원본의 데이터를 64까지 줄여주고 (압축) relu 함수로 인해 0이하의 값들이 소실됨, 노이즈(0이하의 값)을 제거 후 원복

mse와 binary_crossentropy 의 차이는 무엇?