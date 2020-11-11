# 11-11 

### keyword

다차원 행렬의 전치행렬, 다차원 행렬 슬라이싱, verbos, 다차원 행렬의 input_dim 처리, 함수형모델, 앙상블, Concatenate 

## 

숙제 emsemble3  튜닝

#### 다차원 데이터 keras12 ~14(일일이 링크걸기 귀찮...)

###### 행열의 변환

- x = np.array([range(1, 101), range(311, 411), range(100)]) 
  
- 로 진행하게되면 (3,100) => 개의 데이터를 가지고  각데이터는 100개의 속성을 가진다.
  
- 내가 원하는것은 3개의 속성을 가지는 데이터를 100개 원한다. 

  - x.T
  - np.transepose(x)
  - x.transepose()
  - 결과 (100,3)으로 변환 => 100 개의 데이터를 가지며 각 데이터는 3개의 속성을 가짐

  <br>

##### 데이터(x)

| 행/열 | 주식(x1) | 채권(x2) | 환율(x3) |
| ----- | -------- | -------- | -------- |
| 1     | 100      | 105      | 207      |
| 2     | 109      | 208      |          |



##### 주식가격(y)

| num  | 주가 |
| ---- | ---- |
| 1    | 1000 |
| 2    | 2000 |

| 306  |
| ---- |
|      |

y = w1x1 * w2x2 * w3x3 + b

##### 다차원의 행렬슬라이싱

- (3,100 ) -> (100,3) 으로 변환
  - 즉 총 데이터 100개가 있다 이중 60개를 잘라 train과 test로 나누면 된다.
- x_train = np.array(list(x)[:60]) 
- x_test = np.array(list(x)[60:100])

<br>

----------------------------------

<br>

##### 다차원의 입력 

- 3(3,)입력 -> 1출력 (y = w1x1 * w2x2 * w3x3 + b)
  - input_shape(3,)
  - model.add(Dense(1))
- 1입력 -> 3출력(y1, y2, y3 =  wx + b)
  - input_shape(1,)
  - model.add(Dense(3))
- (1000, 100, 10, 1) 입력 -> 1출력 => 예제가 나오면 많이 연습필요
  - input_shape(100,10,1) -> 즉 행을 무시한다! (맨 앞은 데이터의 갯수 )
  - model.add(Dense(3))



**R2가 음수면 모델이 문제다!**



<br>

-----------

<br>

#### verbos 

<a href='https://github.com/Kmmanki/bit_seoul/blob/main/keras/keras14_verbose.py'>keras14</a>

loss 보다 val_loss를 더 많이 보게된다.

fit의 옵션으로  verbos => 진행상황의 출력을 알려주는 옵션 설정

- verbose =2 진행과정의 진행 bar가 사라진다.

-  verbose = 0 => 진행과정 없이 결과만 나온다, defualt = 1
- verbos = 3 => 진행과정중 epoch만 보여준다.

<br>

----------------

<br>

#### 함수형 모델 
<a href='https://github.com/Kmmanki/bit_seoul/blob/main/keras/keras15_hammsu.py'>keras15</a> 

- 모델의 재사용
- 선형회귀에서 마지막output의 활성화 함수는 linear

**Sequential**(3,)

| Layer (type) | Output Shape | Param # |
| ---- | ---- | ---- | ---- |
| dense (Dense) | (None, 5) | 20 |
| dense_1 (Dense) | (None, 4) | 24 |
| dense_4 (Dense) | (None, 3) | 15 |
| dense_3 (Dense)    | (None, 1) | 4 |

_________________________________________________________________
Layer (type)                 Output Shape              Param #

dense (Dense)                (None, 5)                 20
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 24
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 15
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 4
=>input 명시하지 않음
<br>
Total params: 63
Trainable params: 63
Non-trainable params: 0

- param = w를 구하기 + b구하기  이전 노드 * 다음 노드 + 1x다음노드
  - 즉 이전노드 +1 * 다음노드  = 파라미터의 수 

**함수형(Model)(3,)**

Model: "functional_1"

| Layer (type) | Output Shape | Param # |
| ---- | ---- | ---- |
| input_1 (InputLayer) | [(None, 3)] | 0 |
| dense (Dense) | (None, 5) | 20 |
| dense_1 (Dense) | (None, 4) | 24 |
| dense_2 (Dense) | (None, 3) | 415|
| dense_3 (Dense) | (None, 2) | 4 |
=>input 명시
<br>

Total params: 63
Trainable params: 63
Non-trainable params: 0

<br>


함수형 모델은 layer의 이전 레이어를 명시하여 이어나감

<br>

m1_input = Input(shape=(3,), name='model1_input')

m1_dense1 = Dense(300, activation='relu', name='m1_d1')(m1_input)  <- 이전 레이어

m1_dense2 = Dense(200, activation='relu', name='m1_d2')(m1_dense1) <-이전 레이어

<br>

1개의 레이어에서 2개의 레이어 분기 시에도 각 레이어는 동일한 이전 레이어를 명시

<br>

output1 = Dense(32, name='out1_d1')(m1_dense2 ) 

output2 = Dense(16, name='out2_d1')(m1_dense2 )



<- 서로 같은 이전 레이를 받는 서로 다른 2개의 아웃풋 레이어의 시작

<br>

------------

<br>

#### 앙상블 keras16

주식, 채권, 환율 | 온도, 습도 불괘지수  => 주가를 알 수 있다면

2개의 데이터를 1개의 (100,6) 행열로 만들어 진행 할 수 있음 하지만 가중치를 줄 수 없음.



- 그러나 앙상블을 통해 학습한다면 주식, 채권, 환율에 높은 가중치를 줄 수 있다.

- model + model **Concatenate **
  - x2개, y2개 <a href='https://github.com/Kmmanki/bit_seoul/blob/main/keras/keras16_ensemble.py'>ensemble</a>
    - input model 2개  -> Concatenate -> output 2개
  - x2개, y3개 <a href='https://github.com/Kmmanki/bit_seoul/blob/main/keras/keras16_ensemble2.py'>ensemble2</a>
    - input model 2개 -> Concatenate -> output 3개
  - x2개, y1개 <a href='https://github.com/Kmmanki/bit_seoul/blob/main/keras/keras16_ensemble3.py'>ensemble3 **가장 빈번한 케이스**(최적화 됨)</a> 
    - input model 2개   Concatenate -> output 1개
  - x1개, y3개 <a href='https://github.com/Kmmanki/bit_seoul/blob/main/keras/keras16_ensemble4.py'>ensenmble4</a> 
    - input model1개 -> output model 3개 
- model의 최종반환(모델의 연결간 종단 x)의  activation은 linear
  - relu의 경우 다음 노드로 넘어 갈 시 0이하의 값을 전달하지 않음
  - 최종 결과에서는 0이하의 값 또한 필요하기에 linear 활성함수 사용







































