# 11-12

키워드 

RNN, LSTM GRU, simpleRNN

#### RNN (Recurrent Neural Network) keras17~24

Time Series(시계열) 데이터를 모델링 하기 위해 등장

Ex) 음악, 동영상, 에세이, 시, 소스 코드, 주가 차트, 

<br>

##### LSTM(Long Short-Term Memory models)

- 일반적으로 RNN중 가장 좋은 성능을 나타냄
- parameter = 게이트 수 * (자르는 수 + 바이어스 + 인풋노드 ) *인풋 노드 수 
  - `model.add(LSTM(10, activation='relu', input_shape=(3,1))` 
  - 게이트수 = 4, 자르는 수 = 1, 바이어스 = 1 ,인풋 노드=10
- 시계열 처리의 경우 가장 나중의 데이터가 높은 가중치가 있다.
- RNN의 경우 input_shape의 조정이 필요
  - (13,3) 데이터의 RNN에 필요한 shape는 (13,3,1)이됨
  - 13의 데이터의 수 3은 데이터의 속성, 1은 연산 시 묶음단위 연산
  - input_shape에서는 행은 무시하므로 shape = (3,1)
  - predict 시 array([65,75,85])는 (3,) 즉 요소3개인 데이터가 아닌  요소가 1개인 데이터3개
  - 3개의 요소를 갖는 1개의 데이터로 표현하기위해  reshape(1,3)

##### GRU, simpleRNN도 같은 사용하는 방식이 같다.

<br>

------------

<br>

#### EalyStopping keras21

keras.callbakc에 있는 class

과도한 학습이 진행되면 loss값이 높아지고, acc값이 낮아지는 지점이 발생

이 지점을 지나기 전에 학습을 중단하는 객체



​	`earlyStopping = EarlyStopping(monitor='loss', patience=125, mode='min')`

​	`model.fit(    x, y,    batch_size=1,    epochs=10000,    verbose=1,    `

`callbacks=[earlyStopping]  )`



loss의 값을 기준으로 가장 작은 값이 125에포크 만큼 갱신이 되지 않으면 종료

​	fit의 callback파라미터에 사용



<br>

----------------

<br>



#### return_sequensces

RNN은 13,3,1과 같이 마지막에 연산의 단위를 넣어주어야 함 

그러나 반환은 13,3 과 같은 형태로 반환이 되기 때문에 다음 노드에서 RNN을 사용 할 수 없음

Node의 파라미터에 return_sequenses=True로 주어지면 반환의 데이터를 입력과 같은 shape를 유지

<br>

`model.add(LSTM(300, input_shape=(3,1), return_sequences=True))`

`model.add(LSTM(100, input_shape=(3,1)))`

<br>



\#LSTM 80 = 75.03619 , loss = 24.639448165893555 param = 8,577

\#RNN 80 = 74.5879 loss = 29.2908 parm = 1,151

\#GRU 80 = 76.09706 loss = 15.232931137084961 , param = 7,585



