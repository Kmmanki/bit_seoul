# 11-16

키워드 

CNN, Conv2d, 분류 모델, 지난주 복습

지난주 복습

- Deep = Node + Layer

- y = wx + b
  - w는 모든 파라미터에서 연산

  - 연산을 통해 loss를 최소화 
  - 파라미터는 아웃풋 * 다음 노드 + b! b = 다음노드의 갯수 

- Dense

  - 2 차원 => (열,) -> (3,)
  - 1차원 반환

- 모델의 합치기 => 앙상블 => Contcatenate 사용 

- LSTM  => 순차열 데이터 처리 

  - 일반 Dense 보다 많은 연산을 처리
  - 게이트 수 (처리 개수 + 바이어스 + 노드수 )* 노드수
  - 3 차원 => (행,열, 처리 개수) => input_shape = (열, 처리 개수) =>(3,1).reshape(3,1,1)
  - 두 개 이상의 LSTM을 역는 방법 => return_sequenses
  - 2차원 반환

- hisory

  - fit실행 시 에포마다의 loss, metrics  반환

- EalyStopping

  - moitor, patient, mode 파라미터가짐
  - overfit을 방지하기 위해 조기 종료

- 시각화

  - tensorboard, matplotlib을 통한 history의 그래프화

- 데이터 전처리

  - 85%의 데이터 전터리와 15% 모델링 => 데이터 처리의 중요성

  - 모델의 저장과 불러오기 -> 추후 W의 저장

  - Scaler (Minmax, Standard, MaxAbs, Robust)

    - Minmax 

      - 데이터의 값을 0~1로 수정
      - 일반적인 그래프의 밖 데이터 때문에 오차 생김

    - Standard

      - 표준편차 그래프의 중앙을 0으로 수정
      - 중앙의 값을 사용하기에 최소, 최대의 값에 가까울 수록 (이상치) 영향을 줄임

      <a href='https://github.com/Kmmanki/bit_seoul/blob/main/markdown/11-13%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%ED%8C%8C%EC%8B%B1%2C%20%EB%AA%A8%EB%8D%B8%EC%9D%98%20%EC%A0%80%EC%9E%A5%EA%B3%BC%20%EB%A1%9C%EB%93%9C%2C%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%9D%98%20%EC%8B%9C%EA%B0%81%ED%99%94%2C%20%EC%A0%84%EC%B2%98%EB%A6%AC.md#%EC%A0%84%EC%B2%98%EB%A6%ACkeras34_minmax'>11-13 데이터파싱 표 참고</a>

      <br>

      ------------

      <br>

## CNN(Convolutional Neural Network 합성곱 신경망)

컴퓨터가 이해하기 쉽도록 나누어 특징을 찾아냄

### Conv2d

 - model.add(Conv2D(10, (2,2),input_shape=(5,5,1))) 

    - filter
      	- 출력 공간의 차원 = 다음 레이어에 넘길 노드의 갯수 = 노드 수 
    - Kernel_size:
       -  2D Convolutional  창의 높이와 너비 
       - Ex(2,2) 가로2 높이2의 16픽셀 -> 5,5 -> kernel(2,2) ->(4,4)됨
    - strides: 
       - Convolutional 의 스트라이드를 지정하는 정수디폴트 =1 
       - 몇 칸씩 띄워서 연산할 것인가
    - pading: 경계 처리 방법
       - same : 가장자리의 데이터의 연산의 횟수가 중앙의 데이터의 연산 횟수와 차이나는 것 때문에 사용 (5,5) ->kernel(2,2) -> (5,5)
       - vaild: 디폴트 vaild 사용한함
    - input_shape: (rows, cols, channels) => 가로 세로 특성
       - 입력모양: (배치사이즈, rows,cols,channels) 
       - 한번에 처리 할 이미지의 파일의 수 , 가로, 세로, 특성
       - 파일이 6만개 있을 경우 6만개의 파일을 한번에 처리하는 단위  = 배치사이즈
    - param 계산
       - model.add(Conv2D(10, (2,2),input_shape=(10,10,1)))
          - 파라미터의 수 = (채널 * 커널사이즈 * +1(바이어스) ) 나가는 노드
          - ( 1 * 2 * 2 + 1 ) * 10 
<br>
------
<br>
### Maxpolling2d

   - stride 없이 잘라 feature가 높은 픽셀만 남기는 기법
   - stride 없기 때문에 반으로 줄어듬 (3,3,7) 노드는 이전의 노드 가져옴
<br>

------
<br>
### Flatten()

   - Conv2d를 스칼라로 넘겨주어야 함.
   - model.add(Flatten())를 통해  변환
   - rowsxcolsxchennel = 63(노드 수 )

<br>

-------

<br>

### 원핫 인코딩

   - sklearn(skrlearn.preprocessing.OneHotEncoder)
   - keras(tensorflow.keras.utils.to_categorical)
   - (1) 각 단어에 고유한 인덱스를 부여 (정수 인코딩)
     (2) 표현하고 싶은 단어의 인덱스의 위치에 1을 부여, 다른 단어의 인덱스의 위치에는 0을 부여
      - 1 : 10000
      - 2: 01000
      - 3: 00100
      - 4: 00010
      - 5: 00001
      - <br>
      - y_predict의 값은 총합이 1이 되며  각 자리 수 중 가장 큰 값을 추출하여  분류

<br>

-------

<br>

### Conv2d 과정

- 데이터 전처리

   - mnist에서 데이터 불러오기, train, test

   - 분류 모델이기 때문에 y를 oneHotEncoding 하기

   - Conv2d의 reshapesms 이미지 수, 가로, 세로, 채널로 reshape필요

   - 빠르고 정확하 연산을 위해 minmaxScaling

- 모델링

   - CNN 중 Conv2d를 이용한 모델링

   - Conv2d의 input_shape는 가로 세로 채널로 채널은 흑백은 1 컬러는 3을 가진다.

   - Conv2d에서 Dense로 연결 시 Conv2d는 4치원을 반환하므로 Flatten을 사용하여 스칼라로 output으로 반환

   - 최종 레이어에서의 노드 수는 분류의 총 종류 수, activation은 반드시 softmax

- 컴파일 및 훈련

   - 분류의 loss는 반드시 categorical_crossentropy

   - 분류의 메트릭스는 acc로 

- 예측 및 y_predict구하기

   - y_predict는 OneHotEncoding된 상태로 반환되기 때문에 np.argmax를 사용하여 디코딩 필요

   - np.argmax(y_predict, axis=1)를 사용 하여 각 행마다 디코딩

<br>

-------

<br>

### DNN으로 구현하는 분류

- Conv2d와 다른점은 input_shape의 입력과 입력의 reshape
   - x_train.shape가 60000,28,28이라면
   - x_train.reshape(60000,28x28)
   - input_shape=(28x28,) 
   - 나머지는 Conv2d와 같다.















