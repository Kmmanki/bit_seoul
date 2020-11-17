#  11-17



키워드

cifar, fashion, load_boston,diabetes, dropout

## keras.cifar 10 

keras39

다중분류를 위한 학습 이미지 데이터를 제공 10개의 종류를 가짐

## keras.fashion

keras40

다중분류를 위한 학습 옷 이미지 데이터를 제공 10개의 종류를 가짐

## keras.cifar 100

keras41

다중분류를 위한 학습 이미지 데이터를 제공 100개의 종류를 가짐

## sklean.load_boston

keras43

선형회귀를 위한 학습 보스턴 집값 데이터를 제공 

## sklean.load_diabetes

keras44

선형회귀를 위한 학습 당뇨 데이터를 제공 

- Conv2D의 아웃풋이 다음 레이어 Conv2D의 커널만큼 자를 수 없을 시 padding='same'
- model.add(Conv2D(30, (2,2), padding='valid', input_shape=(5,2,1)))
  -  5행 2열을 2,2로 strides=1로 잘라 나가면 => 아웃풋은 (4,1,30)
-  model.add(Conv2D(100, (2,2), padding='valid', activation='relu'))
  - 4행 1열을 2,2로 자르려는데 열이 부족하다???? 에러!!!!!
  - 방법은 2개 두 번째 Conv2d에 padding='same'을 사용하여 5,2로 계산
  - 첫 번째 Conv2D에 padding='same'을 줘서 아웃풋을 5,2,30으로 반환



<br>

----------

<br>



## DropOut

keras 42

기존의 레이어 노드를 변경하지 않고 노드 수를 줄이는 방법

- 기존의 레이어 다음에 Dropout 레이어를 추가 
  - 0.2를 넣으면 기존 노드의 20%를 제거
  - 학습 정확도가 검증 정확도에 비해 너무 높으면 적용

<br>

-----

<br>

