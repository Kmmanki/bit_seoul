# 11-18

키워드

이진분류, sigmoid, mode.save, ModelCheckPoint



## sklean.load_Iris

숫자 데이터로 이루어진 분류 학습용 데이터

- 이미지라면 소프트맥스는 아니다!!!
- 남자 여자 외계인 으로 분류하는데 키로 분류한다! 
  - 이미지가 아니더라도분류이다!! 키가 300cm

train_test_split은 



<br>

---------

<br>

## Sigmoid(이진분류)

|          | 다중분류                 | 이진분류            |
| -------- | ------------------------ | ------------------- |
| 라벨링   | 필요                     | 필요없음            |
| 활성함수 | softmax                  | sigmoid             |
| loss함수 | categorical_crossentropy | Binary_crossentropy |



<br>

---------

<br>



이진분류모델, 회귀모델 다중분류모델 = 결과! 

결과가 이진, 회귀, 다중인 것 일 뿐 DNN RNN CNN -> 레이어 무엇으로 진행하도 된다!



<br>

---------

<br>



## ModelCheckPoint

모델의 weight값을 저장하는 방법으로 fint의 callback에 입력

- modelPath = './model/{epoch:02d}--{val_loss:.4f}.hdf5'

  - 에포크의 2자리 수, val_loss의 4자리 실수로 저장

- checkPoint = ModelCheckpoint(filepath=modelPath, monitor='val_loss', save_best_only=True, mode='auto' )

  - save_best_only는 최소 val_loss가 갱신 될 대마다 저장된다.

- modelCheckPoint는 모델과 가중치 모두 저장

  

<br>

---------

<br>

## model.save()

```
model.save() #1번

model.fit()

model.save()#2번
```

```
model1 = load_model(path+'.h5')
```

- 1번의 저장은 모델만 저장이 된다.

- 2번의 저장은 학습, 모델이 저장됨(모델 + 가중치 저장)

```
model.save_weight()
model3.load_weights(path+"_weight.h5")
```

- 모델의 가중치만 저장

  - 가중치만 저장하기 때문에 모델, 컴파일이 필요

    



<br>

---------

<br>

```
plt.figure(figsize=(10,6))										#가로 10인치 세로 6인치 기본창
plt.subplot(2,1,1) 											#(2행 1열의 첫 번째 그래프 선택)
plt.plot(hist.history['loss'], marker='.', c='red')				# 그래프에 loss를 그리고 각 에포크마다 마커를 . 색상은 red
plt.plot(hist.history['val_loss'], marker='.', c='blue')
plt.grid()													#격자무늬 그리드 표시
plt.title('loss')											#제목
plt.ylabel('loss')											#Y축 제목 (Y축 라벨링)
plt.legend(['loss','val_loss'])								#설명표 plot에 넣은 순서대로 

plt.subplot(2,1,2)
plt.plot(hist.history['mae'], marker='.', c='red')
plt.plot(hist.history['val_mae'], marker='.', c='blue')
plt.grid()
plt.title('mae')
plt.ylabel('mae')
plt.legend(['loss','val_mae'])

plt.show()
```



