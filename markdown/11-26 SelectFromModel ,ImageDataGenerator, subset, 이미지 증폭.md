# 11-26

키워드

SelectFromModel ,ImageDataGenerator, subset, 이미지 증폭



## SelectFromModel

모델에서 피처를 하나씩 제거하며 r2를 확인 할 때 사용

```
for thresh in thresholds:
    selection = SelectFromModel(model, threshold = thresh, prefit=True) # 피처의 개수를 하나씩 제거
    
    select_x_train = selection.transform(x_train) # 피쳐의 개수를 줄인 트레인을 반환

    selection_model = XGBRegressor(n_jobs=-1) # 모델 생성 
    selection_model.fit(select_x_train, y_train) #모델의 핏

    select_x_test = selection.transform(x_test) # 피쳐의 개수를 줄인 테스트 반환
    y_predict = selection_model.predict(select_x_test) # 프레딕트 

    score = r2_score(y_test, y_predict)

    invaild_feature.append(thresh)
    if select_x_train.shape[1] ==9:# 9개만 남기고 나머지 제거 
        break

    print("Thresh%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
    
```



<br>

--------

<br>

## ImageDataGenerator

이미지 데이터를 사용 할 때 

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

#이미지에 대한 생성 옵션 지정
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   horizontal_flip=True, #50% 확률로 수평으로 뒤집음
                                   vertical_flip=True, #50% 확률로 수직으로 뒤집음
                                   width_shift_range=0.1, #왼쪽, 오른쪽 움직임 (평행 이동)
                                   height_shift_range=0.1, #위, 아래 움직임 (평행 이동)
                                   rotation_range=5, #n도 안에서 랜덤으로 이미지 회전
                                   zoom_range=1.2, #range 안에서 랜덤하게 zoom
                                   shear_range=0.7, #0.7 라디안 내외로 시계반대방향으로 변형
                                   fill_mode='nearest', #이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
                                   validation_split=0.2
                                   )

test_datagen = ImageDataGenerator(rescale=1./255)

#flow 또는 flow_from_directory
#실제 데이터르 알려주고 이미지 불러오기

# 하위 디렉토리0= ad, 1= nomal로 라벨링됨
xy_train = train_datagen.flow_from_directory(
    './data/data2', # target directory
    target_size=(200,200), 
    batch_size=8,
    class_mode='binary', # 이진분류 
    subset='training'
    #, save_to_dir='./data/data1_2/test'
    #,subset='training'
)
# next(xy) #save_to_dir에 해당 이미지 저장

xy_test = test_datagen.flow_from_directory(
    './data/data1/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',
    #, save_to_dir='./data/data1_2/test'
    ,#subset='validation'
)


model= Sequential()

model.fit_generator(
   xy_train,
    steps_per_epoch=100, #augmentation한거에서 100개만 뽑음 /= 제너레이터로부터 얼마나 많은 샘플을 뽑을 것인지
    #보통은 데이터셋의 샘플 수를 배치 크기로 나눈 값
    epochs=50,
    validation_data=xy_valid,
    validation_steps=50, #한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정
    #보통은 검증 데이터셋의 샘플 수를 배치 크기로 나눈 값
)
```



- subset
  - 디렉토리 구조가 Train/Data A, Train/ Data B, Test/ Data A, Test/Data B 와 같은경우 사용할 필요 X
  - 디렉토리 구조 ./Data A, Data B 와 같다면 subset을 사용하여 Train, Test를 구분 할 수 있다.