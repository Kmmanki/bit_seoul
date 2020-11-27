import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,Dropout
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import cv2

plt.figure(figsize=(10,10))
#1. 전처리
x=np.load('./homework/project1/npy/proejct1_x.npy')
y=np.load('./homework/project1/npy/proejct1_y.npy')

# print(x)

print(x.shape) #(12633, 178, 218, 3)
print(y.shape) #(12633,)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state =66)

# print(x_train.shape) #(1039, 178, 218, 3)
# print(x_test.shape)


model = Sequential()
model.add(Conv2D(32, (2,2), padding='valid' ,activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
model.add(Dropout(0.2))
model.add(MaxPooling2D())

model.add(Conv2D(32, (2,2), padding='valid' ,activation='relu' ))
model.add(Dropout(0.2))
model.add(MaxPooling2D())

model.add(Conv2D(64, (2,2), padding='valid' ,activation='relu' ))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
modelPath = './homework/project1/models/checkpoints/PJ-Conv2D-{val_loss:.4f}.hdf5'
checkPoint = ModelCheckpoint(filepath=modelPath, monitor='val_loss', save_best_only=True, mode='auto' )

ealyStopping = EarlyStopping(monitor='val_loss', mode='max', patience=20)
for i in range(1,3,2):
    
    hist = model.fit(x_train, y_train, 
                        epochs=300, 
                        batch_size=16, 
                        validation_split=0.2, 
                        callbacks=[ealyStopping, checkPoint]
                        )

    loss, acc = model.evaluate(x_test, y_test)

    x_predict = x_test
    y_predict = model.predict(x_predict)
    
    print('loss', loss)
    print('acc',acc)
    print("예상값", np.round(y_predict).reshape(y_predict.shape[0],))
    print("정답값", y_test)
    print('==========================================')
    print()
    print("예상값", np.sum(np.round(y_predict).reshape(y_test.shape[0],)))
    print('정답값의 합',np.sum( y_test))

    plt.subplot(5,2,i)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['loss, val_loss'])

    plt.subplot(5,2,i+1)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.legend(['acc, val_acc'])

    # if acc > 0.03:
    #     model.save('./homework/project1/models/model_Conv2D_train1_val_loss'+str(val_loss)+".h5")



plt.show()

'''
rsmprop
loss 0.18207214772701263
acc 0.989315390586853
예상값 [1. 1. 1. ... 1. 0. 0.]
정답값 [1. 1. 1. ... 1. 0. 0.]
예상값 1073.0
정답값의 합 1082.0

adam
loss 0.12941297888755798
acc 0.989315390586853
예상값 [1. 1. 1. ... 1. 0. 0.]
정답값 [1. 1. 1. ... 1. 0. 0.]
==========================================

예상값 1079.0
정답값의 합 1082.0
'''

'''
TODO: 
1. 필요없는 파라미터 제거 하는 것 추가
2. 이미지 출력해봐서 맞는지 확인 (했음 1 축구공, 0대머리)
3. 그냥 머리 추가
4. 캐글 이진분류, 다중분류 예제 찾기
5. xgboost 이미지 사용법 찾기
6. 스케일링 두번 한거 아닌가 확인
7. CV 추가

ps
작은 데이터 셋으로 강력한 이미지 분류 모델 설계
[김만기] [오후 9:43] https://keraskorea.github.io/posts/2018-10-24-little_data_powerful_model/

전이학습 모델
[김만기] [오후 9:54] https://jeinalog.tistory.com/m/13

개와 고양이 전이학습
[김만기] [오후 10:07] https://codingcrews.github.io/2019/01/22/transfer-learning-and-find-tuning/

'''
