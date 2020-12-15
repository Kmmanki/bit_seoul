# 12-08

키워드

distribute, pickle, joblib, xgb.save



<br>

---

<br>

## Distribute

분산처리: 여러장의 gpu를 돌리는 것

```
#2. 모델
import tensorflow as tf
strategy = tf.distribute.MirroredStrategy(cross_device_ops=\
    tf.distribute.HierarchicalCopyAllReduce()    
    )

with strategy.scope():
    model = Sequential()
    model.add(Conv2D(10, (2,2), input_shape=(28,28,1), padding='same'))
    model.add(Conv2D(12, (2,2), padding='valid'))
    model.add(Conv2D(13, (3,3))) #27,27,20 받음
    model.add(Conv2D(15, (3,3), strides=2)) 
    model.add(MaxPooling2D(pool_size=2)) 
    model.add(Flatten()) 
    model.add(Dense(20, activation='relu')) 
    model.add(Dense(10, activation='softmax')) 
    model.summary()

    #3. 컴파일 , 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'], ) # 모든 acc를 합치면 1 
```

<br>

---

<br>

## pickle을 이용한 save 

pickle을 사용하여 저장, 로드

저장은 dump, 로드는 load

피클은 bgboost 전용이 아님 각종 저장 가능

```
import pickle
pickle.dump(model, open('./save/xgb_save/cancer.pickle.dat', 'wb'))
print('저장됨')

model2 = pickle.load(open('./save/xgb_save/cancer.pickle.dat', 'rb'))
```

<br>

---

<br>

## joblib을 이용한 save

pickle과 동일하게 각종 저장 가능

```
import joblib
joblib.dump(model, './save/xgb_save/job_cancer.pickle.dat' )

print('저장됨')
model2  = joblib.load('./save/xgb_save/cancer.pickle.dat')
acc2 = model2.score(x_test, y_test)
```

<br>

---

<br>

## xgb 전용 저장

```
model.save_model('./save/xgb_save/xgbmodel_cancer.pickle.model')
print('저장됨')

model2 = XGBClassifier()
model2.load_model('./save/xgb_save/xgbmodel_cancer.pickle.model')
acc2 = model2.score(x_test, y_test)
```

<br>

---

<br>

## python import

```
import p11_car
import p12_tv

print('==================')
print('do.py의 modul 이름 ',__name__)
'''
운전하다
car.py의 modul 이름  p11_car
시청하다
tv.py의 modul 이름  p12_tv
==================
do.py의 modul 이름  __main__


자기 자신이 호출하면 main 다른파일에서 호출하면 파일 명 
'''
```





```
def drive():
    print('운전하다')

if __name__ == '__main__':
    drive()
    
    '''
    다른 모듈에서 호출 시  drive를 호출 하지 않게 한다.
    다른 모듈에서 drive 출력
    ==================
	do.py의 modul 이름  __main__
    '''
```

```
import machine.car
import machine.tv

'''
모듈로 machine 폴더 아래 car, tv 파일이 존재한다.
'''

import sys

print(sys.path)
'''
파일패스 확인
'''
```







<br>

----

<br>

## 네이밍 룰



```
Class 대문자 시작 WachTv(), Wach_TV
함수 소문자 시작, do_you() <== ?, doYou() <= 카멜케이스
상수 전부 대문자 AAA=10
```



