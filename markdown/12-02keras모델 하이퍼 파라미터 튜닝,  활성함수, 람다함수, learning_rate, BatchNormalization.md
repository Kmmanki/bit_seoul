# 12-02

키워드

keras모델 하이퍼 파라미터 튜닝,  활성함수, 람다함수, learning_rate, BatchNormalization

<br>

##  wrapper.sckit_learn

kers66, 72 

sklearn에서 사용하는 모델들을 keras에서 사용하기 위해 래핑

이를 사용하여 GridSearchCV, RandimizedSearchCV, Pipeline 등을 사용 할 수 있다.

```
(x_train, y_train), (x_test, y_test) = mnist.load_data()


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28* 28).astype('float32')/255. 
x_test = x_test.reshape(10000, 28* 28).astype('float32')/255.

#2.모델
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

def build_model(drop=0.5, optimizer=Adam, lr = 0.001
            , loss='categorical_crossentopy', patience=5, layernum=1, nodenum1=10, nodenum2=20):

    input = Input(shape=(28*28, ), name='input')
    x = Dense(512,  activation='relu', name='hidden1')(input)
    x = Dropout(drop)(x)

    for i in range(layernum):
        x = Dense(nodenum1, activation='relu', name='nodenum1_'+str(i))(x)
        x = Dropout(drop)(x)
        x = Dense(nodenum2, activation='relu', name='nodenum2_'+str(i))(x)
        x = Dropout(drop)(x)

    output = Dense(10, activation='softmax', name='output')(x)

    model = Model(inputs=[input], outputs=[output])


    model.compile(optimizer=optimizer(learning_rate=lr), metrics=['acc'], loss=loss)
    model.summary()
    return model

def create_hyperparameter():
    batchs = [40,]
    optimizer = [Adam]
    dropout = [0.1]
    epochs = [10]
    loss = ['categorical_crossentropy']
    patience = [5]
    layernum = [2]
    nodenum1 = [16,]
    nodenum2 = [32,]
    validation_split= [0.1]
    lr = [0.001]
    return {'batch_size': batchs, 'optimizer':optimizer,
             'drop':dropout, 'epochs':epochs, 'loss': loss,
             
             'patience':patience, 'layernum':layernum,
             'nodenum1':nodenum1, 'nodenum2':nodenum2,
             'validation_split':validation_split, 'lr':lr}
  
hyperparameters = create_hyperparameter()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=build_model, verbose=1) #keras를 sklean으로 래핑 시킨다.

search = RandomizedSearchCV(model, hyperparameters, cv=3) #sklearn만 사용할 수 있다. 

search.fit(x_train, y_train)

print("best_estimator: ",search.best_estimator_)
print('best_params: ',search.best_params_)

'''
best_params:  {'validation_split': 0.1, 'patience': 10, 'optimizer': 'nadam', 
'nodenum2': 64, 'nodenum1': 32, 'lr': 0.001, 'loss': 'binary_crossentropy', 
'layernum': 1, 'epochs': 10, 'drop': 0.1, 'batch_size': 50}
'''
```

위와 같이 파라미터를 지정 할 때 모델의 변수명과 일치 해야하며 batch_size와 같이 기본 값으로 넣어주어도 된다.

<br>

-------

<br>



## 람다함수

keras67

함수를 간단하게 만드는 것 

```
gradient = lambda x: 2*x - 4 #람다 함수: 함수 만들어 준거다 x는 입력


def gradient2(x):
    return 2*x - 4

print(gradient(4))#4
print(gradient2(4))#4
```



<br>

-------

<br>

## learning_rate 

keras69

loss를 구하기 위한 파라미터로 loss 그래프에서 기울기가 0인 지점을찾기위한 수 

큰 값을 가지게 되면 0인 지점을 찾을 수 없고 너무 작은 값을 가지면 오랜 학습기간을 가지게 됨

<br>

-------

<br>

## 활성 함수 별 그래프와 특징

keras71

- sigmoid 

  - sigmoid를 통과한 값은 0 ~ 1사이로 수렴

  <img width="560" height= "560" src='https://hvidberrrg.github.io/deep_learning/activation_functions/assets/sigmoid_function.png'>



- tanh

  - 값이 -1 ~ 1로 수렴

  <img width="560" height= "560" src= 'https://www.mathworks.com/help/examples/matlab/win64/GraphHyperbolicTangentFunctionExample_01.png'>



- relu, selu 

<img src="https://www.researchgate.net/profile/Guoqiang_Zhang22/publication/326646583/figure/fig1/AS:652848589197321@1532662641462/Subplot-a-shows-6-different-activation-functions-where-their-main-differences-sit-in.png">



- relu는 0이하의 값은 0으로 손실된다. 

  - 0인 값 또한 손실 된다. 그렇기에 -의 값은 0이 아닌 아주 작은 수로 

  - ELU a(e**x -1) ~ x 

    Leaky Relu 0.1x ~ x

    selu lamda a(e**x -1) ~ x 



<br>

----------

<br>

## BatchNormalization

활성함수 이전에 진행되어야함

```
model.add(Conv2D(64, (2,2),kernel_regularizer=l1(0.01), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
```

위와 같은 방식으로  acvivation을 레이어로 넘기게 되면 batchnormalization을 사용하는것이 좋다.



