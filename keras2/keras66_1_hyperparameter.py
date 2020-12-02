from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.layers import Dropout
import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28* 28).astype('float32')/255. 
x_test = x_test.reshape(10000, 28* 28).astype('float32')/255.

#2.모델

def build_model(drop=0.5, optimizer='adam'):
    input = Input(shape=(28*28, ), name='input')
    x = Dense(512,  activation='relu', name='hidden1')(input)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    output = Dense(10, activation='softmax', name='output')(x)

    model = Model(inputs=[input], outputs=[output])

    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batchs = [10,20,30,40,50]
    optimizer = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1,0.2,0.5]

    return {'batch_size': batchs, 'optimizer':optimizer, 'drop':dropout}
    
hyperparameters = create_hyperparameter()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=build_model, verbose=1) #keras를 sklean으로 래핑 시킨다.

search = GridSearchCV(model, hyperparameters, cv=3) #sklearn만 사용할 수 있다. 

search.fit(x_train, y_train)

print("best_estimator: ",search.best_estimator_)
print('best_params: ',search.best_params_)