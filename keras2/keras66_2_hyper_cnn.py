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

x_train = x_train.reshape(60000, 28, 28,1).astype('float32')/255. 
x_test = x_test.reshape(10000, 28, 28,1).astype('float32')/255.

#2.모델
def build_model(drop=0.5, optimizer='adam'
            , loss='categorical_crossentopy', patience=5, layernum=1, nodenum1=10, nodenum2=20):

    input = Input(shape=(28,28,1 ), name='input')
    x = Conv2D(16,(4,4), padding='valid', activation='relu', name='hidden1')(input)
    x = Dropout(drop)(x)

    for i in range(layernum):
        x = Conv2D(nodenum1, (4,4), padding='valid', activation='relu', name='nodenum1_'+str(i))(x)
        x = Dropout(drop)(x)
        x = Conv2D(nodenum2, (4,4), padding='valid', activation='relu', name='nodenum2_'+str(i))(x)
        x = Dropout(drop)(x)

    x = Flatten()(x)
    output = Dense(10, activation='softmax', name='output')(x)

    model = Model(inputs=[input], outputs=[output])

    # es = EarlyStopping(monitor='val_loss', mode = 'auto', patience=patience)
    model.compile(optimizer=optimizer, metrics=['acc'], loss=loss)
    return model

def create_hyperparameter():
    batchs = [216]
    optimizer = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1]
    epochs = [50]
    loss = ['categorical_crossentropy']
    patience = [5]
    layernum = [1]
    nodenum1 = [16]
    nodenum2 = [32]
    es = [
        EarlyStopping(monitor='loss', mode='auto', patience=1),
        EarlyStopping(monitor='loss', mode='auto', patience=2),
        EarlyStopping(monitor='loss', mode='auto', patience=3),
    ]
    
    return {'batch_size': batchs, 'optimizer':optimizer,
             'drop':dropout, 'epochs':epochs, 'loss': loss,
             
             'patience':patience, 'layernum':layernum,
             'nodenum1':nodenum1, 'nodenum2':nodenum2,
             'callbacks': es
             }
    
hyperparameters = create_hyperparameter()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=build_model, verbose=1) #keras를 sklean으로 래핑 시킨다.

search = RandomizedSearchCV(model, hyperparameters, cv=3) #sklearn만 사용할 수 있다. 
search.fit(x_train, y_train)

print("best_estimator: ",search.best_estimator_)
print('best_params: ',search.best_params_)