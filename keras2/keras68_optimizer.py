from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np

#1. 정제된 데이터 세팅

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([2,3,4,5,6,7,8,9,10,11])


#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=1, activation='relu'))
model.add(Dense(16))
model.add(Dense(1))
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
li = []
for i in [0.1,0.001,0.0001]:
    
    #3. 데이터 모델 훈련
    lr = i
    ot = 'SGD'
    # optimizer = Adam(learning_rate=lr)
    # optimizer = Adadelta(learning_rate=lr)
    # optimizer = Adagrad(learning_rate=lr)
    # optimizer = Adamax(learning_rate=lr)
    # optimizer = RMSprop(learning_rate=lr)
    optimizer = SGD()
    # optimizer = Nadam(learning_rate=lr)




    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    hist = model.fit(x, y, epochs=50, batch_size=2)

    #4평가
    loss, mse = model.evaluate(x, y, batch_size=2)

    y_predict = model.predict([12])
    
    print('loss: ',loss,'\t optimizer: ',ot, '\tlr',lr, '\t value:', y_predict)
    li.append(str('loss: '+str(loss)+'\t optimizer: '+str(ot)+ '\tlr'+str(lr)+ '\t value:'+ str(y_predict)))
    print(y_predict)

for i in li:
    print(i)
# for y in y_predict:
#     print(round(float(y)))

'''
loss:  7.140899106161669e-05     optimizer:  Adam       lr 0.1   value: [[13.009641]]
loss:  0.026386449113488197      optimizer:  Adam       lr 0.001         value: [[13.260897]]
loss:  14.714327812194824        optimizer:  Adam       lr 0.0001        value: [[6.170642]]

loss:  2.057147741317749         optimizer:  Adadelta   lr 0.1   value: [[10.816233]]
loss:  64.73318481445312         optimizer:  Adadelta   lr 0.001         value: [[-1.8229663]]
loss:  72.37083435058594         optimizer:  Adadelta   lr 0.0001        value: [[-2.7142265]]

loss: 5.347053593141027e-05      optimizer: Adagrad     lr0.1    value:[[13.018996]]
loss: 4.021766926598502e-06      optimizer: Adagrad     lr0.001  value:[[13.00341]]
loss: 3.695259920277749e-06      optimizer: Adagrad     lr0.0001         value:[[13.002812]]

loss: 0.00028403152828104794     optimizer: Adamax      lr0.1    value:[[13.002887]]
loss: 0.00015724023978691548     optimizer: Adamax      lr0.001  value:[[13.0111475]]
loss: 0.00013789712102152407     optimizer: Adamax      lr0.0001         value:[[13.015287]]

loss: 0.1815643012523651         optimizer: RMSprop     lr0.1    value:[[13.5161085]]
loss: 0.05402199178934097        optimizer: RMSprop     lr0.001  value:[[13.138687]]
loss: 0.048425644636154175       optimizer: RMSprop     lr0.0001         value:[[13.10718]]

loss: nan        optimizer: SGD lr0.1    value:[[nan]]
loss: nan        optimizer: SGD lr0.001  value:[[nan]]
loss: nan        optimizer: SGD lr0.0001         value:[[nan]]
loss: 8.37477970123291   optimizer: SGD(디폴) lr0.0001         value:[[6.146758]]
'''
