import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def split_x(seq, size):
        aaa = []

        # 11 - 5 =6 
        for i in range(len(seq)-  size +1): # 행 만드는 반복문 
            
            subset =seq[i : (i+size)] #dataset 에서 한칸씩 밀어나면서 size 갯수 만큼 꺼내기
            # [0 : 5] 1 2 3 4 5 
            # [1 : 6] 2 3 4 5 6  
            # [2 : 7] 3 4 5 6 7 
            # [2 : 8] 4 5 6 7 8 
            

            aaa.append([item for item in subset])  #->([subset])
            #append(1 2 3 4 5) i = 0
            #append(2 3 4 5 6) i = 1
            #append(3 4 5 6 7) i = 2 
            #append(4 5 6 7 8) i = 3
            #append(5 6 7 8 9) i = 4
            #append(6 7 8 9 10) i = 5
        print(type(aaa))
        return np.array(aaa)

dataset = np.array(range(1,101))
size = 5

data = split_x(dataset, size)

# x = dataset[0:100, 0:4] 모든 행 4번까지
# y = dataset[0:100, 4] 모든 행, 4번만

l_x = list()
l_y = list()
for i in range(len(data)):
    l_x.append(data[i][:size-1])
    l_y.append(data[i][size-1:])

x =np.array(l_x)
y =np.array(l_y).reshape(96,)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9)

#2.모델 작성

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(4,)))
model.add(Dense(300, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.save("./save/keras27.h5")
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics='mae')

elaystopping = EarlyStopping(monitor='loss', patience=50, mode='min')

hist = model.fit(x_train,y_train, batch_size=2, verbose=1, epochs=800, validation_split=0.1, callbacks=[elaystopping])

loss, mae = model.evaluate(x_test, y_test,batch_size=2)

x_pred = np.array([97,98,99,100]).reshape(1,4)
print(x_pred.shape)

y_pred = model.predict(x_pred)

print(x_pred)
print("loss: ", loss)
print("mse: ", mae)
print("y_pred: ", y_pred)

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])

plt.title('loss, & mae')
plt.ylabel('loss, mae')
plt.xlabel('epoch')
plt.legend(['trin loss','val loss', 'train mae', 'val mae'])

plt.show()

