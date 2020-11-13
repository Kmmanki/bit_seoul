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

l_x = list()
l_y = list()
for i in range(size):
    l_x.append(data[i][:size-1])
    l_y.append(data[i][size-1:])

x =np.array(l_x)
y =np.array(l_y) 

x = x.reshape(5,4,1)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)

#2.모델 작성

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(4,1)))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(300, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics='mse')

elaystopping = EarlyStopping(monitor='loss', patience=30, mode='min')

model.fit(x_train,y_train, batch_size=1, verbose=1, epochs=1000, validation_split=0.2)

loss, mse = model.evaluate(x_test, y_test)

x_pred = np.array([97,98,99,100])
x_pred = x_pred.reshape(1,4,1)
y_pred = model.predict(x_pred)


print(x_pred)
print("loss: ", loss)
print("mse: ", mse)
print("y_pred: ", y_pred)

