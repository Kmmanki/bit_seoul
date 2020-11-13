import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

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

dataset = np.array(range(1,10))
size = 5

data = split_x(dataset, size)

x = data[0][:4]

l_x = list()
l_y = list()
for i in range(size):
    l_x.append(data[i][:size-1])
    l_y.append(data[i][size-1:])

x =np.array(l_x)
y =np.array(l_y) 

print(x)
print(y)

x = x.reshape(5,4,1)
#2.모델 작성

model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(4,1)))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics='mse')
model.fit(x,y, batch_size=1, verbose=1, epochs=100)

x_test = np.array([7,8,9,10]).reshape(4,1)
y_predict = model.predict(x_test)
print(y_predict)

