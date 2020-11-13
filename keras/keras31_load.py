import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,LSTM, Reshape
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

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)

#2.모델 작성

model = load_model("./save/keras30.h5")
model.summary()
model.add(Reshape((1,), input_shape=(4,1) ))
model.summary()

 TODO: 
# 과제 load한 모델의 Input_shape layer를 수정하여 적용하자.


# model = Sequential()
# model.add(Dense(100, activation='relu', input_shape=(4,)))
# model.add(Dense(300, activation= 'relu'))
# model.add(Dense(100, activation= 'relu'))
# model.save("./save/keras27.h5")
# model.add(Dense(1))

# model.compile(loss='mse', optimizer='adam', metrics='mse')

# elaystopping = EarlyStopping(monitor='loss', patience=125, mode='min')

# model.fit(x_train,y_train, batch_size=2, verbose=1, epochs=10000, validation_split=0.1, callbacks=[elaystopping])

# loss, mse = model.evaluate(x_test, y_test,batch_size=2)

# x_pred = np.array([97,98,99,100]).reshape(1,4)
# print(x_pred.shape)

# y_pred = model.predict(x_pred)

# print(x_pred)
# print("loss: ", loss)
# print("mse: ", mse)
# print("y_pred: ", y_pred)

