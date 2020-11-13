import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

#1.데이터
dataset = np.array(range(1,101))
size = 5

def split_x(seq, size):
        aaa = []

        # 11 - 5 =6 
        for i in range(len(seq)-  size +1): # 행 만드는 반복문 
            subset =seq[i : (i+size)] #dataset 에서 한칸씩 밀어나면서 size 갯수 만큼 꺼내기
            aaa.append([item for item in subset])  #->([subset])
        print(type(aaa))
        return np.array(aaa)

#데이터 정제 
dataset =split_x(dataset, size)

x = dataset[0:100, 0:4]
y = dataset[0:100, 4]

x = x.reshape(x.shape[0],x.shape[1],1)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)

#모델링
model = load_model(".\save\keras27.h5")
model.add(Dense(1, name='YEEEEE')
model.summary()

# #컴파일
# model.compile(loss='mse', metrics=['acc'], optimizar='adma')
# ealystopping = EarlyStopping(patience=100, monitor='loss', mode='min')

# #학습
# model.fit(x_train, y_train, vailidation_split=0.1, epochs=1000,batch_size=2, callbacks=[ealystopping])

# #평가
# loss, mse = model.evaluate(x_test, y_test)
# x_pred = np.array([97,98,99,100]).reshape(1,4)

# #결과
# y_pred = model.predict(x_pred)
# print("=============================================")
# print(x_pred)
# print("loss: ", loss)
# print("mse: ", mse)
# print("y_pred: ", y_pred)



# #ValueError: All layers added to a
# #  Sequential model should have unique 
# # names. Name "dense" is already the name  <-이름이 이미 있단다.
# # of a layer in this model. Update the `name`
# #  argument to pass a unique name. <-이름은 유니크 해야한단다.