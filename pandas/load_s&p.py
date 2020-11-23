from pandas_datareader import data, wb  
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#이미 date는 빠지고 float 타입인듯
df = data.DataReader("^GSPC", "yahoo")
# print(df)

print(df[df.values.shape[0]-5:])

df = df.to_numpy()
# print(df.shape)
y = df[5:,3:4] #6번 째 close
y_now = df[df.shape[0]-1:, 3:4] # 마지막 종가

scaler1 = MinMaxScaler()
scaler1.fit(df)
df = scaler1.transform(df)

x_predict = df[df.shape[0]-5:]
print(x_predict)
def spli(data, size):
    result = [] 
    for i in range(data.shape[0]):
        if i+ size <  data.shape[0]:
            result.append(data[i: i+size])
        else:
            return np.array(result)
    return np.array(result)

x = spli(df,5)

# # 데이터 전처리 -> train_test -> scaler 

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, train_size=0.8)

model = RandomForestRegressor()
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
x_predict = x_predict.reshape(1,x_predict.shape[0]*x_predict.shape[1])

wowo = 0

for i in range(10):
    model.fit(x_train, y_train)
    model.predict(x_predict)

    r2 = model.score(x_test, y_test)
    y_predict = model.predict(x_predict)

    print("r2", r2)
    result = y_now - y_predict
    # print ("현재 종가 - 내일종가: ",y_now - y_predict)
    if result < 0:
        print('내일',np.abs(result),'만큼 오름')
        wowo += np.abs(result)
    else: 
        print('내일',np.abs(result),'만큼 떨어짐')
        wowo -= np.abs(result)

print('50회 평균', wowo/50)
