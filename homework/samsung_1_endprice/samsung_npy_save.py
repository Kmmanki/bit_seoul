import pandas as pd
from keras62_split2 import spli
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#파싱 사이즈 
size = 5


###삼성
samsung =pd.read_csv('./data/csv/삼성전자 1120.csv', encoding='CP949', header=0)
samsung = samsung.sort_values(['일자'], ascending=['True'])


for i in range(len(samsung.index)):
    #일자
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace("/",""))
    #시가
    for j in range(1,14):
        if type(samsung.iloc[i,j]) == type('') and j !=5:
            samsung.iloc[i,j] = int(samsung.iloc[i,j].replace(",",""))
    # #고가
    # samsung.iloc[i,2] = int(samsung.iloc[i,2].replace(",",""))
    # #저가
    # samsung.iloc[i,3] = int(samsung.iloc[i,3].replace(",",""))
    # #종가
    # samsung.iloc[i,4] = int(samsung.iloc[i,4].replace(",",""))
    # #전일비
    # samsung.iloc[i,6] = int(samsung.iloc[i,6].replace(",",""))
    # #7은 이미 플롯임
    # samsung.iloc[i,5] = samsung.iloc[i,5].replace(",","")
    
    


#     #액면분할 18년 5월 4일20180504 < 


# print("samsung_np.shape: ",samung_np.shape)#626,14



# print(samung_np)

# print(x_samsung.shape) #620, 5, 14
# print(y_samsung.shape) #620,
 
###삼성

###비트 시작 
bit =pd.read_csv('./data/csv/비트컴퓨터 1120.csv', encoding='CP949')
bit = bit.sort_values(['일자'], ascending=['True'])

for i in range(len(bit.index)):
    #일자
    bit.iloc[i,0] = int(bit.iloc[i,0].replace("/",""))
    for j in range(1,14):
        if type(bit.iloc[i,j]) == type('') and j !=5:
            bit.iloc[i,j] = int(bit.iloc[i,j].replace(",",""))



is_samsung = samsung['일자'] > 20180504
samsung = samsung[is_samsung]

is_samsung = samsung['일자'] != 20201120 #20일 빼기
samsung = samsung[is_samsung]

samsung = samsung['시가 고가 저가 종가 등락률 거래량 금액(백만)'.split()]



#행 갯수 맞추기
is_bit = bit['일자'] > 20180504
bit = bit[is_bit]
# print(bit)

is_bit = bit['일자'] != 20201120 #20일 빼기
bit = bit[is_bit]
bit = bit['시가 고가 저가 종가 등락률 거래량 '.split()]
y_samsung = samsung.values[5:,3]

#52600시작
print(y_samsung)
# print(samsung)
scaler = MinMaxScaler()

scaler.fit(samsung)
samsung = scaler.transform(samsung)

scaler.fit(bit)
bit     = scaler.transform(bit)


x_samsung = spli(samsung, size)

x_bit = spli(bit, size)

print(x_bit.shape)# 620, 5 14

print(type(x_samsung))
print(type(y_samsung))
print(type(x_bit))
print(x_samsung[0])
print(x_bit[0])
x_samsung = x_samsung.astype('float32')
x_bit = x_bit.astype('float32')
y_samsung = y_samsung.astype('float32')

np.save("./homework/samsung_1_endprice/x_samsung.npy", x_samsung)
np.save("./homework/samsung_1_endprice/y_samsung.npy", y_samsung)
np.save("./homework/samsung_1_endprice/x_bit.npy", x_bit)