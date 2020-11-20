import pandas as pd
from keras62_split2 import spli
import numpy as np
from sklearn.preprocessing import MinMaxScaler

size = 5

samsung =pd.read_csv('./data/csv/삼성전자 1120.csv', encoding='CP949', header=0)
samsung = samsung.sort_values(['일자'], ascending=['True'])

for i in range(len(samsung.index)):
    #일자
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace("/",""))
    #시가
    for j in range(1,17):
        if type(samsung.iloc[i,j]) == type('') and j !=5:
            samsung.iloc[i,j] = float(samsung.iloc[i,j].replace(",",""))

bit =pd.read_csv('./data/csv/비트컴퓨터 1120.csv', encoding='CP949')
bit = bit.sort_values(['일자'], ascending=['True'])

for i in range(len(bit.index)):
    #일자
    bit.iloc[i,0] = int(bit.iloc[i,0].replace("/",""))
    for j in range(1,17):
        if type(bit.iloc[i,j]) == type('') and j !=5:
            bit.iloc[i,j] = float(bit.iloc[i,j].replace(",",""))


################################삼성 비트 끝
kosdak =pd.read_csv('./data/csv/코스닥.csv', encoding='CP949', header=0)
kosdak = kosdak.sort_values(['일자'], ascending=['True'])

for i in range(len(kosdak.index)):
    kosdak.iloc[i,0] = int(kosdak.iloc[i,0].replace("/",""))
    for j in range(1,15):
        if type(kosdak.iloc[i,j]) == type('') and j !=5:
            kosdak.iloc[i,j] = float(kosdak.iloc[i,j].replace(",",""))

#######삼성, 비트 floateger로 파싱 끝

gold =pd.read_csv('./data/csv/금현물.csv', encoding='CP949', header=0)
gold = gold.sort_values(['일자'], ascending=['True'])

for i in range(len(gold.index)):
    gold.iloc[i,0] = int(gold.iloc[i,0].replace("/",""))
    for j in range(1,13):
        if type(gold.iloc[i,j]) == type('') and j !=5:
            if j == 7:
                gold.iloc[i,j] = gold.iloc[i,j].replace("%",",")
                gold.iloc[i,j] = float(gold.iloc[i,j].replace(",",""))
            else:
                gold.iloc[i,j] = float(gold.iloc[i,j].replace(",",""))


print(gold)
is_samsung = samsung['일자'] > 20180504
is_bit = bit['일자'] > 20180504
is_kosdak = kosdak['일자'] >20180508
is_gold = kosdak['일자'] >20180508
bit = bit[is_bit]

samsung = samsung[is_samsung]
bit = bit[is_bit]
kosdak = kosdak[is_kosdak]
gold = gold[is_gold]

is_samsung = samsung['일자'] != 20201120 #20일 빼기
is_bit = bit['일자'] != 20201120 #20일 빼기

samsung = samsung[is_samsung]
bit = bit[is_bit]

print(samsung.shape) #(625, 17)
print(bit.shape) #(625, 17)
print(kosdak.shape)# (625, 15)
print(gold.shape) # (625, 13)

#### str -> float 완 컬럼 파싱 시작

samsung = samsung['시가 고가 저가 종가 등락률 개인 기관 외국계'.split()] #8개 컬럼
bit = bit['시가 고가 저가 종가'.split()] # 4개 컬럼
kosdak = kosdak['시가 고가 저가 현재가 상승 보합 하락'.split()] #7개 컬럼
gold = gold['시가 고가 저가 종가 개인 외국인'.split()] #6개 컬럼

y_samsung = samsung.values[5:, 0] #5번 째 이후 시작가 모두 #50200.0
scaler = MinMaxScaler()

scaler.fit(samsung)
samsung = scaler.transform(samsung)

scaler.fit(bit)
bit     = scaler.transform(bit)

scaler.fit(kosdak)
kosdak     = scaler.transform(kosdak)

scaler.fit(gold)
gold     = scaler.transform(gold)

x_samsung =spli(samsung, 5)
x_bit =spli(bit, 5)
x_kosdak=spli(kosdak, 5)
x_gold=spli(gold, 5)

x_samsung = x_samsung.astype('float32')
y_samsung = y_samsung.astype('float32')
x_bit = x_bit.astype('float32')
x_kosdak = x_kosdak.astype('float32')
x_gold = x_gold.astype('float32')

# print(x_gold[0])
# print(y_samsung[0])
# print(x_bit[0])
# print(x_kosdak[0])
# print(x_gold[0]) 전부 숫자임

np.save('./homework/samsung_2_startprice/x_samsung.npy', arr= x_samsung)
np.save('./homework/samsung_2_startprice/y_samsung.npy', arr= y_samsung)
np.save('./homework/samsung_2_startprice/x_bit.npy', arr= x_bit)
np.save('./homework/samsung_2_startprice/x_kosdak.npy', arr= x_kosdak)
np.save('./homework/samsung_2_startprice/x_gold.npy', arr= x_gold)