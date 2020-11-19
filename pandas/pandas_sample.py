import pandas as pd
import numpy as np

from numpy.random import randn

np.random.seed(100)
data = randn(5,4)
print(data)

df = pd.DataFrame(data, index='A B C D E'.split(),       columns='가 나 다 라'.split())

print(df)

data2 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]

df2 = pd.DataFrame(data2, index=['A','B','C','D','E'], columns=['가','나','다','라'])

print(df2)

df3 = pd.DataFrame(np.array([[1,2,3],[4,5,6]]))
print(df3)

print('df2[나]: \n', df2['나']) # '나'의 모든 열을 출력함
print('df2[나,라 ]: \n', df2[['나','라']]) # '나'의 모든 열 + '라'의 모든열 #컬럼명, row명이 있다면 이름으로 주어야한다datetime A combination of a date and a time. Attributes: ()
# print('df2[0]: \n', df2[[0]]) # 에러
# print('df2.loc["나"]: \n', df2.loc['나']) # 에러 loc는 행에서만 사용
print('df2.iloc[:,2]: \n', df2.iloc[:,2]) # index location = iloc 모든 행의 3번 째 인덱스 df2['다']
# print('df2[":,2"]: \n', df2[':,2']) # 에러
print('df2.loc["A"]: \n', df2.loc['A']) # 
print('df2.loc[["A","B"]]: \n', df2.loc[['A','B']]) # ab행 모두
print('df2.iloc[[0,2]]]: \n', df2.iloc[[0,2]]) # index location = iloc 모든 행의 3번 째 인덱스 df2['다']


#행렬

print('df2.loc[["A","B"], ["가","나"]]',df2.loc[["A","B"], ["가","나"]])

print('df2.loc["E", "다"]',df2.loc["E", "다"]) #19
print("df2.iloc[4,2]", df2.iloc[4,2]) #19
print("df2.iloc[4][2]", df2.iloc[4][2]) #19

#loc 컬럼명, 인덱스명
#iloc 일반적인 index 순서