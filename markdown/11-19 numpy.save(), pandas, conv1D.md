# 11-19

키워드 

numpy.save(), pandas, conv1D

## Numpy 의 save

<a href='https://github.com/Kmmanki/bit_seoul/blob/main/keras/keras56_save_npy.py'>keras56</a>

numpy 데이터를 저장 할 수 있음 

저장 위치에 따라 전처리 전, 후 데이터를 저장 할 수 있음.

```
(x_train, y_train), (x_test, y_test) = mnist.load_data() #원본 데이터 불러오기
np.save('./data/mist_x_train', arr=x_train) #데이터 저장
x_train =np.load('./data/mist_x_train.npy') #저장된 데이터 불러오기
```

<br>

-------------------

<br>



## pandas

keras58, 59

pandas/pandas_sample

```
datasets = pd.read_csv('./data/csv/iris_ys.csv',
                        header=0, index_col=0, sep=',')

print(datasets)
=====================================================================
     sepal_length  sepal_width  petal_length  petal_width  species
1             5.1          3.5           1.4          0.2        0
2             4.9          3.0           1.4          0.2        0
3             4.7          3.2           1.3          0.2        0
4             4.6          3.1           1.5          0.2        0
5             5.0          3.6           1.4          0.2        0
..            ...          ...           ...          ...      ...
146           6.7          3.0           5.2          2.3        2
147           6.3          2.5           5.0          1.9        2
148           6.5          3.0           5.2          2.0        2
149           6.2          3.4           5.4          2.3        2
150           5.9          3.0           5.1          1.8        2
#인덱스와 헤더가 출력이 되지만 사용자 편의상 출력 된 것 뿐 실 데이터로 들어가지 않음
```

데이터의 유연성 

- pandas > numpy
- pandas -> 문자, 숫자 등 다양한 데이터 호환
- numpy -> numpy 타입의 데

속도

- pandas < numpy

```
aaa= datasets.values
aaa = datasets.to_numpy() 
#numpy가 반환 빠른 속도 및 저장을 위해 
```



<br>

-------------

<br>

keras61

- Cnv1D 
  - Conv2D(배치사이즈, 가로 , 세로, 채널) 4차원 데이터 -> 3차원 인풋
  - Conv1D 3차원데이터 -> 2차원 인풋
    - 한 행으로 만들어 연속된 값으로 다음 값을 찾을 수 있다.(어? LSTM이랑 비슷하다?)
    - LSTM은 GATE가 많기 때문에 속도가 느리지만 Conv1D는 빠름
    - Conv2D와 LSTM의 중간정도의 성능과 기능을 가지고 있다.

