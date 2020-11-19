import numpy as np
import pandas as pd

datasets = pd.read_csv('./data/csv/iris_ys.csv',
                        header=0, index_col=0, sep=',')

# print(datasets)

# print(datasets.shape) #150, 5

# print(datasets.head()) #위에서 5개

# print(datasets.tail()) #아래서 5개


# aaa= datasets.values
aaa = datasets.to_numpy()
print(type(datasets))
print(type(aaa))
print(aaa.shape)

np.save('./data/iris_ys_pd.npy', arr=aaa)













# header None, 0 1 /index col  None, 0 1 
'''
None /None
         0             1            2             3            4        5
0      NaN  sepal_length  sepal_width  petal_length  petal_width  species
1      1.0           5.1          3.5           1.4          0.2        0
2      2.0           4.9            3           1.4          0.2        0
3      3.0           4.7          3.2           1.3          0.2        0
4      4.0           4.6          3.1           1.5          0.2        0
..     ...           ...          ...           ...          ...      ...
146  146.0           6.7            3           5.2          2.3        2
147  147.0           6.3          2.5             5          1.9        2
148  148.0           6.5            3           5.2            2        2
149  149.0           6.2          3.4           5.4          2.3        2
150  150.0           5.9            3           5.1          1.8        2

[151 rows  x 6 columns]
(151, 6)

========================================================
None / 0

                  1            2             3            4        5
0
NaN    sepal_length  sepal_width  petal_length  petal_width  species
1.0             5.1          3.5           1.4          0.2        0
2.0             4.9            3           1.4          0.2        0
3.0             4.7          3.2           1.3          0.2        0
4.0             4.6          3.1           1.5          0.2        0
...             ...          ...           ...          ...      ...
146.0           6.7            3           5.2          2.3        2
147.0           6.3          2.5             5          1.9        2
148.0           6.5            3           5.2            2        2
149.0           6.2          3.4           5.4          2.3        2
150.0           5.9            3           5.1          1.8        2

[151 rows x 5 columns]
(151, 5)

============================================================
None / 1

                  0            2             3            4        5
1
sepal_length    NaN  sepal_width  petal_length  petal_width  species
5.1             1.0          3.5           1.4          0.2        0
4.9             2.0            3           1.4          0.2        0
4.7             3.0          3.2           1.3          0.2        0
4.6             4.0          3.1           1.5          0.2        0
...             ...          ...           ...          ...      ...
6.7           146.0            3           5.2          2.3        2
6.3           147.0          2.5             5          1.9        2
6.5           148.0            3           5.2            2        2
6.2           149.0          3.4           5.4          2.3        2
5.9           150.0            3           5.1          1.8        2

[151 rows x 5 columns]
(151, 5)


================================================
0/None
     Unnamed: 0  sepal_length  sepal_width  petal_length  petal_width  species
0             1           5.1          3.5           1.4          0.2        0
1             2           4.9          3.0           1.4          0.2        0
2             3           4.7          3.2           1.3          0.2        0
3             4           4.6          3.1           1.5          0.2        0
4             5           5.0          3.6           1.4          0.2        0
..          ...           ...          ...           ...          ...      ...
145         146           6.7          3.0           5.2          2.3        2
146         147           6.3          2.5           5.0          1.9        2
147         148           6.5          3.0           5.2          2.0        2
148         149           6.2          3.4           5.4          2.3        2
149         150           5.9          3.0           5.1          1.8        2

[150 rows x 6 columns]
(150, 6)

==================================================
0/0
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

[150 rows x 5 columns]
(150, 5)





==================================================
0/1
              Unnamed: 0  sepal_width  petal_length  petal_width  species
sepal_length
5.1                    1          3.5           1.4          0.2        0
4.9                    2          3.0           1.4          0.2        0
4.7                    3          3.2           1.3          0.2        0
4.6                    4          3.1           1.5          0.2        0
5.0                    5          3.6           1.4          0.2        0
...                  ...          ...           ...          ...      ...
6.7                  146          3.0           5.2          2.3        2
6.3                  147          2.5           5.0          1.9        2
6.5                  148          3.0           5.2          2.0        2
6.2                  149          3.4           5.4          2.3        2
5.9                  150          3.0           5.1          1.8        2

[150 rows x 5 columns]
(150, 5)

=============================================================
1/None

       1  5.1  3.5  1.4  0.2  0
0      2  4.9  3.0  1.4  0.2  0
1      3  4.7  3.2  1.3  0.2  0
2      4  4.6  3.1  1.5  0.2  0
3      5  5.0  3.6  1.4  0.2  0
4      6  5.4  3.9  1.7  0.4  0
..   ...  ...  ...  ...  ... ..
144  146  6.7  3.0  5.2  2.3  2
145  147  6.3  2.5  5.0  1.9  2
146  148  6.5  3.0  5.2  2.0  2
147  149  6.2  3.4  5.4  2.3  2
148  150  5.9  3.0  5.1  1.8  2

[149 rows x 6 columns]
(149, 6)

==========================================================
1/0

     5.1  3.5  1.4  0.2  0
1
2    4.9  3.0  1.4  0.2  0
3    4.7  3.2  1.3  0.2  0
4    4.6  3.1  1.5  0.2  0
5    5.0  3.6  1.4  0.2  0
6    5.4  3.9  1.7  0.4  0
..   ...  ...  ...  ... ..
146  6.7  3.0  5.2  2.3  2
147  6.3  2.5  5.0  1.9  2
148  6.5  3.0  5.2  2.0  2
149  6.2  3.4  5.4  2.3  2
150  5.9  3.0  5.1  1.8  2

[149 rows x 5 columns]
(149, 5)

====================================================
1/1

       1  3.5  1.4  0.2  0
5.1
4.9    2  3.0  1.4  0.2  0
4.7    3  3.2  1.3  0.2  0
4.6    4  3.1  1.5  0.2  0
5.0    5  3.6  1.4  0.2  0
5.4    6  3.9  1.7  0.4  0
..   ...  ...  ...  ... ..
6.7  146  3.0  5.2  2.3  2
6.3  147  2.5  5.0  1.9  2
6.5  148  3.0  5.2  2.0  2
6.2  149  3.4  5.4  2.3  2
5.9  150  3.0  5.1  1.8  2

[149 rows x 5 columns]
(149, 5)
'''