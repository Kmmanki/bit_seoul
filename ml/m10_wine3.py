import numpy as np
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#1.데이터
datasets = pd.read_csv('./data/csv/winequality-white.csv', header=0, sep=";")

#퀄리티로 그룹바이하고 그 결과의 퀄리티의 카운트
#분류할 때 유용할 것 같다
count_data = datasets.groupby('quality')['quality'].count()
print(count_data)
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''
count_data.plot()
plt.show()
