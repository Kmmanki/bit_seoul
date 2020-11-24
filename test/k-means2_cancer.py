from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

x, y = load_breast_cancer(return_X_y=True)


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8)

x_train = pd.DataFrame(x_train[:10])

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)

model = KMeans(n_clusters=2) # 각 그레프를 그려 군집을 만들고 분류한다. 몇개의 군집을 만들 것 인가 Ex 다중분류라면 종류 수 
model.fit(x_train)


#군집 알고리즘 = 클러스타링
result = model.predict(x_test)

plt.figure(figsize=(10,5))
plt.plot(y_test[:10])
plt.plot(result[:10])
plt.legend(['real', 'predict'])
plt.show()