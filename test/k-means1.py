from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

features = [[140,1], [130,1], [160,0],[170,0]] #무게 표면 1 부드러움 0거침

data = pd.DataFrame(features, columns=['kg','surface'])

print(data)

model = KMeans(n_clusters=2)
model.fit(data)

print(model.predict(data))